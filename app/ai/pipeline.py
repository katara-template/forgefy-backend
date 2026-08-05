"""LangGraph extraction pipeline.

All agent nodes run in parallel (fan-out from START → all nodes → END).
They share no state dependencies so there is no reason to run them sequentially.
LangGraph executes independent nodes in separate threads automatically.

Graph topology (parallel fan-out):
    START ──→ feature_extractor     ──→ END
          ──→ question_detector     ──→ END
          ──→ conflict_detector     ──→ END
          ──→ action_item_extractor ──→ END
          ──→ data_model_extractor  ──→ END

Two entry points:
  run_pipeline    — events only; used by the meeting extraction worker.
  run_extraction  — events + per-request token usage, with optional extractor
                    subsetting; used by the developer extract API.
"""
from __future__ import annotations

import logging
import operator
from collections.abc import Iterable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.ai.agents import (
    action_item_extractor,
    conflict_detector,
    data_model_extractor,
    feature_extractor,
    question_detector,
)

logger = logging.getLogger(__name__)

# Public extractor names (as exposed by the developer API) → graph node names.
EXTRACTORS: dict[str, str] = {
    "features": "feature_extractor",
    "questions": "question_detector",
    "conflicts": "conflict_detector",
    "action_items": "action_item_extractor",
    "entities": "data_model_extractor",
}

# Event sub_state → public extractor/group name (the inverse mapping, used to
# shape API responses).
SUB_STATE_TO_GROUP: dict[str, str] = {
    "FEATURE_FOUND": "features",
    "QUESTION_FOUND": "questions",
    "CONFLICT_FOUND": "conflicts",
    "ACTION_ITEM_FOUND": "action_items",
    "ENTITY_FOUND": "entities",
}


def group_events(events: Iterable[dict], requested: Iterable[str] | None = None) -> dict:
    """Group extraction events by public extractor name, dropping unrequested ones.

    Returns ``{"features": [...], "questions": [...], ...}`` — always every
    key in EXTRACTORS, empty lists for groups with no events or not in
    ``requested``.
    """
    wanted = set(requested) if requested is not None else set(EXTRACTORS)
    groups: dict[str, list[dict]] = {name: [] for name in EXTRACTORS}
    for event in events:
        group = SUB_STATE_TO_GROUP.get(event.get("sub_state", ""))
        if group in wanted:
            groups[group].append(event.get("payload", {}))
    return groups


# ── State schema ──────────────────────────────────────────────────────────────


class PipelineState(TypedDict):
    transcript: str
    api_key: str
    model: str
    # operator.add merges lists from parallel nodes without conflict.
    events: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    usage: Annotated[list[dict], operator.add]


# ── Node functions ────────────────────────────────────────────────────────────


def _make_node(agent, result_key: str, sub_state: str, name: str):
    """Build a graph node that runs one agent and normalizes its output."""

    def _node(state: PipelineState) -> dict:
        usage: list[dict] = []
        try:
            result = agent.run(
                state["transcript"], state["api_key"], state["model"], usage=usage
            )
            events = [
                {"sub_state": sub_state, "payload": item}
                for item in result.get(result_key, [])
            ]
            return {"events": events, "errors": [], "usage": usage}
        except Exception as exc:
            logger.warning("%s failed: %s", name, exc)
            return {"events": [], "errors": [f"{name}: {exc}"], "usage": usage}

    return _node


_NODES = {
    "feature_extractor": _make_node(
        feature_extractor, "features", "FEATURE_FOUND", "feature_extractor"
    ),
    "question_detector": _make_node(
        question_detector, "questions", "QUESTION_FOUND", "question_detector"
    ),
    "conflict_detector": _make_node(
        conflict_detector, "conflicts", "CONFLICT_FOUND", "conflict_detector"
    ),
    "action_item_extractor": _make_node(
        action_item_extractor, "action_items", "ACTION_ITEM_FOUND", "action_item_extractor"
    ),
    "data_model_extractor": _make_node(
        data_model_extractor, "entities", "ENTITY_FOUND", "data_model_extractor"
    ),
}


# ── Graph construction ────────────────────────────────────────────────────────


def build_pipeline(extractors: Iterable[str] | None = None) -> Any:
    """Compile and return the LangGraph extraction pipeline (parallel fan-out).

    ``extractors`` takes public names from EXTRACTORS (e.g. ["features"]);
    None means all of them.
    """
    if extractors is None:
        node_names = list(_NODES)
    else:
        node_names = [EXTRACTORS[e] for e in extractors]
        if not node_names:
            raise ValueError("extractors must not be empty")

    graph: StateGraph = StateGraph(PipelineState)

    # Fan-out: all requested nodes start from START and finish at END
    # independently. LangGraph runs nodes that share the same predecessor in
    # parallel threads.
    for name in node_names:
        graph.add_node(name, _NODES[name])
        graph.add_edge(START, name)
        graph.add_edge(name, END)

    return graph.compile()


def run_extraction(
    transcript: str,
    api_key: str,
    model: str,
    extractors: Iterable[str] | None = None,
) -> dict:
    """Run the pipeline and return events, aggregate token usage, and errors.

    Returns ``{"events": [...], "usage": {"input_tokens", "output_tokens"},
    "errors": [...]}``. Each event has ``sub_state`` (one of the FOUND types
    in SUB_STATE_TO_GROUP) and ``payload``.
    """
    pipeline = build_pipeline(extractors)
    final_state = pipeline.invoke(
        {
            "transcript": transcript,
            "api_key": api_key,
            "model": model,
            "events": [],
            "errors": [],
            "usage": [],
        }
    )
    if not final_state:
        return {"events": [], "usage": {"input_tokens": 0, "output_tokens": 0}, "errors": []}

    errors = final_state.get("errors", [])
    if errors:
        logger.error("Pipeline agent errors: %s", errors)

    usage_entries = final_state.get("usage", [])
    return {
        "events": final_state.get("events", []),
        "usage": {
            "input_tokens": sum(u.get("input_tokens", 0) for u in usage_entries),
            "output_tokens": sum(u.get("output_tokens", 0) for u in usage_entries),
        },
        "errors": errors,
    }


def run_pipeline(transcript: str, api_key: str, model: str) -> list[dict]:
    """Run the extraction pipeline and return a list of extraction events.

    Each event has ``sub_state`` (one of the FOUND types in SUB_STATE_TO_GROUP)
    and ``payload``.
    """
    return run_extraction(transcript, api_key, model)["events"]
