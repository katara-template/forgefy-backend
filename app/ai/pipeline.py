"""LangGraph extraction pipeline.

All four agent nodes run in parallel (fan-out from START → all nodes → END).
They share no state dependencies so there is no reason to run them sequentially.
LangGraph executes independent nodes in separate threads automatically.

Graph topology (parallel fan-out):
    START ──→ feature_extractor     ──→ END
          ──→ question_detector     ──→ END
          ──→ conflict_detector     ──→ END
          ──→ action_item_extractor ──→ END
"""
from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.ai.agents import (
    action_item_extractor,
    conflict_detector,
    feature_extractor,
    question_detector,
)

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────


class PipelineState(TypedDict):
    transcript: str
    api_key: str
    model: str
    # operator.add merges lists from parallel nodes without conflict.
    events: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]


# ── Node functions ────────────────────────────────────────────────────────────


def _node_feature_extractor(state: PipelineState) -> dict:
    try:
        result = feature_extractor.run(state["transcript"], state["api_key"], state["model"])
        events = [
            {"sub_state": "FEATURE_FOUND", "payload": f}
            for f in result.get("features", [])
        ]
        return {"events": events, "errors": []}
    except Exception as exc:
        logger.warning("feature_extractor failed: %s", exc)
        return {"events": [], "errors": [f"feature_extractor: {exc}"]}


def _node_question_detector(state: PipelineState) -> dict:
    try:
        result = question_detector.run(state["transcript"], state["api_key"], state["model"])
        events = [
            {"sub_state": "QUESTION_FOUND", "payload": q}
            for q in result.get("questions", [])
        ]
        return {"events": events, "errors": []}
    except Exception as exc:
        logger.warning("question_detector failed: %s", exc)
        return {"events": [], "errors": [f"question_detector: {exc}"]}


def _node_conflict_detector(state: PipelineState) -> dict:
    try:
        result = conflict_detector.run(state["transcript"], state["api_key"], state["model"])
        events = [
            {"sub_state": "CONFLICT_FOUND", "payload": c}
            for c in result.get("conflicts", [])
        ]
        return {"events": events, "errors": []}
    except Exception as exc:
        logger.warning("conflict_detector failed: %s", exc)
        return {"events": [], "errors": [f"conflict_detector: {exc}"]}


def _node_action_item_extractor(state: PipelineState) -> dict:
    try:
        result = action_item_extractor.run(state["transcript"], state["api_key"], state["model"])
        events = [
            {"sub_state": "ACTION_ITEM_FOUND", "payload": a}
            for a in result.get("action_items", [])
        ]
        return {"events": events, "errors": []}
    except Exception as exc:
        logger.warning("action_item_extractor failed: %s", exc)
        return {"events": [], "errors": [f"action_item_extractor: {exc}"]}


# ── Graph construction ────────────────────────────────────────────────────────


def build_pipeline() -> Any:
    """Compile and return the LangGraph extraction pipeline (parallel fan-out)."""
    graph: StateGraph = StateGraph(PipelineState)

    graph.add_node("feature_extractor", _node_feature_extractor)
    graph.add_node("question_detector", _node_question_detector)
    graph.add_node("conflict_detector", _node_conflict_detector)
    graph.add_node("action_item_extractor", _node_action_item_extractor)

    # Fan-out: all four nodes start from START and finish at END independently.
    # LangGraph runs nodes that share the same predecessor in parallel threads.
    for node in ("feature_extractor", "question_detector", "conflict_detector", "action_item_extractor"):
        graph.add_edge(START, node)
        graph.add_edge(node, END)

    return graph.compile()


def run_pipeline(transcript: str, api_key: str, model: str) -> list[dict]:
    """Run the extraction pipeline and return a list of extraction events.

    Each event has ``sub_state`` (one of the four FOUND types) and ``payload``.
    """
    pipeline = build_pipeline()
    final_state = pipeline.invoke(
        {
            "transcript": transcript,
            "api_key": api_key,
            "model": model,
            "events": [],
            "errors": [],
        }
    )
    if not final_state:
        return []
    errors = final_state.get("errors", [])
    if errors:
        logger.error("Pipeline agent errors: %s", errors)
    return final_state.get("events", [])
