"""LangGraph extraction pipeline.

State flows through four sequential nodes (one per agent type).  Each node
calls Claude independently; events accumulate via the list-append reducer.

Graph topology:
    START → feature_extractor → question_detector → conflict_detector
          → action_item_extractor → END
"""
from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

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
    # Annotated with operator.add so LangGraph merges lists from each node.
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
    """Compile and return the LangGraph extraction pipeline."""
    graph: StateGraph = StateGraph(PipelineState)

    graph.add_node("feature_extractor", _node_feature_extractor)
    graph.add_node("question_detector", _node_question_detector)
    graph.add_node("conflict_detector", _node_conflict_detector)
    graph.add_node("action_item_extractor", _node_action_item_extractor)

    graph.set_entry_point("feature_extractor")
    graph.add_edge("feature_extractor", "question_detector")
    graph.add_edge("question_detector", "conflict_detector")
    graph.add_edge("conflict_detector", "action_item_extractor")
    graph.add_edge("action_item_extractor", END)

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
    return final_state.get("events", []) if final_state else []
