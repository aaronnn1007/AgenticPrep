from __future__ import annotations

"""
InterviewPA Orchestrator (LangGraph)
-----------------------------------
This module wires the existing project components into a single end-to-end pipeline:

  Media Processor → Speech Agent → Answer Segmentation →
  Relevancy Agent → Scoring Agent → Feedback Agent

IMPORTANT CONSTRAINTS (per project requirements):
- Orchestrator state is **dict-based** and **JSON-serializable**.
- Each agent is treated as an isolated node; we only pass plain dict data between nodes.
- No agent internal logic is modified here; we only integrate/wire existing components.
"""

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

# Agent graphs / wrappers (existing code; orchestrator only calls them)
from agents.speech_agent.speech_agent.graph import create_speech_agent_graph
from agents.answer_segmentation_agent.graph import build_graph as build_answer_segmentation_graph
from agents.relevancy_agent.relevancy_agent.graph import create_relevancy_graph
from agents.scoring_agent.scoring_agent.graph import ScoringGraph
from agents.feedback_agent.graph import build_feedback_graph


class OrchestratorState(TypedDict, total=False):
    """
    Shared state passed between nodes.

    Keep this purely dict-based (JSON-serializable primitives only).
    """

    # Input
    session_id: str
    video_path: str
    frame_interval_seconds: int

    # Media processor outputs
    audio_path: str
    frames_dir: str
    fps: float
    duration_seconds: float

    # Speech outputs
    question_id: str
    transcript: str
    speech_metrics: Dict[str, Any]

    # Segmentation outputs
    questions: List[Dict[str, Any]]
    segmented_answers: List[Dict[str, Any]]

    # Relevancy outputs
    relevancy_results: List[Dict[str, Any]]

    # Scoring outputs
    final_score: float
    score_label: str
    breakdown: Dict[str, Any]

    # Feedback outputs
    overall_feedback: str
    positives: List[str]
    areas_to_improve: List[str]
    actionable_suggestions: List[str]


# Build isolated subgraphs/wrappers once. They are invoked from orchestrator nodes.
_SPEECH_GRAPH = create_speech_agent_graph()
_ANSWER_SEGMENTATION_GRAPH = build_answer_segmentation_graph()
_RELEVANCY_GRAPH = create_relevancy_graph()
_SCORING = ScoringGraph()
_FEEDBACK_GRAPH = build_feedback_graph()


def media_processor_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node 1: Media Processor

    Fix 1 (Media Processor integration):
    - Do NOT import/execute the Media Processor as a LangGraph subgraph.
    - Import the existing runner function from its module entrypoint and call it directly.
    - Merge returned dict into orchestrator state with state.update(...).
    """
    # FIX 1: call Media Processor as a plain function (no subgraph wrapping)
    from media_processor.media_processor.node import media_processor  # existing working runner

    # Media processor requires: video_path, frame_interval_seconds
    frame_interval_seconds = int(state.get("frame_interval_seconds", 1))
    media_input: Dict[str, Any] = {
        "session_id": state.get("session_id", ""),
        "video_path": state["video_path"],
        "frame_interval_seconds": frame_interval_seconds,
    }

    media_output = media_processor(media_input)  # returns a JSON-serializable dict

    # FIX 1: merge results into shared state
    state.update(media_output)
    return state


def speech_agent_node(state: OrchestratorState) -> OrchestratorState:
    """Node 2: Speech Agent (audio → transcript + speech metrics)."""
    # For a dry run, treat the entire session as one question.
    question_id = state.get("question_id", "q1")

    speech_input: Dict[str, Any] = {
        "audio_path": state["audio_path"],
        "question_id": question_id,
        "transcript": "",
        "speech_metrics": {},
    }

    speech_output = _SPEECH_GRAPH.invoke(speech_input)

    state["question_id"] = question_id
    state["transcript"] = speech_output.get("transcript", "")
    state["speech_metrics"] = speech_output.get("speech_metrics", {})
    return state


def answer_segmentation_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node 3: Answer Segmentation (full transcript → segmented answers per question).

    Fix 2 (Typed object misuse):
    - Do NOT import or instantiate agent-internal TypedDict/dataclass models here.
    - Use plain dict placeholders for questions/answers.
    """
    transcript = state.get("transcript", "")
    questions = state.get("questions")

    if not questions:
        # FIX 2: placeholder questions must be plain dicts (JSON-serializable)
        questions = [
            {
                "question_id": state.get("question_id", "q1"),
                "question_text": "Interview question",
            }
        ]
        state["questions"] = questions

    seg_input: Dict[str, Any] = {
        "questions": questions,
        "full_transcript": transcript,
        "segmented_answers": [],
    }

    seg_output = _ANSWER_SEGMENTATION_GRAPH.invoke(seg_input)
    state["segmented_answers"] = seg_output.get("segmented_answers", [])
    return state


def relevancy_agent_node(state: OrchestratorState) -> OrchestratorState:
    """Node 4: Relevancy Agent (per answer → relevance scoring)."""
    questions = state.get("questions", [])
    segmented_answers = state.get("segmented_answers", [])

    if not segmented_answers:
        state["relevancy_results"] = []
        return state

    question_by_id: Dict[str, Dict[str, Any]] = {
        q.get("question_id", ""): q for q in questions if isinstance(q, dict)
    }

    results: List[Dict[str, Any]] = []
    for seg in segmented_answers:
        if not isinstance(seg, dict):
            continue

        qid = str(seg.get("question_id", ""))
        qtext = str(question_by_id.get(qid, {}).get("question_text", ""))
        answer_text = str(seg.get("answer_text", ""))

        relevancy_input: Dict[str, Any] = {
            "question_id": qid,
            "question_text": qtext,
            "transcript": answer_text,
            "relevance": "",
            "relevance_score": 0.0,
            "key_points_covered": [],
            "missing_points": [],
        }

        relevancy_output = _RELEVANCY_GRAPH.invoke(relevancy_input)
        results.append(
            {
                "question_id": relevancy_output.get("question_id", qid),
                "relevance": relevancy_output.get("relevance", "low"),
                "relevance_score": float(relevancy_output.get("relevance_score", 0.0)),
                "key_points_covered": relevancy_output.get("key_points_covered", []),
                "missing_points": relevancy_output.get("missing_points", []),
            }
        )

    state["relevancy_results"] = results
    return state


def scoring_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node 5: Scoring Agent (speech + relevancy + segmentation → numeric score).

    Fix 3 (Scoring agent invocation):
    - Prefer `.invoke(...)` if the scoring component is a LangGraph-compiled graph.
    - Otherwise call it safely via `.run(...)` if present, or as a plain callable.
    """
    scoring_input: Dict[str, Any] = {
        "speech_metrics": state.get("speech_metrics", {}),
        "relevancy_results": state.get("relevancy_results", []),
        "segmented_answers": state.get("segmented_answers", []),
    }

    # FIX 3: safest default aligned with LangGraph: use invoke if available, else fallback.
    if hasattr(_SCORING, "invoke") and callable(getattr(_SCORING, "invoke")):
        scoring_output = _SCORING.invoke(scoring_input)  # type: ignore[assignment]
    elif hasattr(_SCORING, "run") and callable(getattr(_SCORING, "run")):
        scoring_output = _SCORING.run(scoring_input)  # existing implementation
    elif callable(_SCORING):
        scoring_output = _SCORING(scoring_input)  # type: ignore[misc]
    else:
        raise TypeError("Scoring agent is not invokable (no invoke/run/callable).")

    state["final_score"] = float(scoring_output.get("final_score", 0.0))
    state["score_label"] = scoring_output.get("score_label", "")
    state["breakdown"] = scoring_output.get("breakdown", {})
    return state


def feedback_agent_node(state: OrchestratorState) -> OrchestratorState:
    """Node 6: Feedback Agent (score → human-readable feedback)."""
    feedback_input: Dict[str, Any] = {
        "final_score": float(state.get("final_score", 0.0)),
        "score_label": state.get("score_label", ""),
        "speech_metrics": state.get("speech_metrics", {}),
        "relevancy_results": [
            {"question_id": r.get("question_id", ""), "relevance": r.get("relevance", "low")}
            for r in state.get("relevancy_results", [])
            if isinstance(r, dict)
        ],
        "breakdown": state.get(
            "breakdown",
            {
                "relevance_component": 0.0,
                "speech_component": 0.0,
                "penalties": [],
            },
        ),
        "overall_feedback": None,
        "positives": None,
        "areas_to_improve": None,
        "actionable_suggestions": None,
    }

    feedback_output = _FEEDBACK_GRAPH.invoke(feedback_input)

    state["overall_feedback"] = feedback_output.get("overall_feedback", "")
    state["positives"] = feedback_output.get("positives", [])
    state["areas_to_improve"] = feedback_output.get("areas_to_improve", [])
    state["actionable_suggestions"] = feedback_output.get("actionable_suggestions", [])
    return state


def build_orchestrator_graph():
    """
    Build and compile the top-level LangGraph orchestrator.

    Node order is fixed (do not change):
      Media Processor → Speech Agent → Answer Segmentation →
      Relevancy Agent → Scoring Agent → Feedback Agent
    """
    graph = StateGraph(OrchestratorState)

    graph.add_node("media_processor", media_processor_node)
    graph.add_node("speech_agent", speech_agent_node)
    graph.add_node("answer_segmentation", answer_segmentation_node)
    graph.add_node("relevancy_agent", relevancy_agent_node)
    graph.add_node("scoring_agent", scoring_agent_node)
    graph.add_node("feedback_agent", feedback_agent_node)

    graph.set_entry_point("media_processor")

    graph.add_edge("media_processor", "speech_agent")
    graph.add_edge("speech_agent", "answer_segmentation")
    graph.add_edge("answer_segmentation", "relevancy_agent")
    graph.add_edge("relevancy_agent", "scoring_agent")
    graph.add_edge("scoring_agent", "feedback_agent")
    graph.add_edge("feedback_agent", END)

    return graph.compile()

