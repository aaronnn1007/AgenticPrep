"""
LangGraph Workflow
==================
Orchestrates all agents in the interview analysis pipeline.

Architecture:
- LangGraph StateGraph with InterviewState
- Sequential + Parallel execution where possible
- No LLM-based orchestration - pure graph logic
- Deterministic flow
- Comprehensive error handling and logging
- Supports single-question and multi-question modes

Workflow (single question):
START → QuestionGeneration → [VoiceAgent → AnswerQuality] (chain) + [BodyLanguage] (parallel)
      → ConfidenceInference → ScoringAggregation → Recommendation → END

Analysis-only subgraph (per question in multi-question mode):
START → VoiceAgent → AnswerQuality + BodyLanguage → ConfidenceInference → ScoringAggregation → END

Note: AnswerQuality MUST run after VoiceAgent because it needs the transcript.
      Running them in parallel causes empty transcript → all scores = 0.0.
"""

import logging
from typing import Dict, Any, List, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END

from backend.models.state import (
    InterviewState, QuestionModel, QuestionResult,
    ScoresModel, RecommendationsModel,
    VoiceAnalysisModel, AnswerQualityModel, BodyLanguageModel,
    ConfidenceBehaviorModel,
)
from backend.agents.question_generation import question_generation_node
from backend.agents.voice_agent import voice_agent_node
from backend.agents.answer_quality import answer_quality_node
from backend.agents.body_language_agent import body_language_node
from backend.agents.confidence_inference import confidence_behavior_node
from backend.agents.scoring_aggregation import scoring_aggregation_node
from backend.agents.recommendation_system import recommendation_node

logger = logging.getLogger(__name__)


class InterviewAnalyzerGraph:
    """
    Main LangGraph workflow for interview analysis.

    Design principles:
    - LangGraph handles orchestration (not an LLM)
    - Agents run in optimal order with parallelism where safe
    - State is passed immutably through graph
    - Each node is a single-responsibility agent
    """

    def __init__(self):
        """Initialize and compile the graph."""
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph.

        Flow:
        1. question_generation: Generate interview question
        2. [parallel] voice_agent + body_language: Start audio/video analysis
        3. answer_quality: Runs AFTER voice_agent (needs transcript)
        4. confidence_behavior: Synthesize behavioral traits (after step 2 & 3)
        5. scoring_aggregation: Compute deterministic scores
        6. recommendation: Generate actionable feedback

        CRITICAL: answer_quality depends on transcript from voice_agent.
        Running them in parallel causes transcript="" → all scores = 0.

        Returns:
            Compiled StateGraph
        """
        logger.info("Building interview analyzer graph")

        # Create graph with InterviewState as state type
        # Note: LangGraph works with dicts, so we use dict as state schema
        workflow = StateGraph(
            InterviewState,
            # This tells LangGraph to merge updates field-by-field
            # instead of treating the whole state as one value
        )

        # Add nodes for each agent
        workflow.add_node("question_generation", question_generation_node)
        workflow.add_node("voice_agent", voice_agent_node)
        workflow.add_node("answer_quality", answer_quality_node)
        workflow.add_node("body_language", body_language_node)
        workflow.add_node("confidence_behavior", confidence_behavior_node)
        workflow.add_node("scoring_aggregation", scoring_aggregation_node)
        workflow.add_node("recommendation", recommendation_node)

        # Set entry point
        workflow.set_entry_point("question_generation")

        # Define edges (workflow sequence)
        # voice_agent runs first — it transcribes the audio and writes state.transcript.
        # answer_quality must run AFTER voice_agent so it receives a populated transcript;
        # running it in parallel meant transcript="" when the LLM was called, which
        # caused all answer-quality metrics (and therefore all scores) to be 0.
        # body_language is independent of transcript so it stays parallel with voice_agent.
        workflow.add_edge("question_generation", "voice_agent")
        workflow.add_edge("question_generation", "body_language")

        # answer_quality depends on the transcript produced by voice_agent
        workflow.add_edge("voice_agent", "answer_quality")

        # After answer_quality and body_language both complete, run confidence inference.
        # Note: voice_agent → answer_quality → confidence_behavior is the chain, so we
        # do NOT add a direct voice_agent → confidence_behavior edge (that would cause
        # a concurrent update error since confidence_behavior would be triggered twice).
        workflow.add_edge("answer_quality", "confidence_behavior")
        workflow.add_edge("body_language", "confidence_behavior")

        # Then scoring (deterministic)
        workflow.add_edge("confidence_behavior", "scoring_aggregation")

        # Then recommendations
        workflow.add_edge("scoring_aggregation", "recommendation")

        # Finally end
        workflow.add_edge("recommendation", END)

        logger.info("Graph built successfully")
        return workflow

    def run(self, initial_state: InterviewState) -> InterviewState:
        """
        Execute the full workflow.

        Args:
            initial_state: Initial interview state with at minimum:
                - interview_id
                - role
                - experience_level
                - audio_path (for analysis)
                - video_path (optional, for body language)

        Returns:
            Final state with all analyses complete
        """
        logger.info(
            f"Starting workflow for interview_id={initial_state.interview_id}")

        # Convert to dict for LangGraph
        state_dict = initial_state.model_dump()

        # Run graph
        final_state_dict = self.compiled_graph.invoke(state_dict)

        # Convert back to Pydantic model
        final_state = InterviewState(**final_state_dict)

        logger.info(
            f"Workflow complete. Overall score: {final_state.scores.overall}")

        return final_state

    async def arun(self, initial_state: InterviewState) -> InterviewState:
        """
        Async execution of workflow.

        Useful for handling multiple interviews concurrently.
        """
        logger.info(
            f"Starting async workflow for interview_id={initial_state.interview_id}")

        state_dict = initial_state.model_dump()
        final_state_dict = await self.compiled_graph.ainvoke(state_dict)
        final_state = InterviewState(**final_state_dict)

        logger.info(
            f"Async workflow complete. Overall score: {final_state.scores.overall}")

        return final_state


# Global graph instance
_graph_instance = None


def get_graph() -> InterviewAnalyzerGraph:
    """
    Get singleton graph instance.

    This ensures we only build the graph once and reuse it.
    """
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = InterviewAnalyzerGraph()
    return _graph_instance


def run_interview_analysis(
    interview_id: str,
    role: str,
    experience_level: str,
    audio_path: str,
    video_path: str = None
) -> InterviewState:
    """
    Convenience function to run full interview analysis (single question).

    Args:
        interview_id: Unique interview identifier
        role: Job role
        experience_level: Candidate experience level
        audio_path: Path to audio recording
        video_path: Optional path to video recording

    Returns:
        Complete analysis results
    """
    # Create initial state
    initial_state = InterviewState(
        interview_id=interview_id,
        role=role,
        experience_level=experience_level,
        audio_path=audio_path,
        video_path=video_path
    )

    # Get graph and run
    graph = get_graph()
    return graph.run(initial_state)


# ==========================================================================
# ANALYSIS-ONLY GRAPH (no question generation / recommendation)
# ==========================================================================

class AnalysisOnlyGraph:
    """
    Subgraph that runs only the analysis pipeline for a single question.

    Used in multi-question mode where question generation is handled
    separately and recommendation runs once after all questions.

    Flow:
    START → voice_agent → answer_quality ─┐
    START → body_language ─────────────────┤
                                           ↓
                                  confidence_behavior
                                           ↓
                                  scoring_aggregation → END
    """

    def __init__(self):
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        logger.info("Building analysis-only graph")

        workflow = StateGraph(InterviewState)

        # Nodes (no question_generation or recommendation)
        workflow.add_node("voice_agent", voice_agent_node)
        workflow.add_node("answer_quality", answer_quality_node)
        workflow.add_node("body_language", body_language_node)
        workflow.add_node("confidence_behavior", confidence_behavior_node)
        workflow.add_node("scoring_aggregation", scoring_aggregation_node)

        # Entry: voice_agent and body_language run in parallel
        workflow.set_entry_point("voice_agent")
        # body_language also starts from the beginning via a parallel edge
        workflow.add_edge("__start__", "body_language")

        # answer_quality depends on transcript from voice_agent
        workflow.add_edge("voice_agent", "answer_quality")

        # confidence_behavior waits for both answer_quality and body_language
        workflow.add_edge("answer_quality", "confidence_behavior")
        workflow.add_edge("body_language", "confidence_behavior")

        # scoring
        workflow.add_edge("confidence_behavior", "scoring_aggregation")
        workflow.add_edge("scoring_aggregation", END)

        logger.info("Analysis-only graph built successfully")
        return workflow

    def run(self, state: InterviewState) -> InterviewState:
        logger.info(
            f"Running analysis-only graph for interview_id={state.interview_id}")
        state_dict = state.model_dump()
        final_state_dict = self.compiled_graph.invoke(state_dict)
        return InterviewState(**final_state_dict)


_analysis_graph_instance = None


def get_analysis_graph() -> AnalysisOnlyGraph:
    """Get singleton analysis-only graph instance."""
    global _analysis_graph_instance
    if _analysis_graph_instance is None:
        _analysis_graph_instance = AnalysisOnlyGraph()
    return _analysis_graph_instance


# ==========================================================================
# MULTI-QUESTION ORCHESTRATION
# ==========================================================================

def run_multi_question_analysis(
    interview_id: str,
    role: str,
    experience_level: str,
    questions: List[QuestionModel],
    audio_paths: List[str],
    video_paths: List[str] = None,
) -> InterviewState:
    """
    Run the analysis pipeline for each question, then aggregate scores
    and generate a single set of recommendations.

    Args:
        interview_id: Unique interview identifier
        role: Job role
        experience_level: Candidate experience level
        questions: List of QuestionModel (one per question)
        audio_paths: List of audio file paths (one per question)
        video_paths: Optional list of video file paths

    Returns:
        InterviewState with question_results populated, aggregate scores,
        and holistic recommendations.
    """
    logger.info(
        f"Starting multi-question analysis: interview_id={interview_id}, "
        f"num_questions={len(questions)}"
    )

    analysis_graph = get_analysis_graph()
    question_results: List[QuestionResult] = []

    for i, question in enumerate(questions):
        audio_path = audio_paths[i] if i < len(audio_paths) else None
        video_path = (video_paths[i] if video_paths and i < len(
            video_paths) else None)

        if not audio_path:
            logger.warning(f"Question {i}: No audio path, skipping analysis")
            question_results.append(QuestionResult(question=question))
            continue

        logger.info(
            f"Analyzing question {i + 1}/{len(questions)}: {question.topic}")

        # Build a per-question state
        per_q_state = InterviewState(
            interview_id=interview_id,
            role=role,
            experience_level=experience_level,
            question=question,
            audio_path=audio_path,
            video_path=video_path,
        )

        # Run analysis subgraph
        result_state = analysis_graph.run(per_q_state)

        # Collect per-question result
        qr = QuestionResult(
            question=question,
            transcript=result_state.transcript,
            voice_analysis=result_state.voice_analysis,
            answer_quality=result_state.answer_quality,
            body_language=result_state.body_language,
            confidence_behavior=result_state.confidence_behavior,
            scores=result_state.scores,
        )
        question_results.append(qr)
        logger.info(f"Question {i + 1} done — overall={qr.scores.overall:.1f}")

    # ── Aggregate scores across all questions ──
    aggregate_scores = _aggregate_scores(question_results)

    # ── Generate holistic recommendations ──
    aggregate_recommendations = _generate_holistic_recommendations(
        question_results, aggregate_scores
    )

    # Build final state
    final_state = InterviewState(
        interview_id=interview_id,
        role=role,
        experience_level=experience_level,
        num_questions=len(questions),
        question_results=question_results,
        scores=aggregate_scores,
        recommendations=aggregate_recommendations,
    )

    logger.info(
        f"Multi-question analysis complete: overall={aggregate_scores.overall:.1f}"
    )
    return final_state


def _aggregate_scores(question_results: List[QuestionResult]) -> ScoresModel:
    """Average per-question scores into aggregate scores."""
    if not question_results:
        return ScoresModel()

    scored = [qr for qr in question_results if qr.scores.overall > 0]
    if not scored:
        return ScoresModel()

    n = len(scored)
    return ScoresModel(
        technical=round(sum(qr.scores.technical for qr in scored) / n, 2),
        communication=round(
            sum(qr.scores.communication for qr in scored) / n, 2),
        behavioral=round(sum(qr.scores.behavioral for qr in scored) / n, 2),
        overall=round(sum(qr.scores.overall for qr in scored) / n, 2),
    )


def _generate_holistic_recommendations(
    question_results: List[QuestionResult],
    aggregate_scores: ScoresModel,
) -> RecommendationsModel:
    """Run the recommendation agent with data from all questions."""
    from backend.agents.recommendation_system import RecommendationSystemAgent

    # Build a summary of all per-question answer quality metrics
    all_relevance = []
    all_correctness = []
    all_depth = []
    all_structure = []
    all_gaps: List[str] = []

    for qr in question_results:
        aq = qr.answer_quality
        all_relevance.append(aq.relevance)
        all_correctness.append(aq.correctness)
        all_depth.append(aq.depth)
        all_structure.append(aq.structure)
        all_gaps.extend(aq.gaps)

    n = len(question_results) or 1
    avg_answer_quality = {
        "relevance": round(sum(all_relevance) / n, 3),
        "correctness": round(sum(all_correctness) / n, 3),
        "depth": round(sum(all_depth) / n, 3),
        "structure": round(sum(all_structure) / n, 3),
        "gaps": list(set(all_gaps)),  # deduplicate
    }

    scores_dict = {
        "technical": aggregate_scores.technical,
        "communication": aggregate_scores.communication,
        "behavioral": aggregate_scores.behavioral,
        "overall": aggregate_scores.overall,
    }

    try:
        agent = RecommendationSystemAgent()
        result = agent.generate(scores_dict, avg_answer_quality)
        if "recommendations" in result:
            result = result["recommendations"]
        return RecommendationsModel(**result)
    except Exception as e:
        logger.error(f"Holistic recommendation failed: {e}", exc_info=True)
        return RecommendationsModel(
            strengths=["Analysis completed across multiple questions"],
            weaknesses=[
                "Detailed recommendation generation encountered an error"],
            improvement_plan=[
                "Review per-question feedback for specific areas",
                "Practice answering under timed conditions",
                "Focus on areas with lowest per-question scores",
            ],
        )
