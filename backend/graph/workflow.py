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

Workflow:
START → QuestionGeneration → [VoiceAgent, AnswerQuality, BodyLanguage] (parallel) 
      → ConfidenceBehavior → ScoringAggregation → Recommendation → END
"""

import logging
from typing import Dict, Any, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END

from backend.models.state import InterviewState
from backend.agents.question_generation import question_generation_node
from backend.agents.voice_agent import voice_agent_node
from backend.agents.answer_quality import answer_quality_node
from backend.agents.body_language import body_language_node
from backend.agents.confidence_inference import confidence_behavior_node
from backend.agents.scoring_aggregation import scoring_aggregation_node
from backend.agents.recommendation import recommendation_node

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
        2. [parallel] voice_agent, answer_quality, body_language: Independent analyses
        3. confidence_behavior: Synthesize behavioral traits
        4. scoring_aggregation: Compute deterministic scores
        5. recommendation: Generate actionable feedback

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
        # After question generation, run three analyses in parallel
        workflow.add_edge("question_generation", "voice_agent")
        workflow.add_edge("question_generation", "answer_quality")
        workflow.add_edge("question_generation", "body_language")

        # After all three complete, run confidence inference
        # Note: In LangGraph, parallel branches automatically wait for all to complete
        workflow.add_edge("voice_agent", "confidence_behavior")
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
    Convenience function to run full interview analysis.

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
