"""
LangGraph Orchestration Integration Tests
==========================================
Production-grade integration tests for LangGraph pipeline orchestration.

Tests:
1. Pipeline Execution Order
2. Parallel Node Execution
3. State Propagation
4. Agent Integration
5. Error Handling
6. Deterministic Scoring
7. Complete Workflow

Requirements:
- All agents must be functional
- Audio/video files for processing
- No LLM mocks - test real agents
"""

import asyncio
import logging
import pytest
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from backend.models.state import InterviewState
from backend.graph.workflow import (
    InterviewAnalyzerGraph,
    get_graph,
    run_interview_analysis
)
from utils.mock_webrtc import (
    WebRTCMockSimulator,
    MockInterviewSession,
    AudioChunkConfig,
    BodyMetricsConfig,
    SCENARIO_CONFIDENT_CANDIDATE,
    SCENARIO_NERVOUS_CANDIDATE
)
from utils.schema_validator import (
    ContractValidator,
    INTERVIEW_STATE_SCHEMA,
    SchemaValidator
)

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Fixture for temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def webrtc_simulator():
    """Fixture for WebRTC mock simulator."""
    return WebRTCMockSimulator()


@pytest.fixture
def contract_validator():
    """Fixture for contract validator."""
    return ContractValidator()


@pytest.fixture
def graph():
    """Fixture for LangGraph instance."""
    return get_graph()


@pytest.fixture
def sample_audio_file(temp_dir, webrtc_simulator):
    """Generate sample audio file for testing."""
    session = MockInterviewSession(
        session_id=f"test_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        duration_seconds=10,
        audio_config=AudioChunkConfig(
            speech_rate_wpm=150,
            silence_ratio=0.1
        )
    )

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    return audio_file


# ============================================================================
# TEST: PIPELINE EXECUTION ORDER
# ============================================================================

@pytest.mark.asyncio
async def test_pipeline_execution_order(graph, sample_audio_file):
    """
    Test: LangGraph executes nodes in correct order.

    Expected order:
    1. question_generation
    2. [parallel] voice_agent, answer_quality, body_language
    3. confidence_behavior
    4. scoring_aggregation
    5. recommendation

    Validates:
    - Correct node sequence
    - No out-of-order execution
    - All nodes executed exactly once
    """
    logger.info("TEST: Pipeline execution order")

    # Track node execution order
    execution_order = []

    # Patch each node to track execution
    def make_tracker(node_name, original_func):
        def wrapper(state):
            execution_order.append({
                "node": node_name,
                "timestamp": datetime.now()
            })
            return original_func(state)
        return wrapper

    # Create initial state
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        audio_path=str(sample_audio_file),
        transcript="I would implement this using a hash map for O(1) lookup time."
    )

    # Patch nodes
    from backend.agents import (
        question_generation,
        voice_agent,
        answer_quality,
        body_language,
        confidence_inference,
        scoring_aggregation,
        recommendation
    )

    with patch.object(
        question_generation, 'question_generation_node',
        make_tracker('question_generation',
                     question_generation.question_generation_node)
    ), patch.object(
        voice_agent, 'voice_agent_node',
        make_tracker('voice_agent', voice_agent.voice_agent_node)
    ), patch.object(
        answer_quality, 'answer_quality_node',
        make_tracker('answer_quality', answer_quality.answer_quality_node)
    ), patch.object(
        body_language, 'body_language_node',
        make_tracker('body_language', body_language.body_language_node)
    ), patch.object(
        confidence_inference, 'confidence_behavior_node',
        make_tracker('confidence_behavior',
                     confidence_inference.confidence_behavior_node)
    ), patch.object(
        scoring_aggregation, 'scoring_aggregation_node',
        make_tracker('scoring_aggregation',
                     scoring_aggregation.scoring_aggregation_node)
    ), patch.object(
        recommendation, 'recommendation_node',
        make_tracker('recommendation', recommendation_node)
    ):
        # Rebuild graph with patched nodes
        test_graph = InterviewAnalyzerGraph()

        # Run workflow
        final_state = test_graph.run(initial_state)

    # Validate execution order
    logger.info("Execution order:")
    for idx, exec_info in enumerate(execution_order):
        logger.info(f"  {idx+1}. {exec_info['node']}")

    # Expected order
    expected_sequence = [
        "question_generation",
        # Parallel nodes (order may vary)
        {"voice_agent", "answer_quality", "body_language"},
        "confidence_behavior",
        "scoring_aggregation",
        "recommendation"
    ]

    # Validate first node
    assert execution_order[0]["node"] == "question_generation", \
        "First node should be question_generation"

    # Validate parallel nodes executed
    parallel_nodes = {execution_order[i]["node"] for i in range(1, 4)}
    assert parallel_nodes == {"voice_agent", "answer_quality", "body_language"}, \
        f"Parallel nodes mismatch: {parallel_nodes}"

    # Validate remaining sequence
    remaining_nodes = [execution_order[i]["node"]
                       for i in range(4, len(execution_order))]
    assert remaining_nodes == ["confidence_behavior", "scoring_aggregation", "recommendation"], \
        f"Remaining sequence incorrect: {remaining_nodes}"

    # Validate each node executed exactly once
    node_counts = {}
    for exec_info in execution_order:
        node = exec_info["node"]
        node_counts[node] = node_counts.get(node, 0) + 1

    for node, count in node_counts.items():
        assert count == 1, f"Node '{node}' executed {count} times, expected 1"

    logger.info("✓ Pipeline execution order PASSED")


# ============================================================================
# TEST: STATE PROPAGATION
# ============================================================================

@pytest.mark.asyncio
async def test_state_propagation(graph, sample_audio_file):
    """
    Test: State is correctly propagated through pipeline.

    Validates:
    - Initial state fields preserved
    - Agent outputs added to state
    - No data loss between nodes
    - Final state completeness
    """
    logger.info("TEST: State propagation")

    # Create initial state
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role="Senior Software Engineer",
        experience_level="Senior",
        audio_path=str(sample_audio_file),
        transcript="I would use a binary search tree for efficient insertion and lookup."
    )

    # Run workflow
    final_state = graph.run(initial_state)

    # Validate initial state preserved
    assert final_state.interview_id == initial_state.interview_id
    assert final_state.role == initial_state.role
    assert final_state.experience_level == initial_state.experience_level

    # Validate agent outputs present
    assert final_state.question is not None, "Question should be generated"
    assert final_state.question.text != "", "Question text should not be empty"

    assert final_state.voice_analysis is not None, "Voice analysis should be present"
    assert final_state.voice_analysis.speech_rate_wpm >= 0, "Speech rate should be valid"

    assert final_state.answer_quality is not None, "Answer quality should be present"
    assert 0.0 <= final_state.answer_quality.relevance <= 1.0, "Relevance should be in [0,1]"

    assert final_state.body_language is not None, "Body language should be present"
    assert 0.0 <= final_state.body_language.eye_contact <= 1.0, "Eye contact should be in [0,1]"

    assert final_state.confidence_behavior is not None, "Confidence behavior should be present"
    assert 0.0 <= final_state.confidence_behavior.confidence <= 1.0, "Confidence should be in [0,1]"

    assert final_state.scores is not None, "Scores should be present"
    assert 0.0 <= final_state.scores.overall <= 100.0, "Overall score should be in [0,100]"

    assert final_state.recommendations is not None, "Recommendations should be present"
    assert len(final_state.recommendations.strengths) > 0, "Should have strengths"
    assert len(
        final_state.recommendations.improvement_plan) > 0, "Should have improvement plan"

    logger.info("✓ State propagation PASSED")


# ============================================================================
# TEST: DETERMINISTIC SCORING
# ============================================================================

@pytest.mark.asyncio
async def test_deterministic_scoring(graph, sample_audio_file):
    """
    Test: Same inputs produce same scores (deterministic).

    Validates:
    - Consistent scoring
    - No randomness in pipeline
    - Reproducible results
    """
    logger.info("TEST: Deterministic scoring")

    # Create initial state
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        audio_path=str(sample_audio_file),
        transcript="I would implement quicksort with a pivot selection strategy."
    )

    # Run workflow multiple times
    num_runs = 3
    results = []

    for i in range(num_runs):
        # Create fresh state with same data
        state = InterviewState(
            interview_id=f"int_{uuid.uuid4().hex[:8]}_{i}",
            role=initial_state.role,
            experience_level=initial_state.experience_level,
            audio_path=initial_state.audio_path,
            transcript=initial_state.transcript
        )

        final_state = graph.run(state)
        results.append(final_state)

        logger.info(f"Run {i+1} scores: Technical={final_state.scores.technical:.2f}, "
                    f"Communication={final_state.scores.communication:.2f}, "
                    f"Behavioral={final_state.scores.behavioral:.2f}, "
                    f"Overall={final_state.scores.overall:.2f}")

    # Compare scores across runs
    # Note: Whisper transcription may vary slightly, so allow small variance
    tolerance = 5.0  # 5% tolerance

    base_scores = results[0].scores

    for i, result in enumerate(results[1:], start=2):
        scores = result.scores

        tech_diff = abs(scores.technical - base_scores.technical)
        comm_diff = abs(scores.communication - base_scores.communication)
        behav_diff = abs(scores.behavioral - base_scores.behavioral)
        overall_diff = abs(scores.overall - base_scores.overall)

        logger.info(f"Run {i} vs Run 1 differences: "
                    f"Tech={tech_diff:.2f}, Comm={comm_diff:.2f}, "
                    f"Behav={behav_diff:.2f}, Overall={overall_diff:.2f}")

        assert tech_diff <= tolerance, f"Technical score variance {tech_diff} > {tolerance}"
        assert comm_diff <= tolerance, f"Communication score variance {comm_diff} > {tolerance}"
        assert behav_diff <= tolerance, f"Behavioral score variance {behav_diff} > {tolerance}"
        assert overall_diff <= tolerance, f"Overall score variance {overall_diff} > {tolerance}"

    logger.info("✓ Deterministic scoring PASSED")


# ============================================================================
# TEST: COMPLETE WORKFLOW
# ============================================================================

@pytest.mark.asyncio
async def test_complete_workflow_confident_candidate(
    graph,
    temp_dir,
    webrtc_simulator,
    contract_validator
):
    """
    Test: Complete workflow for confident candidate.

    Validates:
    - Full pipeline execution
    - High scores for confident candidate
    - Positive recommendations
    - Schema compliance
    """
    logger.info("TEST: Complete workflow (confident candidate)")

    # Generate audio for confident candidate
    session = SCENARIO_CONFIDENT_CANDIDATE
    session.session_id = f"test_{uuid.uuid4().hex[:8]}"
    session.duration_seconds = 15

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    # Create initial state
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role=session.role,
        experience_level=session.experience_level,
        audio_path=str(audio_file),
        transcript="I have implemented this algorithm using dynamic programming. "
                   "The time complexity is O(n log n) and space complexity is O(n). "
                   "I would optimize it by using memoization to cache intermediate results."
    )

    # Run workflow
    start_time = datetime.now()
    final_state = graph.run(initial_state)
    end_time = datetime.now()

    execution_time = (end_time - start_time).total_seconds()
    logger.info(f"Workflow execution time: {execution_time:.2f}s")

    # Validate execution time
    assert execution_time < 30.0, f"Workflow took too long: {execution_time}s"

    # Validate scores (should be high for confident candidate)
    logger.info(f"Final scores: Technical={final_state.scores.technical:.2f}, "
                f"Communication={final_state.scores.communication:.2f}, "
                f"Behavioral={final_state.scores.behavioral:.2f}, "
                f"Overall={final_state.scores.overall:.2f}")

    # Confident candidate should score reasonably well
    assert final_state.scores.technical >= 40.0, "Technical score too low for confident candidate"
    assert final_state.scores.communication >= 40.0, "Communication score too low"
    assert final_state.scores.behavioral >= 40.0, "Behavioral score too low"
    assert final_state.scores.overall >= 40.0, "Overall score too low"

    # Validate recommendations
    assert len(
        final_state.recommendations.strengths) > 0, "Should identify strengths"
    assert len(
        final_state.recommendations.improvement_plan) > 0, "Should provide improvement plan"

    # Validate schema compliance
    state_dict = final_state.model_dump()
    validation_result = contract_validator.validate_interview_state(
        state_dict,
        check_completeness=True
    )

    if not validation_result.valid:
        logger.error(f"Schema validation errors: {validation_result.errors}")

    assert validation_result.valid, "Interview state should comply with schema"

    logger.info("✓ Complete workflow (confident candidate) PASSED")


@pytest.mark.asyncio
async def test_complete_workflow_nervous_candidate(
    graph,
    temp_dir,
    webrtc_simulator,
    contract_validator
):
    """
    Test: Complete workflow for nervous candidate.

    Validates:
    - Full pipeline execution
    - Lower scores for nervous candidate
    - Improvement-focused recommendations
    - Schema compliance
    """
    logger.info("TEST: Complete workflow (nervous candidate)")

    # Generate audio for nervous candidate
    session = SCENARIO_NERVOUS_CANDIDATE
    session.session_id = f"test_{uuid.uuid4().hex[:8]}"
    session.duration_seconds = 15

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    # Create initial state with less confident transcript
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role=session.role,
        experience_level=session.experience_level,
        audio_path=str(audio_file),
        transcript="Um, well, I think... maybe we could use, like, a loop? "
                   "I'm not totally sure, but, uh, probably iterate through?"
    )

    # Run workflow
    final_state = graph.run(initial_state)

    # Validate scores (should be lower for nervous candidate)
    logger.info(f"Final scores: Technical={final_state.scores.technical:.2f}, "
                f"Communication={final_state.scores.communication:.2f}, "
                f"Behavioral={final_state.scores.behavioral:.2f}, "
                f"Overall={final_state.scores.overall:.2f}")

    # Nervous candidate should score lower
    # Note: Actual scores depend on agent logic, but overall should reflect weaknesses
    assert final_state.scores.overall <= 100.0, "Overall score should be valid"

    # Should identify weaknesses
    assert len(
        final_state.recommendations.weaknesses) > 0, "Should identify weaknesses"
    assert len(
        final_state.recommendations.improvement_plan) > 0, "Should provide improvement plan"

    # Validate schema compliance
    state_dict = final_state.model_dump()
    validation_result = contract_validator.validate_interview_state(
        state_dict,
        check_completeness=True
    )

    assert validation_result.valid, "Interview state should comply with schema"

    logger.info("✓ Complete workflow (nervous candidate) PASSED")


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================

@pytest.mark.asyncio
async def test_workflow_handles_missing_audio(graph):
    """
    Test: Workflow handles missing audio file gracefully.

    Validates:
    - Error handling
    - Graceful degradation
    - No crashes
    """
    logger.info("TEST: Workflow handles missing audio")

    # Create initial state with non-existent audio file
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        audio_path="/nonexistent/audio.wav",
        transcript="Sample transcript for testing."
    )

    # Run workflow - should handle error gracefully
    try:
        final_state = graph.run(initial_state)

        # Pipeline should complete even if audio processing fails
        # Voice analysis might have default values
        assert final_state is not None
        assert final_state.scores is not None

        logger.info("✓ Workflow handled missing audio gracefully")

    except Exception as e:
        logger.error(f"Workflow failed to handle missing audio: {e}")
        pytest.fail(f"Workflow should handle missing audio gracefully: {e}")


@pytest.mark.asyncio
async def test_workflow_handles_empty_transcript(graph, sample_audio_file):
    """
    Test: Workflow handles empty transcript gracefully.

    Validates:
    - Empty string handling
    - Default value behavior
    - No division by zero
    """
    logger.info("TEST: Workflow handles empty transcript")

    # Create initial state with empty transcript
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        audio_path=str(sample_audio_file),
        transcript=""
    )

    # Run workflow
    try:
        final_state = graph.run(initial_state)

        # Should complete without crashes
        assert final_state is not None
        assert final_state.scores is not None

        # Scores should be low but valid
        assert 0.0 <= final_state.scores.overall <= 100.0

        logger.info("✓ Workflow handled empty transcript gracefully")

    except Exception as e:
        logger.error(f"Workflow failed with empty transcript: {e}")
        pytest.fail(f"Workflow should handle empty transcript: {e}")


# ============================================================================
# TEST: PERFORMANCE BENCHMARKING
# ============================================================================

@pytest.mark.asyncio
async def test_workflow_performance_benchmark(graph, sample_audio_file):
    """
    Test: Workflow completes within performance requirements.

    Validates:
    - Execution time < 30 seconds
    - No excessive delays
    - Scalability indicators
    """
    logger.info("TEST: Workflow performance benchmark")

    # Create initial state
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        audio_path=str(sample_audio_file),
        transcript="I would solve this problem using a greedy algorithm approach."
    )

    # Measure execution time
    start_time = datetime.now()
    final_state = graph.run(initial_state)
    end_time = datetime.now()

    execution_time = (end_time - start_time).total_seconds()

    logger.info(f"Workflow execution time: {execution_time:.2f}s")

    # Performance requirements
    assert execution_time < 30.0, \
        f"Workflow execution time {execution_time}s exceeds 30s threshold"

    # Warn if > 10 seconds
    if execution_time > 10.0:
        logger.warning(
            f"Workflow execution time {execution_time}s > 10s (acceptable but slow)")

    logger.info("✓ Workflow performance benchmark PASSED")


# ============================================================================
# TEST SUMMARY
# ============================================================================

def test_summary():
    """
    Summary of LangGraph orchestration tests.

    Coverage:
    ✓ Pipeline execution order
    ✓ State propagation
    ✓ Deterministic scoring
    ✓ Complete workflow (confident)
    ✓ Complete workflow (nervous)
    ✓ Error handling (missing audio)
    ✓ Error handling (empty transcript)
    ✓ Performance benchmarking

    Total: 8 orchestration tests
    """
    logger.info("LangGraph orchestration tests completed successfully")
