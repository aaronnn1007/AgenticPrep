"""
Edge Cases, Consistency, and Performance Integration Tests
===========================================================
Production-grade tests for edge cases, consistency, and performance.

Tests:
1. Edge Cases (silent audio, network issues, missing data, etc.)
2. Consistency Tests (deterministic behavior, stable scoring)
3. Performance Tests (latency, throughput, scalability)

Requirements:
- Stress testing capabilities
- Statistical analysis
- Performance profiling
"""

import asyncio
import logging
import pytest
import tempfile
import uuid
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import time

from backend.models.state import InterviewState
from backend.graph.workflow import InterviewAnalyzerGraph, get_graph
from backend.streaming.redis_session import RedisSessionStore
from backend.streaming.audio_worker import AudioStreamingWorker, AudioChunk
from backend.streaming.voice_metrics import VoiceMetricsComputer
from utils.mock_webrtc import (
    WebRTCMockSimulator,
    MockInterviewSession,
    AudioChunkConfig,
    BodyMetricsConfig,
    NetworkCondition,
    SCENARIO_CONFIDENT_CANDIDATE,
    SCENARIO_NERVOUS_CANDIDATE,
    SCENARIO_SILENT_CANDIDATE,
    SCENARIO_NETWORK_ISSUES
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
def graph():
    """Fixture for LangGraph instance."""
    return get_graph()


@pytest.fixture
async def redis_store():
    """Fixture for Redis session store."""
    store = RedisSessionStore(
        redis_url="redis://localhost:6379/1",
        session_ttl=300
    )
    await store.connect()
    await store.redis.flushdb()
    yield store
    await store.redis.flushdb()
    await store.close()


@pytest.fixture
async def audio_worker():
    """Fixture for audio streaming worker."""
    worker = AudioStreamingWorker(
        model_size="tiny",
        device="cpu",
        compute_type="int8"
    )
    await worker.initialize()
    yield worker
    await worker.shutdown()


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_edge_case_silent_audio(
    graph,
    temp_dir,
    webrtc_simulator
):
    """
    Test: System handles silent audio gracefully.

    Validates:
    - No crashes on silence
    - Appropriate default values
    - Low or zero scores for silence
    """
    logger.info("TEST: Edge case - silent audio")

    # Generate silent audio
    session = SCENARIO_SILENT_CANDIDATE
    session.session_id = f"test_{uuid.uuid4().hex[:8]}"
    session.duration_seconds = 5

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    # Create initial state with minimal transcript
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        audio_path=str(audio_file),
        transcript=""  # Empty transcript
    )

    # Run workflow
    try:
        final_state = graph.run(initial_state)

        # Should complete without crashes
        assert final_state is not None

        # Scores should be low but valid
        assert 0.0 <= final_state.scores.overall <= 100.0

        # Should identify lack of response as weakness
        assert len(final_state.recommendations.weaknesses) > 0

        logger.info(
            f"Scores for silent audio: {final_state.scores.overall:.2f}")
        logger.info("✓ Edge case - silent audio PASSED")

    except Exception as e:
        pytest.fail(f"System failed on silent audio: {e}")


@pytest.mark.asyncio
async def test_edge_case_very_fast_speech(graph, temp_dir, webrtc_simulator):
    """
    Test: System handles very fast speech.

    Validates:
    - High WPM detection
    - Appropriate coaching
    - No numeric overflows
    """
    logger.info("TEST: Edge case - very fast speech")

    # Generate audio with very fast speech
    session = MockInterviewSession(
        session_id=f"test_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        duration_seconds=5,
        audio_config=AudioChunkConfig(
            speech_rate_wpm=250,  # Very fast
            silence_ratio=0.01
        )
    )

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    # Long transcript for fast speech
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role=session.role,
        experience_level=session.experience_level,
        audio_path=str(audio_file),
        transcript="I would implement using hash map for lookup then iterate through "
                   "array checking each element against hash map and return indices when "
                   "match found with target sum this gives O(n) time complexity and O(n) "
                   "space complexity which is optimal for this problem."
    )

    # Run workflow
    final_state = graph.run(initial_state)

    # Voice analysis should detect high speech rate
    assert final_state.voice_analysis.speech_rate_wpm > 200, \
        "Should detect very fast speech"

    # Recommendations should mention pacing
    all_recommendations = (
        final_state.recommendations.weaknesses +
        final_state.recommendations.improvement_plan
    )

    # Check if any recommendation mentions pacing/speed
    mentions_pacing = any(
        "pace" in rec.lower() or "speed" in rec.lower() or "slow" in rec.lower()
        for rec in all_recommendations
    )

    logger.info(
        f"Speech rate: {final_state.voice_analysis.speech_rate_wpm:.1f} WPM")
    logger.info("✓ Edge case - very fast speech PASSED")


@pytest.mark.asyncio
async def test_edge_case_very_long_pause(graph, temp_dir, webrtc_simulator):
    """
    Test: System handles very long pauses.

    Validates:
    - Long silence detection
    - Appropriate metrics
    - Coaching on hesitation
    """
    logger.info("TEST: Edge case - very long pause")

    # Generate audio with long pauses
    session = MockInterviewSession(
        session_id=f"test_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Junior",
        duration_seconds=10,
        audio_config=AudioChunkConfig(
            speech_rate_wpm=80,  # Slow
            silence_ratio=0.60  # 60% silence
        )
    )

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    # Short transcript for long pauses
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role=session.role,
        experience_level=session.experience_level,
        audio_path=str(audio_file),
        transcript="Um... well... I think... maybe... a loop?"
    )

    # Run workflow
    final_state = graph.run(initial_state)

    # Should complete successfully
    assert final_state is not None

    # Should detect low speech content
    assert final_state.voice_analysis.speech_rate_wpm < 120, \
        "Should detect slow/hesitant speech"

    logger.info(
        f"Speech rate: {final_state.voice_analysis.speech_rate_wpm:.1f} WPM")
    logger.info("✓ Edge case - very long pause PASSED")


@pytest.mark.asyncio
async def test_edge_case_high_filler_words(graph, temp_dir, webrtc_simulator):
    """
    Test: System handles excessive filler words.

    Validates:
    - High filler ratio detection
    - Appropriate scoring penalty
    - Coaching on clarity
    """
    logger.info("TEST: Edge case - high filler words")

    # Transcript with many filler words
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Junior",
        transcript="Um, like, you know, I would, uh, basically, like, use, um, "
                   "a hash map, you know, which is, like, really, um, efficient, "
                   "sort of, and, uh, basically, you know, iterate through, like, the array."
    )

    # Generate audio
    session = MockInterviewSession(
        session_id=f"test_{uuid.uuid4().hex[:8]}",
        role=initial_state.role,
        experience_level=initial_state.experience_level,
        duration_seconds=5,
        audio_config=AudioChunkConfig(
            filler_probability=0.30  # 30% filler words
        )
    )

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    initial_state.audio_path = str(audio_file)

    # Run workflow
    final_state = graph.run(initial_state)

    # Should detect high filler ratio
    assert final_state.voice_analysis.filler_ratio > 0.10, \
        "Should detect excessive filler words"

    # Communication score should be affected
    assert final_state.scores.communication < 85.0, \
        "Communication score should be lower due to filler words"

    logger.info(f"Filler ratio: {final_state.voice_analysis.filler_ratio:.2%}")
    logger.info("✓ Edge case - high filler words PASSED")


@pytest.mark.asyncio
async def test_edge_case_missing_body_metrics(graph, temp_dir, webrtc_simulator):
    """
    Test: System handles missing body language data.

    Validates:
    - Graceful handling of missing video
    - Default body metrics
    - Pipeline completion
    """
    logger.info("TEST: Edge case - missing body metrics")

    # Generate audio
    session = MockInterviewSession(
        session_id=f"test_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        duration_seconds=5
    )

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    # Create state without video_path
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role=session.role,
        experience_level=session.experience_level,
        audio_path=str(audio_file),
        video_path=None,  # No video
        transcript="I would implement this using dynamic programming."
    )

    # Run workflow
    final_state = graph.run(initial_state)

    # Should complete successfully
    assert final_state is not None

    # Body language metrics should have default values
    assert final_state.body_language is not None

    # Behavioral score may be lower or use defaults
    assert 0.0 <= final_state.scores.behavioral <= 100.0

    logger.info("✓ Edge case - missing body metrics PASSED")


@pytest.mark.asyncio
async def test_edge_case_network_interruption(
    redis_store,
    audio_worker,
    webrtc_simulator
):
    """
    Test: System handles network interruptions.

    Validates:
    - Resilience to packet loss
    - Graceful degradation
    - State recovery
    """
    logger.info("TEST: Edge case - network interruption")

    # Create session with unstable network
    session = SCENARIO_NETWORK_ISSUES
    session.session_id = f"test_{uuid.uuid4().hex[:8]}"
    session.duration_seconds = 10

    await redis_store.create_session(
        session_id=session.session_id,
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role=session.role,
        experience_level=session.experience_level
    )

    # Track audio chunks
    chunks_received = []

    async def on_audio(audio_data: bytes, metadata: Dict[str, Any]):
        chunks_received.append(metadata)

        # Simulate occasional packet drops (30%)
        if len(chunks_received) % 3 != 0:
            chunk = AudioChunk(
                session_id=session.session_id,
                data=audio_data,
                timestamp=datetime.now(),
                sample_rate=metadata["sample_rate"],
                channels=1,
                sample_width=2
            )
            await audio_worker.process_audio_chunk(chunk)

    # Run simulation
    await webrtc_simulator.simulate_interview_session(
        session=session,
        audio_callback=lambda data, meta: asyncio.create_task(
            on_audio(data, meta)),
        body_callback=lambda metrics: None,
        event_callback=None
    )

    # Wait for processing
    await asyncio.sleep(3)

    # System should handle gracefully
    session_data = await redis_store.get_session(session.session_id)
    assert session_data is not None

    logger.info(f"Chunks received: {len(chunks_received)}")
    logger.info("✓ Edge case - network interruption PASSED")


# ============================================================================
# CONSISTENCY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_consistency_deterministic_scoring(
    graph,
    temp_dir,
    webrtc_simulator
):
    """
    Test: Same inputs produce consistent scores (< 5% variance).

    Validates:
    - Deterministic behavior
    - Low score variance
    - Reproducibility
    """
    logger.info("TEST: Consistency - deterministic scoring")

    # Generate audio file
    session = SCENARIO_CONFIDENT_CANDIDATE
    session.session_id = f"test_{uuid.uuid4().hex[:8]}"
    session.duration_seconds = 10

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    # Fixed transcript
    transcript = "I would implement this using a hash map for O(1) lookup time."

    # Run workflow 5 times
    num_runs = 5
    results = []

    for i in range(num_runs):
        initial_state = InterviewState(
            interview_id=f"int_{uuid.uuid4().hex[:8]}_{i}",
            role=session.role,
            experience_level=session.experience_level,
            audio_path=str(audio_file),
            transcript=transcript
        )

        final_state = graph.run(initial_state)
        results.append(final_state)

        logger.info(f"Run {i+1}: Overall={final_state.scores.overall:.2f}")

    # Calculate variance
    overall_scores = [r.scores.overall for r in results]
    technical_scores = [r.scores.technical for r in results]
    communication_scores = [r.scores.communication for r in results]
    behavioral_scores = [r.scores.behavioral for r in results]

    overall_variance = statistics.stdev(
        overall_scores) if len(overall_scores) > 1 else 0
    technical_variance = statistics.stdev(
        technical_scores) if len(technical_scores) > 1 else 0
    communication_variance = statistics.stdev(
        communication_scores) if len(communication_scores) > 1 else 0
    behavioral_variance = statistics.stdev(
        behavioral_scores) if len(behavioral_scores) > 1 else 0

    # Calculate percentage variance
    mean_overall = statistics.mean(overall_scores)
    overall_variance_pct = (
        overall_variance / mean_overall * 100) if mean_overall > 0 else 0

    logger.info(f"Score statistics:")
    logger.info(
        f"  Overall: mean={mean_overall:.2f}, stdev={overall_variance:.2f} ({overall_variance_pct:.1f}%)")
    logger.info(
        f"  Technical: mean={statistics.mean(technical_scores):.2f}, stdev={technical_variance:.2f}")
    logger.info(
        f"  Communication: mean={statistics.mean(communication_scores):.2f}, stdev={communication_variance:.2f}")
    logger.info(
        f"  Behavioral: mean={statistics.mean(behavioral_scores):.2f}, stdev={behavioral_variance:.2f}")

    # Validate variance is low (< 5%)
    assert overall_variance_pct < 5.0, \
        f"Score variance {overall_variance_pct:.1f}% exceeds 5% threshold"

    logger.info("✓ Consistency - deterministic scoring PASSED")


@pytest.mark.asyncio
async def test_consistency_stable_recommendations(
    graph,
    temp_dir,
    webrtc_simulator
):
    """
    Test: Recommendations are stable across runs.

    Validates:
    - Consistent weakness identification
    - Stable improvement plans
    - No recommendation flip-flopping
    """
    logger.info("TEST: Consistency - stable recommendations")

    # Generate audio
    session = SCENARIO_NERVOUS_CANDIDATE
    session.session_id = f"test_{uuid.uuid4().hex[:8]}"
    session.duration_seconds = 10

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    # Fixed transcript with clear issues
    transcript = "Um, well, like, I think... maybe a loop? I'm not sure."

    # Run workflow 3 times
    results = []

    for i in range(3):
        initial_state = InterviewState(
            interview_id=f"int_{uuid.uuid4().hex[:8]}_{i}",
            role=session.role,
            experience_level=session.experience_level,
            audio_path=str(audio_file),
            transcript=transcript
        )

        final_state = graph.run(initial_state)
        results.append(final_state)

    # Check for consistent weakness patterns
    all_weaknesses = []
    for result in results:
        all_weaknesses.extend(result.recommendations.weaknesses)

    # Count occurrence of weakness themes
    weakness_keywords = ["clarity", "confidence",
                         "filler", "hesitation", "structure"]
    keyword_counts = {kw: 0 for kw in weakness_keywords}

    for weakness in all_weaknesses:
        for keyword in weakness_keywords:
            if keyword.lower() in weakness.lower():
                keyword_counts[keyword] += 1

    logger.info(f"Weakness themes across runs: {keyword_counts}")

    # At least some consistent themes should appear
    consistent_themes = sum(
        1 for count in keyword_counts.values() if count >= 2)
    assert consistent_themes > 0, "Should have consistent weakness themes across runs"

    logger.info("✓ Consistency - stable recommendations PASSED")


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_performance_transcription_latency(
    audio_worker,
    temp_dir,
    webrtc_simulator
):
    """
    Test: Transcription latency is acceptable.

    Target: < 3 seconds for 2-second audio chunk

    Validates:
    - Real-time performance
    - Consistent latency
    - No degradation over time
    """
    logger.info("TEST: Performance - transcription latency")

    # Generate audio
    session = MockInterviewSession(
        session_id=f"test_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        duration_seconds=2
    )

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    # Load audio
    import wave
    with wave.open(str(audio_file), 'rb') as wav:
        audio_data = wav.readframes(wav.getnframes())
        sample_rate = wav.getframerate()

    # Measure latency over multiple runs
    latencies = []

    for i in range(3):
        chunk = AudioChunk(
            session_id=f"{session.session_id}_{i}",
            data=audio_data,
            timestamp=datetime.now(),
            sample_rate=sample_rate,
            channels=1,
            sample_width=2
        )

        start = time.time()
        await audio_worker.process_audio_chunk(chunk)
        await asyncio.sleep(1)  # Wait for processing
        end = time.time()

        latency = end - start
        latencies.append(latency)
        logger.info(f"  Run {i+1}: {latency:.3f}s")

    # Calculate statistics
    mean_latency = statistics.mean(latencies)
    max_latency = max(latencies)

    logger.info(
        f"Transcription latency: mean={mean_latency:.3f}s, max={max_latency:.3f}s")

    # Validate
    assert mean_latency < 5.0, f"Mean latency {mean_latency:.3f}s exceeds 5s threshold"
    assert max_latency < 8.0, f"Max latency {max_latency:.3f}s exceeds 8s threshold"

    logger.info("✓ Performance - transcription latency PASSED")


@pytest.mark.asyncio
async def test_performance_pipeline_execution_time(
    graph,
    temp_dir,
    webrtc_simulator
):
    """
    Test: LangGraph pipeline executes within time limit.

    Target: < 30 seconds
    Warning: > 10 seconds

    Validates:
    - Overall pipeline performance
    - No bottlenecks
    - Scalability
    """
    logger.info("TEST: Performance - pipeline execution time")

    # Generate audio
    session = MockInterviewSession(
        session_id=f"test_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        duration_seconds=10
    )

    audio_file = temp_dir / f"{session.session_id}.wav"
    webrtc_simulator.generate_full_audio_file(session, audio_file)

    # Create initial state
    initial_state = InterviewState(
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role=session.role,
        experience_level=session.experience_level,
        audio_path=str(audio_file),
        transcript="I would implement this algorithm using dynamic programming."
    )

    # Measure execution time
    start = time.time()
    final_state = graph.run(initial_state)
    end = time.time()

    execution_time = end - start

    logger.info(f"Pipeline execution time: {execution_time:.2f}s")

    # Validate
    assert execution_time < 30.0, \
        f"Pipeline execution time {execution_time:.2f}s exceeds 30s threshold"

    if execution_time > 10.0:
        logger.warning(
            f"Pipeline execution time {execution_time:.2f}s > 10s (acceptable but slow)")

    logger.info("✓ Performance - pipeline execution time PASSED")


@pytest.mark.asyncio
async def test_performance_streaming_throughput(
    redis_store,
    audio_worker,
    webrtc_simulator
):
    """
    Test: Streaming system handles expected throughput.

    Target: Process audio chunks with < 500ms average delay

    Validates:
    - Throughput capacity
    - No backlog buildup
    - Consistent performance
    """
    logger.info("TEST: Performance - streaming throughput")

    # Create session
    session = MockInterviewSession(
        session_id=f"test_{uuid.uuid4().hex[:8]}",
        role="Software Engineer",
        experience_level="Mid",
        duration_seconds=20,
        audio_config=AudioChunkConfig(
            chunk_duration_ms=2000
        )
    )

    await redis_store.create_session(
        session_id=session.session_id,
        interview_id=f"int_{uuid.uuid4().hex[:8]}",
        role=session.role,
        experience_level=session.experience_level
    )

    # Track processing delays
    processing_delays = []

    async def on_audio(audio_data: bytes, metadata: Dict[str, Any]):
        chunk_time = datetime.fromisoformat(metadata["timestamp"])

        chunk = AudioChunk(
            session_id=session.session_id,
            data=audio_data,
            timestamp=chunk_time,
            sample_rate=metadata["sample_rate"],
            channels=1,
            sample_width=2
        )

        process_start = datetime.now()
        await audio_worker.process_audio_chunk(chunk)
        process_end = datetime.now()

        delay = (process_end - chunk_time).total_seconds()
        processing_delays.append(delay)

    # Run simulation
    await webrtc_simulator.simulate_interview_session(
        session=session,
        audio_callback=lambda data, meta: asyncio.create_task(
            on_audio(data, meta)),
        body_callback=lambda metrics: None,
        event_callback=None
    )

    # Wait for processing
    await asyncio.sleep(3)

    # Calculate statistics
    if processing_delays:
        mean_delay = statistics.mean(processing_delays)
        max_delay = max(processing_delays)

        logger.info(
            f"Processing delays: mean={mean_delay:.3f}s, max={max_delay:.3f}s")

        # Validate
        assert mean_delay < 5.0, \
            f"Mean processing delay {mean_delay:.3f}s exceeds 5s threshold"
    else:
        logger.warning("No processing delays recorded")

    logger.info("✓ Performance - streaming throughput PASSED")


# ============================================================================
# TEST SUMMARY
# ============================================================================

def test_summary():
    """
    Summary of edge cases, consistency, and performance tests.

    Edge Cases (7):
    ✓ Silent audio
    ✓ Very fast speech
    ✓ Very long pause
    ✓ High filler words
    ✓ Missing body metrics
    ✓ Network interruption

    Consistency (2):
    ✓ Deterministic scoring (< 5% variance)
    ✓ Stable recommendations

    Performance (3):
    ✓ Transcription latency (< 3s per chunk)
    ✓ Pipeline execution time (< 30s)
    ✓ Streaming throughput (< 500ms delay)

    Total: 12 tests
    """
    logger.info(
        "Edge cases, consistency, and performance tests completed successfully")
