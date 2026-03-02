"""
Full Pipeline Test Harness
===========================
Comprehensive end-to-end test of the Multi-Agent Interview Performance Analyzer.

This script:
1. Runs mock interview end-to-end
2. Uses synthetic/real audio and video files (if available)
3. Tests all agent integrations
4. Validates state transitions
5. Prints detailed analysis results
6. Tests error handling and edge cases

Run with:
    python scripts/full_pipeline_test.py

Requirements:
- All agents must be implemented
- LangGraph must be configured
- Test audio/video files (or will use synthetic data)
"""

from backend.config import settings
from backend.graph.workflow import get_graph, InterviewAnalyzerGraph
from backend.models.state import InterviewState
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import uuid

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_CASES = [
    {
        "name": "Mid-Level Software Engineer",
        "role": "Software Engineer",
        "experience_level": "Mid",
        "audio_file": "data/uploads/audio/test_audio.wav",
        "video_file": "data/uploads/video/test_video.mp4",
        "expected_score_range": (50, 80)
    },
    {
        "name": "Senior Data Scientist",
        "role": "Data Scientist",
        "experience_level": "Senior",
        "audio_file": "data/uploads/audio/test_audio.wav",
        "video_file": None,  # Test without video
        "expected_score_range": (40, 90)
    },
    {
        "name": "Fresher Developer",
        "role": "Backend Developer",
        "experience_level": "Fresher",
        "audio_file": "data/uploads/audio/test_audio.wav",
        "video_file": "data/uploads/video/test_video.mp4",
        "expected_score_range": (30, 70)
    }
]


# =============================================================================
# TEST UTILITIES
# =============================================================================

def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def validate_state(state: InterviewState) -> Dict[str, bool]:
    """
    Validate that all expected state fields are populated.

    Returns:
        Dict mapping field names to validation status
    """
    validations = {
        "interview_id": bool(state.interview_id),
        "role": bool(state.role),
        "experience_level": bool(state.experience_level),
        "question": state.question is not None,
        "transcript": bool(state.transcript),
        "voice_analysis": state.voice_analysis is not None,
        "answer_quality": state.answer_quality is not None,
        "body_language": state.body_language is not None,
        "confidence_behavior": state.confidence_behavior is not None,
        "scores": state.scores is not None,
        "recommendations": state.recommendations is not None
    }

    return validations


def print_question_details(state: InterviewState) -> None:
    """Print question generation results."""
    if not state.question:
        print("  ⚠️  No question generated")
        return

    print(f"  Topic: {state.question.topic}")
    print(f"  Difficulty: {state.question.difficulty:.2f}")
    print(f"  Text: {state.question.text[:100]}...")
    print(f"  Intent Points: {len(state.question.intent)}")
    for i, intent in enumerate(state.question.intent, 1):
        print(f"    {i}. {intent}")


def print_voice_analysis(state: InterviewState) -> None:
    """Print voice analysis results."""
    va = state.voice_analysis
    if not va:
        print("  ⚠️  No voice analysis")
        return

    print(f"  Speech Rate: {va.speech_rate_wpm:.1f} WPM")
    print(f"  Filler Ratio: {va.filler_ratio:.2%}")
    print(f"  Clarity: {va.clarity:.2f}/1.0")
    print(f"  Tone: {va.tone}")
    print(f"  Transcript Length: {len(state.transcript)} characters")
    if state.transcript:
        print(f"  Transcript Preview: {state.transcript[:150]}...")


def print_answer_quality(state: InterviewState) -> None:
    """Print answer quality analysis results."""
    aq = state.answer_quality
    if not aq:
        print("  ⚠️  No answer quality analysis")
        return

    print(f"  Relevance: {aq.relevance:.2f}/1.0")
    print(f"  Correctness: {aq.correctness:.2f}/1.0")
    print(f"  Depth: {aq.depth:.2f}/1.0")
    print(f"  Structure: {aq.structure:.2f}/1.0")

    if aq.gaps:
        print(f"  Missing Concepts ({len(aq.gaps)}):")
        for gap in aq.gaps[:5]:  # Show first 5
            print(f"    - {gap}")


def print_body_language(state: InterviewState) -> None:
    """Print body language analysis results."""
    bl = state.body_language
    if not bl:
        print("  ⚠️  No body language analysis")
        return

    print(f"  Eye Contact: {bl.eye_contact:.2f}/1.0")
    print(f"  Posture Stability: {bl.posture_stability:.2f}/1.0")
    print(f"  Facial Expressiveness: {bl.facial_expressiveness:.2f}/1.0")

    if bl.distractions:
        print(f"  Distractions ({len(bl.distractions)}):")
        for distraction in bl.distractions:
            print(f"    - {distraction}")


def print_confidence_behavior(state: InterviewState) -> None:
    """Print confidence and behavioral analysis results."""
    cb = state.confidence_behavior
    if not cb:
        print("  ⚠️  No confidence behavior analysis")
        return

    print(f"  Confidence: {cb.confidence:.2f}/1.0")
    print(f"  Nervousness: {cb.nervousness:.2f}/1.0")
    print(f"  Professionalism: {cb.professionalism:.2f}/1.0")

    if cb.behavioral_flags:
        print(f"  Behavioral Flags ({len(cb.behavioral_flags)}):")
        for flag in cb.behavioral_flags:
            print(f"    - {flag}")


def print_scores(state: InterviewState) -> None:
    """Print final scores."""
    scores = state.scores
    if not scores:
        print("  ⚠️  No scores computed")
        return

    print(f"  Technical: {scores.technical:.1f}/100")
    print(f"  Communication: {scores.communication:.1f}/100")
    print(f"  Behavioral: {scores.behavioral:.1f}/100")
    print(f"  ╔{'═' * 40}╗")
    print(f"  ║  OVERALL SCORE: {scores.overall:>6.1f}/100 {' ' * 14}║")
    print(f"  ╚{'═' * 40}╝")


def print_recommendations(state: InterviewState) -> None:
    """Print recommendations."""
    rec = state.recommendations
    if not rec:
        print("  ⚠️  No recommendations generated")
        return

    if rec.strengths:
        print(f"  ✅ Strengths ({len(rec.strengths)}):")
        for i, strength in enumerate(rec.strengths, 1):
            print(f"    {i}. {strength}")

    if rec.weaknesses:
        print(f"\n  ⚠️  Areas for Improvement ({len(rec.weaknesses)}):")
        for i, weakness in enumerate(rec.weaknesses, 1):
            print(f"    {i}. {weakness}")

    if rec.improvement_plan:
        print(f"\n  📋 Improvement Plan ({len(rec.improvement_plan)}):")
        for i, step in enumerate(rec.improvement_plan, 1):
            print(f"    {i}. {step}")


def print_validation_summary(validations: Dict[str, bool]) -> None:
    """Print validation summary."""
    print_subsection("Validation Summary")

    passed = sum(validations.values())
    total = len(validations)

    for field, is_valid in validations.items():
        status = "✅" if is_valid else "❌"
        print(f"  {status} {field}")

    print(f"\n  Passed: {passed}/{total} ({passed/total*100:.1f}%)")


# =============================================================================
# TEST EXECUTION
# =============================================================================

def check_test_files() -> Dict[str, bool]:
    """
    Check if test audio/video files exist.

    Returns:
        Dict mapping file paths to existence status
    """
    print_subsection("Checking Test Files")

    files_to_check = {
        "data/uploads/audio/test_audio.wav": "Test Audio",
        "data/uploads/video/test_video.mp4": "Test Video"
    }

    results = {}
    for file_path, name in files_to_check.items():
        exists = Path(file_path).exists()
        results[file_path] = exists
        status = "✅ Found" if exists else "❌ Missing"
        print(f"  {status}: {name} ({file_path})")

    return results


def run_single_test(test_case: Dict[str, Any], graph: InterviewAnalyzerGraph) -> InterviewState:
    """
    Run a single test case through the pipeline.

    Args:
        test_case: Test case configuration
        graph: Compiled LangGraph instance

    Returns:
        Final interview state
    """
    print_section(f"TEST CASE: {test_case['name']}", "=")

    # Generate unique interview ID
    interview_id = f"test_{uuid.uuid4().hex[:8]}"

    # Check if files exist
    audio_exists = Path(test_case['audio_file']).exists(
    ) if test_case['audio_file'] else False
    video_exists = Path(test_case['video_file']).exists(
    ) if test_case.get('video_file') else False

    if not audio_exists:
        print(f"  ⚠️  SKIP: Audio file not found: {test_case['audio_file']}")
        print(f"     Please provide test audio at: {test_case['audio_file']}")
        return None

    # Create initial state
    print_subsection("Initial State")
    print(f"  Interview ID: {interview_id}")
    print(f"  Role: {test_case['role']}")
    print(f"  Experience Level: {test_case['experience_level']}")
    print(
        f"  Audio File: {test_case['audio_file']} {'✅' if audio_exists else '❌'}")
    if test_case.get('video_file'):
        print(
            f"  Video File: {test_case['video_file']} {'✅' if video_exists else '❌'}")

    initial_state = InterviewState(
        interview_id=interview_id,
        role=test_case['role'],
        experience_level=test_case['experience_level'],
        audio_path=test_case['audio_file'] if audio_exists else None,
        video_path=test_case.get('video_file') if video_exists else None,
        created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ')
    )

    # Run pipeline
    print_subsection("Running Pipeline")
    start_time = time.time()

    try:
        final_state = graph.run(initial_state)
        execution_time = time.time() - start_time

        print(f"  ✅ Pipeline completed in {execution_time:.2f} seconds")

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"  ❌ Pipeline failed after {execution_time:.2f} seconds")
        print(f"  Error: {str(e)}")
        logger.error(f"Pipeline failed for {test_case['name']}", exc_info=True)
        return None

    # Print results
    print_subsection("Question Generation")
    print_question_details(final_state)

    print_subsection("Voice Analysis")
    print_voice_analysis(final_state)

    print_subsection("Answer Quality Analysis")
    print_answer_quality(final_state)

    print_subsection("Body Language Analysis")
    print_body_language(final_state)

    print_subsection("Confidence & Behavioral Analysis")
    print_confidence_behavior(final_state)

    print_subsection("Final Scores")
    print_scores(final_state)

    print_subsection("Recommendations")
    print_recommendations(final_state)

    # Validation
    validations = validate_state(final_state)
    print_validation_summary(validations)

    # Score range check
    if final_state.scores:
        expected_min, expected_max = test_case['expected_score_range']
        actual = final_state.scores.overall

        print_subsection("Score Range Validation")
        in_range = expected_min <= actual <= expected_max
        status = "✅ PASS" if in_range else "⚠️  WARN"
        print(
            f"  {status}: Overall score {actual:.1f} (expected {expected_min}-{expected_max})")

    return final_state


def run_full_pipeline_test():
    """Run complete end-to-end pipeline test."""
    print_section("MULTI-AGENT INTERVIEW ANALYZER - FULL PIPELINE TEST", "═")

    # Check environment
    print_subsection("Environment Check")
    print(f"  Settings Loaded: ✅")
    print(f"  App Name: {settings.APP_NAME}")
    print(f"  App Version: {settings.APP_VERSION}")
    print(f"  Log Level: {settings.LOG_LEVEL}")

    # Check test files
    file_status = check_test_files()

    if not any(file_status.values()):
        print("\n⚠️  WARNING: No test files found!")
        print("   Please create test files:")
        print("   - data/uploads/audio/test_audio.wav")
        print("   - data/uploads/video/test_video.mp4")
        print("\n   You can use any audio/video recording for testing.")
        return

    # Initialize graph
    print_subsection("Initializing LangGraph")
    try:
        graph = get_graph()
        print("  ✅ Graph compiled successfully")
    except Exception as e:
        print(f"  ❌ Graph compilation failed: {e}")
        logger.error("Graph compilation failed", exc_info=True)
        return

    # Run test cases
    results = []
    for test_case in TEST_CASES:
        result = run_single_test(test_case, graph)
        if result:
            results.append({
                'name': test_case['name'],
                'state': result,
                'success': True
            })
        else:
            results.append({
                'name': test_case['name'],
                'state': None,
                'success': False
            })

    # Summary
    print_section("TEST SUMMARY", "═")

    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])

    print(f"  Total Test Cases: {total_tests}")
    print(f"  Successful: {successful_tests}")
    print(f"  Failed/Skipped: {total_tests - successful_tests}")
    print(f"  Success Rate: {successful_tests/total_tests*100:.1f}%")

    print("\n  Individual Results:")
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL/SKIP"
        print(f"    {status}: {result['name']}")
        if result['success'] and result['state']:
            print(
                f"           Score: {result['state'].scores.overall:.1f}/100")

    # Overall status
    if successful_tests == total_tests:
        print("\n🎉 All tests passed! System is operational.")
    elif successful_tests > 0:
        print(
            f"\n⚠️  {total_tests - successful_tests} test(s) failed or skipped.")
    else:
        print("\n❌ All tests failed. Please check configuration and logs.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        run_full_pipeline_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        logger.error("Unexpected error in test harness", exc_info=True)
        sys.exit(1)
