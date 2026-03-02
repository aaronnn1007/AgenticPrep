"""
System Calibration Script
==========================
Comprehensive calibration and benchmarking for Multi-Agent Interview Performance Analyzer.

This script:
1. Loads synthetic interview test cases
2. Runs each case through the LangGraph pipeline
3. Collects and analyzes results
4. Detects anomalies (score inflation, lack of differentiation, instability)
5. Generates comprehensive reports with visualizations

Usage:
    python scripts/system_calibration.py
    python scripts/system_calibration.py --output-dir ./calibration_results
    python scripts/system_calibration.py --no-plots

Author: Senior AI Systems Engineer
Date: 2026-02-16
"""

from backend.config import settings
from backend.utils.reporting import (
    compute_statistics,
    compute_category_statistics,
    analyze_anomalies,
    plot_score_distribution,
    plot_category_comparison,
    plot_score_components,
    generate_text_report,
    save_results_csv
)
from backend.agents.recommendation_system import RecommendationSystemAgent
from backend.agents.scoring_aggregation import ScoringAggregationAgent
from backend.agents.confidence_inference import ConfidenceBehaviorInferenceAgent
from backend.agents.answer_quality import AnswerQualityAnalyser
from backend.models.state import (
    InterviewState,
    QuestionModel,
    VoiceAnalysisModel,
    AnswerQualityModel,
    BodyLanguageModel,
    ConfidenceBehaviorModel
)
import sys
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATA_PATH = Path("data/sample_interviews.json")
DEFAULT_OUTPUT_DIR = Path("output/calibration")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_cases(data_path: Path) -> List[Dict[str, Any]]:
    """
    Load synthetic interview test cases from JSON file.

    Args:
        data_path: Path to sample_interviews.json

    Returns:
        List of test case dictionaries
    """
    logger.info(f"Loading test cases from {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Test cases file not found: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_cases = data.get('test_cases', [])
    logger.info(f"Loaded {len(test_cases)} test cases")

    return test_cases


# =============================================================================
# MOCK DATA GENERATION
# =============================================================================

def create_mock_voice_analysis(
    voice_characteristics: Dict[str, Any]
) -> VoiceAnalysisModel:
    """
    Create mock voice analysis data from characteristics.

    Args:
        voice_characteristics: Dict with speech_rate_wpm, filler_ratio, clarity, tone

    Returns:
        VoiceAnalysisModel instance
    """
    return VoiceAnalysisModel(
        speech_rate_wpm=voice_characteristics.get('speech_rate_wpm', 130.0),
        filler_ratio=voice_characteristics.get('filler_ratio', 0.05),
        clarity=voice_characteristics.get('clarity', 0.7),
        tone=voice_characteristics.get('tone', 'neutral')
    )


def create_mock_body_language() -> BodyLanguageModel:
    """
    Create default mock body language data.

    Returns:
        BodyLanguageModel with default values
    """
    return BodyLanguageModel(
        eye_contact=0.7,
        posture_stability=0.75,
        facial_expressiveness=0.65,
        distractions=[]
    )


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_single_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single test case through the analysis pipeline.

    Note: Since we don't have actual audio/video files, we:
    1. Use the provided transcript directly
    2. Mock voice analysis based on voice_characteristics
    3. Run answer quality, confidence, scoring, and recommendation agents

    Args:
        test_case: Test case dictionary

    Returns:
        Result dictionary with scores and metadata
    """
    test_id = test_case.get('id', 'unknown')
    logger.info(f"Running test case: {test_id}")

    start_time = time.time()

    try:
        # Extract test case data
        question_data = test_case.get('question', {})
        transcript = test_case.get('transcript', '')
        role = test_case.get('role', 'Software Engineer')
        experience_level = test_case.get('experience_level', 'Mid')
        voice_chars = test_case.get('voice_characteristics', {})

        # Create question model
        question = QuestionModel(
            text=question_data.get('text', ''),
            topic=question_data.get('topic', 'General'),
            difficulty=question_data.get('difficulty', 0.5),
            intent=question_data.get('intent', [])
        )

        # Mock voice analysis
        voice_analysis = create_mock_voice_analysis(voice_chars)

        # Mock body language (neutral defaults)
        body_language = create_mock_body_language()

        # Step 1: Answer Quality Analysis
        logger.debug(f"  [{test_id}] Running answer quality analysis...")
        answer_quality_agent = AnswerQualityAnalyser()

        answer_quality_result = answer_quality_agent.evaluate(
            question={
                'text': question.text,
                'intent': question.intent,
                'difficulty': question.difficulty,
                'topic': question.topic
            },
            transcript=transcript,
            role=role,
            experience_level=experience_level
        )

        answer_quality = AnswerQualityModel(
            **answer_quality_result['answer_quality'])

        # Step 2: Confidence & Behavioral Inference
        logger.debug(f"  [{test_id}] Running confidence inference...")
        confidence_agent = ConfidenceBehaviorInferenceAgent()

        confidence_behavior = confidence_agent.infer(
            voice_analysis=voice_analysis.model_dump(),
            answer_quality=answer_quality.model_dump()
        )

        # Step 3: Scoring Aggregation
        logger.debug(f"  [{test_id}] Computing scores...")
        scoring_agent = ScoringAggregationAgent()

        scoring_result = scoring_agent.compute(answer_quality.model_dump())

        # Calculate behavioral score
        behavioral_score = (
            confidence_behavior.confidence * 0.4 +
            confidence_behavior.professionalism * 0.6
        ) * 100
        behavioral_score = max(0.0, min(100.0, behavioral_score))

        # Calculate overall with behavioral component
        overall_score = (
            scoring_result.scores.technical * 0.4 +
            scoring_result.scores.communication * 0.3 +
            behavioral_score * 0.3
        )

        # Step 4: Recommendations
        logger.debug(f"  [{test_id}] Generating recommendations...")
        recommendation_agent = RecommendationSystemAgent()

        recommendation_result = recommendation_agent.generate(
            scores={
                'technical': scoring_result.scores.technical,
                'communication': scoring_result.scores.communication,
                'overall': overall_score
            },
            answer_quality=answer_quality.model_dump()
        )

        execution_time = time.time() - start_time

        # Compile result
        result = {
            'id': test_id,
            'category': test_case.get('category', 'unknown'),
            'description': test_case.get('description', ''),
            'role': role,
            'experience_level': experience_level,
            'transcript': transcript,
            'transcript_length': len(transcript),
            'scores': {
                'technical': round(scoring_result.scores.technical, 2),
                'communication': round(scoring_result.scores.communication, 2),
                'behavioral': round(behavioral_score, 2),
                'overall': round(overall_score, 2)
            },
            'answer_quality': answer_quality.model_dump(),
            'voice_analysis': voice_analysis.model_dump(),
            'confidence_behavior': confidence_behavior.model_dump(),
            'recommendations': recommendation_result.model_dump(),
            'execution_time': round(execution_time, 2),
            'success': True,
            'error': None
        }

        logger.info(
            f"  [{test_id}] Completed - Overall: {overall_score:.1f}, "
            f"Time: {execution_time:.2f}s"
        )

        return result

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"  [{test_id}] Failed: {e}", exc_info=True)

        # Return error result
        return {
            'id': test_id,
            'category': test_case.get('category', 'unknown'),
            'description': test_case.get('description', ''),
            'role': test_case.get('role', ''),
            'experience_level': test_case.get('experience_level', ''),
            'transcript': test_case.get('transcript', ''),
            'transcript_length': len(test_case.get('transcript', '')),
            'scores': {
                'technical': 0.0,
                'communication': 0.0,
                'behavioral': 0.0,
                'overall': 0.0
            },
            'execution_time': round(execution_time, 2),
            'success': False,
            'error': str(e)
        }


def run_calibration(
    test_cases: List[Dict[str, Any]],
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Run calibration on all test cases.

    Args:
        test_cases: List of test case dictionaries
        progress_callback: Optional callback function for progress updates

    Returns:
        List of result dictionaries
    """
    logger.info(f"Starting calibration run on {len(test_cases)} test cases")

    results = []

    for i, test_case in enumerate(test_cases, 1):
        if progress_callback:
            progress_callback(i, len(test_cases),
                              test_case.get('id', 'unknown'))

        result = run_single_test_case(test_case)
        results.append(result)

    successful = sum(1 for r in results if r.get('success', False))
    logger.info(
        f"Calibration complete: {successful}/{len(results)} successful")

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_calibration_report(
    results: List[Dict[str, Any]],
    test_cases: List[Dict[str, Any]],
    output_dir: Path,
    generate_plots: bool = True
) -> None:
    """
    Generate comprehensive calibration report.

    Args:
        results: List of calibration results
        test_cases: Original test cases with expected scores
        output_dir: Directory to save reports
        generate_plots: Whether to generate visualization plots
    """
    logger.info("Generating calibration report...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract expected scores from test cases
    expected_scores = {}
    for test_case in test_cases:
        test_id = test_case.get('id', '')
        expected = test_case.get('expected_scores', {})
        if 'overall' in expected:
            expected_scores[test_id] = tuple(expected['overall'])

    # Filter successful results for analysis
    successful_results = [r for r in results if r.get('success', False)]

    if not successful_results:
        logger.error("No successful results to analyze!")
        return

    # Compute overall statistics
    overall_scores = [r['scores']['overall'] for r in successful_results]
    overall_stats = compute_statistics(overall_scores)

    # Detect anomalies
    anomaly_report = analyze_anomalies(successful_results, expected_scores)

    # Generate text report
    text_report_path = output_dir / "calibration_report.txt"
    generate_text_report(successful_results, overall_stats,
                         anomaly_report, text_report_path)
    logger.info(f"✅ Text report saved to {text_report_path}")

    # Save results to CSV
    csv_path = output_dir / "calibration_results.csv"
    save_results_csv(successful_results, csv_path)
    logger.info(f"✅ CSV results saved to {csv_path}")

    # Save full results to JSON
    json_path = output_dir / "calibration_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ JSON results saved to {json_path}")

    # Generate plots
    if generate_plots:
        logger.info("Generating visualizations...")

        # Score distribution histogram
        plot_path = output_dir / "score_distribution.png"
        if plot_score_distribution(successful_results, plot_path):
            logger.info(f"✅ Score distribution plot saved to {plot_path}")

        # Category comparison box plot
        plot_path = output_dir / "category_comparison.png"
        if plot_category_comparison(successful_results, plot_path):
            logger.info(f"✅ Category comparison plot saved to {plot_path}")

        # Score components scatter plot
        plot_path = output_dir / "score_components.png"
        if plot_score_components(successful_results, plot_path):
            logger.info(f"✅ Score components plot saved to {plot_path}")


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def print_banner(text: str, char: str = "=") -> None:
    """Print a formatted banner."""
    width = 80
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_progress(current: int, total: int, test_id: str) -> None:
    """Print progress update."""
    percentage = (current / total) * 100
    print(f"  [{current}/{total}] ({percentage:.0f}%) Processing: {test_id}")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary of results to console."""
    print_banner("CALIBRATION SUMMARY", "=")

    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print(f"Total Test Cases:     {len(results)}")
    print(f"Successful:           {len(successful)} ✅")
    print(f"Failed:               {len(failed)} ❌")

    if successful:
        overall_scores = [r['scores']['overall'] for r in successful]
        print(f"\nScore Statistics:")
        print(f"  Mean:               {statistics.mean(overall_scores):.2f}")
        print(f"  Median:             {statistics.median(overall_scores):.2f}")
        print(
            f"  Std Dev:            {statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0:.2f}")
        print(
            f"  Range:              {min(overall_scores):.2f} - {max(overall_scores):.2f}")

    if failed:
        print(f"\nFailed Cases:")
        for result in failed:
            print(
                f"  • {result['id']}: {result.get('error', 'Unknown error')}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run system calibration for Interview Performance Analyzer"
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to test cases JSON file (default: {DEFAULT_DATA_PATH})"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for reports (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help="Skip generating visualization plots"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print header
    print_banner("SYSTEM CALIBRATION & BENCHMARKING", "═")

    print(f"Configuration:")
    print(f"  Data File:          {args.data}")
    print(f"  Output Directory:   {args.output_dir}")
    print(f"  Generate Plots:     {not args.no_plots}")
    print(f"  Verbose Logging:    {args.verbose}")
    print(f"  App Version:        {settings.APP_VERSION}")

    try:
        # Step 1: Load test cases
        print_banner("LOADING TEST CASES", "-")
        test_cases = load_test_cases(args.data)

        # Print test case summary
        categories = {}
        for tc in test_cases:
            cat = tc.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        print("Test Case Categories:")
        for category, count in sorted(categories.items()):
            print(f"  • {category}: {count}")

        # Step 2: Run calibration
        print_banner("RUNNING CALIBRATION", "-")
        print("This may take several minutes...\n")

        results = run_calibration(test_cases, progress_callback=print_progress)

        # Step 3: Print summary
        print_summary(results)

        # Step 4: Generate reports
        print_banner("GENERATING REPORTS", "-")
        generate_calibration_report(
            results,
            test_cases,
            args.output_dir,
            generate_plots=not args.no_plots
        )

        # Final message
        print_banner("CALIBRATION COMPLETE", "═")
        print(f"📊 All reports saved to: {args.output_dir}")
        print(f"\nReview the following files:")
        print(f"  • calibration_report.txt  - Comprehensive text report")
        print(f"  • calibration_results.csv - Detailed results in CSV format")
        print(f"  • calibration_results.json - Full results with all metrics")
        if not args.no_plots:
            print(f"  • score_distribution.png  - Score histogram")
            print(f"  • category_comparison.png - Box plot by category")
            print(f"  • score_components.png    - Technical vs Communication scatter")

        print("\n✅ Calibration complete!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)
        print(f"\n❌ Calibration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import statistics  # Import here for print_summary
    main()
