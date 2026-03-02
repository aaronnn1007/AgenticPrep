"""
Body Language Calibration Script
==================================
Calibration script for the Body Language Analyser Agent.

Purpose:
- Run analysis on sample videos
- Print metric averages
- Flag abnormal values
- Validate system configuration

Usage:
    python scripts/body_language_calibration.py

Requirements:
- Place sample videos in: data/uploads/video/
- Minimum 3 sample videos recommended (supports up to any number)
"""

from backend.models.state import BodyLanguageModel
from backend.agents.body_language_agent import BodyLanguageAnalyser
import sys
import logging
from pathlib import Path
from typing import List, Dict
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Calibration thresholds
THRESHOLDS = {
    'eye_contact_warning': 0.2,
    'posture_stability_warning': 0.2,
    'facial_expressiveness_warning_low': 0.15,
    'facial_expressiveness_warning_high': 0.9,
}


def find_sample_videos(video_dir: str = "data/uploads/video") -> List[Path]:
    """
    Find sample video files in the specified directory.

    Args:
        video_dir: Directory to search for videos

    Returns:
        List of Path objects for found video files
    """
    video_dir_path = Path(video_dir)

    if not video_dir_path.exists():
        logger.warning(f"Video directory not found: {video_dir}")
        # Try alternative paths
        alternative_paths = [
            Path("backend/data/uploads/video"),
            Path("data/uploads/audio"),  # Fallback to audio dir for structure
            Path("tests/sample_videos"),
        ]

        for alt_path in alternative_paths:
            if alt_path.exists():
                logger.info(f"Using alternative path: {alt_path}")
                video_dir_path = alt_path
                break

    # Supported video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    videos = []
    for ext in video_extensions:
        videos.extend(video_dir_path.glob(f"*{ext}"))

    return sorted(videos)


def analyze_video_sample(video_path: Path, analyser: BodyLanguageAnalyser) -> Dict:
    """
    Analyze a single video and return results with metadata.

    Args:
        video_path: Path to video file
        analyser: BodyLanguageAnalyser instance

    Returns:
        Dictionary with analysis results and metadata
    """
    logger.info(f"Analyzing: {video_path.name}")

    try:
        result = analyser.analyze(str(video_path))

        return {
            'filename': video_path.name,
            'success': True,
            'metrics': result,
            'error': None
        }

    except Exception as e:
        logger.error(f"Error analyzing {video_path.name}: {e}")
        return {
            'filename': video_path.name,
            'success': False,
            'metrics': None,
            'error': str(e)
        }


def calculate_statistics(results: List[Dict]) -> Dict:
    """
    Calculate statistics across all successful analyses.

    Args:
        results: List of analysis result dictionaries

    Returns:
        Dictionary with mean, median, min, max for each metric
    """
    successful_results = [r for r in results if r['success']]

    if not successful_results:
        return {}

    eye_contacts = [r['metrics'].eye_contact for r in successful_results]
    posture_stabilities = [
        r['metrics'].posture_stability for r in successful_results]
    facial_expressiveness = [
        r['metrics'].facial_expressiveness for r in successful_results]

    return {
        'eye_contact': {
            'mean': statistics.mean(eye_contacts),
            'median': statistics.median(eye_contacts),
            'min': min(eye_contacts),
            'max': max(eye_contacts),
        },
        'posture_stability': {
            'mean': statistics.mean(posture_stabilities),
            'median': statistics.median(posture_stabilities),
            'min': min(posture_stabilities),
            'max': max(posture_stabilities),
        },
        'facial_expressiveness': {
            'mean': statistics.mean(facial_expressiveness),
            'median': statistics.median(facial_expressiveness),
            'min': min(facial_expressiveness),
            'max': max(facial_expressiveness),
        }
    }


def flag_abnormal_values(stats: Dict) -> List[str]:
    """
    Check statistics against thresholds and flag abnormalities.

    Args:
        stats: Statistics dictionary from calculate_statistics()

    Returns:
        List of warning messages
    """
    warnings = []

    if not stats:
        return ["No successful analyses to evaluate"]

    # Check eye contact
    if stats['eye_contact']['mean'] < THRESHOLDS['eye_contact_warning']:
        warnings.append(
            f"⚠️  WARNING: Low average eye contact ({stats['eye_contact']['mean']:.3f}) "
            f"- below threshold {THRESHOLDS['eye_contact_warning']}"
        )

    # Check posture stability
    if stats['posture_stability']['mean'] < THRESHOLDS['posture_stability_warning']:
        warnings.append(
            f"⚠️  WARNING: Low average posture stability ({stats['posture_stability']['mean']:.3f}) "
            f"- below threshold {THRESHOLDS['posture_stability_warning']}"
        )

    # Check facial expressiveness (too low)
    if stats['facial_expressiveness']['mean'] < THRESHOLDS['facial_expressiveness_warning_low']:
        warnings.append(
            f"⚠️  WARNING: Very low facial expressiveness ({stats['facial_expressiveness']['mean']:.3f}) "
            f"- below threshold {THRESHOLDS['facial_expressiveness_warning_low']}"
        )

    # Check facial expressiveness (too high)
    if stats['facial_expressiveness']['mean'] > THRESHOLDS['facial_expressiveness_warning_high']:
        warnings.append(
            f"⚠️  WARNING: Very high facial expressiveness ({stats['facial_expressiveness']['mean']:.3f}) "
            f"- above threshold {THRESHOLDS['facial_expressiveness_warning_high']}"
        )

    return warnings


def print_results(results: List[Dict], stats: Dict, warnings: List[str]):
    """
    Print calibration results in a formatted manner.

    Args:
        results: List of individual analysis results
        stats: Statistics dictionary
        warnings: List of warning messages
    """
    print("\n" + "="*70)
    print("BODY LANGUAGE ANALYSER CALIBRATION RESULTS")
    print("="*70 + "\n")

    # Individual results
    print(f"Analyzed {len(results)} video(s):\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['filename']}")

        if result['success']:
            metrics = result['metrics']
            print(f"   ✓ Eye Contact:            {metrics.eye_contact:.3f}")
            print(
                f"   ✓ Posture Stability:      {metrics.posture_stability:.3f}")
            print(
                f"   ✓ Facial Expressiveness:  {metrics.facial_expressiveness:.3f}")

            if metrics.distractions:
                print(f"   ⚠ Distractions: {', '.join(metrics.distractions)}")
        else:
            print(f"   ✗ Error: {result['error']}")

        print()

    # Statistics
    if stats:
        print("-"*70)
        print("AGGREGATE STATISTICS")
        print("-"*70 + "\n")

        for metric_name, metric_stats in stats.items():
            print(f"{metric_name.replace('_', ' ').title()}:")
            print(f"  Mean:   {metric_stats['mean']:.3f}")
            print(f"  Median: {metric_stats['median']:.3f}")
            print(f"  Min:    {metric_stats['min']:.3f}")
            print(f"  Max:    {metric_stats['max']:.3f}")
            print()

    # Warnings
    print("-"*70)
    print("CALIBRATION WARNINGS")
    print("-"*70 + "\n")

    if warnings:
        for warning in warnings:
            print(warning)
    else:
        print("✓ All metrics within normal ranges")

    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70 + "\n")


def main():
    """Main calibration execution."""
    print("\n🎥 Body Language Analyser Calibration Script")
    print("============================================\n")

    # Find sample videos
    print("Searching for sample videos...")
    videos = find_sample_videos()

    if not videos:
        print("❌ No sample videos found!")
        print("\nTo run calibration:")
        print("1. Place video files (.mp4, .avi, .mov, etc.) in: data/uploads/video/")
        print("2. Run this script again")
        print("\nNote: You can use any interview video recordings for calibration.")
        return

    print(f"✓ Found {len(videos)} video(s)\n")

    # Initialize analyser
    print("Initializing Body Language Analyser...")
    analyser = BodyLanguageAnalyser(frame_sample_rate=5)
    print("✓ Analyser ready\n")

    # Analyze all videos
    print("Running analysis...")
    print("-" * 50)

    results = []
    for video_path in videos:
        result = analyze_video_sample(video_path, analyser)
        results.append(result)

    print("-" * 50)
    print(f"✓ Completed {len(results)} analysis runs\n")

    # Calculate statistics
    stats = calculate_statistics(results)

    # Flag abnormal values
    warnings = flag_abnormal_values(stats)

    # Print comprehensive results
    print_results(results, stats, warnings)

    # Exit code based on warnings
    if warnings and any("⚠️  WARNING" in w for w in warnings):
        print("⚠️  Calibration completed with warnings")
        sys.exit(1)
    else:
        print("✓ Calibration completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
