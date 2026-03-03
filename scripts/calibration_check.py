"""
Calibration Check Script for Scoring & Aggregation Agent
========================================================
This script validates the scoring engine by generating synthetic test cases
and analyzing score distributions to detect potential issues.

Purpose:
--------
1. Generate diverse synthetic answer quality inputs
2. Compute score distributions across the synthetic dataset
3. Detect scoring inflation (average > 85)
4. Detect over-penalization (average < 40)
5. Provide statistical summary of scoring behavior

Usage:
------
    python scripts/calibration_check.py
    
Output:
-------
- Mean scores for technical, communication, and overall
- Standard deviation for each score category
- Min/max values
- Warning flags for inflation or over-penalization
- Detailed score distribution by percentile

Calibration Targets:
--------------------
- Mean overall score: 60-75 (healthy range)
- Standard deviation: 15-25 (good variance)
- No systematic bias toward high or low scores
"""

from backend.agents.scoring_aggregation import ScoringAggregationAgent
import sys
import random
from pathlib import Path
from typing import List, Dict, Any
from statistics import mean, stdev

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_inputs(count: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate synthetic answer quality inputs for calibration.

    Generates a diverse set of realistic answer quality scenarios including:
    - Excellent answers (high scores across all dimensions)
    - Poor answers (low scores across all dimensions)
    - Unbalanced answers (strong in some areas, weak in others)
    - Average answers (moderate scores)

    Args:
        count: Number of synthetic inputs to generate
        seed: Random seed for reproducibility

    Returns:
        List of answer quality dictionaries
    """
    random.seed(seed)

    synthetic_data = []

    # Distribution strategy:
    # - 20% excellent (0.8-1.0)
    # - 20% poor (0.0-0.3)
    # - 30% average (0.4-0.7)
    # - 30% mixed/unbalanced

    for i in range(count):
        category = i % 5

        if category == 0:
            # Excellent answer
            data = {
                "relevance": random.uniform(0.8, 1.0),
                "correctness": random.uniform(0.8, 1.0),
                "depth": random.uniform(0.8, 1.0),
                "structure": random.uniform(0.8, 1.0),
                "gaps": []
            }
        elif category == 1:
            # Poor answer
            data = {
                "relevance": random.uniform(0.0, 0.3),
                "correctness": random.uniform(0.0, 0.3),
                "depth": random.uniform(0.0, 0.3),
                "structure": random.uniform(0.0, 0.3),
                "gaps": ["fundamental concepts", "basic understanding"]
            }
        elif category == 2:
            # Average answer
            data = {
                "relevance": random.uniform(0.4, 0.7),
                "correctness": random.uniform(0.4, 0.7),
                "depth": random.uniform(0.4, 0.7),
                "structure": random.uniform(0.4, 0.7),
                "gaps": ["some details"]
            }
        elif category == 3:
            # Strong technical, weak communication
            data = {
                "relevance": random.uniform(0.3, 0.5),
                "correctness": random.uniform(0.8, 1.0),
                "depth": random.uniform(0.7, 0.9),
                "structure": random.uniform(0.3, 0.5),
                "gaps": ["clear organization"]
            }
        else:
            # Strong communication, weak technical
            data = {
                "relevance": random.uniform(0.8, 1.0),
                "correctness": random.uniform(0.3, 0.5),
                "depth": random.uniform(0.2, 0.4),
                "structure": random.uniform(0.8, 1.0),
                "gaps": ["technical depth", "accuracy"]
            }

        synthetic_data.append(data)

    return synthetic_data


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_statistics(scores: List[float]) -> Dict[str, float]:
    """
    Compute statistical measures for a list of scores.

    Args:
        scores: List of score values

    Returns:
        Dictionary with statistical measures
    """
    if not scores:
        return {
            "mean": 0.0,
            "stdev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0
        }

    sorted_scores = sorted(scores)
    n = len(sorted_scores)

    return {
        "mean": mean(scores),
        "stdev": stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "median": sorted_scores[n // 2],
        "p25": sorted_scores[n // 4],
        "p75": sorted_scores[3 * n // 4]
    }


def detect_calibration_issues(stats: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Detect potential calibration issues based on statistical analysis.

    Args:
        stats: Dictionary of statistics for each score category

    Returns:
        List of warning messages
    """
    warnings = []

    # Check for score inflation
    if stats["overall"]["mean"] > 85:
        warnings.append(
            f"⚠️  SCORE INFLATION DETECTED: "
            f"Average overall score is {stats['overall']['mean']:.1f} (threshold: 85). "
            f"Scores may be artificially high."
        )

    if stats["technical"]["mean"] > 85:
        warnings.append(
            f"⚠️  TECHNICAL SCORE INFLATION: "
            f"Average technical score is {stats['technical']['mean']:.1f}. "
            f"Consider adjusting technical weights."
        )

    if stats["communication"]["mean"] > 85:
        warnings.append(
            f"⚠️  COMMUNICATION SCORE INFLATION: "
            f"Average communication score is {stats['communication']['mean']:.1f}. "
            f"Consider adjusting communication weights."
        )

    # Check for over-penalization
    if stats["overall"]["mean"] < 40:
        warnings.append(
            f"⚠️  OVER-PENALIZATION DETECTED: "
            f"Average overall score is {stats['overall']['mean']:.1f} (threshold: 40). "
            f"Scores may be too harsh."
        )

    if stats["technical"]["mean"] < 40:
        warnings.append(
            f"⚠️  TECHNICAL OVER-PENALIZATION: "
            f"Average technical score is {stats['technical']['mean']:.1f}. "
            f"Consider adjusting technical weights."
        )

    if stats["communication"]["mean"] < 40:
        warnings.append(
            f"⚠️  COMMUNICATION OVER-PENALIZATION: "
            f"Average communication score is {stats['communication']['mean']:.1f}. "
            f"Consider adjusting communication weights."
        )

    # Check for insufficient variance
    if stats["overall"]["stdev"] < 10:
        warnings.append(
            f"⚠️  LOW VARIANCE: "
            f"Overall score standard deviation is {stats['overall']['stdev']:.1f}. "
            f"Scores may not be differentiating well between answers."
        )

    # Check for healthy range
    if 60 <= stats["overall"]["mean"] <= 75 and 15 <= stats["overall"]["stdev"] <= 25:
        warnings.append(
            f"✅ HEALTHY CALIBRATION: "
            f"Scores are well-calibrated (mean: {stats['overall']['mean']:.1f}, "
            f"stdev: {stats['overall']['stdev']:.1f})"
        )

    return warnings


# =============================================================================
# VISUALIZATION
# =============================================================================

def print_score_distribution(scores: List[float], label: str, width: int = 50):
    """
    Print a simple ASCII histogram of score distribution.

    Args:
        scores: List of scores
        label: Label for the distribution
        width: Width of the histogram bars
    """
    if not scores:
        return

    # Create bins: 0-20, 20-40, 40-60, 60-80, 80-100
    bins = [0, 20, 40, 60, 80, 100]
    counts = [0] * (len(bins) - 1)

    for score in scores:
        for i in range(len(bins) - 1):
            if bins[i] <= score < bins[i + 1]:
                counts[i] += 1
                break
        else:
            # Handle score == 100
            if score == 100:
                counts[-1] += 1

    max_count = max(counts) if counts else 1

    print(f"\n{label} Distribution:")
    print("-" * (width + 20))

    for i in range(len(bins) - 1):
        bar_length = int((counts[i] / max_count) *
                         width) if max_count > 0 else 0
        bar = "█" * bar_length
        percentage = (counts[i] / len(scores)) * 100
        print(
            f"  {bins[i]:3d}-{bins[i+1]:3d}: {bar:50s} {counts[i]:2d} ({percentage:5.1f}%)")


# =============================================================================
# MAIN CALIBRATION CHECK
# =============================================================================

def run_calibration_check(num_samples: int = 20, verbose: bool = True):
    """
    Run complete calibration check on the scoring engine.

    Args:
        num_samples: Number of synthetic inputs to generate
        verbose: Whether to print detailed output

    Returns:
        Dictionary with calibration results
    """
    if verbose:
        print("=" * 70)
        print("SCORING ENGINE CALIBRATION CHECK")
        print("=" * 70)
        print(f"\nGenerating {num_samples} synthetic answer quality inputs...")

    # Generate synthetic data
    synthetic_inputs = generate_synthetic_inputs(num_samples)

    # Initialize agent
    agent = ScoringAggregationAgent()

    if verbose:
        print(f"Computing scores using configured weights...")
        print(f"  Technical: correctness={agent.weights['technical']['correctness']}, "
              f"depth={agent.weights['technical']['depth']}")
        print(f"  Communication: structure={agent.weights['communication']['structure']}, "
              f"relevance={agent.weights['communication']['relevance']}")
        print(f"  Overall: technical={agent.weights['overall']['technical']}, "
              f"communication={agent.weights['overall']['communication']}")

    # Compute scores
    technical_scores = []
    communication_scores = []
    overall_scores = []

    for i, input_data in enumerate(synthetic_inputs):
        result = agent.compute(input_data)
        technical_scores.append(result.scores.technical)
        communication_scores.append(result.scores.communication)
        overall_scores.append(result.scores.overall)

    # Compute statistics
    technical_stats = compute_statistics(technical_scores)
    communication_stats = compute_statistics(communication_scores)
    overall_stats = compute_statistics(overall_scores)

    stats = {
        "technical": technical_stats,
        "communication": communication_stats,
        "overall": overall_stats
    }

    # Print results
    if verbose:
        print("\n" + "=" * 70)
        print("STATISTICAL SUMMARY")
        print("=" * 70)

        print(f"\nTechnical Scores:")
        print(f"  Mean:     {technical_stats['mean']:6.2f}")
        print(f"  Std Dev:  {technical_stats['stdev']:6.2f}")
        print(f"  Min:      {technical_stats['min']:6.2f}")
        print(f"  25th %:   {technical_stats['p25']:6.2f}")
        print(f"  Median:   {technical_stats['median']:6.2f}")
        print(f"  75th %:   {technical_stats['p75']:6.2f}")
        print(f"  Max:      {technical_stats['max']:6.2f}")

        print(f"\nCommunication Scores:")
        print(f"  Mean:     {communication_stats['mean']:6.2f}")
        print(f"  Std Dev:  {communication_stats['stdev']:6.2f}")
        print(f"  Min:      {communication_stats['min']:6.2f}")
        print(f"  25th %:   {communication_stats['p25']:6.2f}")
        print(f"  Median:   {communication_stats['median']:6.2f}")
        print(f"  75th %:   {communication_stats['p75']:6.2f}")
        print(f"  Max:      {communication_stats['max']:6.2f}")

        print(f"\nOverall Scores:")
        print(f"  Mean:     {overall_stats['mean']:6.2f}")
        print(f"  Std Dev:  {overall_stats['stdev']:6.2f}")
        print(f"  Min:      {overall_stats['min']:6.2f}")
        print(f"  25th %:   {overall_stats['p25']:6.2f}")
        print(f"  Median:   {overall_stats['median']:6.2f}")
        print(f"  75th %:   {overall_stats['p75']:6.2f}")
        print(f"  Max:      {overall_stats['max']:6.2f}")

        # Print distributions
        print_score_distribution(technical_scores, "Technical")
        print_score_distribution(communication_scores, "Communication")
        print_score_distribution(overall_scores, "Overall")

        # Check for calibration issues
        print("\n" + "=" * 70)
        print("CALIBRATION ANALYSIS")
        print("=" * 70)

        warnings = detect_calibration_issues(stats)

        if warnings:
            print()
            for warning in warnings:
                print(warning)
        else:
            print("\n✅ No calibration issues detected.")

        print("\n" + "=" * 70)
        print("CALIBRATION CHECK COMPLETE")
        print("=" * 70)

        # Summary recommendation
        print("\nRecommendations:")
        if overall_stats['mean'] > 85:
            print(
                "  • Consider reducing weights for components that consistently score high")
            print("  • Ensure upstream metrics are properly calibrated")
        elif overall_stats['mean'] < 40:
            print("  • Consider increasing weights for components that are too harsh")
            print("  • Review if input metrics are correctly normalized")
        else:
            print("  • Scoring appears well-calibrated")
            print("  • Continue monitoring as more data becomes available")

        print()

    return {
        "stats": stats,
        "scores": {
            "technical": technical_scores,
            "communication": communication_scores,
            "overall": overall_scores
        },
        "warnings": detect_calibration_issues(stats)
    }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Main entry point for calibration check script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibration check for Scoring & Aggregation Agent"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of synthetic samples to generate (default: 20)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    results = run_calibration_check(
        num_samples=args.samples,
        verbose=not args.quiet
    )

    # Exit with error code if serious calibration issues detected
    serious_warnings = [
        w for w in results["warnings"]
        if "INFLATION" in w or "OVER-PENALIZATION" in w
    ]

    if serious_warnings:
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
