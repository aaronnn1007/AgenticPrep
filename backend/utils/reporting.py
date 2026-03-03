"""
Reporting Utility for System Calibration
=========================================
Statistical analysis and reporting for interview analyzer calibration.

This module provides:
- Score distribution analysis
- Statistical metrics (mean, std dev, percentiles)
- Anomaly detection (score inflation, lack of differentiation)
- Visualization (histograms, scatter plots)
- Report generation

Author: Senior AI Systems Engineer
Date: 2026-02-16
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import statistics
import json
import csv

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning(
        "matplotlib not available - visualizations will be skipped")

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ScoreStatistics:
    """Statistical summary of scores."""
    mean: float
    median: float
    std_dev: float
    min_score: float
    max_score: float
    percentile_25: float
    percentile_75: float
    range: float
    coefficient_variation: float  # std_dev / mean


@dataclass
class AnomalyReport:
    """Report of detected anomalies."""
    score_inflation: bool
    lack_differentiation: bool
    high_instability: bool
    warnings: List[str]
    suggestions: List[str]


# =============================================================================
# STATISTICS COMPUTATION
# =============================================================================

def compute_statistics(scores: List[float]) -> ScoreStatistics:
    """
    Compute comprehensive statistics for a list of scores.

    Args:
        scores: List of score values

    Returns:
        ScoreStatistics object
    """
    if not scores:
        raise ValueError("Cannot compute statistics on empty score list")

    mean = statistics.mean(scores)
    median = statistics.median(scores)
    std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    min_score = min(scores)
    max_score = max(scores)

    # Compute percentiles
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    percentile_25 = sorted_scores[n // 4] if n >= 4 else sorted_scores[0]
    percentile_75 = sorted_scores[3 * n // 4] if n >= 4 else sorted_scores[-1]

    score_range = max_score - min_score
    coefficient_variation = (std_dev / mean) if mean != 0 else 0.0

    return ScoreStatistics(
        mean=mean,
        median=median,
        std_dev=std_dev,
        min_score=min_score,
        max_score=max_score,
        percentile_25=percentile_25,
        percentile_75=percentile_75,
        range=score_range,
        coefficient_variation=coefficient_variation
    )


def compute_category_statistics(
    results: List[Dict[str, Any]],
    score_key: str = 'overall'
) -> Dict[str, ScoreStatistics]:
    """
    Compute statistics grouped by category.

    Args:
        results: List of result dictionaries with 'category' and 'scores'
        score_key: Which score to analyze (technical, communication, overall)

    Returns:
        Dict mapping category to ScoreStatistics
    """
    # Group scores by category
    category_scores = {}
    for result in results:
        category = result.get('category', 'unknown')
        score = result.get('scores', {}).get(score_key, 0.0)

        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)

    # Compute statistics for each category
    category_stats = {}
    for category, scores in category_scores.items():
        if scores:
            category_stats[category] = compute_statistics(scores)

    return category_stats


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def detect_score_inflation(
    results: List[Dict[str, Any]],
    threshold: float = 75.0
) -> Tuple[bool, str]:
    """
    Detect if scores are inflated (too many high scores).

    Args:
        results: List of result dictionaries
        threshold: Score threshold for considering "high" (default: 75)

    Returns:
        Tuple of (is_inflated, message)
    """
    overall_scores = [r.get('scores', {}).get('overall', 0.0) for r in results]

    if not overall_scores:
        return False, "No scores to analyze"

    high_score_ratio = sum(1 for s in overall_scores if s >=
                           threshold) / len(overall_scores)
    mean_score = statistics.mean(overall_scores)

    # Inflation detected if >60% of scores are above threshold OR mean > 80
    is_inflated = high_score_ratio > 0.6 or mean_score > 80.0

    if is_inflated:
        message = (
            f"⚠️  SCORE INFLATION detected: "
            f"{high_score_ratio:.1%} of scores >= {threshold}, "
            f"mean = {mean_score:.1f}"
        )
    else:
        message = f"✅ No score inflation detected (mean = {mean_score:.1f})"

    return is_inflated, message


def detect_lack_of_differentiation(
    results: List[Dict[str, Any]],
    min_std_dev: float = 10.0
) -> Tuple[bool, str]:
    """
    Detect if scores lack differentiation (all too similar).

    Args:
        results: List of result dictionaries
        min_std_dev: Minimum acceptable standard deviation

    Returns:
        Tuple of (lacks_differentiation, message)
    """
    overall_scores = [r.get('scores', {}).get('overall', 0.0) for r in results]

    if len(overall_scores) < 2:
        return False, "Not enough scores to analyze differentiation"

    std_dev = statistics.stdev(overall_scores)
    score_range = max(overall_scores) - min(overall_scores)

    # Lack of differentiation if std dev is too low or range is narrow
    lacks_diff = std_dev < min_std_dev or score_range < 20

    if lacks_diff:
        message = (
            f"⚠️  LACK OF DIFFERENTIATION detected: "
            f"std_dev = {std_dev:.1f}, range = {score_range:.1f}"
        )
    else:
        message = (
            f"✅ Good score differentiation "
            f"(std_dev = {std_dev:.1f}, range = {score_range:.1f})"
        )

    return lacks_diff, message


def detect_instability(
    results: List[Dict[str, Any]],
    expected_scores: Dict[str, Tuple[float, float]]
) -> Tuple[bool, List[str]]:
    """
    Detect instability by comparing actual scores to expected ranges.

    Args:
        results: List of result dictionaries with 'id' and 'scores'
        expected_scores: Dict mapping test_id to (min, max) expected score

    Returns:
        Tuple of (is_unstable, list of warnings)
    """
    warnings = []
    out_of_range_count = 0

    for result in results:
        test_id = result.get('id', '')
        actual_score = result.get('scores', {}).get('overall', 0.0)

        if test_id in expected_scores:
            min_expected, max_expected = expected_scores[test_id]

            if actual_score < min_expected or actual_score > max_expected:
                out_of_range_count += 1
                warnings.append(
                    f"  • {test_id}: score {actual_score:.1f} outside expected "
                    f"range [{min_expected:.1f}, {max_expected:.1f}]"
                )

    instability_ratio = out_of_range_count / len(results) if results else 0.0
    is_unstable = instability_ratio > 0.3  # More than 30% out of range

    if is_unstable:
        warnings.insert(
            0,
            f"⚠️  INSTABILITY detected: {out_of_range_count}/{len(results)} "
            f"({instability_ratio:.1%}) scores outside expected ranges"
        )
    else:
        warnings.insert(
            0,
            f"✅ Stable scoring: {out_of_range_count}/{len(results)} "
            f"({instability_ratio:.1%}) scores outside expected ranges"
        )

    return is_unstable, warnings


def analyze_anomalies(
    results: List[Dict[str, Any]],
    expected_scores: Dict[str, Tuple[float, float]]
) -> AnomalyReport:
    """
    Comprehensive anomaly detection and suggestion generation.

    Args:
        results: List of calibration results
        expected_scores: Expected score ranges for each test case

    Returns:
        AnomalyReport with findings and suggestions
    """
    warnings = []
    suggestions = []

    # Check for score inflation
    is_inflated, inflation_msg = detect_score_inflation(results)
    warnings.append(inflation_msg)

    if is_inflated:
        suggestions.append(
            "Consider reducing scoring weights or tightening evaluation criteria"
        )
        suggestions.append(
            "Review LLM prompts to ensure they're not overly generous in evaluation"
        )

    # Check for lack of differentiation
    lacks_diff, diff_msg = detect_lack_of_differentiation(results)
    warnings.append(diff_msg)

    if lacks_diff:
        suggestions.append(
            "Increase weight differences between score components to amplify distinctions"
        )
        suggestions.append(
            "Review evaluation criteria to ensure they distinguish answer quality better"
        )

    # Check for instability
    is_unstable, instability_warnings = detect_instability(
        results, expected_scores)
    warnings.extend(instability_warnings)

    if is_unstable:
        suggestions.append(
            "Increase LLM temperature for evaluation agents (currently may be too deterministic)"
        )
        suggestions.append(
            "Review prompts for consistency and clarity"
        )
        suggestions.append(
            "Consider adding more examples in prompts for calibration"
        )

    # Check category separation
    category_stats = compute_category_statistics(results)

    if 'excellent' in category_stats and 'poor' in category_stats:
        excellent_mean = category_stats['excellent'].mean
        poor_mean = category_stats['poor'].mean
        separation = excellent_mean - poor_mean

        if separation < 30:
            warnings.append(
                f"⚠️  Poor category separation: 'excellent' ({excellent_mean:.1f}) "
                f"vs 'poor' ({poor_mean:.1f}) = {separation:.1f} point gap"
            )
            suggestions.append(
                "Increase scoring differentiation between quality levels"
            )
        else:
            warnings.append(
                f"✅ Good category separation: 'excellent' ({excellent_mean:.1f}) "
                f"vs 'poor' ({poor_mean:.1f}) = {separation:.1f} point gap"
            )

    return AnomalyReport(
        score_inflation=is_inflated,
        lack_differentiation=lacks_diff,
        high_instability=is_unstable,
        warnings=warnings,
        suggestions=suggestions
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_score_distribution(
    results: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Score Distribution"
) -> bool:
    """
    Create histogram of score distribution.

    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
        title: Plot title

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available - skipping visualization")
        return False

    try:
        overall_scores = [r.get('scores', {}).get(
            'overall', 0.0) for r in results]

        plt.figure(figsize=(10, 6))
        plt.hist(overall_scores, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Overall Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)

        # Add mean and median lines
        mean_score = statistics.mean(overall_scores)
        median_score = statistics.median(overall_scores)

        plt.axvline(mean_score, color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {mean_score:.1f}')
        plt.axvline(median_score, color='green', linestyle='--',
                    linewidth=2, label=f'Median: {median_score:.1f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Score distribution plot saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create score distribution plot: {e}")
        return False


def plot_category_comparison(
    results: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Score by Category"
) -> bool:
    """
    Create box plot comparing scores across categories.

    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
        title: Plot title

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        return False

    try:
        # Group scores by category
        category_scores = {}
        for result in results:
            category = result.get('category', 'unknown')
            score = result.get('scores', {}).get('overall', 0.0)

            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)

        # Create box plot
        plt.figure(figsize=(12, 6))
        categories = list(category_scores.keys())
        scores = [category_scores[cat] for cat in categories]

        plt.boxplot(scores, labels=categories)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Overall Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Category comparison plot saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create category comparison plot: {e}")
        return False


def plot_score_components(
    results: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Score Components"
) -> bool:
    """
    Create scatter plot of score components.

    Args:
        results: List of result dictionaries
        output_path: Path to save the plot
        title: Plot title

    Returns:
        True if successful, False otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        return False

    try:
        technical = [r.get('scores', {}).get('technical', 0.0)
                     for r in results]
        communication = [r.get('scores', {}).get(
            'communication', 0.0) for r in results]
        categories = [r.get('category', 'unknown') for r in results]

        plt.figure(figsize=(10, 8))

        # Create color map for categories
        unique_categories = list(set(categories))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
        category_color_map = {cat: colors[i]
                              for i, cat in enumerate(unique_categories)}

        # Plot each category with different color
        for category in unique_categories:
            cat_tech = [t for t, c in zip(
                technical, categories) if c == category]
            cat_comm = [co for co, c in zip(
                communication, categories) if c == category]
            plt.scatter(cat_tech, cat_comm, label=category, alpha=0.6, s=100,
                        color=category_color_map[category])

        plt.xlabel('Technical Score', fontsize=12)
        plt.ylabel('Communication Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)

        # Add diagonal line (y=x)
        max_val = max(max(technical), max(communication))
        plt.plot([0, max_val], [0, max_val], 'k--',
                 alpha=0.3, label='Equal Scores')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Score components plot saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create score components plot: {e}")
        return False


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_text_report(
    results: List[Dict[str, Any]],
    statistics: ScoreStatistics,
    anomaly_report: AnomalyReport,
    output_path: Path
) -> None:
    """
    Generate comprehensive text report.

    Args:
        results: List of calibration results
        statistics: Overall score statistics
        anomaly_report: Anomaly detection results
        output_path: Path to save the report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SYSTEM CALIBRATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Test Cases: {len(results)}\n")
        f.write(f"Report Generated: 2026-02-16\n\n")

        # Overall Statistics
        f.write("OVERALL SCORE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean:               {statistics.mean:.2f}\n")
        f.write(f"Median:             {statistics.median:.2f}\n")
        f.write(f"Std Deviation:      {statistics.std_dev:.2f}\n")
        f.write(f"Min Score:          {statistics.min_score:.2f}\n")
        f.write(f"Max Score:          {statistics.max_score:.2f}\n")
        f.write(f"25th Percentile:    {statistics.percentile_25:.2f}\n")
        f.write(f"75th Percentile:    {statistics.percentile_75:.2f}\n")
        f.write(f"Range:              {statistics.range:.2f}\n")
        f.write(
            f"Coeff. Variation:   {statistics.coefficient_variation:.3f}\n\n")

        # Category Statistics
        f.write("SCORE BY CATEGORY\n")
        f.write("-" * 80 + "\n")
        category_stats = compute_category_statistics(results)
        for category, stats in sorted(category_stats.items()):
            f.write(f"\n{category.upper()}:\n")
            f.write(f"  Mean:       {stats.mean:.2f}\n")
            f.write(f"  Std Dev:    {stats.std_dev:.2f}\n")
            f.write(
                f"  Range:      {stats.min_score:.2f} - {stats.max_score:.2f}\n")
        f.write("\n")

        # Anomaly Detection
        f.write("ANOMALY DETECTION\n")
        f.write("-" * 80 + "\n")
        for warning in anomaly_report.warnings:
            f.write(f"{warning}\n")
        f.write("\n")

        # Suggestions
        if anomaly_report.suggestions:
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            for i, suggestion in enumerate(anomaly_report.suggestions, 1):
                f.write(f"{i}. {suggestion}\n")
            f.write("\n")

        # Individual Results
        f.write("INDIVIDUAL TEST CASE RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'ID':<20} {'Category':<18} {'Tech':<8} {'Comm':<8} {'Behav':<8} {'Overall':<8}\n")
        f.write("-" * 80 + "\n")

        for result in sorted(results, key=lambda x: x.get('scores', {}).get('overall', 0.0), reverse=True):
            test_id = result.get('id', 'unknown')
            category = result.get('category', 'unknown')
            scores = result.get('scores', {})

            f.write(
                f"{test_id:<20} {category:<18} "
                f"{scores.get('technical', 0.0):<8.1f} "
                f"{scores.get('communication', 0.0):<8.1f} "
                f"{scores.get('behavioral', 0.0):<8.1f} "
                f"{scores.get('overall', 0.0):<8.1f}\n"
            )

    logger.info(f"Text report saved to {output_path}")


def save_results_csv(
    results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save calibration results to CSV file.

    Args:
        results: List of calibration results
        output_path: Path to save the CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'id', 'category', 'role', 'experience_level',
            'technical_score', 'communication_score', 'behavioral_score', 'overall_score',
            'transcript_length', 'execution_time'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            scores = result.get('scores', {})
            writer.writerow({
                'id': result.get('id', ''),
                'category': result.get('category', ''),
                'role': result.get('role', ''),
                'experience_level': result.get('experience_level', ''),
                'technical_score': scores.get('technical', 0.0),
                'communication_score': scores.get('communication', 0.0),
                'behavioral_score': scores.get('behavioral', 0.0),
                'overall_score': scores.get('overall', 0.0),
                'transcript_length': result.get('transcript_length', 0),
                'execution_time': result.get('execution_time', 0.0)
            })

    logger.info(f"Results saved to CSV: {output_path}")
