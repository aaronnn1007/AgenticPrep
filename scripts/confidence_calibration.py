"""
Confidence & Behavioral Inference Agent - Calibration Script
=============================================================
Production-grade calibration script for detecting bias and score inflation.

Purpose:
- Run 15 synthetic metric combinations
- Calculate average confidence, nervousness, professionalism
- Detect score inflation (avg confidence > 0.85)
- Detect over-penalization (avg nervousness > 0.7)
- Identify bias patterns

Usage:
    python scripts/confidence_calibration.py

Author: Senior AI Backend Engineer
"""

from backend.models.state import ConfidenceBehaviorModel
from backend.agents.confidence_inference import ConfidenceBehaviorInferenceAgent
import sys
import os
import logging
from typing import Dict, Any, List
from statistics import mean, stdev

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================================================
# SYNTHETIC TEST CASES
# =========================================================

SYNTHETIC_TEST_CASES = [
    {
        "name": "High Performance (Low Fillers, High Correctness)",
        "voice_analysis": {
            "speech_rate_wpm": 145.0,
            "filler_ratio": 0.02,
            "clarity": 0.91,
            "tone": "confident"
        },
        "answer_quality": {
            "relevance": 0.93,
            "correctness": 0.90,
            "depth": 0.88,
            "structure": 0.92,
            "gaps": []
        }
    },
    {
        "name": "Moderate Performance (Medium Metrics)",
        "voice_analysis": {
            "speech_rate_wpm": 140.0,
            "filler_ratio": 0.08,
            "clarity": 0.72,
            "tone": "neutral"
        },
        "answer_quality": {
            "relevance": 0.75,
            "correctness": 0.70,
            "depth": 0.68,
            "structure": 0.73,
            "gaps": ["optimization"]
        }
    },
    {
        "name": "Nervous Performance (High Fillers, Low Clarity)",
        "voice_analysis": {
            "speech_rate_wpm": 175.0,
            "filler_ratio": 0.19,
            "clarity": 0.45,
            "tone": "nervous"
        },
        "answer_quality": {
            "relevance": 0.62,
            "correctness": 0.55,
            "depth": 0.50,
            "structure": 0.48,
            "gaps": ["core_concept", "error_handling", "edge_cases"]
        }
    },
    {
        "name": "Fast Speech + High Clarity (Bias Test)",
        "voice_analysis": {
            "speech_rate_wpm": 180.0,
            "filler_ratio": 0.04,
            "clarity": 0.89,
            "tone": "confident"
        },
        "answer_quality": {
            "relevance": 0.87,
            "correctness": 0.85,
            "depth": 0.82,
            "structure": 0.88,
            "gaps": []
        }
    },
    {
        "name": "Slow Speech + Low Fillers",
        "voice_analysis": {
            "speech_rate_wpm": 110.0,
            "filler_ratio": 0.03,
            "clarity": 0.80,
            "tone": "calm"
        },
        "answer_quality": {
            "relevance": 0.78,
            "correctness": 0.74,
            "depth": 0.71,
            "structure": 0.76,
            "gaps": ["advanced_concept"]
        }
    },
    {
        "name": "High Structure, Moderate Content",
        "voice_analysis": {
            "speech_rate_wpm": 138.0,
            "filler_ratio": 0.06,
            "clarity": 0.78,
            "tone": "confident"
        },
        "answer_quality": {
            "relevance": 0.82,
            "correctness": 0.68,
            "depth": 0.65,
            "structure": 0.91,
            "gaps": ["depth_missing"]
        }
    },
    {
        "name": "Poor Performance (Multiple Issues)",
        "voice_analysis": {
            "speech_rate_wpm": 95.0,
            "filler_ratio": 0.22,
            "clarity": 0.38,
            "tone": "uncertain"
        },
        "answer_quality": {
            "relevance": 0.45,
            "correctness": 0.40,
            "depth": 0.35,
            "structure": 0.42,
            "gaps": ["fundamental_concept", "logic_error", "incomplete"]
        }
    },
    {
        "name": "Excellent Technical + Some Fillers",
        "voice_analysis": {
            "speech_rate_wpm": 155.0,
            "filler_ratio": 0.09,
            "clarity": 0.75,
            "tone": "neutral"
        },
        "answer_quality": {
            "relevance": 0.95,
            "correctness": 0.93,
            "depth": 0.90,
            "structure": 0.88,
            "gaps": []
        }
    },
    {
        "name": "Low Correctness + High Clarity",
        "voice_analysis": {
            "speech_rate_wpm": 142.0,
            "filler_ratio": 0.05,
            "clarity": 0.86,
            "tone": "confident"
        },
        "answer_quality": {
            "relevance": 0.55,
            "correctness": 0.48,
            "depth": 0.45,
            "structure": 0.70,
            "gaps": ["key_misunderstanding", "incorrect_approach"]
        }
    },
    {
        "name": "High Filler But Good Content",
        "voice_analysis": {
            "speech_rate_wpm": 148.0,
            "filler_ratio": 0.15,
            "clarity": 0.68,
            "tone": "nervous"
        },
        "answer_quality": {
            "relevance": 0.85,
            "correctness": 0.82,
            "depth": 0.78,
            "structure": 0.75,
            "gaps": []
        }
    },
    {
        "name": "Balanced Average Performance",
        "voice_analysis": {
            "speech_rate_wpm": 145.0,
            "filler_ratio": 0.07,
            "clarity": 0.70,
            "tone": "neutral"
        },
        "answer_quality": {
            "relevance": 0.72,
            "correctness": 0.68,
            "depth": 0.65,
            "structure": 0.70,
            "gaps": ["minor_gap"]
        }
    },
    {
        "name": "Very Fast Speech + Low Correctness",
        "voice_analysis": {
            "speech_rate_wpm": 195.0,
            "filler_ratio": 0.12,
            "clarity": 0.55,
            "tone": "rushed"
        },
        "answer_quality": {
            "relevance": 0.60,
            "correctness": 0.52,
            "depth": 0.48,
            "structure": 0.55,
            "gaps": ["rushed_explanation", "incomplete"]
        }
    },
    {
        "name": "Professional Delivery + Minor Gaps",
        "voice_analysis": {
            "speech_rate_wpm": 142.0,
            "filler_ratio": 0.04,
            "clarity": 0.84,
            "tone": "confident"
        },
        "answer_quality": {
            "relevance": 0.88,
            "correctness": 0.79,
            "depth": 0.74,
            "structure": 0.90,
            "gaps": ["optimization", "edge_case"]
        }
    },
    {
        "name": "Low Clarity Despite Good Structure",
        "voice_analysis": {
            "speech_rate_wpm": 135.0,
            "filler_ratio": 0.11,
            "clarity": 0.52,
            "tone": "neutral"
        },
        "answer_quality": {
            "relevance": 0.78,
            "correctness": 0.72,
            "depth": 0.68,
            "structure": 0.85,
            "gaps": ["clarification_needed"]
        }
    },
    {
        "name": "Strong Delivery + Complete Answer",
        "voice_analysis": {
            "speech_rate_wpm": 148.0,
            "filler_ratio": 0.03,
            "clarity": 0.88,
            "tone": "confident"
        },
        "answer_quality": {
            "relevance": 0.92,
            "correctness": 0.89,
            "depth": 0.86,
            "structure": 0.90,
            "gaps": []
        }
    }
]


# =========================================================
# CALIBRATION LOGIC
# =========================================================

def run_calibration():
    """
    Run calibration on synthetic test cases.

    Returns:
        Dictionary with calibration results
    """
    print("=" * 70)
    print("CONFIDENCE & BEHAVIORAL INFERENCE AGENT - CALIBRATION")
    print("=" * 70)
    print(f"\nRunning {len(SYNTHETIC_TEST_CASES)} synthetic test cases...\n")

    # Initialize agent
    try:
        agent = ConfidenceBehaviorInferenceAgent()
        print(f"✓ Agent initialized with model: {agent.config['model']}")
        print(f"✓ Temperature: {agent.config['temperature']}")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        return None

    # Run all test cases
    results = []

    for idx, test_case in enumerate(SYNTHETIC_TEST_CASES, 1):
        print(f"[{idx}/{len(SYNTHETIC_TEST_CASES)}] {test_case['name']}")

        try:
            result = agent.infer(
                test_case["voice_analysis"],
                test_case["answer_quality"]
            )

            results.append({
                "name": test_case["name"],
                "confidence": result.confidence,
                "nervousness": result.nervousness,
                "professionalism": result.professionalism,
                "flags": result.behavioral_flags
            })

            print(f"    Confidence: {result.confidence:.3f}")
            print(f"    Nervousness: {result.nervousness:.3f}")
            print(f"    Professionalism: {result.professionalism:.3f}")
            print(f"    Flags: {result.behavioral_flags}")
            print()

        except Exception as e:
            print(f"    ✗ ERROR: {e}\n")
            logger.error(
                f"Test case '{test_case['name']}' failed: {e}", exc_info=True)
            continue

    if not results:
        print("✗ No successful test cases. Calibration failed.")
        return None

    # Calculate statistics
    return analyze_calibration_results(results)


def analyze_calibration_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze calibration results and detect issues.

    Args:
        results: List of result dictionaries

    Returns:
        Analysis dictionary
    """
    print("=" * 70)
    print("CALIBRATION ANALYSIS")
    print("=" * 70)
    print()

    # Extract metrics
    confidences = [r["confidence"] for r in results]
    nervousness_scores = [r["nervousness"] for r in results]
    professionalism_scores = [r["professionalism"] for r in results]

    # Calculate statistics
    avg_confidence = mean(confidences)
    avg_nervousness = mean(nervousness_scores)
    avg_professionalism = mean(professionalism_scores)

    std_confidence = stdev(confidences) if len(confidences) > 1 else 0.0
    std_nervousness = stdev(nervousness_scores) if len(
        nervousness_scores) > 1 else 0.0
    std_professionalism = stdev(professionalism_scores) if len(
        professionalism_scores) > 1 else 0.0

    print("AVERAGE SCORES:")
    print(
        f"  Confidence:      {avg_confidence:.3f} (σ = {std_confidence:.3f})")
    print(
        f"  Nervousness:     {avg_nervousness:.3f} (σ = {std_nervousness:.3f})")
    print(
        f"  Professionalism: {avg_professionalism:.3f} (σ = {std_professionalism:.3f})")
    print()

    # Detect issues
    issues = []

    # Check for score inflation
    if avg_confidence > 0.85:
        issues.append(
            "⚠️  SCORE INFLATION DETECTED: Average confidence > 0.85")
        print(f"⚠️  SCORE INFLATION DETECTED")
        print(
            f"    Average confidence ({avg_confidence:.3f}) is too high (> 0.85)")
        print(f"    Model may be over-optimistic.\n")

    # Check for over-penalization
    if avg_nervousness > 0.7:
        issues.append(
            "⚠️  OVER-PENALIZATION DETECTED: Average nervousness > 0.7")
        print(f"⚠️  OVER-PENALIZATION DETECTED")
        print(
            f"    Average nervousness ({avg_nervousness:.3f}) is too high (> 0.7)")
        print(f"    Model may be too harsh.\n")

    # Check for low variance (potential bias toward middle)
    if std_confidence < 0.1 and std_nervousness < 0.1:
        issues.append(
            "⚠️  LOW VARIANCE: Model may not be discriminating between cases")
        print(f"⚠️  LOW VARIANCE DETECTED")
        print(f"    Confidence and nervousness have very low standard deviations")
        print(f"    Model may be biased toward middle values.\n")

    # Check for healthy range
    if 0.55 <= avg_confidence <= 0.75 and 0.25 <= avg_nervousness <= 0.55:
        print("✓ HEALTHY RANGE: Averages are within expected bounds")
        print("  Model appears well-calibrated.\n")

    # Distribution analysis
    print("SCORE DISTRIBUTION:")
    print(
        f"  Confidence range: [{min(confidences):.3f}, {max(confidences):.3f}]")
    print(
        f"  Nervousness range: [{min(nervousness_scores):.3f}, {max(nervousness_scores):.3f}]")
    print(
        f"  Professionalism range: [{min(professionalism_scores):.3f}, {max(professionalism_scores):.3f}]")
    print()

    # Behavioral flags analysis
    all_flags = []
    for r in results:
        all_flags.extend(r["flags"])

    unique_flags = set(all_flags)
    print(f"BEHAVIORAL FLAGS:")
    print(f"  Total flags generated: {len(all_flags)}")
    print(f"  Unique flags: {len(unique_flags)}")
    if unique_flags:
        print(f"  Most common flags:")
        from collections import Counter
        flag_counts = Counter(all_flags)
        for flag, count in flag_counts.most_common(5):
            print(f"    - {flag}: {count}x")
    print()

    # Summary
    print("=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    if not issues:
        print("✓ PASSED: No calibration issues detected")
        print("✓ Model is well-calibrated and ready for production")
    else:
        print(f"⚠️  ISSUES FOUND: {len(issues)} calibration issue(s)")
        for issue in issues:
            print(f"  {issue}")
        print("\nRecommendation: Review prompt or model temperature")

    print("=" * 70)

    return {
        "avg_confidence": avg_confidence,
        "avg_nervousness": avg_nervousness,
        "avg_professionalism": avg_professionalism,
        "std_confidence": std_confidence,
        "std_nervousness": std_nervousness,
        "std_professionalism": std_professionalism,
        "issues": issues,
        "test_cases_run": len(results),
        "unique_flags": len(unique_flags)
    }


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():
    """Main execution function."""
    try:
        analysis = run_calibration()

        if analysis:
            # Exit with appropriate code
            if analysis["issues"]:
                print("\n⚠️  Calibration completed with warnings")
                sys.exit(1)
            else:
                print("\n✓ Calibration completed successfully")
                sys.exit(0)
        else:
            print("\n✗ Calibration failed")
            sys.exit(2)

    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Calibration error: {e}")
        logger.error("Calibration failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
