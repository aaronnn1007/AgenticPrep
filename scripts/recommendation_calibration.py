"""
Recommendation System Calibration Script
========================================
Validates that the recommendation system generates appropriately
varied and calibrated outputs across different performance profiles.

Purpose:
- Detect bias or inflation in recommendations
- Ensure output varies appropriately with input scores
- Flag if all candidates receive similar recommendations
- Verify strength/weakness balance matches performance

Author: Senior AI Backend Engineer
Date: 2026-02-13
"""

from backend.agents.recommendation_system import RecommendationSystemAgent
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # dotenv not installed, will use system environment variables

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================================================
# SYNTHETIC TEST PROFILES
# =========================================================

CALIBRATION_PROFILES = [
    {
        "name": "Exceptional Performer",
        "scores": {
            "technical": 95.0,
            "communication": 93.0,
            "overall": 94.0
        },
        "answer_quality": {
            "relevance": 0.96,
            "correctness": 0.95,
            "depth": 0.92,
            "structure": 0.94,
            "gaps": []
        }
    },
    {
        "name": "Strong Technical, Moderate Communication",
        "scores": {
            "technical": 88.0,
            "communication": 68.0,
            "overall": 78.0
        },
        "answer_quality": {
            "relevance": 0.90,
            "correctness": 0.88,
            "depth": 0.85,
            "structure": 0.72,
            "gaps": ["communication clarity"]
        }
    },
    {
        "name": "Moderate All-Around",
        "scores": {
            "technical": 72.0,
            "communication": 70.0,
            "overall": 71.0
        },
        "answer_quality": {
            "relevance": 0.75,
            "correctness": 0.72,
            "depth": 0.68,
            "structure": 0.74,
            "gaps": ["edge cases", "optimization"]
        }
    },
    {
        "name": "Weak Technical, Strong Communication",
        "scores": {
            "technical": 55.0,
            "communication": 82.0,
            "overall": 68.0
        },
        "answer_quality": {
            "relevance": 0.70,
            "correctness": 0.55,
            "depth": 0.50,
            "structure": 0.85,
            "gaps": ["algorithm correctness", "data structure choice"]
        }
    },
    {
        "name": "Poor Performer",
        "scores": {
            "technical": 42.0,
            "communication": 45.0,
            "overall": 43.0
        },
        "answer_quality": {
            "relevance": 0.52,
            "correctness": 0.45,
            "depth": 0.40,
            "structure": 0.48,
            "gaps": ["fundamental concepts", "problem approach", "error handling"]
        }
    },
    {
        "name": "Competent but with Gaps",
        "scores": {
            "technical": 75.0,
            "communication": 78.0,
            "overall": 76.0
        },
        "answer_quality": {
            "relevance": 0.80,
            "correctness": 0.76,
            "depth": 0.72,
            "structure": 0.82,
            "gaps": ["scalability", "testing strategy"]
        }
    },
    {
        "name": "Nervous but Knowledgeable",
        "scores": {
            "technical": 80.0,
            "communication": 58.0,
            "overall": 69.0
        },
        "answer_quality": {
            "relevance": 0.82,
            "correctness": 0.80,
            "depth": 0.78,
            "structure": 0.60,
            "gaps": ["communication flow"]
        }
    },
    {
        "name": "Overconfident Junior",
        "scores": {
            "technical": 58.0,
            "communication": 72.0,
            "overall": 65.0
        },
        "answer_quality": {
            "relevance": 0.68,
            "correctness": 0.58,
            "depth": 0.55,
            "structure": 0.75,
            "gaps": ["technical depth", "edge case handling"]
        }
    },
    {
        "name": "Experienced Professional",
        "scores": {
            "technical": 90.0,
            "communication": 88.0,
            "overall": 89.0
        },
        "answer_quality": {
            "relevance": 0.92,
            "correctness": 0.90,
            "depth": 0.88,
            "structure": 0.91,
            "gaps": []
        }
    },
    {
        "name": "Inconsistent Performer",
        "scores": {
            "technical": 65.0,
            "communication": 75.0,
            "overall": 70.0
        },
        "answer_quality": {
            "relevance": 0.72,
            "correctness": 0.65,
            "depth": 0.62,
            "structure": 0.78,
            "gaps": ["consistency in approach"]
        }
    }
]


# =========================================================
# CALIBRATION CHECKS
# =========================================================

class CalibrationChecker:
    """Analyze recommendations for bias and calibration issues."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.flags: List[str] = []

    def check_strength_weakness_balance(self, profile_name: str, scores: Dict, recommendations: Dict):
        """Check if strength/weakness ratio matches performance level."""
        overall_score = scores["overall"]
        strengths_count = len(recommendations["strengths"])
        weaknesses_count = len(recommendations["weaknesses"])

        # High performers should have more strengths
        if overall_score > 85:
            if weaknesses_count > strengths_count:
                self.flags.append(
                    f"⚠️  {profile_name}: High score ({overall_score}) but more weaknesses than strengths"
                )

        # Low performers should have more weaknesses
        if overall_score < 60:
            if strengths_count > weaknesses_count:
                self.flags.append(
                    f"⚠️  {profile_name}: Low score ({overall_score}) but more strengths than weaknesses"
                )

    def check_gap_coverage(self, profile_name: str, gaps: List[str], recommendations: Dict):
        """Check if identified gaps are mentioned in recommendations."""
        if not gaps:
            return

        all_text = " ".join(
            recommendations["weaknesses"] + recommendations["improvement_plan"]
        ).lower()

        uncovered_gaps = []
        for gap in gaps:
            gap_keywords = gap.lower().split()
            covered = any(
                keyword in all_text for keyword in gap_keywords if len(keyword) > 3)
            if not covered:
                uncovered_gaps.append(gap)

        if uncovered_gaps:
            self.flags.append(
                f"⚠️  {profile_name}: Gaps not covered in recommendations: {uncovered_gaps}"
            )

    def check_recommendation_diversity(self):
        """Check if recommendations are sufficiently diverse across profiles."""
        if len(self.results) < 3:
            return

        # Compare first strength across all profiles
        first_strengths = [r["recommendations"]
                           ["strengths"][0].lower() for r in self.results]

        # Check for too much similarity
        from difflib import SequenceMatcher

        similar_pairs = 0
        total_pairs = 0

        for i in range(len(first_strengths)):
            for j in range(i + 1, len(first_strengths)):
                total_pairs += 1
                similarity = SequenceMatcher(
                    None, first_strengths[i], first_strengths[j]).ratio()
                if similarity > 0.7:
                    similar_pairs += 1

        if total_pairs > 0 and similar_pairs / total_pairs > 0.4:
            self.flags.append(
                f"⚠️  High similarity detected: {similar_pairs}/{total_pairs} recommendation pairs are >70% similar"
            )

    def check_inflation_bias(self):
        """Check if system inflates positive recommendations."""
        if len(self.results) < 5:
            return

        avg_strengths = sum(len(r["recommendations"]["strengths"])
                            for r in self.results) / len(self.results)
        avg_weaknesses = sum(len(r["recommendations"]["weaknesses"])
                             for r in self.results) / len(self.results)

        if avg_strengths > avg_weaknesses * 1.5:
            self.flags.append(
                f"⚠️  Potential inflation bias: avg strengths ({avg_strengths:.1f}) >> avg weaknesses ({avg_weaknesses:.1f})"
            )

    def add_result(self, profile: Dict, recommendations: Dict):
        """Add a result for analysis."""
        self.results.append({
            "profile": profile["name"],
            "scores": profile["scores"],
            "recommendations": recommendations
        })

        # Run individual checks
        self.check_strength_weakness_balance(
            profile["name"],
            profile["scores"],
            recommendations
        )
        self.check_gap_coverage(
            profile["name"],
            profile["answer_quality"]["gaps"],
            recommendations
        )

    def finalize(self):
        """Run aggregate checks."""
        self.check_recommendation_diversity()
        self.check_inflation_bias()

    def get_flags(self) -> List[str]:
        """Return all detected issues."""
        return self.flags


# =========================================================
# MAIN CALIBRATION RUNNER
# =========================================================

def run_calibration():
    """
    Run calibration across all test profiles.

    Returns:
        int: 0 if successful, 1 if issues detected
    """
    print("=" * 80)
    print("RECOMMENDATION SYSTEM CALIBRATION")
    print("=" * 80)
    print(
        f"\nRunning calibration with {len(CALIBRATION_PROFILES)} profiles...")
    print()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        return 1

    # Initialize agent
    try:
        agent = RecommendationSystemAgent(model_name="gpt-4o-mini")
        print(f"✓ Agent initialized with model: gpt-4o-mini")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return 1

    # Initialize checker
    checker = CalibrationChecker()

    # Process each profile
    success_count = 0
    failure_count = 0

    for i, profile in enumerate(CALIBRATION_PROFILES, 1):
        print(f"[{i}/{len(CALIBRATION_PROFILES)}] Testing: {profile['name']}")
        print(f"  Scores: T={profile['scores']['technical']:.0f}, "
              f"C={profile['scores']['communication']:.0f}, "
              f"O={profile['scores']['overall']:.0f}")

        try:
            # Generate recommendations
            result = agent.generate(
                scores=profile["scores"],
                answer_quality=profile["answer_quality"]
            )

            recommendations = result["recommendations"]

            # Display summary
            print(f"  ✓ Generated: {len(recommendations['strengths'])} strengths, "
                  f"{len(recommendations['weaknesses'])} weaknesses, "
                  f"{len(recommendations['improvement_plan'])} actions")

            # Add to checker
            checker.add_result(profile, recommendations)

            success_count += 1

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failure_count += 1

        print()

    # Run aggregate checks
    checker.finalize()

    # Print summary
    print("=" * 80)
    print("CALIBRATION SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {success_count}/{len(CALIBRATION_PROFILES)}")
    print(f"✗ Failed: {failure_count}/{len(CALIBRATION_PROFILES)}")
    print()

    # Print flags
    flags = checker.get_flags()
    if flags:
        print("⚠️  ISSUES DETECTED:")
        print()
        for flag in flags:
            print(f"  {flag}")
        print()
        print(f"Total issues: {len(flags)}")
    else:
        print("✓ No calibration issues detected")
        print("✓ System appears well-calibrated")

    print()
    print("=" * 80)

    # Return status
    if failure_count > 0 or len(flags) > 3:
        print("❌ CALIBRATION FAILED")
        return 1
    elif len(flags) > 0:
        print("⚠️  CALIBRATION PASSED WITH WARNINGS")
        return 0
    else:
        print("✅ CALIBRATION PASSED")
        return 0


# =========================================================
# DETAILED PROFILE ANALYSIS
# =========================================================

def analyze_profile(profile_name: str):
    """
    Analyze a single profile in detail.

    Args:
        profile_name: Name of the profile to analyze
    """
    # Find profile
    profile = None
    for p in CALIBRATION_PROFILES:
        if p["name"].lower() == profile_name.lower():
            profile = p
            break

    if not profile:
        print(f"❌ Profile '{profile_name}' not found")
        print(f"\nAvailable profiles:")
        for p in CALIBRATION_PROFILES:
            print(f"  - {p['name']}")
        return

    print("=" * 80)
    print(f"ANALYZING: {profile['name']}")
    print("=" * 80)
    print()

    # Initialize agent
    agent = RecommendationSystemAgent(model_name="gpt-4o-mini")

    # Generate recommendations
    print("Generating recommendations...")
    result = agent.generate(
        scores=profile["scores"],
        answer_quality=profile["answer_quality"]
    )

    recommendations = result["recommendations"]

    # Display results
    print()
    print("INPUT:")
    print(f"  Technical Score: {profile['scores']['technical']:.0f}")
    print(f"  Communication Score: {profile['scores']['communication']:.0f}")
    print(f"  Overall Score: {profile['scores']['overall']:.0f}")
    print(
        f"  Gaps: {', '.join(profile['answer_quality']['gaps']) if profile['answer_quality']['gaps'] else 'None'}")
    print()

    print("STRENGTHS:")
    for i, strength in enumerate(recommendations["strengths"], 1):
        print(f"  {i}. {strength}")
    print()

    print("WEAKNESSES:")
    for i, weakness in enumerate(recommendations["weaknesses"], 1):
        print(f"  {i}. {weakness}")
    print()

    print("IMPROVEMENT PLAN:")
    for i, action in enumerate(recommendations["improvement_plan"], 1):
        print(f"  {i}. {action}")
    print()
    print("=" * 80)


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate the Recommendation System Agent"
    )
    parser.add_argument(
        "--profile",
        help="Analyze a specific profile in detail",
        type=str
    )
    parser.add_argument(
        "--list",
        help="List all available profiles",
        action="store_true"
    )

    args = parser.parse_args()

    if args.list:
        print("Available calibration profiles:")
        for i, profile in enumerate(CALIBRATION_PROFILES, 1):
            print(f"{i:2d}. {profile['name']}")
            print(f"    Overall Score: {profile['scores']['overall']:.0f}")
        sys.exit(0)

    if args.profile:
        analyze_profile(args.profile)
        sys.exit(0)

    # Run full calibration
    exit_code = run_calibration()
    sys.exit(exit_code)
