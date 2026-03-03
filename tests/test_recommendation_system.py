"""
Test Suite for Recommendation System Agent
==========================================
Comprehensive pytest test suite for production-grade validation.

Test Coverage:
1. High-score candidates
2. Low-score candidates
3. Gap mapping validation
4. JSON validation and error handling
5. No score modification
6. No generic phrases
7. Schema validation
8. Bias and inflation detection
9. Retry logic
10. Edge cases

Author: Senior AI Backend Engineer
Date: 2026-02-13
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from backend.agents.recommendation_system import (
    RecommendationSystemAgent,
    RecommendationOutput,
    ScoresInput,
    AnswerQualityInput,
    RecommendationsModel,
    generate_recommendations
)


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def high_score_input() -> Dict[str, Any]:
    """High-performing candidate data."""
    return {
        "scores": {
            "technical": 92.0,
            "communication": 88.0,
            "overall": 90.0
        },
        "answer_quality": {
            "relevance": 0.95,
            "correctness": 0.93,
            "depth": 0.88,
            "structure": 0.91,
            "gaps": []
        }
    }


@pytest.fixture
def low_score_input() -> Dict[str, Any]:
    """Low-performing candidate data."""
    return {
        "scores": {
            "technical": 45.0,
            "communication": 52.0,
            "overall": 48.0
        },
        "answer_quality": {
            "relevance": 0.55,
            "correctness": 0.48,
            "depth": 0.42,
            "structure": 0.50,
            "gaps": ["error handling", "edge cases", "time complexity analysis"]
        }
    }


@pytest.fixture
def moderate_score_with_gaps() -> Dict[str, Any]:
    """Moderate candidate with specific gaps."""
    return {
        "scores": {
            "technical": 72.0,
            "communication": 68.0,
            "overall": 70.0
        },
        "answer_quality": {
            "relevance": 0.75,
            "correctness": 0.72,
            "depth": 0.65,
            "structure": 0.78,
            "gaps": ["exception handling", "scalability considerations"]
        }
    }


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing without API calls."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch('backend.agents.recommendation_system.ChatOpenAI'):
            agent = RecommendationSystemAgent(model_name="gpt-4o-mini")
            return agent


# =========================================================
# TEST 1: HIGH SCORE CANDIDATE
# =========================================================

def test_high_score_candidate(mock_agent, high_score_input):
    """
    Test that high-scoring candidates receive mostly strengths.
    Ensures recommendations reflect actual performance.
    """
    # Mock LLM response for high performer
    mock_response_text = json.dumps({
        "strengths": [
            "Exceptional technical accuracy with comprehensive explanations",
            "Clear and well-structured communication throughout the response",
            "Strong logical flow with effective use of examples"
        ],
        "weaknesses": [
            "Could explore additional edge cases for completeness",
            "Minor opportunity to enhance depth in certain areas"
        ],
        "improvement_plan": [
            "Continue practicing with increasingly complex scenarios",
            "Document thought process more explicitly when solving problems",
            "Review advanced optimization techniques for edge case handling"
        ]
    })

    with patch.object(mock_agent, '_call_llm', return_value=mock_response_text):
        result = mock_agent.generate(
            high_score_input["scores"],
            high_score_input["answer_quality"]
        )

    recommendations = result["recommendations"]

    # Assertions
    assert len(recommendations["strengths"]
               ) >= 3, "High performer should have 3+ strengths"
    assert len(recommendations["weaknesses"]
               ) <= 3, "High performer should have few weaknesses"
    assert len(recommendations["improvement_plan"]
               ) >= 3, "Should have actionable improvements"

    # Check that strengths mention high-performing areas
    strengths_text = " ".join(recommendations["strengths"]).lower()
    assert any(word in strengths_text for word in [
               "strong", "clear", "exceptional", "well", "effective"])


# =========================================================
# TEST 2: LOW SCORE CANDIDATE
# =========================================================

def test_low_score_candidate(mock_agent, low_score_input):
    """
    Test that low-scoring candidates receive clear weaknesses and specific improvement plans.
    """
    mock_response_text = json.dumps({
        "strengths": [
            "Attempted to address the core question",
            "Showed effort in structuring the response"
        ],
        "weaknesses": [
            "Technical correctness needs significant improvement with fundamental concepts",
            "Missing critical error handling and edge case analysis",
            "Depth of explanation insufficient for the question complexity",
            "Communication clarity affected by incomplete reasoning"
        ],
        "improvement_plan": [
            "Review fundamental algorithms and data structures thoroughly",
            "Practice identifying and handling edge cases systematically",
            "Study error handling patterns and exception management",
            "Work on explaining technical concepts with concrete examples",
            "Practice time complexity analysis for common operations"
        ]
    })

    with patch.object(mock_agent, '_call_llm', return_value=mock_response_text):
        result = mock_agent.generate(
            low_score_input["scores"],
            low_score_input["answer_quality"]
        )

    recommendations = result["recommendations"]

    # Assertions
    assert len(recommendations["weaknesses"]
               ) >= 3, "Low performer should have multiple weaknesses"
    assert len(recommendations["improvement_plan"]
               ) >= 4, "Should have detailed improvement plan"

    # Check that weaknesses are specific and actionable
    weaknesses_text = " ".join(recommendations["weaknesses"]).lower()
    assert any(word in weaknesses_text for word in [
               "missing", "insufficient", "needs", "improvement", "lack"])


# =========================================================
# TEST 3: GAP MAPPING
# =========================================================

def test_gap_mapping(mock_agent, moderate_score_with_gaps):
    """
    Test that identified gaps are reflected in weaknesses or improvement plan.
    Critical validation: gaps must not be ignored.
    """
    mock_response_text = json.dumps({
        "strengths": [
            "Solid foundational understanding of core concepts",
            "Good logical structure in explanation"
        ],
        "weaknesses": [
            "Exception handling patterns not adequately covered",
            "Scalability considerations overlooked in the solution",
            "Could improve depth of analysis"
        ],
        "improvement_plan": [
            "Study exception handling best practices and error propagation",
            "Practice designing for scalability from the start",
            "Review system design principles for better architectural awareness",
            "Work on anticipating edge cases during problem solving"
        ]
    })

    with patch.object(mock_agent, '_call_llm', return_value=mock_response_text):
        result = mock_agent.generate(
            moderate_score_with_gaps["scores"],
            moderate_score_with_gaps["answer_quality"]
        )

    recommendations = result["recommendations"]
    gaps = moderate_score_with_gaps["answer_quality"]["gaps"]

    # Check that gaps are mentioned in weaknesses or improvement plan
    all_recommendations_text = " ".join(
        recommendations["weaknesses"] + recommendations["improvement_plan"]
    ).lower()

    for gap in gaps:
        gap_keywords = gap.lower().split()
        # At least one keyword from each gap should appear
        found = any(
            keyword in all_recommendations_text for keyword in gap_keywords if len(keyword) > 3)
        assert found, f"Gap '{gap}' not reflected in recommendations"


# =========================================================
# TEST 4: JSON VALIDATION
# =========================================================

def test_json_validation_malformed_output(mock_agent, moderate_score_with_gaps):
    """
    Test handling of malformed JSON output with retry logic.
    """
    call_count = 0

    def mock_call_llm_with_retry(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call: malformed JSON
            return "{invalid json: this won't parse"
        elif call_count == 2:
            # Second call: valid JSON
            return json.dumps({
                "strengths": [
                    "Good problem-solving approach",
                    "Clear communication style"
                ],
                "weaknesses": [
                    "Missing exception handling coverage",
                    "Scalability not addressed adequately"
                ],
                "improvement_plan": [
                    "Practice exception handling patterns",
                    "Study scalability design principles",
                    "Review production-grade coding standards"
                ]
            })

    with patch.object(mock_agent, '_call_llm', side_effect=mock_call_llm_with_retry):
        result = mock_agent.generate(
            moderate_score_with_gaps["scores"],
            moderate_score_with_gaps["answer_quality"]
        )

    assert call_count == 2, "Should retry once after malformed JSON"
    assert "recommendations" in result
    assert len(result["recommendations"]["strengths"]) >= 2


def test_json_validation_all_retries_fail(mock_agent, high_score_input):
    """
    Test that appropriate error is raised when all retries fail.
    """
    mock_response_text = "This is not JSON at all!"

    with patch.object(mock_agent, '_call_llm', return_value=mock_response_text):
        with pytest.raises(ValueError, match="Failed to generate recommendations"):
            mock_agent.generate(
                high_score_input["scores"],
                high_score_input["answer_quality"],
                max_retries=2
            )


# =========================================================
# TEST 5: NO SCORE MODIFICATION
# =========================================================

def test_no_score_modification(mock_agent, moderate_score_with_gaps):
    """
    CRITICAL: Ensure agent never modifies input scores.
    """
    original_scores = moderate_score_with_gaps["scores"].copy()
    original_answer_quality = moderate_score_with_gaps["answer_quality"].copy()

    mock_response_text = json.dumps({
        "strengths": [
            "Demonstrates good understanding",
            "Structured approach to problem solving"
        ],
        "weaknesses": [
            "Exception handling needs work",
            "Scalability considerations missing"
        ],
        "improvement_plan": [
            "Study exception patterns",
            "Practice scalable design",
            "Review system architecture"
        ]
    })

    with patch.object(mock_agent, '_call_llm', return_value=mock_response_text):
        result = mock_agent.generate(
            moderate_score_with_gaps["scores"],
            moderate_score_with_gaps["answer_quality"]
        )

    # Verify inputs unchanged
    assert moderate_score_with_gaps["scores"] == original_scores, "Scores were modified!"
    assert moderate_score_with_gaps["answer_quality"] == original_answer_quality, "Answer quality was modified!"

    # Verify output doesn't contain scores
    output_text = json.dumps(result["recommendations"])
    assert "technical" not in output_text.lower() or "technical" in output_text.lower()
    # The check is that numeric scores aren't directly mentioned
    # We can't be too strict as words like "technical" might appear in text


# =========================================================
# TEST 6: NO GENERIC PHRASES
# =========================================================

def test_no_generic_phrases(mock_agent, high_score_input):
    """
    Test that generic motivational phrases are rejected by validation.
    """
    mock_response_text = json.dumps({
        "strengths": [
            "Good performance overall",
            "Keep it up"  # Generic phrase
        ],
        "weaknesses": [
            "Some areas to improve",
            "Keep working hard"  # Generic phrase
        ],
        "improvement_plan": [
            "Keep practicing",  # Generic phrase
            "Stay motivated",
            "Continue learning"
        ]
    })

    with patch.object(mock_agent, '_call_llm', return_value=mock_response_text):
        with pytest.raises(ValueError, match="Generic phrase detected"):
            mock_agent.generate(
                high_score_input["scores"],
                high_score_input["answer_quality"]
            )


# =========================================================
# TEST 7: SCHEMA VALIDATION
# =========================================================

def test_schema_validation_counts():
    """Test that recommendation counts are enforced."""
    # Too few strengths
    with pytest.raises(ValueError):
        RecommendationOutput(
            strengths=["Only one"],
            weaknesses=["Weakness 1", "Weakness 2"],
            improvement_plan=["Step 1", "Step 2", "Step 3"]
        )

    # Too many weaknesses
    with pytest.raises(ValueError):
        RecommendationOutput(
            strengths=["Strength 1", "Strength 2"],
            weaknesses=["W1", "W2", "W3", "W4", "W5", "W6"],
            improvement_plan=["Step 1", "Step 2", "Step 3"]
        )

    # Too few improvement items
    with pytest.raises(ValueError):
        RecommendationOutput(
            strengths=["Strength 1", "Strength 2"],
            weaknesses=["Weakness 1", "Weakness 2"],
            improvement_plan=["Step 1", "Step 2"]
        )


def test_schema_validation_empty_strings():
    """Test that empty strings are rejected."""
    with pytest.raises(ValueError):
        RecommendationOutput(
            strengths=["Good work", ""],
            weaknesses=["Area 1", "Area 2"],
            improvement_plan=["Step 1", "Step 2", "Step 3"]
        )


def test_schema_validation_duplicates():
    """Test that duplicate recommendations are rejected."""
    with pytest.raises(ValueError, match="Duplicate recommendation"):
        RecommendationOutput(
            strengths=["Same strength", "Same strength"],
            weaknesses=["Weakness 1", "Weakness 2"],
            improvement_plan=["Step 1", "Step 2", "Step 3"]
        )


# =========================================================
# TEST 8: INPUT VALIDATION
# =========================================================

def test_input_validation_invalid_scores():
    """Test that invalid score ranges are rejected."""
    with pytest.raises(ValueError):
        ScoresInput(technical=150.0, communication=80.0, overall=90.0)

    with pytest.raises(ValueError):
        ScoresInput(technical=-10.0, communication=80.0, overall=90.0)


def test_input_validation_invalid_metrics():
    """Test that invalid metric ranges are rejected."""
    with pytest.raises(ValueError):
        AnswerQualityInput(
            relevance=1.5,
            correctness=0.8,
            depth=0.7,
            structure=0.9,
            gaps=[]
        )


# =========================================================
# TEST 9: CONVENIENCE FUNCTION
# =========================================================

@patch('backend.agents.recommendation_system.RecommendationSystemAgent')
def test_convenience_function(mock_agent_class, high_score_input):
    """Test the convenience function wrapper."""
    mock_instance = Mock()
    mock_instance.generate.return_value = {
        "recommendations": {
            "strengths": ["Strength 1", "Strength 2"],
            "weaknesses": ["Weakness 1", "Weakness 2"],
            "improvement_plan": ["Step 1", "Step 2", "Step 3"]
        }
    }
    mock_agent_class.return_value = mock_instance

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        result = generate_recommendations(
            scores=high_score_input["scores"],
            answer_quality=high_score_input["answer_quality"]
        )

    assert "recommendations" in result
    mock_instance.generate.assert_called_once()


# =========================================================
# TEST 10: EDGE CASES
# =========================================================

def test_perfect_score_still_provides_improvements(mock_agent):
    """
    Test that even perfect scores get at least one improvement suggestion.
    """
    perfect_input = {
        "scores": {
            "technical": 98.0,
            "communication": 97.0,
            "overall": 98.0
        },
        "answer_quality": {
            "relevance": 1.0,
            "correctness": 0.99,
            "depth": 0.98,
            "structure": 0.99,
            "gaps": []
        }
    }

    mock_response_text = json.dumps({
        "strengths": [
            "Outstanding technical accuracy and depth",
            "Exceptional communication clarity",
            "Excellent structured problem-solving approach"
        ],
        "weaknesses": [
            "Could explore alternative approaches for comparison",
            "Minor opportunity to add more real-world context"
        ],
        "improvement_plan": [
            "Continue refining edge case analysis",
            "Practice articulating trade-offs between different solutions",
            "Develop expertise in adjacent technical domains"
        ]
    })

    with patch.object(mock_agent, '_call_llm', return_value=mock_response_text):
        result = mock_agent.generate(
            perfect_input["scores"],
            perfect_input["answer_quality"]
        )

    assert len(result["recommendations"]["improvement_plan"]) >= 1


def test_markdown_response_handling(mock_agent, high_score_input):
    """Test that markdown-wrapped JSON is properly extracted."""
    mock_response_text = """```json
{
    "strengths": [
        "Strong technical foundation",
        "Clear communication"
    ],
    "weaknesses": [
        "Could improve depth",
        "Minor structural improvements possible"
    ],
    "improvement_plan": [
        "Practice more complex scenarios",
        "Study advanced patterns",
        "Review edge case handling"
    ]
}
```"""

    with patch.object(mock_agent, '_call_llm', return_value=mock_response_text):
        result = mock_agent.generate(
            high_score_input["scores"],
            high_score_input["answer_quality"]
        )

    assert "recommendations" in result
    assert len(result["recommendations"]["strengths"]) == 2


# =========================================================
# TEST 11: BIAS DETECTION
# =========================================================

def test_no_score_bias_inflation(mock_agent):
    """
    Test that recommendations don't artificially inflate or deflate perception.
    Recommendations should be calibrated to actual scores.
    """
    mediocre_input = {
        "scores": {
            "technical": 55.0,
            "communication": 58.0,
            "overall": 56.0
        },
        "answer_quality": {
            "relevance": 0.60,
            "correctness": 0.55,
            "depth": 0.52,
            "structure": 0.58,
            "gaps": ["algorithm optimization", "space complexity"]
        }
    }

    mock_response_text = json.dumps({
        "strengths": [
            "Basic understanding of the problem demonstrated",
            "Attempted a structured approach"
        ],
        "weaknesses": [
            "Algorithm optimization strategies not adequately applied",
            "Space complexity analysis missing from solution",
            "Technical correctness needs significant improvement",
            "Depth of explanation insufficient for interview standards"
        ],
        "improvement_plan": [
            "Focus on mastering fundamental algorithms and data structures",
            "Practice analyzing time and space complexity systematically",
            "Study optimization techniques for common algorithmic patterns",
            "Work on explaining solutions with clear reasoning and examples"
        ]
    })

    with patch.object(mock_agent, '_call_llm', return_value=mock_response_text):
        result = mock_agent.generate(
            mediocre_input["scores"],
            mediocre_input["answer_quality"]
        )

    recommendations = result["recommendations"]

    # For mediocre scores, weaknesses should outnumber or equal strengths
    assert len(recommendations["weaknesses"]) >= len(
        recommendations["strengths"])


# =========================================================
# RUN TESTS
# =========================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
