"""
Test Suite for Scoring & Aggregation Agent
==========================================
Comprehensive test coverage for deterministic scoring engine.

Test Categories:
1. Perfect answer (all 1.0 values)
2. Zero answer (all 0.0 values)
3. Partial answer (mixed values)
4. Out-of-bounds input (clamping validation)
5. Missing fields (validation errors)
6. Weight adjustment (configuration testing)
7. Edge cases (boundary protection)
8. Precision testing (decimal accuracy)
"""

import pytest
from pydantic import ValidationError

from backend.agents.scoring_aggregation import (
    ScoringAggregationAgent,
    AnswerQualityInput,
    ScoreOutput,
    compute_scores
)
from backend.config import SCORING_WEIGHTS


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def agent():
    """Create a fresh agent instance for each test."""
    return ScoringAggregationAgent()


@pytest.fixture
def perfect_answer():
    """Perfect answer with all metrics at maximum."""
    return {
        "relevance": 1.0,
        "correctness": 1.0,
        "depth": 1.0,
        "structure": 1.0,
        "gaps": []
    }


@pytest.fixture
def zero_answer():
    """Zero answer with all metrics at minimum."""
    return {
        "relevance": 0.0,
        "correctness": 0.0,
        "depth": 0.0,
        "structure": 0.0,
        "gaps": ["everything"]
    }


@pytest.fixture
def partial_answer():
    """Partial answer with mixed metric values."""
    return {
        "relevance": 0.8,
        "correctness": 0.9,
        "depth": 0.7,
        "structure": 0.85,
        "gaps": ["advanced concepts"]
    }


# =============================================================================
# TEST 1: PERFECT ANSWER
# =============================================================================

def test_perfect_answer(agent, perfect_answer):
    """
    Test that perfect answer produces 100.0 scores.

    Expected behavior:
    - All input metrics = 1.0
    - All output scores = 100.0
    - No rounding errors
    """
    result = agent.compute(perfect_answer)

    assert result.scores.technical == 100.0, \
        "Perfect answer should yield technical score of 100.0"

    assert result.scores.communication == 100.0, \
        "Perfect answer should yield communication score of 100.0"

    assert result.scores.overall == 100.0, \
        "Perfect answer should yield overall score of 100.0"


def test_perfect_answer_convenience_function(perfect_answer):
    """Test convenience function with perfect answer."""
    scores = compute_scores(perfect_answer)

    assert scores["technical"] == 100.0
    assert scores["communication"] == 100.0
    assert scores["overall"] == 100.0


# =============================================================================
# TEST 2: ZERO ANSWER
# =============================================================================

def test_zero_answer(agent, zero_answer):
    """
    Test that zero answer produces 0.0 scores.

    Expected behavior:
    - All input metrics = 0.0
    - All output scores = 0.0
    - No negative values
    """
    result = agent.compute(zero_answer)

    assert result.scores.technical == 0.0, \
        "Zero answer should yield technical score of 0.0"

    assert result.scores.communication == 0.0, \
        "Zero answer should yield communication score of 0.0"

    assert result.scores.overall == 0.0, \
        "Zero answer should yield overall score of 0.0"


def test_zero_answer_gaps_preserved(agent, zero_answer):
    """Test that gaps are preserved but don't affect scoring."""
    # Gaps should not affect numerical scores
    result = agent.compute(zero_answer)
    assert result.scores.overall == 0.0


# =============================================================================
# TEST 3: PARTIAL ANSWER
# =============================================================================

def test_partial_answer_technical_score(agent, partial_answer):
    """
    Test technical score calculation with partial answer.

    Formula: (correctness * 0.6 + depth * 0.4) * 100
    Expected: (0.9 * 0.6 + 0.7 * 0.4) * 100 = 82.0
    """
    result = agent.compute(partial_answer)

    # Calculate expected technical score
    w_correctness = SCORING_WEIGHTS['technical']['correctness']
    w_depth = SCORING_WEIGHTS['technical']['depth']
    expected_technical = (
        partial_answer['correctness'] * w_correctness +
        partial_answer['depth'] * w_depth
    ) * 100.0

    assert result.scores.technical == round(expected_technical, 2), \
        f"Technical score mismatch. Expected {expected_technical:.2f}"


def test_partial_answer_communication_score(agent, partial_answer):
    """
    Test communication score calculation with partial answer.

    Formula: (structure * 0.7 + relevance * 0.3) * 100
    Expected: (0.85 * 0.7 + 0.8 * 0.3) * 100 = 83.5
    """
    result = agent.compute(partial_answer)

    # Calculate expected communication score
    w_structure = SCORING_WEIGHTS['communication']['structure']
    w_relevance = SCORING_WEIGHTS['communication']['relevance']
    expected_communication = (
        partial_answer['structure'] * w_structure +
        partial_answer['relevance'] * w_relevance
    ) * 100.0

    assert result.scores.communication == round(expected_communication, 2), \
        f"Communication score mismatch. Expected {expected_communication:.2f}"


def test_partial_answer_overall_score(agent, partial_answer):
    """
    Test overall score calculation with partial answer.

    Formula: technical * 0.6 + communication * 0.4
    """
    result = agent.compute(partial_answer)

    # Calculate expected scores
    w_tech_corr = SCORING_WEIGHTS['technical']['correctness']
    w_tech_depth = SCORING_WEIGHTS['technical']['depth']
    technical = (
        partial_answer['correctness'] * w_tech_corr +
        partial_answer['depth'] * w_tech_depth
    ) * 100.0

    w_comm_struct = SCORING_WEIGHTS['communication']['structure']
    w_comm_rel = SCORING_WEIGHTS['communication']['relevance']
    communication = (
        partial_answer['structure'] * w_comm_struct +
        partial_answer['relevance'] * w_comm_rel
    ) * 100.0

    w_overall_tech = SCORING_WEIGHTS['overall']['technical']
    w_overall_comm = SCORING_WEIGHTS['overall']['communication']
    expected_overall = (
        technical * w_overall_tech +
        communication * w_overall_comm
    )

    assert result.scores.overall == round(expected_overall, 2), \
        f"Overall score mismatch. Expected {expected_overall:.2f}"


# =============================================================================
# TEST 4: OUT-OF-BOUNDS INPUT (CLAMPING)
# =============================================================================

def test_out_of_bounds_high_values(agent):
    """
    Test that values > 1.0 are automatically clamped to 1.0.

    Expected behavior:
    - Input values > 1.0 are clamped to 1.0
    - No validation errors raised
    - Scores computed correctly with clamped values
    """
    out_of_bounds_input = {
        "relevance": 1.5,
        "correctness": 2.0,
        "depth": 1.2,
        "structure": 1.8,
        "gaps": []
    }

    # Should not raise error - values are clamped
    result = agent.compute(out_of_bounds_input)

    # With all values clamped to 1.0, scores should be 100.0
    assert result.scores.technical == 100.0
    assert result.scores.communication == 100.0
    assert result.scores.overall == 100.0


def test_out_of_bounds_low_values(agent):
    """
    Test that values < 0.0 are automatically clamped to 0.0.

    Expected behavior:
    - Input values < 0.0 are clamped to 0.0
    - No validation errors raised
    - Scores computed correctly with clamped values
    """
    out_of_bounds_input = {
        "relevance": -0.5,
        "correctness": -1.0,
        "depth": -0.3,
        "structure": -0.2,
        "gaps": []
    }

    # Should not raise error - values are clamped
    result = agent.compute(out_of_bounds_input)

    # With all values clamped to 0.0, scores should be 0.0
    assert result.scores.technical == 0.0
    assert result.scores.communication == 0.0
    assert result.scores.overall == 0.0


def test_mixed_out_of_bounds_values(agent):
    """Test clamping with mixed in-bounds and out-of-bounds values."""
    mixed_input = {
        "relevance": 1.2,    # Will be clamped to 1.0
        "correctness": 0.5,  # Valid
        "depth": -0.1,       # Will be clamped to 0.0
        "structure": 0.8,    # Valid
        "gaps": []
    }

    result = agent.compute(mixed_input)

    # Technical: (0.5 * 0.6 + 0.0 * 0.4) * 100 = 30.0
    expected_technical = (0.5 * 0.6 + 0.0 * 0.4) * 100.0
    assert result.scores.technical == round(expected_technical, 2)

    # Communication: (0.8 * 0.7 + 1.0 * 0.3) * 100 = 86.0
    expected_communication = (0.8 * 0.7 + 1.0 * 0.3) * 100.0
    assert result.scores.communication == round(expected_communication, 2)


# =============================================================================
# TEST 5: MISSING FIELDS (VALIDATION ERRORS)
# =============================================================================

def test_missing_required_field_relevance(agent):
    """Test that missing 'relevance' field raises validation error."""
    incomplete_input = {
        "correctness": 0.8,
        "depth": 0.7,
        "structure": 0.9,
        "gaps": []
    }

    with pytest.raises(ValidationError) as exc_info:
        agent.compute(incomplete_input)

    # Check that error mentions missing field
    assert "relevance" in str(exc_info.value).lower()


def test_missing_required_field_correctness(agent):
    """Test that missing 'correctness' field raises validation error."""
    incomplete_input = {
        "relevance": 0.8,
        "depth": 0.7,
        "structure": 0.9,
        "gaps": []
    }

    with pytest.raises(ValidationError) as exc_info:
        agent.compute(incomplete_input)

    assert "correctness" in str(exc_info.value).lower()


def test_missing_required_field_depth(agent):
    """Test that missing 'depth' field raises validation error."""
    incomplete_input = {
        "relevance": 0.8,
        "correctness": 0.9,
        "structure": 0.9,
        "gaps": []
    }

    with pytest.raises(ValidationError) as exc_info:
        agent.compute(incomplete_input)

    assert "depth" in str(exc_info.value).lower()


def test_missing_required_field_structure(agent):
    """Test that missing 'structure' field raises validation error."""
    incomplete_input = {
        "relevance": 0.8,
        "correctness": 0.9,
        "depth": 0.7,
        "gaps": []
    }

    with pytest.raises(ValidationError) as exc_info:
        agent.compute(incomplete_input)

    assert "structure" in str(exc_info.value).lower()


def test_gaps_field_optional(agent):
    """Test that 'gaps' field is optional and defaults to empty list."""
    input_without_gaps = {
        "relevance": 0.8,
        "correctness": 0.9,
        "depth": 0.7,
        "structure": 0.85
    }

    # Should not raise error
    result = agent.compute(input_without_gaps)
    assert result.scores.overall > 0


# =============================================================================
# TEST 6: WEIGHT ADJUSTMENT
# =============================================================================

def test_weight_adjustment_technical():
    """
    Test that modifying technical weights affects scores correctly.

    This test verifies that the agent properly loads and applies
    configured weights.
    """
    # Create agent with default weights
    agent = ScoringAggregationAgent()

    # Test input with different correctness and depth
    test_input = {
        "relevance": 0.5,
        "correctness": 1.0,  # High correctness
        "depth": 0.0,        # Low depth
        "structure": 0.5,
        "gaps": []
    }

    result = agent.compute(test_input)

    # With default weights (correctness: 0.6, depth: 0.4):
    # Technical = (1.0 * 0.6 + 0.0 * 0.4) * 100 = 60.0
    expected_technical = (1.0 * 0.6 + 0.0 * 0.4) * 100.0
    assert result.scores.technical == round(expected_technical, 2)


def test_weight_adjustment_communication():
    """Test that communication weights are properly applied."""
    agent = ScoringAggregationAgent()

    # Test input with different structure and relevance
    test_input = {
        "relevance": 0.0,    # Low relevance
        "correctness": 0.5,
        "depth": 0.5,
        "structure": 1.0,    # High structure
        "gaps": []
    }

    result = agent.compute(test_input)

    # With default weights (structure: 0.7, relevance: 0.3):
    # Communication = (1.0 * 0.7 + 0.0 * 0.3) * 100 = 70.0
    expected_communication = (1.0 * 0.7 + 0.0 * 0.3) * 100.0
    assert result.scores.communication == round(expected_communication, 2)


def test_weight_adjustment_overall():
    """Test that overall weights are properly applied."""
    agent = ScoringAggregationAgent()

    # Create scenario where technical and communication differ significantly
    test_input = {
        "relevance": 0.0,
        "correctness": 1.0,  # High technical components
        "depth": 1.0,
        "structure": 0.0,    # Low communication components
        "gaps": []
    }

    result = agent.compute(test_input)

    # Technical = 100.0, Communication = 0.0
    # Overall = 100.0 * 0.6 + 0.0 * 0.4 = 60.0
    assert result.scores.technical == 100.0
    assert result.scores.communication == 0.0
    assert result.scores.overall == 60.0


# =============================================================================
# TEST 7: EDGE CASES AND BOUNDARY PROTECTION
# =============================================================================

def test_all_boundary_values(agent):
    """Test all valid boundary combinations."""
    boundary_cases = [
        {"relevance": 0.0, "correctness": 0.0, "depth": 0.0, "structure": 0.0},
        {"relevance": 1.0, "correctness": 1.0, "depth": 1.0, "structure": 1.0},
        {"relevance": 0.5, "correctness": 0.5, "depth": 0.5, "structure": 0.5},
    ]

    for case in boundary_cases:
        case["gaps"] = []
        result = agent.compute(case)

        # All scores should be in valid range
        assert 0.0 <= result.scores.technical <= 100.0
        assert 0.0 <= result.scores.communication <= 100.0
        assert 0.0 <= result.scores.overall <= 100.0


def test_score_never_exceeds_100(agent):
    """Test that scores never exceed 100.0 even with extreme inputs."""
    extreme_input = {
        "relevance": 1.0,
        "correctness": 1.0,
        "depth": 1.0,
        "structure": 1.0,
        "gaps": []
    }

    result = agent.compute(extreme_input)

    assert result.scores.technical <= 100.0
    assert result.scores.communication <= 100.0
    assert result.scores.overall <= 100.0


def test_score_never_negative(agent):
    """Test that scores are never negative even with extreme inputs."""
    extreme_input = {
        "relevance": 0.0,
        "correctness": 0.0,
        "depth": 0.0,
        "structure": 0.0,
        "gaps": []
    }

    result = agent.compute(extreme_input)

    assert result.scores.technical >= 0.0
    assert result.scores.communication >= 0.0
    assert result.scores.overall >= 0.0


# =============================================================================
# TEST 8: PRECISION TESTING
# =============================================================================

def test_two_decimal_precision(agent, partial_answer):
    """Test that all scores are rounded to exactly 2 decimal places."""
    result = agent.compute(partial_answer)

    # Check that scores have at most 2 decimal places
    def count_decimals(value):
        str_value = str(value)
        if '.' in str_value:
            return len(str_value.split('.')[1])
        return 0

    assert count_decimals(result.scores.technical) <= 2
    assert count_decimals(result.scores.communication) <= 2
    assert count_decimals(result.scores.overall) <= 2


def test_deterministic_output(agent, partial_answer):
    """
    Test that multiple computations produce identical results.

    Critical for reproducibility and auditability.
    """
    result1 = agent.compute(partial_answer)
    result2 = agent.compute(partial_answer)
    result3 = agent.compute(partial_answer)

    assert result1.scores.technical == result2.scores.technical
    assert result1.scores.communication == result2.scores.communication
    assert result1.scores.overall == result2.scores.overall

    assert result1.scores.technical == result3.scores.technical
    assert result1.scores.communication == result3.scores.communication
    assert result1.scores.overall == result3.scores.overall


# =============================================================================
# TEST 9: PYDANTIC MODEL VALIDATION
# =============================================================================

def test_answer_quality_input_model_validation():
    """Test AnswerQualityInput Pydantic model validation."""
    # Valid input
    valid_input = AnswerQualityInput(
        relevance=0.8,
        correctness=0.9,
        depth=0.7,
        structure=0.85,
        gaps=["some gap"]
    )

    assert valid_input.relevance == 0.8
    assert valid_input.correctness == 0.9
    assert valid_input.gaps == ["some gap"]


def test_score_output_model_validation():
    """Test ScoreOutput Pydantic model validation."""
    # Valid output
    valid_output = ScoreOutput(
        technical=85.5,
        communication=90.25,
        overall=87.35
    )

    assert valid_output.technical == 85.5
    assert valid_output.communication == 90.25
    assert valid_output.overall == 87.35


def test_invalid_score_output_out_of_range():
    """Test that ScoreOutput rejects out-of-range values."""
    # Try to create output with score > 100
    with pytest.raises(ValidationError):
        ScoreOutput(
            technical=105.0,
            communication=90.0,
            overall=95.0
        )

    # Try to create output with negative score
    with pytest.raises(ValidationError):
        ScoreOutput(
            technical=-5.0,
            communication=90.0,
            overall=85.0
        )


# =============================================================================
# TEST 10: INTEGRATION TESTS
# =============================================================================

def test_full_workflow_integration(agent):
    """Test complete workflow from input to output."""
    answer_quality = {
        "relevance": 0.85,
        "correctness": 0.92,
        "depth": 0.78,
        "structure": 0.88,
        "gaps": ["error handling edge cases"]
    }

    result = agent.compute(answer_quality)

    # Verify structure
    assert hasattr(result, 'scores')
    assert hasattr(result.scores, 'technical')
    assert hasattr(result.scores, 'communication')
    assert hasattr(result.scores, 'overall')

    # Verify all scores are valid
    assert 0.0 <= result.scores.technical <= 100.0
    assert 0.0 <= result.scores.communication <= 100.0
    assert 0.0 <= result.scores.overall <= 100.0

    # Verify scores make sense (overall should be weighted average)
    w_tech = SCORING_WEIGHTS['overall']['technical']
    w_comm = SCORING_WEIGHTS['overall']['communication']
    expected_overall = (
        result.scores.technical * w_tech +
        result.scores.communication * w_comm
    )
    assert abs(result.scores.overall - expected_overall) < 0.01


def test_multiple_agents_independent():
    """Test that multiple agent instances are independent."""
    agent1 = ScoringAggregationAgent()
    agent2 = ScoringAggregationAgent()

    input_data = {
        "relevance": 0.7,
        "correctness": 0.8,
        "depth": 0.6,
        "structure": 0.75,
        "gaps": []
    }

    result1 = agent1.compute(input_data)
    result2 = agent2.compute(input_data)

    # Both should produce identical results
    assert result1.scores.technical == result2.scores.technical
    assert result1.scores.communication == result2.scores.communication
    assert result1.scores.overall == result2.scores.overall


# =============================================================================
# TEST 11: ERROR HANDLING
# =============================================================================

def test_invalid_input_type():
    """Test that invalid input types raise appropriate errors."""
    agent = ScoringAggregationAgent()

    # Try with string instead of dict
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        agent.compute("invalid input")

    # Try with None
    with pytest.raises((ValidationError, TypeError, AttributeError)):
        agent.compute(None)


def test_non_numeric_values():
    """Test that non-numeric values in fields raise validation errors."""
    agent = ScoringAggregationAgent()

    invalid_input = {
        "relevance": "high",  # String instead of float
        "correctness": 0.9,
        "depth": 0.8,
        "structure": 0.85,
        "gaps": []
    }

    with pytest.raises(ValidationError):
        agent.compute(invalid_input)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
