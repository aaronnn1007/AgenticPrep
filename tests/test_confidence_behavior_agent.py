"""
Test Suite for Confidence & Behavioral Inference Agent
=======================================================
Comprehensive pytest test suite for production-grade confidence inference.

Test Coverage:
- High confidence cases
- Nervous behavior detection
- Professional demeanor assessment
- JSON validation failures
- Bias protection
- Output bounds validation
- Edge cases
- Retry logic

Author: Senior AI Backend Engineer
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from backend.agents.confidence_inference import (
    ConfidenceBehaviorInferenceAgent,
    VoiceAnalysisInput,
    AnswerQualityInput
)
from backend.models.state import ConfidenceBehaviorModel
from backend.utils.json_parser import JSONParseError


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def high_confidence_metrics():
    """Metrics indicating high confidence performance."""
    return {
        "voice_analysis": {
            "speech_rate_wpm": 145.0,
            "filler_ratio": 0.03,  # Very low fillers
            "clarity": 0.88,  # High clarity
            "tone": "confident"
        },
        "answer_quality": {
            "relevance": 0.92,
            "correctness": 0.89,  # High correctness
            "depth": 0.85,
            "structure": 0.91,
            "gaps": []
        }
    }


@pytest.fixture
def nervous_metrics():
    """Metrics indicating nervous behavior."""
    return {
        "voice_analysis": {
            "speech_rate_wpm": 180.0,  # Fast speech
            "filler_ratio": 0.18,  # High filler ratio
            "clarity": 0.42,  # Low clarity
            "tone": "nervous"
        },
        "answer_quality": {
            "relevance": 0.65,
            "correctness": 0.58,
            "depth": 0.52,
            "structure": 0.48,  # Poor structure
            "gaps": ["key_concept", "error_handling", "edge_cases"]
        }
    }


@pytest.fixture
def professional_metrics():
    """Metrics indicating professional demeanor."""
    return {
        "voice_analysis": {
            "speech_rate_wpm": 140.0,
            "filler_ratio": 0.05,
            "clarity": 0.85,
            "tone": "confident"
        },
        "answer_quality": {
            "relevance": 0.88,
            "correctness": 0.82,
            "depth": 0.78,
            "structure": 0.92,  # Very high structure
            "gaps": ["optimization"]
        }
    }


@pytest.fixture
def fast_speech_high_clarity_metrics():
    """Fast speech but high clarity - should NOT trigger nervousness."""
    return {
        "voice_analysis": {
            "speech_rate_wpm": 175.0,  # Fast speech
            "filler_ratio": 0.04,  # Low fillers
            "clarity": 0.87,  # High clarity
            "tone": "confident"
        },
        "answer_quality": {
            "relevance": 0.85,
            "correctness": 0.88,
            "depth": 0.82,
            "structure": 0.86,
            "gaps": []
        }
    }


@pytest.fixture
def mock_agent():
    """Mock agent for testing without actual LLM calls."""
    with patch('backend.agents.confidence_inference.get_llm_config') as mock_config:
        mock_config.return_value = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "base_url": None,
            "temperature": 0.2
        }

        with patch('backend.agents.confidence_inference.ChatOpenAI'):
            agent = ConfidenceBehaviorInferenceAgent()
            return agent


# =========================================================
# TEST 1: HIGH CONFIDENCE CASE
# =========================================================

def test_high_confidence_case(mock_agent, high_confidence_metrics):
    """
    Test that high correctness + low filler ratio → high confidence.

    Expected:
    - confidence > 0.7
    - nervousness < 0.4
    - professionalism > 0.7
    """
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = json.dumps({
        "confidence": 0.85,
        "nervousness": 0.18,
        "professionalism": 0.82,
        "behavioral_flags": ["confident_technical_delivery", "strong_communication"]
    })

    mock_agent.llm.invoke = Mock(return_value=mock_response)

    # Execute inference
    result = mock_agent.infer(
        high_confidence_metrics["voice_analysis"],
        high_confidence_metrics["answer_quality"]
    )

    # Assertions
    assert isinstance(result, ConfidenceBehaviorModel)
    assert result.confidence > 0.7, f"Expected confidence > 0.7, got {result.confidence}"
    assert result.nervousness < 0.4, f"Expected nervousness < 0.4, got {result.nervousness}"
    assert result.professionalism > 0.7, f"Expected professionalism > 0.7, got {result.professionalism}"
    assert isinstance(result.behavioral_flags, list)

    print(f"✓ High confidence test passed: confidence={result.confidence:.2f}")


# =========================================================
# TEST 2: NERVOUS CASE
# =========================================================

def test_nervous_case(mock_agent, nervous_metrics):
    """
    Test that high filler ratio + low clarity → increased nervousness.

    Expected:
    - nervousness > 0.6
    - confidence < 0.5
    - behavioral_flags contain nervousness indicators
    """
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = json.dumps({
        "confidence": 0.42,
        "nervousness": 0.71,
        "professionalism": 0.48,
        "behavioral_flags": ["needs_composure_improvement", "high_filler_usage"]
    })

    mock_agent.llm.invoke = Mock(return_value=mock_response)

    # Execute inference
    result = mock_agent.infer(
        nervous_metrics["voice_analysis"],
        nervous_metrics["answer_quality"]
    )

    # Assertions
    assert result.nervousness > 0.6, f"Expected nervousness > 0.6, got {result.nervousness}"
    assert result.confidence < 0.5, f"Expected confidence < 0.5, got {result.confidence}"
    assert len(
        result.behavioral_flags) > 0, "Expected behavioral flags for nervous case"

    print(f"✓ Nervous case test passed: nervousness={result.nervousness:.2f}")


# =========================================================
# TEST 3: PROFESSIONAL STRUCTURE
# =========================================================

def test_professional_structure(mock_agent, professional_metrics):
    """
    Test that high structure + steady tone → high professionalism.

    Expected:
    - professionalism > 0.7
    - confidence > 0.6
    """
    # Mock LLM response
    mock_response = Mock()
    mock_response.content = json.dumps({
        "confidence": 0.78,
        "nervousness": 0.24,
        "professionalism": 0.86,
        "behavioral_flags": ["professional_demeanor", "strong_structural_communication"]
    })

    mock_agent.llm.invoke = Mock(return_value=mock_response)

    # Execute inference
    result = mock_agent.infer(
        professional_metrics["voice_analysis"],
        professional_metrics["answer_quality"]
    )

    # Assertions
    assert result.professionalism > 0.7, f"Expected professionalism > 0.7, got {result.professionalism}"
    assert result.confidence > 0.6, f"Expected confidence > 0.6, got {result.confidence}"

    print(
        f"✓ Professional structure test passed: professionalism={result.professionalism:.2f}")


# =========================================================
# TEST 4: JSON VALIDATION FAILURE
# =========================================================

def test_json_validation_failure(mock_agent, high_confidence_metrics):
    """
    Test that malformed JSON triggers retry logic and eventually fails gracefully.

    Expected:
    - JSONParseError raised after max retries
    """
    # Mock LLM to return malformed JSON
    mock_response = Mock()
    mock_response.content = "This is not valid JSON at all, just some text."

    mock_agent.llm.invoke = Mock(return_value=mock_response)

    # Execute and expect failure
    with pytest.raises(JSONParseError):
        mock_agent.infer(
            high_confidence_metrics["voice_analysis"],
            high_confidence_metrics["answer_quality"]
        )

    # Verify retry attempts were made
    assert mock_agent.llm.invoke.call_count >= 2, "Expected at least 2 retry attempts"

    print("✓ JSON validation failure test passed (correctly raised JSONParseError)")


# =========================================================
# TEST 5: BIAS PROTECTION
# =========================================================

def test_bias_protection(mock_agent, fast_speech_high_clarity_metrics):
    """
    Test that fast speech + high clarity does NOT trigger nervousness spike.

    This tests bias protection: fast speech alone should NOT be penalized.

    Expected:
    - nervousness < 0.4 (should not be high)
    - confidence should remain reasonable (> 0.6)
    """
    # Mock LLM response - should recognize confidence despite fast speech
    mock_response = Mock()
    mock_response.content = json.dumps({
        "confidence": 0.81,
        "nervousness": 0.22,  # Low nervousness despite fast speech
        "professionalism": 0.79,
        "behavioral_flags": ["confident_technical_delivery", "fast_articulate_communicator"]
    })

    mock_agent.llm.invoke = Mock(return_value=mock_response)

    # Execute inference
    result = mock_agent.infer(
        fast_speech_high_clarity_metrics["voice_analysis"],
        fast_speech_high_clarity_metrics["answer_quality"]
    )

    # Assertions - bias protection check
    assert result.nervousness < 0.4, (
        f"BIAS VIOLATION: Fast speech + high clarity should NOT trigger high nervousness. "
        f"Got nervousness={result.nervousness}"
    )
    assert result.confidence > 0.6, (
        f"Fast speech with high clarity should maintain confidence. Got {result.confidence}"
    )

    print(
        f"✓ Bias protection test passed: nervousness={result.nervousness:.2f} (no false penalty)")


# =========================================================
# TEST 6: OUTPUT BOUNDS VALIDATION
# =========================================================

def test_output_bounds(mock_agent, high_confidence_metrics):
    """
    Test that all numeric outputs are clamped between 0 and 1.

    Expected:
    - All numeric values in [0.0, 1.0]
    - Out-of-bounds values are clamped
    """
    # Mock LLM response with out-of-bounds values
    mock_response = Mock()
    mock_response.content = json.dumps({
        "confidence": 1.5,  # Out of bounds
        "nervousness": -0.2,  # Out of bounds
        "professionalism": 0.7,
        "behavioral_flags": ["test"]
    })

    mock_agent.llm.invoke = Mock(return_value=mock_response)

    # Execute inference
    result = mock_agent.infer(
        high_confidence_metrics["voice_analysis"],
        high_confidence_metrics["answer_quality"]
    )

    # Assertions - bounds checking
    assert 0.0 <= result.confidence <= 1.0, f"Confidence out of bounds: {result.confidence}"
    assert 0.0 <= result.nervousness <= 1.0, f"Nervousness out of bounds: {result.nervousness}"
    assert 0.0 <= result.professionalism <= 1.0, f"Professionalism out of bounds: {result.professionalism}"

    # Verify clamping occurred
    assert result.confidence == 1.0, "Expected confidence to be clamped to 1.0"
    assert result.nervousness == 0.0, "Expected nervousness to be clamped to 0.0"

    print("✓ Output bounds test passed (values correctly clamped)")


# =========================================================
# TEST 7: INPUT VALIDATION
# =========================================================

def test_input_validation_voice_analysis(mock_agent):
    """Test that invalid voice analysis input is rejected."""
    invalid_voice = {
        "speech_rate_wpm": -50.0,  # Invalid: negative
        "filler_ratio": 1.5,  # Invalid: > 1.0
        "clarity": 0.8,
        "tone": "neutral"
    }

    valid_answer = {
        "relevance": 0.8,
        "correctness": 0.7,
        "depth": 0.6,
        "structure": 0.75,
        "gaps": []
    }

    with pytest.raises(ValueError):
        mock_agent.infer(invalid_voice, valid_answer)

    print("✓ Input validation test passed (rejected invalid voice analysis)")


def test_input_validation_answer_quality(mock_agent):
    """Test that invalid answer quality input is rejected."""
    valid_voice = {
        "speech_rate_wpm": 145.0,
        "filler_ratio": 0.05,
        "clarity": 0.8,
        "tone": "confident"
    }

    invalid_answer = {
        "relevance": 1.2,  # Invalid: > 1.0
        "correctness": 0.7,
        "depth": -0.1,  # Invalid: < 0.0
        "structure": 0.75,
        "gaps": []
    }

    with pytest.raises(ValueError):
        mock_agent.infer(valid_voice, invalid_answer)

    print("✓ Input validation test passed (rejected invalid answer quality)")


# =========================================================
# TEST 8: BEHAVIORAL FLAGS DEDUPLICATION
# =========================================================

def test_behavioral_flags_deduplication(mock_agent, high_confidence_metrics):
    """Test that duplicate behavioral flags are removed."""
    # Mock LLM response with duplicate flags
    mock_response = Mock()
    mock_response.content = json.dumps({
        "confidence": 0.8,
        "nervousness": 0.2,
        "professionalism": 0.75,
        "behavioral_flags": [
            "confident_delivery",
            "Confident_Delivery",  # Duplicate (case-insensitive)
            "strong_technical",
            "confident_delivery"  # Duplicate (exact)
        ]
    })

    mock_agent.llm.invoke = Mock(return_value=mock_response)

    # Execute inference
    result = mock_agent.infer(
        high_confidence_metrics["voice_analysis"],
        high_confidence_metrics["answer_quality"]
    )

    # Assertions
    assert len(result.behavioral_flags) == 2, (
        f"Expected 2 unique flags after deduplication, got {len(result.behavioral_flags)}"
    )

    print(
        f"✓ Deduplication test passed: {len(result.behavioral_flags)} unique flags")


# =========================================================
# TEST 9: EMPTY GAPS HANDLING
# =========================================================

def test_empty_gaps_handling(mock_agent):
    """Test that empty gaps list is handled correctly."""
    voice_analysis = {
        "speech_rate_wpm": 150.0,
        "filler_ratio": 0.04,
        "clarity": 0.85,
        "tone": "confident"
    }

    answer_quality = {
        "relevance": 0.9,
        "correctness": 0.88,
        "depth": 0.85,
        "structure": 0.9,
        "gaps": []  # Empty gaps
    }

    # Mock response
    mock_response = Mock()
    mock_response.content = json.dumps({
        "confidence": 0.88,
        "nervousness": 0.15,
        "professionalism": 0.85,
        "behavioral_flags": ["excellent_performance"]
    })

    mock_agent.llm.invoke = Mock(return_value=mock_response)

    # Should not raise exception
    result = mock_agent.infer(voice_analysis, answer_quality)

    assert isinstance(result, ConfidenceBehaviorModel)
    print("✓ Empty gaps handling test passed")


# =========================================================
# TEST 10: MARKDOWN CODE BLOCK HANDLING
# =========================================================

def test_markdown_code_block_handling(mock_agent, high_confidence_metrics):
    """Test that JSON wrapped in markdown code blocks is parsed correctly."""
    # Mock LLM response with markdown code blocks
    mock_response = Mock()
    mock_response.content = """```json
{
    "confidence": 0.82,
    "nervousness": 0.21,
    "professionalism": 0.79,
    "behavioral_flags": ["confident_delivery"]
}
```"""

    mock_agent.llm.invoke = Mock(return_value=mock_response)

    # Execute inference
    result = mock_agent.infer(
        high_confidence_metrics["voice_analysis"],
        high_confidence_metrics["answer_quality"]
    )

    # Assertions
    assert isinstance(result, ConfidenceBehaviorModel)
    assert result.confidence == 0.82

    print("✓ Markdown code block handling test passed")


# =========================================================
# TEST 11: TEMPERATURE CONFIGURATION
# =========================================================

def test_temperature_configuration():
    """Test that temperature is enforced at 0.2 for consistency."""
    with patch('backend.agents.confidence_inference.get_llm_config') as mock_config:
        mock_config.return_value = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "base_url": None,
            "temperature": 0.7  # Different temperature
        }

        with patch('backend.agents.confidence_inference.ChatOpenAI') as mock_llm_class:
            agent = ConfidenceBehaviorInferenceAgent()

            # Verify temperature was overridden to 0.2
            assert agent.config["temperature"] == 0.2, (
                f"Expected temperature=0.2, got {agent.config['temperature']}"
            )

    print("✓ Temperature configuration test passed (enforced 0.2)")


# =========================================================
# TEST 12: MODEL OVERRIDE
# =========================================================

def test_model_override():
    """Test that model can be overridden via constructor."""
    with patch('backend.agents.confidence_inference.get_llm_config') as mock_config:
        mock_config.return_value = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "base_url": None,
            "temperature": 0.2
        }

        with patch('backend.agents.confidence_inference.ChatOpenAI'):
            agent = ConfidenceBehaviorInferenceAgent(model_name="gpt-4o")

            # Verify model was overridden
            assert agent.config["model"] == "gpt-4o", (
                f"Expected model='gpt-4o', got {agent.config['model']}"
            )

    print("✓ Model override test passed")


# =========================================================
# RUN ALL TESTS
# =========================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
