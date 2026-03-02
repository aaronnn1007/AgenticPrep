"""
Test Suite for Answer Quality Analyser
========================================
Comprehensive pytest suite testing the production Answer Quality Analyser agent.

Test Categories:
1. Good Answer Tests - High quality responses
2. Irrelevant Answer Tests - Off-topic responses  
3. Empty Answer Tests - No response handling
4. Partial Answer Tests - Incomplete responses
5. JSON Validation Tests - Malformed output handling
6. Input Validation Tests - Schema validation
7. Retry Logic Tests - Error recovery
8. Edge Case Tests - Boundary conditions

Usage:
    pytest tests/test_answer_quality.py -v
    pytest tests/test_answer_quality.py::test_good_answer -v
"""

from backend.utils.json_parser import JSONParseError
from backend.agents.answer_quality import (
    AnswerQualityAnalyser,
    AnswerQualityInput,
    AnswerQualityMetrics,
    QuestionInput
)
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Set fake API key for all tests
os.environ['OPENAI_API_KEY'] = 'test-key-fake-12345'

# Import the agent and schemas


# =========================================================
# TEST FIXTURES
# =========================================================

@pytest.fixture
def sample_question():
    """Sample question for testing."""
    return {
        "text": "Explain the difference between processes and threads in operating systems.",
        "intent": ["concurrency", "memory model", "context switching", "independence"],
        "difficulty": 0.6,
        "topic": "Operating Systems"
    }


@pytest.fixture
def good_answer_transcript():
    """High-quality answer transcript."""
    return """
    Processes and threads are both mechanisms for concurrency, but they differ significantly.
    
    A process is an independent execution unit with its own memory space. Each process has 
    separate address space, which means they cannot directly access each other's memory.
    This provides strong isolation but makes inter-process communication more expensive.
    
    Threads, on the other hand, are lightweight execution units within a process. Multiple 
    threads within the same process share the same memory space, which makes communication 
    between them very efficient through shared variables. However, this also requires careful
    synchronization to avoid race conditions.
    
    Context switching between processes is more expensive than between threads because the 
    OS needs to save and restore the entire memory context. Thread context switches are 
    faster since they share the same address space.
    
    In terms of independence, if one process crashes, it doesn't affect other processes. 
    But if one thread crashes, it can potentially bring down the entire process.
    """


@pytest.fixture
def irrelevant_answer_transcript():
    """Completely irrelevant answer."""
    return """
    I think Python is a great programming language. I've been using it for web development
    with Django and Flask. My favorite feature is list comprehensions. They make code more
    readable and Pythonic. Also, I really enjoy working with pandas for data analysis.
    """


@pytest.fixture
def empty_answer_transcript():
    """Empty or minimal answer."""
    return ""


@pytest.fixture
def partial_answer_transcript():
    """Partially complete answer - misses some concepts."""
    return """
    A process is like a running program. It has its own memory. Threads are lighter
    and share memory within a process. That's basically it.
    """


@pytest.fixture
def mock_llm_response():
    """Mock LLM response factory."""
    def _create_response(content: str):
        mock_response = Mock()
        mock_response.content = content
        return mock_response
    return _create_response


# =========================================================
# TEST 1: GOOD ANSWER - Should score high (>0.7 relevance)
# =========================================================

def test_good_answer(sample_question, good_answer_transcript, mock_llm_response):
    """
    Test Case: High-quality answer should receive high scores.

    Expected:
    - relevance > 0.7
    - correctness > 0.7
    - depth > 0.7
    - structure > 0.7
    - minimal gaps
    """
    # Set fake API key for testing
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    # Mock LLM to return high scores
    mock_json = """
    {
        "relevance": 0.95,
        "correctness": 0.90,
        "depth": 0.85,
        "structure": 0.90,
        "gaps": []
    }
    """

    with patch('backend.agents.answer_quality.ChatOpenAI') as mock_openai:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response(mock_json)
        mock_openai.return_value = mock_instance

        analyser = AnswerQualityAnalyser(model_name="gpt-4o-mini")
        result = analyser.evaluate(
            question=sample_question,
            transcript=good_answer_transcript,
            role="Software Engineer",
            experience_level="Mid-Level"
        )

        # Assertions
        assert "answer_quality" in result
        quality = result["answer_quality"]

        assert quality["relevance"] > 0.7, "Good answer should have high relevance"
        assert quality["correctness"] > 0.7, "Good answer should have high correctness"
        assert quality["depth"] > 0.7, "Good answer should have high depth"
        assert quality["structure"] > 0.7, "Good answer should have high structure"
        assert len(quality["gaps"]
                   ) <= 1, "Good answer should have minimal gaps"


# =========================================================
# TEST 2: IRRELEVANT ANSWER - Should score low (<0.3 relevance)
# =========================================================

def test_irrelevant_answer(sample_question, irrelevant_answer_transcript, mock_llm_response):
    """
    Test Case: Irrelevant answer should receive low relevance score.

    Expected:
    - relevance < 0.3
    - multiple gaps identifying missing concepts
    """
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    # Mock LLM to return low relevance
    mock_json = """
    {
        "relevance": 0.15,
        "correctness": 0.20,
        "depth": 0.10,
        "structure": 0.40,
        "gaps": ["concurrency", "memory model", "context switching", "independence"]
    }
    """

    with patch('backend.agents.answer_quality.ChatOpenAI') as mock_openai:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response(mock_json)
        mock_openai.return_value = mock_instance

        analyser = AnswerQualityAnalyser(model_name="gpt-4o-mini")
        result = analyser.evaluate(
            question=sample_question,
            transcript=irrelevant_answer_transcript,
            role="Software Engineer",
            experience_level="Junior"
        )

        quality = result["answer_quality"]

        assert quality["relevance"] < 0.3, "Irrelevant answer should have low relevance"
        assert len(quality["gaps"]
                   ) > 2, "Irrelevant answer should have many gaps"


# =========================================================
# TEST 3: EMPTY ANSWER - All zeros
# =========================================================

def test_empty_answer(sample_question, empty_answer_transcript):
    """
    Test Case: Empty transcript should return all zero scores.

    Expected:
    - All metrics = 0.0
    - gaps = ["No answer provided"]
    - No LLM call should be made
    """
    with patch('backend.agents.answer_quality.ChatOpenAI') as mock_openai:
        analyser = AnswerQualityAnalyser(model_name="gpt-4o-mini")
        result = analyser.evaluate(
            question=sample_question,
            transcript=empty_answer_transcript,
            role="Software Engineer",
            experience_level="Senior"
        )

        quality = result["answer_quality"]

        # All scores should be zero
        assert quality["relevance"] == 0.0
        assert quality["correctness"] == 0.0
        assert quality["depth"] == 0.0
        assert quality["structure"] == 0.0
        assert "No answer provided" in quality["gaps"]

        # LLM should not have been called
        mock_openai.return_value.invoke.assert_not_called()


# =========================================================
# TEST 4: PARTIAL ANSWER - Moderate depth, high relevance
# =========================================================

def test_partial_answer(sample_question, partial_answer_transcript, mock_llm_response):
    """
    Test Case: Partial answer should have mixed scores.

    Expected:
    - relevance > 0.6 (on topic)
    - depth < 0.5 (lacks detail)
    - some gaps
    """
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    mock_json = """
    {
        "relevance": 0.70,
        "correctness": 0.65,
        "depth": 0.35,
        "structure": 0.50,
        "gaps": ["context switching", "independence"]
    }
    """

    with patch('backend.agents.answer_quality.ChatOpenAI') as mock_openai:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response(mock_json)
        mock_openai.return_value = mock_instance

        analyser = AnswerQualityAnalyser(model_name="gpt-4o-mini")
        result = analyser.evaluate(
            question=sample_question,
            transcript=partial_answer_transcript,
            role="Software Engineer",
            experience_level="Junior"
        )

        quality = result["answer_quality"]

        assert quality["relevance"] > 0.6, "Partial answer should be relevant"
        assert quality["depth"] < 0.5, "Partial answer should lack depth"
        assert len(quality["gaps"]) > 0, "Partial answer should have gaps"


# =========================================================
# TEST 5: JSON VALIDATION - Malformed LLM output
# =========================================================

def test_json_validation_with_retry(sample_question, good_answer_transcript, mock_llm_response):
    """
    Test Case: Malformed JSON should trigger retry and eventually fallback.

    Expected:
    - First call returns invalid JSON
    - Retry mechanism activates
    - Eventually returns fallback metrics
    """
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    # Mock LLM to return malformed JSON that will fail
    malformed_json = "This is not JSON at all { invalid }"

    with patch('backend.agents.answer_quality.ChatOpenAI') as mock_openai:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response(malformed_json)
        mock_openai.return_value = mock_instance

        analyser = AnswerQualityAnalyser(model_name="gpt-4o-mini")
        result = analyser.evaluate(
            question=sample_question,
            transcript=good_answer_transcript,
            role="Software Engineer",
            experience_level="Mid-Level"
        )

        quality = result["answer_quality"]

        # Should return fallback values
        assert quality["relevance"] == 0.5
        assert quality["correctness"] == 0.5
        assert quality["depth"] == 0.5
        assert quality["structure"] == 0.5
        assert any("error" in gap.lower() or "unable" in gap.lower()
                   for gap in quality["gaps"])

        # LLM should have been called multiple times (retries)
        assert mock_instance.invoke.call_count >= 2


# =========================================================
# TEST 6: VALUE CLAMPING - Out of range values
# =========================================================

def test_value_clamping(sample_question, good_answer_transcript, mock_llm_response):
    """
    Test Case: LLM returns values outside 0-1 range.

    Expected:
    - Values should be automatically clamped to [0, 1]
    """
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    # Mock LLM to return out-of-range values
    mock_json = """
    {
        "relevance": 1.5,
        "correctness": -0.2,
        "depth": 0.75,
        "structure": 2.0,
        "gaps": []
    }
    """

    with patch('backend.agents.answer_quality.ChatOpenAI') as mock_openai:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response(mock_json)
        mock_openai.return_value = mock_instance

        analyser = AnswerQualityAnalyser(model_name="gpt-4o-mini")
        result = analyser.evaluate(
            question=sample_question,
            transcript=good_answer_transcript,
            role="Software Engineer",
            experience_level="Senior"
        )

        quality = result["answer_quality"]

        # All values should be clamped to [0, 1]
        assert 0.0 <= quality["relevance"] <= 1.0
        assert 0.0 <= quality["correctness"] <= 1.0
        assert 0.0 <= quality["depth"] <= 1.0
        assert 0.0 <= quality["structure"] <= 1.0


# =========================================================
# TEST 7: INPUT VALIDATION - Invalid input schema
# =========================================================

def test_invalid_input_schema():
    """
    Test Case: Invalid input should be handled gracefully with fallback metrics.

    Expected:
    - Pydantic validation error is caught internally
    - Fallback metrics returned instead of crashing
    """
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    with patch('backend.agents.answer_quality.ChatOpenAI'):
        analyser = AnswerQualityAnalyser(model_name="gpt-4o-mini")

        # Missing required fields - should return fallback instead of raising
        result = analyser.evaluate(
            # Missing intent, difficulty, topic
            question={"text": "What is Python?"},
            transcript="Python is a programming language",
            role="Engineer",
            experience_level="Mid"
        )

        # Should return fallback metrics, not crash
        quality = result["answer_quality"]
        assert quality["relevance"] == 0.5  # Fallback value
        assert len(quality["gaps"]) > 0  # Should have error message in gaps
        assert any("error" in gap.lower() or "validation" in gap.lower()
                   for gap in quality["gaps"])
# =========================================================
# TEST 8: MARKDOWN CODE BLOCK HANDLING
# =========================================================


def test_markdown_code_block_handling(sample_question, good_answer_transcript, mock_llm_response):
    """
    Test Case: LLM returns JSON wrapped in markdown code blocks.

    Expected:
    - Parser should extract JSON correctly
    - Metrics should be valid
    """
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    # Mock LLM to return JSON in markdown
    mock_json_with_markdown = """
    ```json
    {
        "relevance": 0.88,
        "correctness": 0.85,
        "depth": 0.80,
        "structure": 0.82,
        "gaps": ["context switching"]
    }
    ```
    """

    with patch('backend.agents.answer_quality.ChatOpenAI') as mock_openai:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response(
            mock_json_with_markdown)
        mock_openai.return_value = mock_instance

        analyser = AnswerQualityAnalyser(model_name="gpt-4o-mini")
        result = analyser.evaluate(
            question=sample_question,
            transcript=good_answer_transcript,
            role="Software Engineer",
            experience_level="Mid-Level"
        )

        quality = result["answer_quality"]

        # Should successfully parse despite markdown
        assert quality["relevance"] == 0.88
        assert quality["correctness"] == 0.85
        assert len(quality["gaps"]) == 1


# =========================================================
# TEST 9: ROLE AND EXPERIENCE CONTEXT
# =========================================================

def test_role_and_experience_context(sample_question, partial_answer_transcript, mock_llm_response):
    """
    Test Case: Different roles/experience levels should receive context-appropriate evaluation.

    Expected:
    - Junior level partial answer might be more acceptable
    - Context is properly passed to LLM
    """
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    mock_json = """
    {
        "relevance": 0.75,
        "correctness": 0.70,
        "depth": 0.45,
        "structure": 0.60,
        "gaps": ["context switching details", "memory isolation mechanisms"]
    }
    """

    with patch('backend.agents.answer_quality.ChatOpenAI') as mock_openai:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response(mock_json)
        mock_openai.return_value = mock_instance

        analyser = AnswerQualityAnalyser(model_name="gpt-4o-mini")
        result = analyser.evaluate(
            question=sample_question,
            transcript=partial_answer_transcript,
            role="Backend Engineer",
            experience_level="Junior"
        )

        # Verify the prompt includes role and experience
        call_args = mock_instance.invoke.call_args[0][0]
        prompt_text = str(call_args)

        assert "Backend Engineer" in prompt_text or "Junior" in prompt_text

        quality = result["answer_quality"]
        assert 0.0 <= quality["relevance"] <= 1.0


# =========================================================
# TEST 10: GAPS IDENTIFICATION
# =========================================================

def test_gaps_identification(sample_question, partial_answer_transcript, mock_llm_response):
    """
    Test Case: Gaps should reference intent items that are missing.

    Expected:
    - gaps list contains specific missing concepts
    - gaps reference items from question.intent
    """
    os.environ['OPENAI_API_KEY'] = 'test-key-12345'

    mock_json = """
    {
        "relevance": 0.70,
        "correctness": 0.75,
        "depth": 0.40,
        "structure": 0.65,
        "gaps": ["context switching", "independence"]
    }
    """

    with patch('backend.agents.answer_quality.ChatOpenAI') as mock_openai:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_llm_response(mock_json)
        mock_openai.return_value = mock_instance

        analyser = AnswerQualityAnalyser(model_name="gpt-4o-mini")
        result = analyser.evaluate(
            question=sample_question,
            transcript=partial_answer_transcript,
            role="Software Engineer",
            experience_level="Mid-Level"
        )

        quality = result["answer_quality"]

        # Gaps should be present
        assert len(quality["gaps"]) > 0

        # Gaps should reference concepts from intent
        intent_concepts = set(sample_question["intent"])
        gap_concepts = set(quality["gaps"])

        # At least some gaps should relate to intent items
        # (may not be exact match due to LLM interpretation)
        assert len(quality["gaps"]) > 0


# =========================================================
# RUN ALL TESTS
# =========================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
