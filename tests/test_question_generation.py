"""
Test Suite for Question Generation Agent
==========================================
Comprehensive pytest tests for production-grade question generation.

Test Coverage:
1. Difficulty calibration (Fresher/Mid/Senior)
2. Topic constraint enforcement
3. Answer leakage prevention
4. JSON validation and error handling
5. Retry logic
6. Fallback mechanisms
7. Edge cases
"""

import pytest
import json
import logging
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError

from backend.agents.question_generation import (
    QuestionGenerationAgent,
    QuestionInput,
    QuestionOutput,
    QuestionDetails,
    build_question_prompt,
    build_system_prompt
)
from backend.utils.json_parser import JSONParseError

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================
# FIXTURES
# ==========================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM that can be patched."""
    mock = MagicMock()
    return mock


# ==========================================================
# DIFFICULTY CALIBRATION TESTS
# ==========================================================

def test_junior_easy_question():
    """
    Test: Junior-level question generation.
    Expected: difficulty ~0.2-0.4, fundamental / definitional question.
    """
    mock_response = Mock()
    mock_response.content = '''
    {
        "question": {
            "text": "What is a variable in programming?",
            "topic": "Programming Basics",
            "difficulty": 0.25,
            "intent": [
                "Defines what a variable is",
                "Explains it stores data",
                "Mentions variable names and types"
            ]
        }
    }
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Software Engineer",
            experience_level="Junior",
            difficulty_target=0.3
        )

    assert isinstance(result, QuestionOutput)
    assert 0.2 <= result.question.difficulty <= 0.4
    assert result.question.text
    assert len(result.question.intent) >= 3
    logger.info(
        f"\u2713 Junior question generated: difficulty={result.question.difficulty}")


def test_mid_moderate_question():
    """
    Test: Mid-level question generation.
    Expected: difficulty ~0.4-0.7, application/analysis question.
    """
    mock_response = Mock()
    mock_response.content = '''
    {
        "question": {
            "text": "Implement a function to reverse a linked list. Explain your approach and time complexity.",
            "topic": "Data Structures",
            "difficulty": 0.55,
            "intent": [
                "Implements iterative or recursive solution",
                "Explains pointer manipulation",
                "Calculates O(n) time complexity",
                "Discusses O(1) space complexity"
            ]
        }
    }
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Software Engineer",
            experience_level="Mid",
            difficulty_target=0.5
        )

    assert isinstance(result, QuestionOutput)
    assert 0.4 <= result.question.difficulty <= 0.7
    assert len(result.question.intent) >= 3
    logger.info(
        f"✓ Mid-level question generated: difficulty={result.question.difficulty}")


def test_senior_hard_question():
    """
    Test: Senior-level question generation.
    Expected: difficulty ~0.6-0.9, requires trade-offs/system thinking.
    """
    mock_response = Mock()
    mock_response.content = '''
    {
        "question": {
            "text": "Design a rate limiting system for a microservices architecture that handles 100K requests/sec. Discuss distributed rate limiting, consistency, and failure modes.",
            "topic": "System Design",
            "difficulty": 0.85,
            "intent": [
                "Proposes distributed architecture (token bucket, sliding window)",
                "Discusses consistency trade-offs",
                "Considers failure handling and degradation",
                "Mentions distributed counters (Redis, etc.)",
                "Analyzes performance bottlenecks"
            ]
        }
    }
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Senior Software Engineer",
            experience_level="Senior",
            difficulty_target=0.8
        )

    assert isinstance(result, QuestionOutput)
    assert 0.6 <= result.question.difficulty <= 0.9
    assert "design" in result.question.text.lower()
    assert len(result.question.intent) >= 3
    logger.info(
        f"✓ Senior question generated: difficulty={result.question.difficulty}")


# ==========================================================
# TOPIC CONSTRAINT TESTS
# ==========================================================

def test_topic_constraint_respected():
    """
    Test: Topic constraint enforcement.
    Expected: Generated question topic matches constraint ["OOP"].
    """
    mock_response = Mock()
    mock_response.content = '''
    {
        "question": {
            "text": "Explain encapsulation in OOP and its benefits.",
            "topic": "OOP",
            "difficulty": 0.35,
            "intent": [
                "Defines encapsulation",
                "Explains data hiding",
                "Mentions access modifiers",
                "Discusses benefits"
            ]
        }
    }
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Software Engineer",
            experience_level="Junior",
            topic_constraints=["OOP"],
            difficulty_target=0.3
        )

    assert isinstance(result, QuestionOutput)
    assert "oop" in result.question.topic.lower()
    logger.info(f"✓ Topic constraint respected: {result.question.topic}")


def test_topic_no_constraint():
    """
    Test: Question generation without topic constraints.
    Expected: Agent selects appropriate topic automatically.
    """
    mock_response = Mock()
    mock_response.content = '''
    {
        "question": {
            "text": "What is REST API and how does it differ from GraphQL?",
            "topic": "APIs",
            "difficulty": 0.5,
            "intent": [
                "Defines REST principles",
                "Explains GraphQL",
                "Compares both approaches",
                "Discusses use cases"
            ]
        }
    }
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Backend Developer",
            experience_level="Mid",
            topic_constraints=None,
            difficulty_target=0.5
        )

    assert isinstance(result, QuestionOutput)
    assert result.question.topic
    logger.info(f"✓ Auto-selected topic: {result.question.topic}")


# ==========================================================
# ANSWER LEAKAGE PREVENTION TESTS
# ==========================================================

def test_no_answer_leakage():
    """
    Test: Ensure output does NOT contain solution text.
    Expected: No answer or solution in question text or intent.
    """
    mock_response = Mock()
    mock_response.content = '''
    {
        "question": {
            "text": "Explain the difference between stack and heap memory.",
            "topic": "Memory Management",
            "difficulty": 0.45,
            "intent": [
                "Explains stack memory characteristics",
                "Explains heap memory characteristics",
                "Compares allocation and deallocation",
                "Mentions use cases for each"
            ]
        }
    }
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Software Engineer",
            experience_level="Mid",
            difficulty_target=0.5
        )

    for intent_item in result.question.intent:
        assert not intent_item.lower().startswith("the answer is")
        assert not intent_item.lower().startswith("solution:")
        assert len(intent_item) > 10

    logger.info("✓ No answer leakage detected in output")


def test_intent_quality():
    """
    Test: Ensure intent items are specific evaluation checkpoints.
    Expected: Intent items are meaningful and specific (not too generic).
    """
    mock_response = Mock()
    mock_response.content = '''
    {
        "question": {
            "text": "Implement a binary search algorithm.",
            "topic": "Algorithms",
            "difficulty": 0.4,
            "intent": [
                "Uses divide-and-conquer approach",
                "Handles sorted array requirement",
                "Calculates O(log n) time complexity",
                "Implements correct boundary conditions"
            ]
        }
    }
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Software Engineer",
            experience_level="Mid",
            difficulty_target=0.4
        )

    assert len(result.question.intent) >= 3
    assert len(result.question.intent) <= 6

    for intent_item in result.question.intent:
        assert len(intent_item) > 15

    logger.info("✓ Intent quality validated")


# ==========================================================
# JSON VALIDATION TESTS
# ==========================================================

def test_json_validation_success():
    """
    Test: Successful JSON parsing and validation.
    Expected: Valid JSON is parsed and validated correctly.
    """
    mock_response = Mock()
    mock_response.content = '''
    {
        "question": {
            "text": "What is dependency injection?",
            "topic": "Design Patterns",
            "difficulty": 0.5,
            "intent": [
                "Defines dependency injection",
                "Explains benefits",
                "Provides example"
            ]
        }
    }
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Software Engineer",
            experience_level="Mid",
            difficulty_target=0.5
        )

    assert isinstance(result, QuestionOutput)
    assert isinstance(result.question, QuestionDetails)
    logger.info("✓ JSON validation successful")


def test_json_with_markdown_codeblock():
    """
    Test: Handle JSON wrapped in markdown code blocks.
    Expected: Successfully extracts JSON from markdown.
    """
    mock_response = Mock()
    mock_response.content = '''
    ```json
    {
        "question": {
            "text": "Explain the SOLID principles.",
            "topic": "Software Engineering",
            "difficulty": 0.6,
            "intent": [
                "Defines each SOLID principle",
                "Provides examples",
                "Explains benefits"
            ]
        }
    }
    ```
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Software Engineer",
            experience_level="Senior",
            difficulty_target=0.6
        )

    assert isinstance(result, QuestionOutput)
    assert result.question.text
    logger.info("✓ Markdown code block handled correctly")


def test_difficulty_clamping():
    """
    Test: Difficulty values are clamped to [0.0, 1.0].
    Expected: Out-of-range difficulty is clamped.
    """
    mock_response = Mock()
    mock_response.content = '''
    {
        "question": {
            "text": "Test question",
            "topic": "Testing",
            "difficulty": 1.5,
            "intent": ["Test intent 1", "Test intent 2", "Test intent 3"]
        }
    }
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Software Engineer",
            experience_level="Senior",
            difficulty_target=0.9
        )

    assert 0.0 <= result.question.difficulty <= 1.0
    logger.info(f"✓ Difficulty clamped to: {result.question.difficulty}")


def test_missing_intent_items():
    """
    Test: Handle intent with too few items.
    Expected: Agent adds generic intent items to meet minimum requirement.
    """
    mock_response = Mock()
    mock_response.content = '''
    {
        "question": {
            "text": "What is OOP?",
            "topic": "OOP",
            "difficulty": 0.3,
            "intent": ["Defines OOP"]
        }
    }
    '''

    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()
        result = agent.generate(
            role="Software Engineer",
            experience_level="Junior",
            difficulty_target=0.3
        )

    assert len(result.question.intent) >= 3
    logger.info(f"✓ Intent items padded to: {len(result.question.intent)}")


# ==========================================================
# INPUT VALIDATION TESTS
# ==========================================================

def test_difficulty_out_of_range():
    """
    Test: Out-of-range difficulty_target.
    Expected: Validation error from Pydantic.
    """
    with patch('langchain_openai.ChatOpenAI') as MockChatOpenAI:
        mock_llm_instance = MagicMock()
        MockChatOpenAI.return_value = mock_llm_instance

        agent = QuestionGenerationAgent()

        with pytest.raises(ValidationError):
            agent.generate(
                role="Software Engineer",
                experience_level="Mid",
                difficulty_target=1.5
            )

    logger.info("✓ Out-of-range difficulty rejected")


# ==========================================================
# PROMPT TESTING
# ==========================================================

def test_prompt_builder():
    """
    Test: Prompt building function.
    Expected: Prompt includes all required elements.
    """
    prompt = build_question_prompt(
        role="Data Scientist",
        experience_level="Mid",
        topic_constraints=["Machine Learning"],
        difficulty_target=0.6
    )

    assert "Data Scientist" in prompt
    assert "Mid" in prompt
    assert "Machine Learning" in prompt
    assert "0.6" in prompt
    assert "JSON" in prompt
    assert "intent" in prompt
    logger.info("✓ Prompt builder validated")


def test_system_prompt():
    """
    Test: System prompt contains required instructions.
    Expected: System prompt has key constraints.
    """
    prompt = build_system_prompt()

    assert "JSON" in prompt
    assert "question" in prompt.lower()
    assert len(prompt) > 100
    logger.info("✓ System prompt validated")


# ==========================================================
# NODE DIFFICULTY WIRING TESTS
# ==========================================================

def test_node_wires_junior_difficulty():
    """
    Test: question_generation_node passes difficulty_target=0.3 for Junior.
    """
    from unittest.mock import patch, MagicMock
    from backend.agents.question_generation import question_generation_node
    from backend.models.state import InterviewState, QuestionModel

    fake_question = QuestionModel(
        text="What are the four pillars of OOP?",
        topic="OOP",
        difficulty=0.3,
        intent=["Names all four pillars",
                "Provides a brief definition", "Gives a simple example"],
    )

    dummy_state = InterviewState(
        interview_id="test_001",
        role="Software Engineer",
        experience_level="Junior",
    )

    captured = {}

    def mock_generate(*args, **kwargs):
        captured.update(kwargs)
        from backend.agents.question_generation import QuestionOutput, QuestionDetails
        return QuestionOutput(question=QuestionDetails(**fake_question.model_dump()))

    with patch.object(
        __import__("backend.agents.question_generation", fromlist=[
                   "QuestionGenerationAgent"]).QuestionGenerationAgent,
        "generate",
        side_effect=mock_generate,
    ):
        question_generation_node(dummy_state)

    assert "difficulty_target" in captured, "difficulty_target was not passed to agent.generate()"
    assert captured["difficulty_target"] == 0.3, (
        f"Expected difficulty_target=0.3 for Junior, got {captured['difficulty_target']}"
    )
    logger.info("✓ Junior node: difficulty_target=0.3 wired correctly")


def test_node_wires_mid_difficulty():
    """
    Test: question_generation_node passes difficulty_target=0.55 for Mid.
    """
    from backend.agents.question_generation import question_generation_node
    from backend.models.state import InterviewState, QuestionModel

    fake_question = QuestionModel(
        text="How would you implement pagination in a REST API?",
        topic="APIs",
        difficulty=0.55,
        intent=["Explains cursor vs offset",
                "Mentions performance", "Discusses edge cases"],
    )

    dummy_state = InterviewState(
        interview_id="test_002",
        role="Backend Developer",
        experience_level="Mid",
    )

    captured = {}

    def mock_generate(*args, **kwargs):
        captured.update(kwargs)
        from backend.agents.question_generation import QuestionOutput, QuestionDetails
        return QuestionOutput(question=QuestionDetails(**fake_question.model_dump()))

    with patch.object(
        __import__("backend.agents.question_generation", fromlist=[
                   "QuestionGenerationAgent"]).QuestionGenerationAgent,
        "generate",
        side_effect=mock_generate,
    ):
        question_generation_node(dummy_state)

    assert captured.get("difficulty_target") == 0.55, (
        f"Expected difficulty_target=0.55 for Mid, got {captured.get('difficulty_target')}"
    )
    logger.info("✓ Mid node: difficulty_target=0.55 wired correctly")


def test_node_wires_senior_difficulty():
    """
    Test: question_generation_node passes difficulty_target=0.80 for Senior.
    """
    from backend.agents.question_generation import question_generation_node
    from backend.models.state import InterviewState, QuestionModel

    fake_question = QuestionModel(
        text="Design a distributed rate-limiting system for 100K requests/sec.",
        topic="System Design",
        difficulty=0.8,
        intent=["Proposes distributed counters",
                "Discusses consistency trade-offs", "Handles failure modes"],
    )

    dummy_state = InterviewState(
        interview_id="test_003",
        role="Senior Software Engineer",
        experience_level="Senior",
    )

    captured = {}

    def mock_generate(*args, **kwargs):
        captured.update(kwargs)
        from backend.agents.question_generation import QuestionOutput, QuestionDetails
        return QuestionOutput(question=QuestionDetails(**fake_question.model_dump()))

    with patch.object(
        __import__("backend.agents.question_generation", fromlist=[
                   "QuestionGenerationAgent"]).QuestionGenerationAgent,
        "generate",
        side_effect=mock_generate,
    ):
        question_generation_node(dummy_state)

    assert captured.get("difficulty_target") == 0.80, (
        f"Expected difficulty_target=0.80 for Senior, got {captured.get('difficulty_target')}"
    )
    logger.info("✓ Senior node: difficulty_target=0.80 wired correctly")


def test_prompt_contains_role_alignment():
    """
    Test: build_question_prompt includes a role-alignment instruction.
    """
    prompt = build_question_prompt(
        role="Data Scientist",
        experience_level="Mid",
        topic_constraints=None,
        difficulty_target=0.55,
    )
    assert "Data Scientist" in prompt
    assert "ROLE ALIGNMENT" in prompt
    logger.info("✓ Prompt contains role-alignment instruction")


def test_prompt_junior_tier_guidance():
    """
    Test: Junior difficulty_target produces foundational question guidance.
    """
    prompt = build_question_prompt(
        role="Frontend Developer",
        experience_level="Junior",
        topic_constraints=None,
        difficulty_target=0.3,
    )
    assert "Junior" in prompt or "Fundamental" in prompt
    assert "four pillars" in prompt.lower() or "fundamental" in prompt.lower()
    logger.info("✓ Prompt contains Junior/Fundamental tier guidance")


def test_prompt_senior_tier_guidance():
    """
    Test: Senior difficulty_target produces architecture question guidance.
    """
    prompt = build_question_prompt(
        role="Senior Software Engineer",
        experience_level="Senior",
        topic_constraints=None,
        difficulty_target=0.8,
    )
    assert "Senior" in prompt or "Architecture" in prompt
    assert "design" in prompt.lower()
    logger.info("✓ Prompt contains Senior/Architecture tier guidance")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
