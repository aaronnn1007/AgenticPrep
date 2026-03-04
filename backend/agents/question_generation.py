"""
Question Generation Agent
=========================
Production-grade agent for generating interview questions tailored to role and experience level.

Architecture:
- Single responsibility: Generate relevant interview questions
- Uses LLM (GPT-4o-mini primary, GPT-4o fallback, configurable)
- Enforces strict JSON output schema with automatic retry
- Difficulty calibration aligned with experience level
- Topic constraint validation and enforcement
- No scoring/evaluation logic - only question generation
- No orchestration logic

Design Principles:
- Deterministic structure (no prose in output)
- Automatic retry on JSON parsing failures (max 2 retries)
- Topic validation with regeneration
- Configurable via environment variables
- Production-grade error handling and logging
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

from backend.config import get_llm_config
from backend.utils.json_parser import parse_llm_json_with_retry, JSONParseError
from backend.models.state import InterviewState

from dotenv import load_dotenv
load_dotenv()  # ← Ensure .env is loaded in this module

logger = logging.getLogger(__name__)


# ==========================================================
# PYDANTIC SCHEMAS
# ==========================================================

class QuestionInput(BaseModel):
    """
    Input schema for question generation.

    Contract:
    - role: Target job role
    - experience_level: "Fresher", "Mid", or "Senior"
    - topic_constraints: Optional list of allowed topics (e.g., ["OOP", "Algorithms"])
    - difficulty_target: Target difficulty from 0.0 to 1.0 (default: 0.5)
    """
    role: str = Field(..., description="Job role for the interview (e.g., 'Software Engineer', 'Data Scientist')")
    experience_level: str = Field(
        ..., description="Experience level: 'Junior', 'Mid', or 'Senior'")
    topic_constraints: Optional[List[str]] = Field(
        None, description="Optional list of allowed topics")
    difficulty_target: float = Field(
        0.5, ge=0.0, le=1.0, description="Target difficulty (0.0-1.0)")


class QuestionOutput(BaseModel):
    """
    Output schema for generated question.

    Contract (STRICT):
    - question.text: The question text
    - question.topic: Topic category (must match constraints if provided)
    - question.difficulty: Actual difficulty (0.0-1.0, clamped)
    - question.intent: 3-6 evaluation checkpoints (NOT full answers)

    No prose, no markdown, no explanations outside JSON.
    """
    question: 'QuestionDetails'


class QuestionDetails(BaseModel):
    """Nested question details matching output contract."""
    text: str = Field(..., description="The interview question")
    topic: str = Field(..., description="Topic category")
    difficulty: float = Field(..., ge=0.0, le=1.0,
                              description="Difficulty rating 0.0-1.0")
    intent: List[str] = Field(..., min_length=3, max_length=6,
                              description="3-6 evaluation checkpoints")


# ==========================================================
# PROMPT ENGINEERING
# ==========================================================

def build_system_prompt() -> str:
    """
    Build system prompt with clear role definition and constraints.

    Enforces:
    - JSON-only output
    - No answer generation
    - No subjective questions
    """
    return """You are an expert technical interviewer and question designer.

Your ONLY task is to generate interview questions. You MUST:
1. Return ONLY valid JSON - no prose, no markdown, no explanations
2. NEVER generate the answer or solution
3. Generate clear, objective questions (no vague or subjective questions)
4. Align question complexity with the specified difficulty level
5. Calibrate question depth strictly to the candidate's experience level and ensure all questions are directly relevant to the stated role

You will receive question parameters and must return a JSON object exactly matching the specified format."""


def build_question_prompt(
    role: str,
    experience_level: str,
    topic_constraints: Optional[List[str]],
    difficulty_target: float,
    previous_questions: Optional[List[str]] = None
) -> str:
    """
    Build question generation prompt with difficulty calibration.

    Difficulty Scale Definition:
    - 0.0-0.4: Junior / Fundamental — definitions, recall, basic application
    - 0.4-0.7: Mid / Application — practical skills, comparison, moderate problem-solving
    - 0.7-1.0: Senior / Architecture — system design, trade-offs, advanced optimization

    Experience Level Mapping:
    - Junior: Target 0.2-0.4 (fundamentals and basic application)
    - Mid: Target 0.4-0.7 (application and analysis)
    - Senior: Target 0.6-0.9 (design and architecture)

    Args:
        role: Job role
        experience_level: Fresher, Mid, or Senior
        topic_constraints: Optional list of allowed topics
        difficulty_target: Target difficulty 0.0-1.0

    Returns:
        Formatted prompt string
    """
    # Topic constraint section
    topic_section = ""
    if topic_constraints:
        topics_str = ", ".join(topic_constraints)
        topic_section = f"""
REQUIRED: The question MUST be about one of these topics: {topics_str}
Set the 'topic' field to one of these exact values: {topics_str}
"""
    else:
        topic_section = """
Select an appropriate topic based on the role and experience level.
Common topics: Algorithms, Data Structures, OOP, System Design, Databases, APIs, Testing, etc.
"""

    # Difficulty calibration — three tiers aligned with experience level
    if difficulty_target <= 0.4:
        difficulty_guidance = """
DIFFICULTY TARGET: {:.1f} (Junior / Fundamental)
- Ask for definitions, basic recall, or very simple application
- Focus on foundational knowledge a new graduate should know
- Questions must be answerable without professional experience
- GOOD examples (match this style):
    "What are the four pillars of OOP?"
    "What is the difference between a list and a tuple in Python?"
    "What does HTTP status code 404 mean?"
    "What is a primary key in a database?"
    "Explain what version control is and why it is used."
- AVOID questions requiring design decisions, system experience, or advanced trade-offs
""".format(difficulty_target)
    elif difficulty_target <= 0.7:
        difficulty_guidance = """
DIFFICULTY TARGET: {:.1f} (Mid / Application)
- Ask to apply concepts, solve moderate problems, or compare approaches
- Focus on practical skills a developer with 2-4 years of experience would have
- GOOD examples (match this style):
    "How would you implement pagination in a REST API?"
    "Explain the difference between SQL JOINs and when to use each."
    "How does React's virtual DOM work and why is it useful?"
    "What are ACID properties and why do they matter?"
    "Describe how you would approach debugging a slow database query."
- AVOID questions requiring large-scale system design or senior-level architecture
""".format(difficulty_target)
    else:
        difficulty_guidance = """
DIFFICULTY TARGET: {:.1f} (Senior / Architecture)
- Ask about system design, architecture trade-offs, or advanced optimization
- Focus on skills a senior engineer with 5+ years of experience would have
- GOOD examples (match this style):
    "Design a distributed rate-limiting system for 100K requests/sec."
    "How would you handle eventual consistency in a microservices architecture?"
    "Walk me through how you would architect a real-time notifications service."
    "What trade-offs would you consider when choosing between a monolith and microservices?"
    "How would you design a data pipeline that handles late-arriving events?"
- AVOID trivial or foundational questions — the candidate must be challenged
""".format(difficulty_target)

    # Previous questions section (for diversity in multi-question sessions)
    previous_section = ""
    if previous_questions:
        prev_list = "\n".join(f"  - {q}" for q in previous_questions)
        previous_section = f"""
PREVIOUS QUESTIONS ALREADY ASKED (DO NOT repeat these or ask about the same topic):
{prev_list}

You MUST ask about a DIFFERENT topic than all previous questions.
"""

    prompt = f"""Generate ONE interview question with these parameters:

ROLE: {role}
EXPERIENCE LEVEL: {experience_level}

ROLE ALIGNMENT:
All questions, terminology, and examples MUST be directly relevant to the role of {role}.
Prioritize topics that are central to day-to-day work in this role.

{topic_section}

{previous_section}

{difficulty_guidance}

INTENT FIELD REQUIREMENTS:
- List 3-6 key concepts or checkpoints that indicate a good answer
- These are evaluation points, NOT the full answer
- Be specific (e.g., "Explains time complexity", "Mentions race conditions")
- DO NOT write the actual answer - only what to look for

OUTPUT FORMAT (STRICT):
Return ONLY this JSON structure with NO additional text:

{{
    "question": {{
        "text": "Your interview question here",
        "topic": "topic_name",
        "difficulty": {difficulty_target},
        "intent": [
            "First concept to evaluate",
            "Second concept to evaluate",
            "Third concept to evaluate"
        ]
    }}
}}

CRITICAL RULES:
1. Return ONLY the JSON object above - no markdown, no prose
2. NEVER include the answer or solution in any field
3. The 'difficulty' value should be close to {difficulty_target} (within ±0.1)
4. Ensure 3-6 items in the 'intent' array
5. Make the question clear and objective (no vague questions)

Generate the question now:"""

    return prompt


# ==========================================================
# QUESTION GENERATION AGENT
# ==========================================================


class QuestionGenerationAgent:
    """
    Production-grade agent for generating interview questions.

    Features:
    - Configurable LLM (GPT-4o-mini primary, GPT-4o fallback)
    - Automatic retry on JSON parsing failures (max 2 retries)
    - Difficulty calibration aligned with experience level
    - Topic constraint validation and enforcement
    - Deterministic JSON output (no prose)
    - Comprehensive error handling and logging

    Design Principles:
    - Single responsibility: Generate questions only
    - No evaluation or scoring logic
    - No orchestration logic
    - Stateless operation
    - Environment-variable configurable
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the agent with configured LLM.

        Args:
            model_name: Optional model override (default: from config)
        """
        self.config = get_llm_config("question_generation")
        self.model_name = model_name or self.config.get("model", "gpt-4o-mini")
        self.llm = self._create_llm(self.model_name)
        self.fallback_llm = None  # Lazy initialization
        self.retry_count = 0
        self.max_retries = 2

        logger.info(
            f"QuestionGenerationAgent initialized with model: {self.model_name}")

    def _create_llm(self, model: str) -> ChatOpenAI:
        """
        Create LLM instance with configuration.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")

        Returns:
            Configured ChatOpenAI instance
        """
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set QUESTION_GENERATION_API_KEY or OPENAI_API_KEY environment variable."
            )

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=self.config.get("base_url"),
            temperature=self.config.get("temperature", 0.7),
        )

    def _get_fallback_llm(self) -> ChatOpenAI:
        """Get or create fallback LLM (GPT-4o)."""
        if self.fallback_llm is None:
            fallback_model = "gpt-4o" if self.model_name != "gpt-4o" else "gpt-4o-mini"
            logger.info(f"Initializing fallback LLM: {fallback_model}")
            self.fallback_llm = self._create_llm(fallback_model)
        return self.fallback_llm

    def _call_llm_raw(self, llm: ChatOpenAI, user_prompt: str) -> str:
        """
        Call LLM and return raw response.

        Args:
            llm: LLM instance to use
            user_prompt: User prompt string

        Returns:
            Raw LLM response as string
        """
        messages = [
            SystemMessage(content=build_system_prompt()),
            HumanMessage(content=user_prompt)
        ]

        response = llm.invoke(messages)
        return response.content

    def _validate_topic_constraint(
        self,
        returned_topic: str,
        topic_constraints: Optional[List[str]]
    ) -> bool:
        """
        Validate if returned topic matches constraints.

        Args:
            returned_topic: Topic returned by LLM
            topic_constraints: Required topic constraints

        Returns:
            True if valid or no constraints, False otherwise
        """
        if not topic_constraints:
            return True

        # Case-insensitive partial match
        returned_lower = returned_topic.lower()
        for constraint in topic_constraints:
            if constraint.lower() in returned_lower or returned_lower in constraint.lower():
                return True

        logger.warning(
            f"Topic validation failed: returned '{returned_topic}', "
            f"constraints: {topic_constraints}"
        )
        return False

    def _clamp_difficulty(self, difficulty: float) -> float:
        """
        Clamp difficulty to valid range [0.0, 1.0].

        Args:
            difficulty: Raw difficulty value

        Returns:
            Clamped difficulty
        """
        clamped = max(0.0, min(1.0, difficulty))
        if clamped != difficulty:
            logger.warning(
                f"Difficulty clamped from {difficulty} to {clamped}")
        return clamped

    def generate(
        self,
        role: str,
        experience_level: str,
        topic_constraints: Optional[List[str]] = None,
        difficulty_target: float = 0.5,
        previous_questions: Optional[List[str]] = None
    ) -> QuestionOutput:
        """
        Generate an interview question.

        Args:
            role: Job role (e.g., "Software Engineer")
            experience_level: "Fresher", "Mid", or "Senior"
            topic_constraints: Optional list of allowed topics
            difficulty_target: Target difficulty 0.0-1.0 (default: 0.5)
            previous_questions: List of previously asked question texts (for diversity)

        Returns:
            QuestionOutput with validated question

        Raises:
            ValueError: If input validation fails
            JSONParseError: If JSON parsing fails after all retries
        """
        # Validate input
        input_data = QuestionInput(
            role=role,
            experience_level=experience_level,
            topic_constraints=topic_constraints,
            difficulty_target=difficulty_target
        )

        logger.info(
            f"Generating question: role={role}, level={experience_level}, "
            f"topics={topic_constraints}, difficulty={difficulty_target}, "
            f"previous_count={len(previous_questions) if previous_questions else 0}"
        )

        # Build prompt
        prompt = build_question_prompt(
            role=input_data.role,
            experience_level=input_data.experience_level,
            topic_constraints=input_data.topic_constraints,
            difficulty_target=input_data.difficulty_target,
            previous_questions=previous_questions
        )

        # Attempt generation with retry
        for attempt in range(self.max_retries + 1):
            try:
                # Select LLM (fallback on retry)
                current_llm = self.llm if attempt == 0 else self._get_fallback_llm()

                # Call LLM
                raw_response = self._call_llm_raw(current_llm, prompt)
                logger.debug(
                    f"Attempt {attempt + 1} - Raw LLM response: {raw_response[:300]}...")

                # Parse JSON with retry utility
                def llm_call():
                    return raw_response

                parsed_data = parse_llm_json_with_retry(
                    llm_call=llm_call,
                    max_retries=0,  # We handle retries at this level
                    numeric_keys=["difficulty"],
                    clamp_range=(0.0, 1.0)
                )

                # Extract question data
                if "question" in parsed_data:
                    question_data = parsed_data["question"]
                else:
                    question_data = parsed_data

                # Validate with Pydantic
                question_details = QuestionDetails(**question_data)

                # Validate topic constraint
                if not self._validate_topic_constraint(
                    question_details.topic,
                    input_data.topic_constraints
                ):
                    if attempt < self.max_retries:
                        logger.warning(
                            f"Topic constraint violated, retrying (attempt {attempt + 1})")
                        continue
                    else:
                        # Override topic on final attempt
                        logger.warning(
                            "Topic constraint violated on final attempt, overriding topic")
                        question_details.topic = input_data.topic_constraints[
                            0] if input_data.topic_constraints else question_details.topic

                # Ensure difficulty is clamped
                question_details.difficulty = self._clamp_difficulty(
                    question_details.difficulty)

                # Validate intent count
                if len(question_details.intent) < 3:
                    logger.warning(
                        f"Intent has only {len(question_details.intent)} items, should be 3-6")
                    question_details.intent.extend([
                        "Demonstrates understanding",
                        "Provides clear explanation",
                        "Shows practical knowledge"
                    ][:3 - len(question_details.intent)])

                # Success!
                output = QuestionOutput(question=question_details)
                logger.info(
                    f"Question generated successfully: topic={question_details.topic}, "
                    f"difficulty={question_details.difficulty:.2f}"
                )

                return output

            except (json.JSONDecodeError, JSONParseError, ValidationError) as e:
                logger.error(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt >= self.max_retries:
                    # Final fallback: return rule-based question
                    logger.error(
                        "All retries exhausted, using fallback question")
                    return self._generate_fallback_question(input_data)
                # Retry with next attempt
                continue

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt >= self.max_retries:
                    return self._generate_fallback_question(input_data)
                continue

        # Should never reach here
        return self._generate_fallback_question(input_data)

    def _generate_fallback_question(self, input_data: QuestionInput) -> QuestionOutput:
        """
        Generate a rule-based fallback question if LLM fails.

        Args:
            input_data: Original input parameters

        Returns:
            QuestionOutput with fallback question
        """
        logger.warning("Generating rule-based fallback question")

        # Map experience to difficulty
        difficulty_map = {
            "Junior": 0.3,
            "Mid": 0.6,
            "Senior": 0.8
        }
        difficulty = difficulty_map.get(
            input_data.experience_level, input_data.difficulty_target)

        # Use first topic constraint or default
        topic = input_data.topic_constraints[0] if input_data.topic_constraints else "General"

        # Generate basic question
        if input_data.experience_level == "Junior":
            text = f"Explain the fundamental concepts of {topic} and provide a simple example from a project you've worked on."
        elif input_data.experience_level == "Mid":
            text = f"Describe a challenging problem you've solved involving {topic}. What approach did you take and why?"
        else:  # Senior
            text = f"Design a system or architecture that effectively uses {topic}. Discuss the trade-offs and alternatives you considered."

        question = QuestionDetails(
            text=text,
            topic=topic,
            difficulty=difficulty,
            intent=[
                "Demonstrates relevant experience",
                "Shows technical understanding",
                "Provides concrete examples",
                "Explains reasoning clearly"
            ]
        )

        return QuestionOutput(question=question)


# =========================================================
# LANGGRAPH NODE WRAPPER
# =========================================================

def question_generation_node(state: "InterviewState") -> "InterviewState":
    """
    LangGraph node wrapper for QuestionGenerationAgent.

    Args:
        state: InterviewState object containing role and experience_level

    Returns:
        Updated InterviewState with question field populated
    """
    from backend.models.state import InterviewState, QuestionModel

    logger.info(
        f"QuestionGenerationNode: Starting for interview_id={state.interview_id}")

    try:
        # Initialize agent
        agent = QuestionGenerationAgent()

        # Derive difficulty_target from experience level
        _difficulty_map = {"Junior": 0.3, "Mid": 0.55, "Senior": 0.80}
        difficulty_target = _difficulty_map.get(state.experience_level, 0.5)

        # Generate question
        result = agent.generate(
            role=state.role,
            experience_level=state.experience_level,
            difficulty_target=difficulty_target
        )

        # Create updated state copy
        updated_state = state.model_copy(deep=True)
        updated_state.question = QuestionModel(
            text=result.question.text,
            topic=result.question.topic,
            difficulty=result.question.difficulty,
            intent=result.question.intent
        )

        logger.info(
            f"QuestionGenerationNode: Generated question on topic '{result.question.topic}'")

        # Return only the key we're updating (LangGraph merges it into state)
        return {"question": updated_state.question}

    except Exception as e:
        logger.error(f"QuestionGenerationNode: Failed - {e}", exc_info=True)
        # Return empty question on error
        return {"question": None}
