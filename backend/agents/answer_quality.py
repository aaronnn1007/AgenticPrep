"""
Answer Quality Analyser Agent
==============================
Production-grade component for evaluating interview answer quality using LLM analysis.

ARCHITECTURE:
- Language: Python with strict type hints
- Schema validation: Pydantic V2
- LLM access: LangChain with OpenAI
- Output: Strict JSON-only
- Retry logic: Automatic on JSON parse failures (max 2 retries)
- Error handling: Comprehensive logging and fallback strategies

DESIGN PRINCIPLES:
- Uses GPT-4o-mini (configurable) for semantic quality evaluation
- NO SCORING LOGIC - only provides quality metrics (0-1 scale)
- NO ORCHESTRATION - pure evaluation agent
- Modular, testable, production-ready

INPUT CONTRACT:
{
  "question": {
      "text": str,
      "intent": List[str],
      "difficulty": float,
      "topic": str
  },
  "transcript": str,
  "role": str,
  "experience_level": str
}

OUTPUT CONTRACT:
{
  "answer_quality": {
    "relevance": float (0-1),
    "correctness": float (0-1),
    "depth": float (0-1),
    "structure": float (0-1),
    "gaps": List[str]
  }
}
"""

import os
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field, field_validator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from backend.utils.json_parser import parse_llm_json_with_retry, JSONParseError
from backend.models.state import InterviewState

from dotenv import load_dotenv
load_dotenv()  # ← Ensure .env is loaded in this module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# PYDANTIC SCHEMAS
# =========================================================

class QuestionInput(BaseModel):
    """Question details for evaluation."""
    text: str = Field(..., description="The interview question text")
    intent: List[str] = Field(...,
                              description="Expected concepts/topics to be covered")
    difficulty: float = Field(..., ge=0.0, le=1.0,
                              description="Question difficulty level")
    topic: str = Field(..., description="Main topic of the question")


class AnswerQualityInput(BaseModel):
    """Input schema for answer quality evaluation."""
    question: QuestionInput
    transcript: str = Field(..., description="Candidate's answer transcript")
    role: str = Field(...,
                      description="Target role (e.g., 'Software Engineer')")
    experience_level: str = Field(...,
                                  description="Experience level (e.g., 'Mid-Level')")


class AnswerQualityMetrics(BaseModel):
    """Quality metrics for answer evaluation (output schema)."""
    relevance: float = Field(..., ge=0.0, le=1.0,
                             description="Alignment with question")
    correctness: float = Field(..., ge=0.0, le=1.0,
                               description="Factual accuracy")
    depth: float = Field(..., ge=0.0, le=1.0,
                         description="Thoroughness of explanation")
    structure: float = Field(..., ge=0.0, le=1.0,
                             description="Logical organization")
    gaps: List[str] = Field(default_factory=list,
                            description="Missing concepts from intent")

    @field_validator('gaps', mode='before')
    @classmethod
    def ensure_list(cls, v):
        """Ensure gaps is always a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        return v


class AnswerQualityOutput(BaseModel):
    """Complete output schema."""
    answer_quality: AnswerQualityMetrics


# =========================================================
# LLM PROMPT TEMPLATE
# =========================================================

ANSWER_QUALITY_SYSTEM_PROMPT = """You are an expert technical interviewer evaluating interview answers with precision and objectivity.

Your task is to analyze candidate responses and provide structured quality metrics.

CRITICAL INSTRUCTIONS:
1. Respond ONLY with valid JSON - no additional text, explanations, or markdown
2. All numeric scores must be between 0.0 and 1.0
3. Be objective and evidence-based
4. Do NOT hallucinate - only evaluate what's actually said
5. Consider the candidate's experience level and role context"""


ANSWER_QUALITY_EVALUATION_PROMPT = """QUESTION CONTEXT:
-------------------
Question: {question_text}
Topic: {topic}
Expected Concepts: {intent}
Difficulty Level: {difficulty}

CANDIDATE CONTEXT:
------------------
Role: {role}
Experience Level: {experience_level}

CANDIDATE'S ANSWER:
-------------------
{transcript}

EVALUATION CRITERIA:
--------------------

1. RELEVANCE (0.0 - 1.0):
   Does the answer directly address the question asked?
   - 1.0 = Perfectly on-topic, addresses all aspects of the question
   - 0.7 = Mostly relevant with minor tangents
   - 0.5 = Partially relevant, some off-topic content
   - 0.3 = Tangentially related but misses main point
   - 0.0 = Completely off-topic or no answer

2. CORRECTNESS (0.0 - 1.0):
   Is the technical content accurate and valid?
   - 1.0 = Entirely correct, no factual errors
   - 0.7 = Mostly correct with minor inaccuracies
   - 0.5 = Mix of correct and incorrect information
   - 0.3 = Significant errors or misconceptions
   - 0.0 = Fundamentally incorrect or misleading

3. DEPTH (0.0 - 1.0):
   How thorough and comprehensive is the explanation?
   - 1.0 = Comprehensive with examples, edge cases, trade-offs
   - 0.7 = Good explanation with some detail
   - 0.5 = Surface-level explanation, basic understanding shown
   - 0.3 = Minimal detail, very brief response
   - 0.0 = No substantial content or single word answer

4. STRUCTURE (0.0 - 1.0):
   Is the answer well-organized and clearly communicated?
   - 1.0 = Excellent structure, logical flow, clear communication
   - 0.7 = Well-organized with minor clarity issues
   - 0.5 = Acceptable structure but somewhat disorganized
   - 0.3 = Poor structure, hard to follow
   - 0.0 = Incoherent or rambling

5. GAPS:
   List specific concepts from Expected Concepts that are MISSING or inadequately covered.
   - If all concepts are covered, return empty list: []
   - Format: ["concept_1", "concept_2"]
   - Reference concepts from the Expected Concepts list above

RESPONSE FORMAT (STRICT):
{{
    "relevance": 0.85,
    "correctness": 0.90,
    "depth": 0.75,
    "structure": 0.80,
    "gaps": ["concept_1", "concept_2"]
}}

GUARDRAILS:
- All scores must be floats between 0.0 and 1.0
- gaps must be a list (empty if none)
- Return ONLY the JSON object
- No markdown code blocks
- No explanatory text before or after JSON

Evaluate the answer now:"""


# =========================================================
# ANSWER QUALITY ANALYSER AGENT
# =========================================================

class AnswerQualityAnalyser:
    """
    Production-grade agent for evaluating interview answer quality.

    Responsibilities:
    - Semantic analysis of answer quality
    - Multi-dimensional scoring (relevance, correctness, depth, structure)
    - Gap identification (missing concepts)
    - Robust error handling with retry logic

    NOT responsible for:
    - Final scoring or grade calculation
    - Orchestration or routing decisions
    - Confidence or body language analysis
    """

    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        """
        Initialize the Answer Quality Analyser.

        Args:
            model_name: LLM model to use (default: from env or 'gpt-4o-mini')
            api_key: API key (default: from env)
            base_url: Base URL for API (default: from env, supports GitHub Models)
        """
        self.model_name = model_name or os.getenv(
            "ANSWER_QUALITY_MODEL", "gpt-4o-mini")

        # Try multiple environment variables for API key
        self.api_key = api_key or os.getenv(
            "ANSWER_QUALITY_API_KEY") or os.getenv("OPENAI_API_KEY")

        # Try multiple environment variables for base URL
        self.base_url = base_url or os.getenv(
            "ANSWER_QUALITY_BASE_URL") or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            logger.warning("API key not found. LLM calls will fail.")

        self.llm = self._create_llm()
        logger.info(
            f"AnswerQualityAnalyser initialized with model: {self.model_name}")
        if self.base_url:
            logger.info(f"Using custom endpoint: {self.base_url}")

    def _create_llm(self) -> ChatOpenAI:
        """
        Create LangChain LLM instance.

        Returns:
            Configured ChatOpenAI instance
        """
        llm_config = {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": 0.2,  # Low temperature for consistent, objective evaluation
            "max_tokens": 500,   # Limit output size for JSON-only responses
        }

        # Add base_url if provided (for GitHub Models, Azure, etc.)
        if self.base_url:
            llm_config["base_url"] = self.base_url

        return ChatOpenAI(**llm_config)

    def _build_prompt(self, input_data: AnswerQualityInput) -> str:
        """
        Build evaluation prompt from input data.

        Args:
            input_data: Validated input containing question and answer

        Returns:
            Formatted prompt string
        """
        return ANSWER_QUALITY_EVALUATION_PROMPT.format(
            question_text=input_data.question.text,
            topic=input_data.question.topic,
            intent=", ".join(input_data.question.intent),
            difficulty=input_data.question.difficulty,
            role=input_data.role,
            experience_level=input_data.experience_level,
            transcript=input_data.transcript if input_data.transcript.strip(
            ) else "[No answer provided]"
        )

    def _call_llm_with_retry(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM with automatic retry on JSON parse failure.

        Uses the json_parser utility to handle:
        - Markdown code block extraction
        - JSON parsing with retries
        - Automatic value clamping

        Args:
            prompt: Formatted evaluation prompt

        Returns:
            Parsed quality metrics dictionary

        Raises:
            JSONParseError: If parsing fails after all retries
        """
        def llm_call() -> str:
            """Inner function for LLM invocation."""
            messages = [
                SystemMessage(content=ANSWER_QUALITY_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            raw_output = response.content

            # Log raw LLM output for debugging
            logger.debug(f"Raw LLM output: {raw_output}")

            return raw_output

        # Use retry parser with automatic clamping
        parsed_data = parse_llm_json_with_retry(
            llm_call=llm_call,
            max_retries=2,
            numeric_keys=["relevance", "correctness", "depth", "structure"],
            clamp_range=(0.0, 1.0)
        )

        return parsed_data

    def _generate_empty_answer_metrics(self) -> AnswerQualityMetrics:
        """
        Generate metrics for empty/no answer case.

        Returns:
            Zero scores with appropriate gap message
        """
        return AnswerQualityMetrics(
            relevance=0.0,
            correctness=0.0,
            depth=0.0,
            structure=0.0,
            gaps=["No answer provided"]
        )

    def _generate_fallback_metrics(self, error_msg: str = None) -> AnswerQualityMetrics:
        """
        Generate fallback metrics when LLM fails.

        Args:
            error_msg: Optional error message to include in gaps

        Returns:
            Neutral scores with error indication
        """
        gap_msg = f"Evaluation error: {error_msg}" if error_msg else "Unable to evaluate due to system error"

        logger.warning(f"Using fallback metrics: {gap_msg}")

        return AnswerQualityMetrics(
            relevance=0.5,
            correctness=0.5,
            depth=0.5,
            structure=0.5,
            gaps=[gap_msg]
        )

    def evaluate(
        self,
        question: Dict[str, Any],
        transcript: str,
        role: str,
        experience_level: str
    ) -> Dict[str, Any]:
        """
        Evaluate answer quality based on question and transcript.

        This is the main public interface for the agent.

        Args:
            question: Dict with keys: text, intent, difficulty, topic
            transcript: Candidate's answer transcript
            role: Target job role
            experience_level: Candidate's experience level

        Returns:
            Dict matching OUTPUT CONTRACT with answer_quality metrics

        Example:
            >>> analyser = AnswerQualityAnalyser()
            >>> result = analyser.evaluate(
            ...     question={
            ...         "text": "Explain the difference between processes and threads",
            ...         "intent": ["concurrency", "memory model", "context switching"],
            ...         "difficulty": 0.6,
            ...         "topic": "Operating Systems"
            ...     },
            ...     transcript="A process is an independent program...",
            ...     role="Software Engineer",
            ...     experience_level="Mid-Level"
            ... )
            >>> print(result["answer_quality"]["relevance"])
            0.85
        """
        logger.info(
            f"Evaluating answer quality for role={role}, level={experience_level}")

        try:
            # Validate input using Pydantic
            input_data = AnswerQualityInput(
                question=QuestionInput(**question),
                transcript=transcript,
                role=role,
                experience_level=experience_level
            )

            # Handle empty transcript case
            if not transcript or not transcript.strip():
                logger.info("Empty transcript - returning zero scores")
                metrics = self._generate_empty_answer_metrics()
                return AnswerQualityOutput(answer_quality=metrics).model_dump()

            # Build prompt
            prompt = self._build_prompt(input_data)

            # Call LLM with retry logic
            quality_data = self._call_llm_with_retry(prompt)

            # Validate output using Pydantic
            metrics = AnswerQualityMetrics(**quality_data)

            logger.info(
                f"Evaluation complete: relevance={metrics.relevance:.2f}, "
                f"correctness={metrics.correctness:.2f}, "
                f"depth={metrics.depth:.2f}, "
                f"structure={metrics.structure:.2f}, "
                f"gaps={len(metrics.gaps)}"
            )

            # Return in specified output format
            return AnswerQualityOutput(answer_quality=metrics).model_dump()

        except JSONParseError as e:
            logger.error(f"JSON parsing failed after retries: {e}")
            metrics = self._generate_fallback_metrics("JSON parsing failed")
            return AnswerQualityOutput(answer_quality=metrics).model_dump()

        except Exception as e:
            logger.error(f"Unexpected error in evaluate: {e}", exc_info=True)
            metrics = self._generate_fallback_metrics(str(e))
            return AnswerQualityOutput(answer_quality=metrics).model_dump()


# =========================================================
# CONVENIENCE FUNCTIONS
# =========================================================

def create_answer_quality_analyser(model_name: str = None, api_key: str = None, base_url: str = None) -> AnswerQualityAnalyser:
    """
    Factory function to create analyser instance.

    Args:
        model_name: Optional model name override
        api_key: Optional API key override
        base_url: Optional base URL override (for GitHub Models, Azure, etc.)

    Returns:
        Configured AnswerQualityAnalyser instance
    """
    return AnswerQualityAnalyser(model_name=model_name, api_key=api_key, base_url=base_url)


# =========================================================
# LEGACY COMPATIBILITY (for existing LangGraph integration)
# =========================================================

def answer_quality_node(state: InterviewState) -> InterviewState:
    """
    LangGraph node wrapper for backward compatibility.

    Note: This adapter converts between the legacy InterviewState format
    and the new INPUT/OUTPUT contracts.

    Args:
        state: InterviewState object

    Returns:
        Updated state with answer_quality
    """
    try:
        # Import here to avoid circular dependency
        from backend.models.state import InterviewState, AnswerQualityModel

        if not state.question:
            raise ValueError(
                "Question must be generated before answer quality analysis")

        # Create analyser
        analyser = AnswerQualityAnalyser(
            api_key=os.getenv("ANSWER_QUALITY_API_KEY") or os.getenv(
                "OPENAI_API_KEY"),
            base_url=os.getenv("ANSWER_QUALITY_BASE_URL") or os.getenv(
                "OPENAI_BASE_URL")
        )

        # Convert to new input format
        question_dict = {
            "text": state.question.text,
            "intent": state.question.intent,
            "difficulty": state.question.difficulty,
            "topic": state.question.topic
        }

        # Call evaluate with new contract
        result = analyser.evaluate(
            question=question_dict,
            transcript=state.transcript or "",
            role=state.role,
            experience_level=state.experience_level
        )

        # Convert back to legacy format
        answer_quality = AnswerQualityModel(**result["answer_quality"])

        # Update state
        updated_state = state.model_copy(deep=True)
        updated_state.answer_quality = answer_quality

        return {"answer_quality": updated_state.answer_quality}

    except Exception as e:
        logger.error(f"Error in answer_quality_node: {e}", exc_info=True)
        raise
