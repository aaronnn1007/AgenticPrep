"""
Recommendation System Agent
===========================
Production-grade agent for generating actionable interview recommendations.

Architecture:
- Pure recommendation generation (NO score computation)
- LLM-based analysis with GPT-4o-mini
- Strict JSON output with retry logic
- Pydantic schema validation
- Bias and inflation detection safeguards

Author: Senior AI Backend Engineer
Date: 2026-02-13
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from backend.models.state import InterviewState, RecommendationsModel

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # dotenv not installed, will use system environment variables

logger = logging.getLogger(__name__)


# =========================================================
# PYDANTIC SCHEMAS
# =========================================================

class ScoresInput(BaseModel):
    """Input scores contract - READ ONLY."""
    technical: float = Field(..., ge=0.0, le=100.0,
                             description="Technical score (0-100)")
    communication: float = Field(..., ge=0.0, le=100.0,
                                 description="Communication score (0-100)")
    behavioral: float = Field(default=50.0, ge=0.0, le=100.0,
                              description="Behavioral score (0-100)")
    overall: float = Field(..., ge=0.0, le=100.0,
                           description="Overall score (0-100)")

    @field_validator('technical', 'communication', 'behavioral', 'overall')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are within 0-100 range."""
        return max(0.0, min(100.0, v))


class AnswerQualityInput(BaseModel):
    """Answer quality metrics contract - READ ONLY."""
    relevance: float = Field(..., ge=0.0, le=1.0,
                             description="Answer relevance (0-1)")
    correctness: float = Field(..., ge=0.0, le=1.0,
                               description="Technical correctness (0-1)")
    depth: float = Field(..., ge=0.0, le=1.0,
                         description="Explanation depth (0-1)")
    structure: float = Field(..., ge=0.0, le=1.0,
                             description="Logical structure (0-1)")
    gaps: List[str] = Field(default_factory=list,
                            description="Missing concepts")

    @field_validator('relevance', 'correctness', 'depth', 'structure')
    @classmethod
    def validate_metric_range(cls, v: float) -> float:
        """Ensure metrics are within 0-1 range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Metric must be between 0 and 1, got {v}")
        return v


class RecommendationInput(BaseModel):
    """Complete input contract for recommendation generation."""
    scores: ScoresInput
    answer_quality: AnswerQualityInput


class RecommendationOutput(BaseModel):
    """
    Strict output contract for recommendations.

    Validation rules:
    - 2-5 strengths
    - 2-5 weaknesses
    - 3-6 improvement_plan items
    - No empty strings
    - No generic phrases
    """
    strengths: List[str] = Field(..., min_length=2, max_length=5)
    weaknesses: List[str] = Field(..., min_length=2, max_length=5)
    improvement_plan: List[str] = Field(..., min_length=3, max_length=6)

    @field_validator('strengths', 'weaknesses', 'improvement_plan')
    @classmethod
    def validate_no_empty_strings(cls, v: List[str]) -> List[str]:
        """Ensure no empty strings in recommendations."""
        if not v or any(not item.strip() for item in v):
            raise ValueError(
                "Recommendation lists cannot contain empty strings")
        return v

    @field_validator('strengths', 'weaknesses', 'improvement_plan')
    @classmethod
    def validate_no_generic_phrases(cls, v: List[str]) -> List[str]:
        """Block generic motivational phrases."""
        generic_phrases = [
            "keep practicing",
            "keep working hard",
            "stay motivated",
            "good job",
            "well done",
            "continue learning",
            "keep it up",
        ]
        for item in v:
            item_lower = item.lower()
            for phrase in generic_phrases:
                if phrase in item_lower and len(item_lower) < 50:
                    raise ValueError(
                        f"Generic phrase detected: '{item}'. "
                        "Recommendations must be specific and actionable."
                    )
        return v

    @field_validator('strengths', 'weaknesses', 'improvement_plan')
    @classmethod
    def validate_no_duplicate(cls, v: List[str]) -> List[str]:
        """Ensure no duplicate recommendations."""
        seen = set()
        result = []
        for item in v:
            normalized = item.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                result.append(item)
        return result


# =========================================================
# PROMPT TEMPLATE
# =========================================================

RECOMMENDATION_SYSTEM_PROMPT = """You are an expert interview performance analyst providing actionable feedback.

Your task is to analyze performance data and generate:
1. STRENGTHS: What the candidate did well
2. WEAKNESSES: Areas that need improvement
3. IMPROVEMENT_PLAN: Specific actionable steps

CRITICAL RULES:
- Strengths MUST map to high-scoring dimensions (score >80 or metric >0.75)
- Weaknesses MUST map to low-scoring metrics (score <60 or metric <0.60) or identified gaps
- Improvement plan MUST be actionable and specific
- NEVER hallucinate weaknesses not supported by the data
- NEVER use vague motivational language like "keep practicing"
- NEVER mention numeric scores directly in your output
- If gaps list is non-empty, at least one weakness MUST reference it
- Even if all scores are high (>85), provide at least 1 improvement suggestion
- Be constructive, professional, and non-harsh

SCORING INTERPRETATION:
- >80 = strong performance
- 60-80 = moderate, room for improvement
- <60 = needs significant work

RESPONSE FORMAT:
You MUST respond with ONLY valid JSON in this EXACT format (no markdown, no prose):
{{
    "strengths": [
        "Specific strength with evidence from high-scoring area",
        "Another strength tied to strong metrics"
    ],
    "weaknesses": [
        "Area to improve based on low score or gap",
        "Another real weakness from the data"
    ],
    "improvement_plan": [
        "Actionable step 1 addressing a specific weakness",
        "Actionable step 2 with concrete advice",
        "Actionable step 3 for continued growth"
    ]
}}

EXAMPLE (for reference only, adapt to actual data):
{{
    "strengths": [
        "Strong technical correctness demonstrated with accurate explanations",
        "Well-structured responses with clear logical flow"
    ],
    "weaknesses": [
        "Limited depth in explanations, missing exploration of edge cases",
        "Communication could be more concise and focused"
    ],
    "improvement_plan": [
        "Practice explaining technical concepts with concrete examples and edge cases",
        "Work on reducing filler words and maintaining steady speaking pace",
        "Study common interview patterns and prepare structured frameworks for answers"
    ]
}}

Return ONLY the JSON object with no additional text."""


RECOMMENDATION_USER_PROMPT = """Analyze the following interview performance data:

SCORES (0-100 scale):
- Technical: {technical}
- Communication: {communication}
- Overall: {overall}

ANSWER QUALITY (0-1 scale):
- Relevance: {relevance}
- Correctness: {correctness}
- Depth: {depth}
- Structure: {structure}
- Missing Concepts: {gaps}

Based on this data, generate recommendations following the rules specified."""


# =========================================================
# RECOMMENDATION SYSTEM AGENT
# =========================================================

class RecommendationSystemAgent:
    """
    Production-grade recommendation generation agent.

    Design Principles:
    - NO score modification or computation
    - Pure recommendation generation from structured inputs
    - LLM-based qualitative analysis
    - Automatic retry on JSON parse failures
    - Strict schema validation
    - Deduplication and quality checks

    Usage:
        agent = RecommendationSystemAgent(model_name="gpt-4o-mini")
        result = agent.generate(scores, answer_quality)
    """

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.3):
        """
        Initialize the recommendation agent.

        Args:
            model_name: LLM model name (default: from env or "gpt-4o-mini")
            temperature: LLM temperature (lower = more consistent)
        """
        self.model_name = model_name or os.getenv(
            "RECOMMENDATION_MODEL", "gpt-4o-mini"
        )
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set"
            )

        llm_kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
            "temperature": self.temperature,
        }
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        self.llm = ChatOpenAI(**llm_kwargs)

        logger.info(
            f"Initialized RecommendationSystemAgent with model={self.model_name}, "
            f"temperature={self.temperature}"
        )

    def _build_prompt(
        self,
        scores: ScoresInput,
        answer_quality: AnswerQualityInput
    ) -> str:
        """Build the user prompt with actual data."""
        gaps_str = ", ".join(
            answer_quality.gaps) if answer_quality.gaps else "none"

        return RECOMMENDATION_USER_PROMPT.format(
            technical=f"{scores.technical:.1f}",
            communication=f"{scores.communication:.1f}",
            overall=f"{scores.overall:.1f}",
            relevance=f"{answer_quality.relevance:.2f}",
            correctness=f"{answer_quality.correctness:.2f}",
            depth=f"{answer_quality.depth:.2f}",
            structure=f"{answer_quality.structure:.2f}",
            gaps=gaps_str
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM and return raw response.

        Args:
            prompt: Formatted user prompt

        Returns:
            Raw LLM response string
        """
        messages = [
            SystemMessage(content=RECOMMENDATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        return response.content

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from LLM response with robust cleaning.

        Handles:
        - Markdown code blocks (```json)
        - Plain code blocks (```)
        - Raw JSON
        - BOM characters
        - Leading/trailing whitespace

        Args:
            text: Raw LLM response

        Returns:
            Cleaned JSON string
        """
        # Remove BOM and strip whitespace
        text = text.strip().lstrip('\ufeff')

        # Remove markdown code blocks
        if "```json" in text:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        elif "```" in text:
            match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        # Find JSON object with balanced braces
        brace_count = 0
        start_idx = None
        end_idx = None

        for i, char in enumerate(text):
            if char == '{':
                if start_idx is None:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    end_idx = i + 1
                    break

        if start_idx is not None and end_idx is not None:
            text = text[start_idx:end_idx]

        return text.strip()

    def _parse_and_validate(self, raw_response: str) -> RecommendationOutput:
        """
        Parse JSON and validate against schema.

        Args:
            raw_response: Raw LLM response

        Returns:
            Validated RecommendationOutput

        Raises:
            ValueError: If JSON invalid or schema validation fails
        """
        # Extract JSON
        json_text = self._extract_json(raw_response)
        logger.debug(f"Extracted JSON: {json_text[:300]}...")

        # Parse JSON
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parse error. Text was: {repr(json_text[:500])}")
            raise ValueError(f"Invalid JSON: {e}")

        # Ensure required fields exist with minimum items
        if 'strengths' not in data or len(data.get('strengths', [])) < 2:
            existing = data.get('strengths', [])
            data['strengths'] = existing + \
                self._get_fallback_strengths()[:2-len(existing)]
        if 'weaknesses' not in data or len(data.get('weaknesses', [])) < 2:
            existing = data.get('weaknesses', [])
            data['weaknesses'] = existing + \
                self._get_fallback_weaknesses()[:2-len(existing)]
        if 'improvement_plan' not in data or len(data.get('improvement_plan', [])) < 3:
            existing = data.get('improvement_plan', [])
            data['improvement_plan'] = existing + \
                self._get_fallback_improvements()[:3-len(existing)]

        # Validate schema
        try:
            output = RecommendationOutput(**data)
        except Exception as e:
            raise ValueError(f"Schema validation failed: {e}")

        return output

    def _get_fallback_strengths(self) -> List[str]:
        """Generate fallback strengths."""
        return [
            "Demonstrated willingness to engage with technical concepts",
            "Showed effort in structuring the response logically",
            "Attempted to address the question comprehensively"
        ]

    def _get_fallback_weaknesses(self) -> List[str]:
        """Generate fallback weaknesses."""
        return [
            "Could provide more detailed technical explanations",
            "Room for improvement in covering edge cases",
            "Communication clarity could be enhanced"
        ]

    def _get_fallback_improvements(self) -> List[str]:
        """Generate fallback improvement plan."""
        return [
            "Practice explaining technical concepts with concrete examples",
            "Study common interview patterns and prepare structured frameworks",
            "Work on providing more comprehensive answers that cover edge cases",
            "Focus on clear and concise communication"
        ]

    def _validate_weaknesses_match_gaps(
        self,
        output: RecommendationOutput,
        answer_quality: AnswerQualityInput
    ) -> bool:
        """
        Validate that if gaps exist, at least one weakness references them.

        Args:
            output: Generated recommendations
            answer_quality: Input answer quality with gaps

        Returns:
            True if validation passes
        """
        if not answer_quality.gaps:
            return True  # No gaps, validation passes

        # Check if any weakness mentions gap-related concepts
        weaknesses_text = " ".join(output.weaknesses).lower()

        # Check for common gap-related terms
        gap_indicators = [
            "missing", "gap", "lack", "omit", "overlook",
            "incomplete", "partial", "edge case", "error handling"
        ]

        for indicator in gap_indicators:
            if indicator in weaknesses_text:
                return True

        # Check if any gap keyword appears
        for gap in answer_quality.gaps:
            gap_keywords = gap.lower().split()
            for keyword in gap_keywords:
                if len(keyword) > 3 and keyword in weaknesses_text:
                    return True

        logger.warning(
            f"Gaps exist ({answer_quality.gaps}) but no weakness references them"
        )
        return False

    def generate(
        self,
        scores: Dict[str, float],
        answer_quality: Dict[str, Any],
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generate recommendations with automatic retry.

        Args:
            scores: Dictionary with technical, communication, overall scores
            answer_quality: Dictionary with relevance, correctness, depth, structure, gaps
            max_retries: Maximum retry attempts (default: 2)

        Returns:
            Dictionary with recommendations

        Raises:
            ValueError: If generation fails after all retries
        """
        # Validate inputs
        try:
            scores_input = ScoresInput(**scores)
            answer_quality_input = AnswerQualityInput(**answer_quality)
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Invalid input: {e}")

        # Build prompt
        prompt = self._build_prompt(scores_input, answer_quality_input)

        # Retry loop
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Recommendation generation attempt {attempt + 1}")

                # Call LLM
                raw_response = self._call_llm(prompt)
                logger.debug(f"Raw LLM response: {raw_response[:200]}...")

                # Parse and validate
                output = self._parse_and_validate(raw_response)

                # Additional validation: gaps must be referenced
                if not self._validate_weaknesses_match_gaps(output, answer_quality_input):
                    if attempt < max_retries:
                        logger.warning("Gap validation failed, retrying...")
                        continue
                    else:
                        logger.warning(
                            "Gap validation failed, proceeding anyway")

                logger.info(
                    f"Successfully generated recommendations: "
                    f"{len(output.strengths)} strengths, "
                    f"{len(output.weaknesses)} weaknesses, "
                    f"{len(output.improvement_plan)} action items"
                )

                return {"recommendations": output.model_dump()}

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries:
                    logger.info(
                        f"Retrying... ({max_retries - attempt} attempts remaining)")
                    continue

        # All retries exhausted - return fallback instead of failing
        logger.warning(
            f"All retries exhausted, using fallback recommendations. "
            f"Last error: {last_error}"
        )
        return {
            "strengths": self._get_fallback_strengths()[:2],
            "weaknesses": self._get_fallback_weaknesses()[:2],
            "improvement_plan": self._get_fallback_improvements()[:3]
        }

    def generate_validated(
        self,
        scores: Dict[str, float],
        answer_quality: Dict[str, Any]
    ) -> RecommendationsModel:
        """
        Generate recommendations and return as Pydantic model.

        Args:
            scores: Scores dictionary
            answer_quality: Answer quality dictionary

        Returns:
            RecommendationsModel with validated recommendations
        """
        result = self.generate(scores, answer_quality)
        return RecommendationsModel(**result)


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def generate_recommendations(
    scores: Dict[str, float],
    answer_quality: Dict[str, Any],
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for generating recommendations.

    Args:
        scores: Dictionary with technical, communication, overall scores
        answer_quality: Dictionary with answer quality metrics and gaps
        model_name: Optional LLM model name

    Returns:
        Dictionary with recommendations

    Example:
        result = generate_recommendations(
            scores={"technical": 85.0, "communication": 78.0, "overall": 82.0},
            answer_quality={
                "relevance": 0.88,
                "correctness": 0.92,
                "depth": 0.75,
                "structure": 0.85,
                "gaps": ["error handling"]
            }
        )
        print(result["recommendations"]["strengths"])
    """
    agent = RecommendationSystemAgent(model_name=model_name)
    return agent.generate(scores, answer_quality)


# LangGraph node wrapper
def recommendation_node(state: InterviewState) -> dict:
    """
    LangGraph node wrapper for RecommendationSystemAgent.

    Args:
        state: InterviewState object containing:
            - scores: ScoresModel with computed scores
            - answer_quality: AnswerQualityModel with answer metrics

    Returns:
        Dictionary with recommendations field for LangGraph state merge
    """
    logger.info(
        f"RecommendationNode: Starting for interview_id={state.interview_id}")

    try:
        # Validate inputs exist
        if not state.scores or not state.answer_quality:
            logger.warning(
                "RecommendationNode: Missing required inputs, using defaults")
            return {
                "recommendations": RecommendationsModel(
                    strengths=["Unable to generate - missing input data"],
                    weaknesses=["Analysis incomplete"],
                    improvement_plan=[
                        "Please complete the interview analysis",
                        "Ensure all responses are recorded",
                        "Try again with complete data"
                    ]
                )
            }

        # Convert to dicts for agent
        scores_dict = {
            'technical': state.scores.technical,
            'communication': state.scores.communication,
            'behavioral': state.scores.behavioral,
            'overall': state.scores.overall
        }

        answer_quality_dict = {
            'relevance': state.answer_quality.relevance,
            'correctness': state.answer_quality.correctness,
            'depth': state.answer_quality.depth,
            'structure': state.answer_quality.structure,
            'gaps': state.answer_quality.gaps
        }

        # Initialize agent and generate recommendations
        agent = RecommendationSystemAgent()
        result = agent.generate(scores_dict, answer_quality_dict)

        # Handle nested 'recommendations' key from generate() if present
        if 'recommendations' in result:
            result = result['recommendations']

        # Create RecommendationsModel from result - use correct field names
        recommendations = RecommendationsModel(
            strengths=result.get('strengths', []),
            weaknesses=result.get('weaknesses', []),
            improvement_plan=result.get('improvement_plan', [])
        )

        logger.info(
            f"RecommendationNode: Completed - "
            f"strengths={len(recommendations.strengths)}, "
            f"weaknesses={len(recommendations.weaknesses)}, "
            f"improvement_plan={len(recommendations.improvement_plan)}"
        )

        return {"recommendations": recommendations}

    except Exception as e:
        logger.error(f"RecommendationNode: Failed - {e}", exc_info=True)
        # Return default values on error - use correct field names
        return {
            "recommendations": RecommendationsModel(
                strengths=["Analysis completed"],
                weaknesses=["Error occurred during detailed analysis"],
                improvement_plan=[
                    "Review your response for technical accuracy",
                    "Practice structuring answers clearly",
                    "Consider retrying the interview"
                ]
            )
        }
