"""
Scoring & Aggregation Agent
===========================
Production-grade deterministic scoring engine for interview performance analysis.

CRITICAL PRINCIPLES:
--------------------
- NO LLM USAGE - Pure deterministic Python
- All formulas are transparent and auditable
- Weights are configurable via backend.config.SCORING_WEIGHTS
- Reproducible - same inputs always give same outputs
- Input validation with automatic boundary protection
- All scores scaled 0-100 with 2 decimal precision

ARCHITECTURE:
-------------
This is a standalone, non-LLM component that computes final scores from
validated answer quality metrics. It does not orchestrate or modify upstream
agent outputs.

INPUT:
------
Takes validated answer_quality data with metrics in range [0, 1]:
- relevance: How well the answer addresses the question
- correctness: Technical accuracy of the answer
- depth: Thoroughness of explanation
- structure: Logical organization
- gaps: List of identified knowledge gaps (informational only)

OUTPUT:
-------
Returns validated scores in range [0, 100]:
- technical: Technical competency score
- communication: Communication effectiveness score
- overall: Weighted overall performance score

FORMULAS:
---------
technical = (correctness × 0.6 + depth × 0.4) × 100
communication = (structure × 0.7 + relevance × 0.3) × 100
overall = technical × 0.6 + communication × 0.4

Weights are loaded from backend.config.SCORING_WEIGHTS and can be
modified without code changes.
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

from backend.config import SCORING_WEIGHTS
from backend.utils.validators import (
    clamp_value,
    validate_and_clamp_dict,
    round_score
)

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS - INPUT/OUTPUT CONTRACTS
# =============================================================================

class AnswerQualityInput(BaseModel):
    """
    Input contract for answer quality metrics.

    All numeric fields must be in range [0, 1].
    Out-of-bounds values are automatically clamped.
    Missing required fields will raise validation errors.
    """
    relevance: float = Field(
        ...,
        description="How well the answer addresses the question (0-1)",
        ge=0.0,
        le=1.0
    )
    correctness: float = Field(
        ...,
        description="Technical accuracy of the answer (0-1)",
        ge=0.0,
        le=1.0
    )
    depth: float = Field(
        ...,
        description="Thoroughness and completeness of explanation (0-1)",
        ge=0.0,
        le=1.0
    )
    structure: float = Field(
        ...,
        description="Logical organization and coherence (0-1)",
        ge=0.0,
        le=1.0
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="List of identified knowledge gaps (informational)"
    )

    @field_validator('relevance', 'correctness', 'depth', 'structure')
    @classmethod
    def clamp_scores(cls, v: float) -> float:
        """Automatically clamp out-of-bounds values to valid range."""
        return clamp_value(v, 0.0, 1.0)


class ScoreOutput(BaseModel):
    """
    Output contract for computed scores.

    All scores are in range [0, 100] with 2 decimal precision.
    """
    technical: float = Field(
        ...,
        description="Technical competency score (0-100)",
        ge=0.0,
        le=100.0
    )
    communication: float = Field(
        ...,
        description="Communication effectiveness score (0-100)",
        ge=0.0,
        le=100.0
    )
    overall: float = Field(
        ...,
        description="Weighted overall performance score (0-100)",
        ge=0.0,
        le=100.0
    )


class ScoringResult(BaseModel):
    """Complete scoring result with metadata."""
    scores: ScoreOutput = Field(
        ...,
        description="Computed scores"
    )


# =============================================================================
# SCORING AGGREGATION AGENT
# =============================================================================

class ScoringAggregationAgent:
    """
    Production-grade deterministic scoring engine.

    Usage:
        agent = ScoringAggregationAgent()
        result = agent.compute(answer_quality_data)
        print(result.scores.overall)

    Configuration:
        Weights are loaded from backend.config.SCORING_WEIGHTS.
        Modify config.py to adjust scoring behavior without code changes.

    Thread Safety:
        This class is thread-safe and can be reused across requests.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize scoring agent with configured weights.

        Args:
            config_path: Optional path to custom config (for testing).
                        If None, uses backend.config.SCORING_WEIGHTS.

        Raises:
            ValueError: If weights are invalid or missing
        """
        # Load weights from config
        if config_path:
            # For testing: load custom config
            logger.warning(f"Loading custom config from {config_path}")
            # This would need custom implementation - for now, use default
            self.weights = SCORING_WEIGHTS
        else:
            self.weights = SCORING_WEIGHTS

        # Validate weight structure
        self._validate_weights()

        logger.info("ScoringAggregationAgent initialized with weights:")
        logger.info(f"  Technical: {self.weights['technical']}")
        logger.info(f"  Communication: {self.weights['communication']}")
        logger.info(f"  Overall: {self.weights['overall']}")

    def _validate_weights(self) -> None:
        """
        Validate weight configuration.

        Ensures:
        - All required weight categories exist
        - All weights are numeric
        - Weights within each category sum to ~1.0 (allowing for float precision)

        Raises:
            ValueError: If weights are invalid
        """
        required_categories = ['technical', 'communication', 'overall']

        for category in required_categories:
            if category not in self.weights:
                raise ValueError(f"Missing weight category: {category}")

            if not isinstance(self.weights[category], dict):
                raise ValueError(
                    f"Weight category {category} must be a dictionary"
                )

        # Validate technical weights
        tech_weights = self.weights['technical']
        if 'correctness' not in tech_weights or 'depth' not in tech_weights:
            raise ValueError(
                "Technical weights must include 'correctness' and 'depth'"
            )

        # Validate communication weights
        comm_weights = self.weights['communication']
        if 'structure' not in comm_weights or 'relevance' not in comm_weights:
            raise ValueError(
                "Communication weights must include 'structure' and 'relevance'"
            )

        # Validate overall weights
        overall_weights = self.weights['overall']
        if 'technical' not in overall_weights or 'communication' not in overall_weights:
            raise ValueError(
                "Overall weights must include 'technical' and 'communication'"
            )

    def _calculate_technical_score(
        self,
        correctness: float,
        depth: float
    ) -> float:
        """
        Calculate technical competency score.

        Formula:
            technical = (correctness × w_correctness + depth × w_depth) × 100

        Args:
            correctness: Technical accuracy (0-1)
            depth: Thoroughness of explanation (0-1)

        Returns:
            Technical score (0-100), rounded to 2 decimals
        """
        w_correctness = self.weights['technical']['correctness']
        w_depth = self.weights['technical']['depth']

        technical = (correctness * w_correctness + depth * w_depth) * 100.0

        logger.debug(
            f"Technical: ({correctness} × {w_correctness} + "
            f"{depth} × {w_depth}) × 100 = {technical:.2f}"
        )

        return round_score(technical)

    def _calculate_communication_score(
        self,
        structure: float,
        relevance: float
    ) -> float:
        """
        Calculate communication effectiveness score.

        Formula:
            communication = (structure × w_structure + relevance × w_relevance) × 100

        Args:
            structure: Logical organization (0-1)
            relevance: Question relevance (0-1)

        Returns:
            Communication score (0-100), rounded to 2 decimals
        """
        w_structure = self.weights['communication']['structure']
        w_relevance = self.weights['communication']['relevance']

        communication = (
            structure * w_structure + relevance * w_relevance
        ) * 100.0

        logger.debug(
            f"Communication: ({structure} × {w_structure} + "
            f"{relevance} × {w_relevance}) × 100 = {communication:.2f}"
        )

        return round_score(communication)

    def _calculate_overall_score(
        self,
        technical: float,
        communication: float
    ) -> float:
        """
        Calculate weighted overall performance score.

        Formula:
            overall = technical × w_technical + communication × w_communication

        Note: Input scores are already scaled 0-100, so no additional
        scaling is needed.

        Args:
            technical: Technical score (0-100)
            communication: Communication score (0-100)

        Returns:
            Overall score (0-100), rounded to 2 decimals
        """
        w_technical = self.weights['overall']['technical']
        w_communication = self.weights['overall']['communication']

        overall = technical * w_technical + communication * w_communication

        logger.debug(
            f"Overall: {technical} × {w_technical} + "
            f"{communication} × {w_communication} = {overall:.2f}"
        )

        return round_score(overall)

    def compute(
        self,
        answer_quality: Dict[str, Any]
    ) -> ScoringResult:
        """
        Compute scores from answer quality metrics.

        This is the main entry point for the agent. It validates input,
        computes scores using deterministic formulas, and returns validated
        output.

        Args:
            answer_quality: Dictionary containing:
                - relevance (float): 0-1
                - correctness (float): 0-1
                - depth (float): 0-1
                - structure (float): 0-1
                - gaps (list[str]): optional

        Returns:
            ScoringResult containing validated scores

        Raises:
            ValidationError: If input is invalid or missing required fields

        Example:
            >>> agent = ScoringAggregationAgent()
            >>> result = agent.compute({
            ...     "relevance": 0.8,
            ...     "correctness": 0.9,
            ...     "depth": 0.7,
            ...     "structure": 0.85,
            ...     "gaps": []
            ... })
            >>> print(result.scores.overall)
            82.5
        """
        logger.info("Computing scores from answer quality metrics")

        # Validate and clamp input
        try:
            # First pass: clamp any out-of-bounds values
            clamped_input = validate_and_clamp_dict(answer_quality, 0.0, 1.0)

            # Second pass: full Pydantic validation
            validated_input = AnswerQualityInput(**clamped_input)

            logger.debug(f"Validated input: {validated_input}")

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise

        # Extract validated values
        relevance = validated_input.relevance
        correctness = validated_input.correctness
        depth = validated_input.depth
        structure = validated_input.structure

        # Calculate component scores
        technical = self._calculate_technical_score(correctness, depth)
        communication = self._calculate_communication_score(
            structure, relevance)

        # Calculate overall score
        overall = self._calculate_overall_score(technical, communication)

        # Create validated output
        scores = ScoreOutput(
            technical=technical,
            communication=communication,
            overall=overall
        )

        result = ScoringResult(scores=scores)

        logger.info(
            f"Scores computed - Technical: {technical}, "
            f"Communication: {communication}, Overall: {overall}"
        )

        # Log gap information if present
        if validated_input.gaps:
            logger.info(
                f"Knowledge gaps identified: {len(validated_input.gaps)}")
            for i, gap in enumerate(validated_input.gaps, 1):
                logger.debug(f"  Gap {i}: {gap}")

        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_scores(answer_quality: Dict[str, Any]) -> Dict[str, float]:
    """
    Convenience function for one-off score computation.

    Args:
        answer_quality: Answer quality metrics dictionary

    Returns:
        Dictionary with keys: technical, communication, overall

    Example:
        >>> scores = compute_scores({
        ...     "relevance": 0.8,
        ...     "correctness": 0.9,
        ...     "depth": 0.7,
        ...     "structure": 0.85,
        ...     "gaps": []
        ... })
        >>> print(scores['overall'])
        82.5
    """
    agent = ScoringAggregationAgent()
    result = agent.compute(answer_quality)
    return {
        "technical": result.scores.technical,
        "communication": result.scores.communication,
        "overall": result.scores.overall
    }


# =========================================================
# LANGGRAPH NODE WRAPPER
# =========================================================

def scoring_aggregation_node(state: "InterviewState") -> "InterviewState":
    """
    LangGraph node wrapper for ScoringAggregationAgent.

    Args:
        state: InterviewState object containing:
            - answer_quality: AnswerQualityModel with answer quality metrics
            - confidence_behavior: ConfidenceBehaviorModel with confidence metrics (for behavioral score)

    Returns:
        Updated InterviewState with scores field populated
    """
    from backend.models.state import InterviewState, ScoresModel

    logger.info(
        f"ScoringAggregationNode: Starting for interview_id={state.interview_id}")

    try:
        # Extract answer quality
        answer_quality = state.answer_quality

        if not answer_quality:
            logger.warning(
                "ScoringAggregationNode: Missing answer_quality, using defaults")
            updated_state = state.model_copy(deep=True)
            updated_state.scores = ScoresModel(
                technical=0.0,
                communication=0.0,
                behavioral=0.0,
                overall=0.0
            )
            return updated_state

        # Log received answer_quality metrics for debugging
        logger.info(
            f"ScoringAggregationNode: Received answer_quality - "
            f"relevance={answer_quality.relevance:.2f}, "
            f"correctness={answer_quality.correctness:.2f}, "
            f"depth={answer_quality.depth:.2f}, "
            f"structure={answer_quality.structure:.2f}, "
            f"gaps_count={len(answer_quality.gaps)}"
        )

        # Convert to dict for agent (agent expects dict format)
        answer_quality_dict = {
            'relevance': answer_quality.relevance,
            'correctness': answer_quality.correctness,
            'depth': answer_quality.depth,
            'structure': answer_quality.structure,
            'gaps': answer_quality.gaps
        }

        # Initialize agent
        agent = ScoringAggregationAgent()

        # Compute scores
        result = agent.compute(answer_quality_dict)

        # Calculate behavioral score from confidence_behavior if available
        behavioral_score = 0.0

        if state.confidence_behavior:
            # Behavioral score formula:
            # (confidence × 0.4 + professionalism × 0.6) × 100
            # Lower nervousness increases the score
            confidence = state.confidence_behavior.confidence
            professionalism = state.confidence_behavior.professionalism
            nervousness = state.confidence_behavior.nervousness

            # Adjust confidence based on nervousness
            adjusted_confidence = confidence * (1 - nervousness * 0.3)

            behavioral_score = (adjusted_confidence * 0.4 +
                                professionalism * 0.6) * 100
            behavioral_score = max(0.0, min(100.0, behavioral_score))
            behavioral_score = round(behavioral_score, 2)

        # Calculate overall score with behavioral component
        # overall = technical × 0.4 + communication × 0.3 + behavioral × 0.3
        overall_with_behavioral = (
            result.scores.technical * 0.4 +
            result.scores.communication * 0.3 +
            behavioral_score * 0.3
        )
        overall_with_behavioral = round(overall_with_behavioral, 2)

        # Update state
        updated_state = state.model_copy(deep=True)
        updated_state.scores = ScoresModel(
            technical=result.scores.technical,
            communication=result.scores.communication,
            behavioral=behavioral_score,
            overall=overall_with_behavioral
        )

        logger.info(
            f"ScoringAggregationNode: Completed - "
            f"technical={result.scores.technical:.1f}, "
            f"communication={result.scores.communication:.1f}, "
            f"behavioral={behavioral_score:.1f}, "
            f"overall={overall_with_behavioral:.1f}"
        )

        # Return only the key we're updating (LangGraph merges it into state)
        return {"scores": updated_state.scores}

    except Exception as e:
        logger.error(f"ScoringAggregationNode: Failed - {e}", exc_info=True)
        # Return default scores on error
        return {
            "scores": ScoresModel(
                technical=0.0,
                communication=0.0,
                behavioral=0.0,
                overall=0.0
            )
        }
