"""
Confidence & Behavioral Inference Agent
========================================
Production-grade agent for inferring confidence and behavioral traits from structured metrics.

CRITICAL DESIGN PRINCIPLES:
- NO recomputation of voice metrics
- NO recomputation of answer quality
- NO scoring logic
- NO access to raw audio/video
- ONLY reads structured upstream outputs

Architecture:
- Input: voice_analysis + answer_quality (structured metrics)
- LLM: GPT-4o-mini (temperature=0.2 for low variance)
- Output: confidence, nervousness, professionalism, behavioral_flags
- Strict JSON-only output with retry logic
- Comprehensive bias protection

Author: Senior AI Backend Engineer
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from backend.models.state import ConfidenceBehaviorModel, InterviewState
from backend.config import get_llm_config
from backend.utils.json_parser import parse_llm_json_with_retry, clamp_value

from dotenv import load_dotenv
load_dotenv()  # ← Ensure .env is loaded in this module

logger = logging.getLogger(__name__)


# =========================================================
# PROMPT TEMPLATE WITH COMPREHENSIVE BIAS PROTECTIONS
# =========================================================

CONFIDENCE_INFERENCE_PROMPT = """You are an expert behavioral psychologist analyzing interview performance.

You have access to STRUCTURED METRICS about the candidate's interview performance:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOICE ANALYSIS METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Speech Rate: {speech_rate_wpm} WPM
- Filler Ratio: {filler_ratio} (0.0 = no fillers, 1.0 = all fillers)
- Clarity: {clarity} (0.0-1.0 scale)
- Tone: {tone}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANSWER QUALITY METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Relevance: {relevance} (0.0-1.0)
- Correctness: {correctness} (0.0-1.0)
- Depth: {depth} (0.0-1.0)
- Structure: {structure} (0.0-1.0)
- Missing Concepts: {gaps}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFERENCE RULES (Evidence-Based Assessment)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONFIDENCE INDICATORS:
✓ High correctness + low filler_ratio → Increased confidence
✓ High structure + steady tone → Increased confidence  
✓ High depth + high clarity → Increased confidence
✗ Low correctness does NOT automatically mean low confidence
✗ Fast speech rate alone is NOT a negative indicator

NERVOUSNESS INDICATORS:
✓ High filler_ratio (>0.1) + low clarity (<0.5) → Increased nervousness
✓ Very high filler_ratio (>0.15) alone → Moderate nervousness
✓ Erratic tone patterns → Possible nervousness
✗ Fast speech WITHOUT high fillers is NOT nervousness
✗ Low filler words alone does NOT indicate low nervousness

PROFESSIONALISM INDICATORS:
✓ High structure (>0.7) + relevant answers → Increased professionalism
✓ Low filler_ratio + organized responses → Increased professionalism
✓ Consistent clarity → Professionalism
✗ Speech rate variation alone does NOT affect professionalism

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BIAS PROTECTION GUIDELINES (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ DO NOT penalize fast speech rate alone (120-180 WPM is normal range)
⚠️ DO NOT assume low filler words = high intelligence
⚠️ DO NOT penalize accents or speech patterns
⚠️ DO NOT infer demographic traits (age, gender, ethnicity)
⚠️ DO NOT speculate about personality beyond evidence
⚠️ DO NOT assume nervousness from speech rate variation alone
⚠️ ONLY use evidence-based patterns from the metrics provided

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORE INTERPRETATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0.8 - 1.0  = Strong presence/performance
0.5 - 0.8  = Moderate presence/performance  
0.0 - 0.5  = Needs improvement

Adjust scores based on MULTIPLE indicators, not single metrics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIORAL FLAGS GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Provide 0-5 specific, evidence-based flags. Examples:
- "confident_technical_delivery" (high correctness + low fillers)
- "needs_composure_improvement" (high fillers + low clarity)  
- "strong_structural_communication" (high structure)
- "knowledge_gaps_present" (low correctness + missing concepts)
- "professional_demeanor" (high structure + clarity)

If no notable patterns, return empty list.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT JSON ONLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You MUST respond with ONLY valid JSON in this EXACT format:

{{
    "confidence": 0.75,
    "nervousness": 0.25,
    "professionalism": 0.82,
    "behavioral_flags": ["confident_technical_delivery", "strong_structural_communication"]
}}

RULES:
- All numeric values MUST be floats between 0.0 and 1.0
- behavioral_flags MUST be a list of strings (0-5 items)
- Return ONLY the JSON object, no additional text
- Use evidence-based assessment
- Be consistent and fair

Analyze the metrics now and return ONLY the JSON output:
"""


# =========================================================
# INPUT VALIDATION MODELS
# =========================================================

class VoiceAnalysisInput(BaseModel):
    """Voice analysis input schema."""
    speech_rate_wpm: float = Field(..., ge=0.0, description="Words per minute")
    filler_ratio: float = Field(..., ge=0.0, le=1.0,
                                description="Ratio of filler words")
    clarity: float = Field(..., ge=0.0, le=1.0,
                           description="Speech clarity score")
    tone: str = Field(..., description="Detected tone")


class AnswerQualityInput(BaseModel):
    """Answer quality input schema."""
    relevance: float = Field(..., ge=0.0, le=1.0,
                             description="Answer relevance")
    correctness: float = Field(..., ge=0.0, le=1.0,
                               description="Technical correctness")
    depth: float = Field(..., ge=0.0, le=1.0, description="Explanation depth")
    structure: float = Field(..., ge=0.0, le=1.0,
                             description="Logical structure")
    gaps: List[str] = Field(default_factory=list,
                            description="Missing concepts")


# =========================================================
# CONFIDENCE & BEHAVIORAL INFERENCE AGENT
# =========================================================

class ConfidenceBehaviorInferenceAgent:
    """
    Production-grade agent for confidence and behavioral trait inference.

    Design Principles:
    - Reads ONLY structured metrics (no raw audio/video access)
    - NO recomputation of upstream metrics
    - NO scoring logic
    - LLM-based reasoning with strict JSON output
    - Comprehensive bias protection
    - Retry logic for robustness

    Architecture:
    - Input: voice_analysis + answer_quality
    - Processing: LLM inference with structured prompt
    - Output: ConfidenceBehaviorModel (validated Pydantic model)

    Temperature: 0.2 (low variance for consistent inference)
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the confidence inference agent.

        Args:
            model_name: Optional model override (default: from config)
        """
        self.config = get_llm_config("confidence_inference")

        # Allow model override
        if model_name:
            self.config["model"] = model_name

        # Enforce low temperature for consistency
        self.config["temperature"] = 0.2

        self.llm = self._create_llm()
        logger.info(
            f"Initialized ConfidenceBehaviorInferenceAgent with model: {self.config['model']}")

    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance with strict configuration."""
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "No API key found. Set CONFIDENCE_INFERENCE_API_KEY or OPENAI_API_KEY environment variable"
            )

        return ChatOpenAI(
            model=self.config["model"],
            api_key=api_key,
            base_url=self.config.get("base_url"),
            temperature=self.config["temperature"],
        )

    def _build_prompt(
        self,
        voice_analysis: Dict[str, Any],
        answer_quality: Dict[str, Any]
    ) -> str:
        """
        Build inference prompt from structured metrics.

        Args:
            voice_analysis: Voice metrics dictionary
            answer_quality: Answer quality metrics dictionary

        Returns:
            Formatted prompt string
        """
        gaps_str = ", ".join(answer_quality.get("gaps", [])) if answer_quality.get(
            "gaps") else "none identified"

        return CONFIDENCE_INFERENCE_PROMPT.format(
            speech_rate_wpm=voice_analysis["speech_rate_wpm"],
            filler_ratio=voice_analysis["filler_ratio"],
            clarity=voice_analysis["clarity"],
            tone=voice_analysis["tone"],
            relevance=answer_quality["relevance"],
            correctness=answer_quality["correctness"],
            depth=answer_quality["depth"],
            structure=answer_quality["structure"],
            gaps=gaps_str
        )

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Call LLM with retry logic for JSON parsing.

        Args:
            prompt: Formatted prompt
            max_retries: Maximum retry attempts (default: 2)

        Returns:
            Parsed and validated dictionary

        Raises:
            Exception: If all retries fail
        """
        def llm_call() -> str:
            """Wrapper for LLM call."""
            messages = [
                SystemMessage(
                    content="You are a behavioral psychologist. Always return ONLY valid JSON, no additional text."
                ),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            raw_output = response.content

            # Log raw output for debugging
            logger.debug(f"Raw LLM output: {raw_output[:300]}...")

            return raw_output

        # Use retry-enabled JSON parser
        parsed_data = parse_llm_json_with_retry(
            llm_call=llm_call,
            max_retries=max_retries,
            numeric_keys=["confidence", "nervousness", "professionalism"],
            clamp_range=(0.0, 1.0)
        )

        return parsed_data

    def _deduplicate_flags(self, flags: List[str]) -> List[str]:
        """
        Deduplicate behavioral flags while preserving order.

        Args:
            flags: List of behavioral flags

        Returns:
            Deduplicated list
        """
        seen = set()
        deduplicated = []
        for flag in flags:
            flag_lower = flag.lower().strip()
            if flag_lower and flag_lower not in seen:
                seen.add(flag_lower)
                deduplicated.append(flag)
        return deduplicated

    def _validate_and_clamp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clamp numeric outputs to ensure bounds.

        Args:
            data: Parsed data dictionary

        Returns:
            Validated and clamped dictionary
        """
        # Clamp numeric values
        data["confidence"] = clamp_value(data.get("confidence", 0.5), 0.0, 1.0)
        data["nervousness"] = clamp_value(
            data.get("nervousness", 0.5), 0.0, 1.0)
        data["professionalism"] = clamp_value(
            data.get("professionalism", 0.5), 0.0, 1.0)

        # Deduplicate and validate flags
        flags = data.get("behavioral_flags", [])
        if not isinstance(flags, list):
            logger.warning(
                f"behavioral_flags is not a list: {type(flags)}, converting to list")
            flags = []

        data["behavioral_flags"] = self._deduplicate_flags(flags)

        return data

    def infer(
        self,
        voice_analysis: Dict[str, Any],
        answer_quality: Dict[str, Any]
    ) -> ConfidenceBehaviorModel:
        """
        Perform confidence and behavioral inference on structured metrics.

        Args:
            voice_analysis: Voice analysis metrics
                - speech_rate_wpm: float
                - filler_ratio: float (0-1)
                - clarity: float (0-1)
                - tone: str
            answer_quality: Answer quality metrics
                - relevance: float (0-1)
                - correctness: float (0-1)
                - depth: float (0-1)
                - structure: float (0-1)
                - gaps: List[str]

        Returns:
            ConfidenceBehaviorModel with validated outputs

        Raises:
            ValueError: If input validation fails
            JSONParseError: If JSON parsing fails after retries
        """
        logger.info("Starting confidence and behavioral inference")

        # Validate inputs
        try:
            VoiceAnalysisInput(**voice_analysis)
            AnswerQualityInput(**answer_quality)
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Invalid input data: {e}")

        # Build prompt
        prompt = self._build_prompt(voice_analysis, answer_quality)

        try:
            # Call LLM with retry logic
            parsed_data = self._call_llm_with_retry(prompt, max_retries=2)

            # Validate and clamp outputs
            validated_data = self._validate_and_clamp(parsed_data)

            # Create Pydantic model for final validation
            confidence_behavior = ConfidenceBehaviorModel(**validated_data)

            logger.info(
                f"Inference complete: confidence={confidence_behavior.confidence:.2f}, "
                f"nervousness={confidence_behavior.nervousness:.2f}, "
                f"professionalism={confidence_behavior.professionalism:.2f}"
            )

            return confidence_behavior

        except Exception as e:
            logger.error(f"Confidence inference failed: {e}", exc_info=True)
            raise


# =========================================================
# EXAMPLE USAGE
# =========================================================

def example_usage():
    """Example usage of the ConfidenceBehaviorInferenceAgent."""

    # Sample structured metrics
    voice_analysis = {
        "speech_rate_wpm": 145.0,
        "filler_ratio": 0.08,
        "clarity": 0.82,
        "tone": "confident"
    }

    answer_quality = {
        "relevance": 0.85,
        "correctness": 0.78,
        "depth": 0.72,
        "structure": 0.88,
        "gaps": ["error_handling", "edge_cases"]
    }

    # Initialize agent
    agent = ConfidenceBehaviorInferenceAgent()

    # Perform inference
    result = agent.infer(voice_analysis, answer_quality)

    print("Confidence & Behavioral Inference Result:")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Nervousness: {result.nervousness:.2f}")
    print(f"  Professionalism: {result.professionalism:.2f}")
    print(f"  Behavioral Flags: {result.behavioral_flags}")

    return result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    example_usage()


# =========================================================
# LANGGRAPH NODE WRAPPER
# =========================================================

def confidence_behavior_node(state: InterviewState) -> InterviewState:
    """
    LangGraph node wrapper for ConfidenceBehaviorInferenceAgent.

    Args:
        state: InterviewState object containing:
            - voice_analysis: VoiceAnalysisModel with voice metrics
            - answer_quality: AnswerQualityModel with answer quality metrics

    Returns:
        Updated InterviewState with confidence_behavior field populated
    """
    from backend.models.state import InterviewState, ConfidenceBehaviorModel

    logger = logging.getLogger(__name__)
    logger.info(
        f"ConfidenceBehaviorNode: Starting for interview_id={state.interview_id}")

    try:
        # Validate inputs exist
        if not state.voice_analysis or not state.answer_quality:
            logger.warning(
                "ConfidenceBehaviorNode: Missing required inputs, using defaults")
            updated_state = state.model_copy(deep=True)
            updated_state.confidence_behavior = ConfidenceBehaviorModel(
                confidence=0.5,
                nervousness=0.5,
                professionalism=0.5,
                behavioral_flags=['insufficient_data']
            )
            return updated_state

        # Convert to dicts for agent (agent expects dict format)
        voice_analysis_dict = {
            'speech_rate_wpm': state.voice_analysis.speech_rate_wpm,
            'filler_ratio': state.voice_analysis.filler_ratio,
            'clarity': state.voice_analysis.clarity,
            'tone': state.voice_analysis.tone
        }

        answer_quality_dict = {
            'relevance': state.answer_quality.relevance,
            'correctness': state.answer_quality.correctness,
            'depth': state.answer_quality.depth,
            'structure': state.answer_quality.structure,
            'gaps': state.answer_quality.gaps
        }

        # Initialize agent
        agent = ConfidenceBehaviorInferenceAgent()

        # Perform inference
        result = agent.infer(voice_analysis_dict, answer_quality_dict)

        # Update state
        updated_state = state.model_copy(deep=True)
        updated_state.confidence_behavior = ConfidenceBehaviorModel(
            confidence=result.confidence,
            nervousness=result.nervousness,
            professionalism=result.professionalism,
            behavioral_flags=result.behavioral_flags
        )

        logger.info(
            f"ConfidenceBehaviorNode: Completed - "
            f"confidence={result.confidence:.2f}, "
            f"professionalism={result.professionalism:.2f}"
        )

        return updated_state

    except Exception as e:
        logger.error(f"ConfidenceBehaviorNode: Failed - {e}", exc_info=True)
        # Set defaults with error
        updated_state = state.model_copy(deep=True)
        updated_state.confidence_behavior = ConfidenceBehaviorModel(
            confidence=0.5,
            nervousness=0.5,
            professionalism=0.5,
            behavioral_flags=[f'error: {str(e)}']
        )
        return updated_state
