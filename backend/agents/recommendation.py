"""
Recommendation System Agent
===========================
Generates actionable recommendations from analysis results.

Architecture:
- Uses LLM (GPT-4o-mini) to generate human-readable recommendations
- Converts scores and gaps into strengths, weaknesses, improvement plans
- Outputs structured, actionable feedback
"""

import json
import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from backend.models.state import InterviewState, RecommendationsModel
from backend.config import get_llm_config

logger = logging.getLogger(__name__)


RECOMMENDATION_PROMPT = """You are an expert career coach providing interview feedback.

INTERVIEW PERFORMANCE DATA:

SCORES:
- Technical: {technical}/100
- Communication: {communication}/100
- Behavioral: {behavioral}/100
- Overall: {overall}/100

VOICE ANALYSIS:
- Speech Rate: {speech_rate_wpm} WPM (ideal: 130-160)
- Filler Ratio: {filler_ratio} (should be < 0.05)
- Clarity: {clarity}
- Tone: {tone}

ANSWER QUALITY:
- Relevance: {relevance}
- Correctness: {correctness}
- Depth: {depth}
- Structure: {structure}
- Missing Concepts: {gaps}

BODY LANGUAGE:
- Eye Contact: {eye_contact}
- Posture Stability: {posture_stability}
- Facial Expressiveness: {facial_expressiveness}

BEHAVIORAL:
- Confidence: {confidence}
- Nervousness: {nervousness}
- Professionalism: {professionalism}
- Flags: {behavioral_flags}

Based on this data, provide constructive feedback:

1. STRENGTHS (3-5 items): What the candidate did well
2. WEAKNESSES (2-4 items): Areas that need improvement
3. IMPROVEMENT_PLAN (3-6 items): Specific actionable steps

Be:
- Specific and evidence-based
- Constructive and encouraging
- Actionable (not vague advice)
- Professional

You MUST respond with valid JSON in this EXACT format:
{{
    "strengths": [
        "Specific strength with evidence",
        "Another strength"
    ],
    "weaknesses": [
        "Area to improve with context",
        "Another area"
    ],
    "improvement_plan": [
        "Specific actionable step 1",
        "Specific actionable step 2"
    ]
}}

Return ONLY the JSON object."""


class RecommendationSystemAgent:
    """
    Agent responsible for generating actionable recommendations.
    
    Design principles:
    - Convert quantitative data into qualitative feedback
    - Specific, actionable recommendations
    - Balanced (strengths + areas to improve)
    - Evidence-based from actual metrics
    """
    
    def __init__(self):
        """Initialize with configured LLM."""
        self.config = get_llm_config("recommendation")
        self.llm = self._create_llm()
    
    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance."""
        return ChatOpenAI(
            model=self.config.get("model", "gpt-4o-mini"),
            api_key=self.config.get("api_key"),
            base_url=self.config.get("base_url"),
            temperature=self.config.get("temperature", 0.7),
        )
    
    def _build_prompt(self, state: InterviewState) -> str:
        """Build recommendation prompt from complete state."""
        return RECOMMENDATION_PROMPT.format(
            technical=state.scores.technical,
            communication=state.scores.communication,
            behavioral=state.scores.behavioral,
            overall=state.scores.overall,
            speech_rate_wpm=state.voice_analysis.speech_rate_wpm,
            filler_ratio=state.voice_analysis.filler_ratio,
            clarity=state.voice_analysis.clarity,
            tone=state.voice_analysis.tone,
            relevance=state.answer_quality.relevance,
            correctness=state.answer_quality.correctness,
            depth=state.answer_quality.depth,
            structure=state.answer_quality.structure,
            gaps=", ".join(state.answer_quality.gaps) if state.answer_quality.gaps else "none",
            eye_contact=state.body_language.eye_contact,
            posture_stability=state.body_language.posture_stability,
            facial_expressiveness=state.body_language.facial_expressiveness,
            confidence=state.confidence_behavior.confidence,
            nervousness=state.confidence_behavior.nervousness,
            professionalism=state.confidence_behavior.professionalism,
            behavioral_flags=", ".join(state.confidence_behavior.behavioral_flags)
        )
    
    def _call_llm(self, prompt: str, state: InterviewState) -> Dict[str, Any]:
        """Call LLM for recommendation generation."""
        try:
            messages = [
                SystemMessage(content="You are a professional career coach. Always return valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # Handle markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            
            # Validate
            recommendations = RecommendationsModel(**parsed)
            return recommendations.model_dump()
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return self._generate_fallback(state)
    
    def _generate_fallback(self, state: InterviewState) -> Dict[str, Any]:
        """Generate fallback recommendations based on scores."""
        logger.warning("Using fallback recommendation generation")
        
        strengths = []
        weaknesses = []
        improvement_plan = []
        
        # Simple rule-based fallback
        if state.scores.technical >= 70:
            strengths.append("Strong technical knowledge demonstrated")
        else:
            weaknesses.append("Technical depth needs improvement")
            improvement_plan.append("Review core concepts and practice explaining them")
        
        if state.scores.communication >= 70:
            strengths.append("Effective communication skills")
        else:
            weaknesses.append("Communication clarity could be enhanced")
            improvement_plan.append("Practice structuring answers with clear introduction, body, and conclusion")
        
        if state.scores.behavioral >= 70:
            strengths.append("Professional demeanor and confidence")
        else:
            weaknesses.append("Confidence and professionalism need work")
            improvement_plan.append("Practice mock interviews to build confidence")
        
        return {
            "strengths": strengths or ["Participated in the interview process"],
            "weaknesses": weaknesses or ["Continue developing interview skills"],
            "improvement_plan": improvement_plan or ["Practice regularly with mock interviews"]
        }
    
    def execute(self, state: InterviewState) -> InterviewState:
        """
        Execute recommendation generation.
        
        Prerequisites: All previous agents must have run (need scores and all metrics)
        
        Returns:
            Updated state with recommendations populated
        """
        logger.info("Generating recommendations")
        
        prompt = self._build_prompt(state)
        rec_data = self._call_llm(prompt, state)
        recommendations = RecommendationsModel(**rec_data)
        
        updated_state = state.model_copy(deep=True)
        updated_state.recommendations = recommendations
        
        logger.info(
            f"Recommendations generated: {len(recommendations.strengths)} strengths, "
            f"{len(recommendations.weaknesses)} weaknesses, "
            f"{len(recommendations.improvement_plan)} action items"
        )
        
        return updated_state


# LangGraph node wrapper
def recommendation_node(state: InterviewState) -> InterviewState:
    """LangGraph node wrapper for RecommendationSystemAgent."""
    agent = RecommendationSystemAgent()
    updated_state = agent.execute(state)
    return updated_state
