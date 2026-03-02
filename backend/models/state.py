"""
InterviewState - Global State Contract
=======================================
Pydantic models defining the shared state passed through all agents.
This is the single source of truth for the interview analysis pipeline.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QuestionModel(BaseModel):
    """Question generated for the interview."""
    text: str = Field(..., description="The interview question text")
    topic: str = Field(..., description="Topic category (e.g., 'algorithms', 'system design')")
    difficulty: float = Field(..., ge=0.0, le=1.0, description="Difficulty rating from 0 to 1")
    intent: List[str] = Field(default_factory=list, description="Expected answer components")


class VoiceAnalysisModel(BaseModel):
    """Voice and speech characteristics analysis."""
    clarity: float = Field(0.0, ge=0.0, le=1.0, description="Speech clarity score")
    speech_rate_wpm: float = Field(0.0, ge=0.0, description="Words per minute")
    filler_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Ratio of filler words to total words")
    tone: str = Field("neutral", description="Detected tone (confident, neutral, nervous)")


class AnswerQualityModel(BaseModel):
    """Quality assessment of the answer content."""
    relevance: float = Field(0.0, ge=0.0, le=1.0, description="How relevant the answer is to the question")
    correctness: float = Field(0.0, ge=0.0, le=1.0, description="Technical correctness of the answer")
    depth: float = Field(0.0, ge=0.0, le=1.0, description="Depth of explanation")
    structure: float = Field(0.0, ge=0.0, le=1.0, description="Logical structure and organization")
    gaps: List[str] = Field(default_factory=list, description="Missing concepts or information")


class BodyLanguageModel(BaseModel):
    """Body language and visual behavior analysis."""
    eye_contact: float = Field(0.0, ge=0.0, le=1.0, description="Consistency of eye contact")
    posture_stability: float = Field(0.0, ge=0.0, le=1.0, description="Stability and confidence in posture")
    facial_expressiveness: float = Field(0.0, ge=0.0, le=1.0, description="Appropriate facial expressions")
    distractions: List[str] = Field(default_factory=list, description="Observed distracting behaviors")


class ConfidenceBehaviorModel(BaseModel):
    """Confidence and behavioral inference."""
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence level")
    nervousness: float = Field(0.0, ge=0.0, le=1.0, description="Signs of nervousness")
    professionalism: float = Field(0.0, ge=0.0, le=1.0, description="Professional demeanor")
    behavioral_flags: List[str] = Field(default_factory=list, description="Notable behavioral patterns")


class ScoresModel(BaseModel):
    """Final aggregated scores."""
    technical: float = Field(0.0, ge=0.0, le=100.0, description="Technical competency score (0-100)")
    communication: float = Field(0.0, ge=0.0, le=100.0, description="Communication effectiveness score (0-100)")
    behavioral: float = Field(0.0, ge=0.0, le=100.0, description="Behavioral professionalism score (0-100)")
    overall: float = Field(0.0, ge=0.0, le=100.0, description="Weighted overall score (0-100)")


class RecommendationsModel(BaseModel):
    """Actionable recommendations based on analysis."""
    strengths: List[str] = Field(default_factory=list, description="Identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Areas needing improvement")
    improvement_plan: List[str] = Field(default_factory=list, description="Specific actionable steps")


class InterviewState(BaseModel):
    """
    Global state object passed through all LangGraph nodes.
    
    Design principles:
    - Single source of truth for all interview data
    - Immutable between agent executions (each agent returns updated state)
    - JSON-serializable for persistence and API responses
    - Strongly typed for validation and error prevention
    """
    
    # Core identifiers
    interview_id: str = Field(..., description="Unique interview session identifier")
    role: str = Field(..., description="Job role for the interview")
    experience_level: str = Field(..., description="Candidate experience level (Fresher, Junior, Mid, Senior)")
    
    # Agent outputs - initialized as empty/default, populated by respective agents
    question: Optional[QuestionModel] = None
    transcript: str = Field("", description="Full transcription of candidate's answer")
    
    voice_analysis: VoiceAnalysisModel = Field(default_factory=VoiceAnalysisModel)
    answer_quality: AnswerQualityModel = Field(default_factory=AnswerQualityModel)
    body_language: BodyLanguageModel = Field(default_factory=BodyLanguageModel)
    confidence_behavior: ConfidenceBehaviorModel = Field(default_factory=ConfidenceBehaviorModel)
    
    scores: ScoresModel = Field(default_factory=ScoresModel)
    recommendations: RecommendationsModel = Field(default_factory=RecommendationsModel)
    
    # Internal metadata
    audio_path: Optional[str] = Field(None, description="Path to uploaded audio file")
    video_path: Optional[str] = Field(None, description="Path to uploaded video file")
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "interview_id": "int_123456",
                "role": "Software Engineer",
                "experience_level": "Mid",
                "question": {
                    "text": "Explain the difference between Thread and Process",
                    "topic": "operating_systems",
                    "difficulty": 0.6,
                    "intent": ["memory_isolation", "resource_sharing", "context_switching"]
                }
            }
        }


class InterviewStateUpdate(BaseModel):
    """
    Lightweight update model for partial state updates.
    Used when agents need to update only their namespace.
    """
    interview_id: str
    updates: Dict[str, Any] = Field(..., description="Fields to update")
