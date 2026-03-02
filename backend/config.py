"""
Backend Configuration
=====================
Centralized configuration for the Multi-Agent Interview Analyzer.
All model routing, API keys, and system parameters are defined here.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # ← Load .env before any other code runs


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Principles:
    - All secrets via environment variables
    - Sensible defaults for development
    - Production overrides via .env file
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Allow extra env variables not defined in model
    )

    # Application
    APP_NAME: str = "Interview Performance Analyzer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False

    # LLM Configuration - Swappable models
    # Question Generation Agent
    QUESTION_GENERATION_MODEL: str = "gpt-4o-mini"
    QUESTION_GENERATION_API_KEY: Optional[str] = None
    QUESTION_GENERATION_BASE_URL: Optional[str] = None

    # Answer Quality Analyser (production default: gpt-4o-mini per spec)
    # Note: Can be upgraded to gpt-4o for enhanced analysis
    ANSWER_QUALITY_MODEL: str = "gpt-4o-mini"
    ANSWER_QUALITY_API_KEY: Optional[str] = None
    ANSWER_QUALITY_BASE_URL: Optional[str] = None

    # Confidence Behaviour Inference
    CONFIDENCE_INFERENCE_MODEL: str = "gpt-4o-mini"
    CONFIDENCE_INFERENCE_API_KEY: Optional[str] = None
    CONFIDENCE_INFERENCE_BASE_URL: Optional[str] = None

    # Recommendation System
    RECOMMENDATION_MODEL: str = "gpt-4o-mini"
    RECOMMENDATION_API_KEY: Optional[str] = None
    RECOMMENDATION_BASE_URL: Optional[str] = None

    # Whisper Configuration
    WHISPER_MODEL_SIZE: str = "small"  # tiny, base, small, medium, large
    WHISPER_DEVICE: str = "cpu"  # cpu or cuda
    WHISPER_COMPUTE_TYPE: str = "float32"  # int8, float16, float32

    # Scoring Configuration - Deterministic weights
    SCORE_TECHNICAL_CORRECTNESS_WEIGHT: float = 0.6
    SCORE_TECHNICAL_DEPTH_WEIGHT: float = 0.4

    SCORE_COMMUNICATION_STRUCTURE_WEIGHT: float = 0.6
    SCORE_COMMUNICATION_CLARITY_WEIGHT: float = 0.4

    SCORE_BEHAVIORAL_CONFIDENCE_WEIGHT: float = 0.5
    SCORE_BEHAVIORAL_PROFESSIONALISM_WEIGHT: float = 0.5

    SCORE_OVERALL_TECHNICAL_WEIGHT: float = 0.4
    SCORE_OVERALL_COMMUNICATION_WEIGHT: float = 0.3
    SCORE_OVERALL_BEHAVIORAL_WEIGHT: float = 0.3

    # Voice Analysis Configuration
    VOICE_IDEAL_WPM_MIN: float = 130.0
    VOICE_IDEAL_WPM_MAX: float = 160.0
    VOICE_MAX_FILLER_RATIO: float = 0.05  # 5% max filler words

    # Body Language Configuration
    BODY_FRAME_SAMPLE_RATE: int = 2  # Sample 1 frame every N seconds
    BODY_MIN_EYE_CONTACT_RATIO: float = 0.6  # 60% eye contact minimum

    # File Upload Configuration
    UPLOAD_DIR: str = "data/uploads"
    MAX_AUDIO_SIZE_MB: int = 50
    MAX_VIDEO_SIZE_MB: int = 200
    ALLOWED_AUDIO_EXTENSIONS: set = {"mp3", "wav", "m4a", "ogg"}
    ALLOWED_VIDEO_EXTENSIONS: set = {"mp4", "avi", "mov", "webm"}

    # Database (optional - for persistence)
    DATABASE_URL: Optional[str] = None

    # Logging
    LOG_FILE: str = "logs/interview_analyzer.log"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"


# Global settings instance
settings = Settings()


# Scoring Weights Configuration
# ===============================
# Deterministic scoring weights for the Scoring & Aggregation Agent.
# These weights define how different components contribute to final scores.
# All weights must sum to 1.0 within each category.
SCORING_WEIGHTS = {
    "technical": {
        "correctness": 0.6,  # Weight for correctness in technical score
        "depth": 0.4         # Weight for depth in technical score
    },
    "communication": {
        "structure": 0.7,    # Weight for structure in communication score
        "relevance": 0.3     # Weight for relevance in communication score
    },
    "overall": {
        "technical": 0.6,    # Weight for technical in overall score
        "communication": 0.4  # Weight for communication in overall score
    }
}


# Model routing helper
def get_llm_config(agent_name: str) -> dict:
    """
    Get LLM configuration for a specific agent.

    Returns dict with:
    - model: model name
    - api_key: API key
    - base_url: optional base URL
    - temperature: default temperature
    """
    config_map = {
        "question_generation": {
            "model": settings.QUESTION_GENERATION_MODEL,
            "api_key": settings.QUESTION_GENERATION_API_KEY,
            "base_url": settings.QUESTION_GENERATION_BASE_URL,
            "temperature": 0.7,
        },
        "answer_quality": {
            "model": settings.ANSWER_QUALITY_MODEL,
            "api_key": settings.ANSWER_QUALITY_API_KEY,
            "base_url": settings.ANSWER_QUALITY_BASE_URL,
            "temperature": 0.3,  # Lower for more consistent analysis
        },
        "confidence_inference": {
            "model": settings.CONFIDENCE_INFERENCE_MODEL,
            "api_key": settings.CONFIDENCE_INFERENCE_API_KEY,
            "base_url": settings.CONFIDENCE_INFERENCE_BASE_URL,
            "temperature": 0.5,
        },
        "recommendation": {
            "model": settings.RECOMMENDATION_MODEL,
            "api_key": settings.RECOMMENDATION_API_KEY,
            "base_url": settings.RECOMMENDATION_BASE_URL,
            "temperature": 0.7,
        },
    }

    return config_map.get(agent_name, {})
