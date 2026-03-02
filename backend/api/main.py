"""
FastAPI Application
==================
REST API for the Multi-Agent Interview Analyzer.

Endpoints:
- POST /start-interview: Generate a question
- POST /submit-answer: Analyze complete interview
- GET /health: Health check
"""

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import settings
from backend.models.state import InterviewState, QuestionModel, ScoresModel, RecommendationsModel
from backend.graph.workflow import get_graph, run_interview_analysis
from backend.services.file_handler import save_upload_file, validate_file

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown)."""
    # ── STARTUP ──────────────────────────────────────────────────────────
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Create upload directory
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory: {upload_dir}")

    # Pre-compile graph
    get_graph()
    logger.info("Graph compiled and ready")

    yield  # Application runs here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────
    logger.info("Shutting down application...")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multi-Agent Interview Performance Analyzer",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js frontend
        "http://localhost:3001",  # Next.js frontend (alt port)
        "http://localhost:8000",  # Backend (for docs/testing)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class StartInterviewRequest(BaseModel):
    """Request model for starting an interview."""
    role: str
    experience_level: str

    class Config:
        json_schema_extra = {
            "example": {
                "role": "Software Engineer",
                "experience_level": "Mid"
            }
        }


class StartInterviewResponse(BaseModel):
    """Response model for start-interview endpoint."""
    interview_id: str
    question: QuestionModel
    message: str = "Question generated successfully. Record your answer and submit."


class SubmitAnswerResponse(BaseModel):
    """Response model for submit-answer endpoint."""
    interview_id: str
    scores: ScoresModel
    recommendations: RecommendationsModel
    transcript: str
    message: str = "Analysis complete"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    timestamp: str


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns system status and version information.
    """
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/start-interview", response_model=StartInterviewResponse)
async def start_interview(request: StartInterviewRequest):
    """
    Generate an interview question.

    This endpoint:
    1. Creates a new interview session
    2. Generates a question appropriate for the role and experience level
    3. Returns interview_id and question for the candidate to answer

    Args:
        request: Contains role and experience_level

    Returns:
        interview_id and generated question
    """
    logger.info(
        f"Starting interview for role={request.role}, level={request.experience_level}")

    try:
        # Generate unique interview ID
        interview_id = f"int_{uuid.uuid4().hex[:12]}"

        # Run question generation agent
        from backend.agents.question_generation import QuestionGenerationAgent
        agent = QuestionGenerationAgent()
        result = agent.generate(
            role=request.role,
            experience_level=request.experience_level,
        )

        # Map QuestionDetails → QuestionModel
        q = result.question
        question_model = QuestionModel(
            text=q.text,
            topic=q.topic,
            difficulty=q.difficulty,
            intent=q.intent,
        )

        # Return response
        return StartInterviewResponse(
            interview_id=interview_id,
            question=question_model
        )

    except Exception as e:
        logger.error(f"Error in start_interview: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate question: {str(e)}")


@app.post("/submit-answer", response_model=SubmitAnswerResponse)
async def submit_answer(
    interview_id: str = Form(...),
    role: str = Form(...),
    experience_level: str = Form(...),
    audio_file: UploadFile = File(...),
    video_file: Optional[UploadFile] = File(None)
):
    """
    Submit interview answer for complete analysis.

    This endpoint:
    1. Receives audio (required) and video (optional) files
    2. Runs full analysis pipeline through all agents
    3. Returns comprehensive results with scores and recommendations

    Workflow:
    - Question Generation
    - Voice Analysis (transcription + metrics)
    - Answer Quality Analysis
    - Body Language Analysis (if video provided)
    - Confidence Behavior Inference
    - Deterministic Scoring
    - Recommendation Generation

    Args:
        interview_id: Interview session ID
        role: Job role
        experience_level: Experience level
        audio_file: Recorded audio answer
        video_file: (Optional) Recorded video answer

    Returns:
        Complete analysis with scores, transcript, and recommendations
    """
    logger.info(f"Processing submission for interview_id={interview_id}")

    try:
        # Validate files
        validate_file(audio_file, "audio")
        if video_file:
            validate_file(video_file, "video")

        # Save uploaded files
        audio_path = await save_upload_file(audio_file, interview_id, "audio")
        video_path = None
        if video_file:
            video_path = await save_upload_file(video_file, interview_id, "video")

        logger.info(f"Files saved: audio={audio_path}, video={video_path}")

        # Run full analysis pipeline
        final_state = run_interview_analysis(
            interview_id=interview_id,
            role=role,
            experience_level=experience_level,
            audio_path=str(audio_path),
            video_path=str(video_path) if video_path else None
        )

        # Return results
        return SubmitAnswerResponse(
            interview_id=final_state.interview_id,
            scores=final_state.scores,
            recommendations=final_state.recommendations,
            transcript=final_state.transcript
        )

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error in submit_answer: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "start_interview": "/start-interview (POST)",
            "submit_answer": "/submit-answer (POST)",
            "docs": "/docs"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
