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
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.config import settings
from backend.models.state import (
    InterviewState, QuestionModel, ScoresModel, RecommendationsModel,
    QuestionResult,
)
from backend.graph.workflow import get_graph, run_interview_analysis, run_multi_question_analysis
from backend.services.file_handler import save_upload_file, validate_file
from backend.services.session_store import get_session_store, SessionData

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS
# ============================================================================

# Maps every incoming experience-level string to the canonical backend value.
_EXPERIENCE_LEVEL_MAP: dict[str, str] = {
    "fresher": "Junior",
    "junior": "Junior",
    "mid": "Mid",
    "middle": "Mid",
    "intermediate": "Mid",
    "senior": "Senior",
    "lead": "Senior",
    "staff": "Senior",
}

# Maps canonical experience level to a sensible default difficulty_target.
_DIFFICULTY_BY_LEVEL: dict[str, float] = {
    "Junior": 0.30,
    "Mid": 0.55,
    "Senior": 0.80,
}


def normalize_experience_level(level: str) -> str:
    """Normalize any incoming experience-level string to 'Junior', 'Mid', or 'Senior'."""
    canonical = _EXPERIENCE_LEVEL_MAP.get(level.strip().lower())
    if canonical:
        return canonical
    # Best-effort fallback: title-case the raw value so it at least looks consistent.
    return level.strip().title()


def derive_difficulty(experience_level: str) -> float:
    """Return the default difficulty_target for a normalized experience level."""
    return _DIFFICULTY_BY_LEVEL.get(experience_level, 0.5)


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
    num_questions: int = 5
    time_per_question: int = 120  # seconds

    class Config:
        json_schema_extra = {
            "example": {
                "role": "Software Engineer",
                "experience_level": "Mid",
                "num_questions": 5,
                "time_per_question": 120
            }
        }


class StartInterviewResponse(BaseModel):
    """Response model for start-interview endpoint."""
    interview_id: str
    question: QuestionModel
    num_questions: int
    time_per_question: int
    current_question_index: int = 0
    message: str = "Question generated successfully. Record your answer and submit."


class NextQuestionResponse(BaseModel):
    """Response model for next-question endpoint."""
    interview_id: str
    question: QuestionModel
    current_question_index: int
    is_last: bool
    message: str = "Next question generated."


class SubmitQuestionAnswerResponse(BaseModel):
    """Response model for submit-question-answer endpoint."""
    interview_id: str
    question_index: int
    status: str = "saved"
    message: str = "Answer saved. Move to the next question or finish."


class QuestionResultResponse(BaseModel):
    """Per-question result in the complete-interview response."""
    question: Optional[QuestionModel] = None
    transcript: str = ""
    scores: ScoresModel = ScoresModel()


class CompleteInterviewResponse(BaseModel):
    """Response model for complete-interview endpoint."""
    interview_id: str
    question_results: list[QuestionResultResponse]
    aggregate_scores: ScoresModel
    recommendations: RecommendationsModel
    message: str = "Multi-question analysis complete"


class SubmitAnswerResponse(BaseModel):
    """Response model for submit-answer endpoint (single-question legacy)."""
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
    Generate the first interview question and create a session.

    This endpoint:
    1. Creates a new interview session
    2. Validates num_questions (3-10) and time_per_question (30-600)
    3. Generates the first question
    4. Stores session data for subsequent /next-question calls
    5. Returns interview_id, question, and session config

    Args:
        request: Contains role, experience_level, num_questions, time_per_question

    Returns:
        interview_id, first question, and session configuration
    """
    # Clamp values
    num_questions = max(3, min(10, request.num_questions))
    time_per_question = max(30, min(600, request.time_per_question))

    # Normalise experience level before anything else.
    experience_level = normalize_experience_level(request.experience_level)
    difficulty_target = derive_difficulty(experience_level)

    logger.info(
        f"Starting interview for role={request.role}, level={experience_level} "
        f"(raw: {request.experience_level!r}), difficulty={difficulty_target}, "
        f"num_questions={num_questions}, time_per_question={time_per_question}s"
    )

    try:
        # Generate unique interview ID
        interview_id = f"int_{uuid.uuid4().hex[:12]}"

        # Run question generation agent
        from backend.agents.question_generation import QuestionGenerationAgent
        agent = QuestionGenerationAgent()
        result = agent.generate(
            role=request.role,
            experience_level=experience_level,
            difficulty_target=difficulty_target,
        )

        # Map QuestionDetails → QuestionModel
        q = result.question
        question_model = QuestionModel(
            text=q.text,
            topic=q.topic,
            difficulty=q.difficulty,
            intent=q.intent,
        )

        # Create session and store it
        session = SessionData(
            interview_id=interview_id,
            role=request.role,
            experience_level=experience_level,
            num_questions=num_questions,
            time_per_question=time_per_question,
            questions=[question_model],
            current_question_index=0,
        )
        get_session_store().create(session)

        # Return response
        return StartInterviewResponse(
            interview_id=interview_id,
            question=question_model,
            num_questions=num_questions,
            time_per_question=time_per_question,
            current_question_index=0,
        )

    except Exception as e:
        logger.error(f"Error in start_interview: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate question: {str(e)}")


# ── Next Question ──────────────────────────────────────────────────────

class NextQuestionRequest(BaseModel):
    interview_id: str


@app.post("/next-question", response_model=NextQuestionResponse)
async def next_question(request: NextQuestionRequest):
    """
    Generate the next question for an ongoing multi-question session.

    Looks up the session, generates a new question that avoids repeating
    previous topics, and advances the question index.
    """
    store = get_session_store()
    session = store.get(request.interview_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    next_idx = session.current_question_index + 1
    if next_idx >= session.num_questions:
        raise HTTPException(
            status_code=400, detail="All questions already generated")

    try:
        from backend.agents.question_generation import QuestionGenerationAgent
        agent = QuestionGenerationAgent()

        # Normalise level (session may have been created before the fix was deployed)
        exp_level = normalize_experience_level(session.experience_level)
        diff_target = derive_difficulty(exp_level)

        # Gather previous question texts for diversity
        previous_texts = [q.text for q in session.questions]

        result = agent.generate(
            role=session.role,
            experience_level=exp_level,
            difficulty_target=diff_target,
            previous_questions=previous_texts,
        )

        q = result.question
        question_model = QuestionModel(
            text=q.text, topic=q.topic,
            difficulty=q.difficulty, intent=q.intent,
        )

        # Update session
        session.questions.append(question_model)
        session.current_question_index = next_idx
        store.update(request.interview_id, session)

        is_last = (next_idx == session.num_questions - 1)

        return NextQuestionResponse(
            interview_id=request.interview_id,
            question=question_model,
            current_question_index=next_idx,
            is_last=is_last,
        )

    except Exception as e:
        logger.error(f"Error in next_question: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate next question: {str(e)}")


# ── Submit Question Answer ─────────────────────────────────────────────

@app.post("/submit-question-answer", response_model=SubmitQuestionAnswerResponse)
async def submit_question_answer(
    interview_id: str = Form(...),
    question_index: int = Form(...),
    audio_file: UploadFile = File(...),
    video_file: Optional[UploadFile] = File(None),
):
    """
    Save the audio/video answer for a specific question.

    No analysis runs here — files are stored and the session is updated.
    Analysis happens when /complete-interview is called.
    """
    store = get_session_store()
    session = store.get(interview_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if question_index < 0 or question_index >= session.num_questions:
        raise HTTPException(status_code=400, detail="Invalid question_index")

    try:
        validate_file(audio_file, "audio")
        if video_file:
            validate_file(video_file, "video")

        # Save with question-index-aware filename
        sub_id = f"{interview_id}/q{question_index}"
        audio_path = await save_upload_file(audio_file, sub_id, "audio")
        video_path = None
        if video_file:
            video_path = await save_upload_file(video_file, sub_id, "video")

        # Ensure lists are long enough
        while len(session.audio_paths) <= question_index:
            session.audio_paths.append("")
        while len(session.video_paths) <= question_index:
            session.video_paths.append(None)

        session.audio_paths[question_index] = str(audio_path)
        session.video_paths[question_index] = str(
            video_path) if video_path else None
        store.update(interview_id, session)

        logger.info(f"Answer saved for {interview_id} Q{question_index}")

        return SubmitQuestionAnswerResponse(
            interview_id=interview_id,
            question_index=question_index,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in submit_question_answer: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to save answer: {str(e)}")


# ── Complete Interview ─────────────────────────────────────────────────

class CompleteInterviewRequest(BaseModel):
    interview_id: str


@app.post("/complete-interview", response_model=CompleteInterviewResponse)
async def complete_interview(request: CompleteInterviewRequest):
    """
    Trigger analysis for all submitted answers and return aggregate results.

    Runs the analysis pipeline for each question, aggregates scores,
    and generates holistic recommendations.
    """
    store = get_session_store()
    session = store.get(request.interview_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Ensure at least one answer was submitted
    if not session.audio_paths or all(not p for p in session.audio_paths):
        raise HTTPException(status_code=400, detail="No answers submitted yet")

    try:
        final_state = run_multi_question_analysis(
            interview_id=session.interview_id,
            role=session.role,
            experience_level=session.experience_level,
            questions=session.questions,
            audio_paths=session.audio_paths,
            video_paths=session.video_paths,
        )

        # Build per-question response
        qr_responses = [
            QuestionResultResponse(
                question=qr.question,
                transcript=qr.transcript,
                scores=qr.scores,
            )
            for qr in final_state.question_results
        ]

        # Optionally clean up session
        # store.delete(request.interview_id)

        return CompleteInterviewResponse(
            interview_id=final_state.interview_id,
            question_results=qr_responses,
            aggregate_scores=final_state.scores,
            recommendations=final_state.recommendations,
        )

    except Exception as e:
        logger.error(f"Error in complete_interview: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


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
            "next_question": "/next-question (POST)",
            "submit_question_answer": "/submit-question-answer (POST)",
            "complete_interview": "/complete-interview (POST)",
            "submit_answer": "/submit-answer (POST) [single-question legacy]",
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
