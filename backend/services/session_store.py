"""
Session Store Service
=====================
In-memory session storage for multi-question interview sessions.

Stores session data (questions, audio/video paths, config) across
multiple HTTP requests within a single interview session.

Note: This is ephemeral — sessions are lost on server restart.
Swap to Redis or a database for production persistence.
"""

import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from backend.models.state import QuestionModel

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Data for a single interview session spanning multiple questions."""
    interview_id: str
    role: str
    experience_level: str
    num_questions: int
    time_per_question: int  # seconds
    questions: List[QuestionModel] = field(default_factory=list)
    audio_paths: List[str] = field(default_factory=list)
    video_paths: List[Optional[str]] = field(default_factory=list)
    current_question_index: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat())


class SessionStore:
    """
    Thread-safe in-memory session store.

    Usage:
        store = get_session_store()
        store.create(session_data)
        session = store.get("int_abc123")
        store.delete("int_abc123")
    """

    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._lock = threading.Lock()
        logger.info("SessionStore initialized (in-memory)")

    def create(self, session: SessionData) -> None:
        """Store a new session."""
        with self._lock:
            self._sessions[session.interview_id] = session
        logger.info(f"Session created: {session.interview_id}")

    def get(self, interview_id: str) -> Optional[SessionData]:
        """Retrieve a session by ID."""
        with self._lock:
            return self._sessions.get(interview_id)

    def update(self, interview_id: str, session: SessionData) -> None:
        """Update an existing session."""
        with self._lock:
            self._sessions[interview_id] = session
        logger.debug(f"Session updated: {interview_id}")

    def delete(self, interview_id: str) -> None:
        """Remove a session."""
        with self._lock:
            self._sessions.pop(interview_id, None)
        logger.info(f"Session deleted: {interview_id}")

    def exists(self, interview_id: str) -> bool:
        """Check if a session exists."""
        with self._lock:
            return interview_id in self._sessions

    def count(self) -> int:
        """Return the number of active sessions."""
        with self._lock:
            return len(self._sessions)


# Singleton instance
_store_instance: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Get the global singleton session store."""
    global _store_instance
    if _store_instance is None:
        _store_instance = SessionStore()
    return _store_instance
