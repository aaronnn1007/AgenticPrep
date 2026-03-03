"""
Logger Utility
==============
Production-grade logging utility for LangGraph node execution tracking.

Features:
- Per-node execution timing
- Structured logging with context
- Performance monitoring
- Error tracking with stack traces
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime
import json
from pathlib import Path


# Configure structured logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup a configured logger with file and console handlers.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_node_execution(node_name: str):
    """
    Decorator for logging LangGraph node execution with timing.

    Logs:
    - Node start
    - Node end
    - Execution time
    - Key outputs summary
    - Errors with stack traces

    Args:
        node_name: Human-readable name of the node

    Example:
        @log_node_execution("Voice Analysis")
        def voice_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # ... node logic
            return state
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            logger = logging.getLogger(func.__module__)

            # Extract interview_id if available
            interview_id = state.get('interview_id', 'unknown')

            # Log node start
            logger.info(
                f"[{node_name}] Starting execution for interview_id={interview_id}")
            start_time = time.time()

            try:
                # Execute node
                result = func(state)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Log success
                logger.info(
                    f"[{node_name}] Completed successfully in {execution_time:.2f}s "
                    f"for interview_id={interview_id}"
                )

                # Log key outputs summary (if available)
                _log_node_outputs(logger, node_name, result)

                return result

            except Exception as e:
                # Calculate execution time
                execution_time = time.time() - start_time

                # Log error with details
                logger.error(
                    f"[{node_name}] Failed after {execution_time:.2f}s "
                    f"for interview_id={interview_id}: {str(e)}",
                    exc_info=True
                )

                # Optionally, set error flag in state
                if isinstance(state, dict):
                    state['_errors'] = state.get('_errors', [])
                    state['_errors'].append({
                        'node': node_name,
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    })

                # Re-raise or return state with error (depending on requirements)
                raise

        return wrapper
    return decorator


def _log_node_outputs(logger: logging.Logger, node_name: str, state: Dict[str, Any]) -> None:
    """
    Log summary of key outputs from a node.

    Args:
        logger: Logger instance
        node_name: Name of the node
        state: Updated state after node execution
    """
    try:
        summary_parts = []

        # Question Generation
        if 'question' in state and state['question']:
            q = state['question']
            if isinstance(q, dict):
                summary_parts.append(f"question_topic={q.get('topic', 'N/A')}")
                summary_parts.append(
                    f"difficulty={q.get('difficulty', 0.0):.2f}")

        # Voice Analysis
        if 'voice_analysis' in state and state['voice_analysis']:
            va = state['voice_analysis']
            if isinstance(va, dict):
                summary_parts.append(
                    f"speech_rate={va.get('speech_rate_wpm', 0.0):.1f}wpm")
                summary_parts.append(f"clarity={va.get('clarity', 0.0):.2f}")

        # Answer Quality
        if 'answer_quality' in state and state['answer_quality']:
            aq = state['answer_quality']
            if isinstance(aq, dict):
                summary_parts.append(
                    f"correctness={aq.get('correctness', 0.0):.2f}")
                summary_parts.append(
                    f"relevance={aq.get('relevance', 0.0):.2f}")

        # Body Language
        if 'body_language' in state and state['body_language']:
            bl = state['body_language']
            if isinstance(bl, dict):
                summary_parts.append(
                    f"eye_contact={bl.get('eye_contact', 0.0):.2f}")
                summary_parts.append(
                    f"posture_stability={bl.get('posture_stability', 0.0):.2f}")

        # Confidence Behavior
        if 'confidence_behavior' in state and state['confidence_behavior']:
            cb = state['confidence_behavior']
            if isinstance(cb, dict):
                summary_parts.append(
                    f"confidence={cb.get('confidence', 0.0):.2f}")
                summary_parts.append(
                    f"professionalism={cb.get('professionalism', 0.0):.2f}")

        # Scores
        if 'scores' in state and state['scores']:
            s = state['scores']
            if isinstance(s, dict):
                summary_parts.append(
                    f"overall_score={s.get('overall', 0.0):.1f}")
                summary_parts.append(
                    f"technical={s.get('technical', 0.0):.1f}")
                summary_parts.append(
                    f"communication={s.get('communication', 0.0):.1f}")

        # Recommendations
        if 'recommendations' in state and state['recommendations']:
            r = state['recommendations']
            if isinstance(r, dict):
                num_strengths = len(r.get('strengths', []))
                num_weaknesses = len(r.get('weaknesses', []))
                num_improvements = len(r.get('improvement_plan', []))
                summary_parts.append(
                    f"recommendations=({num_strengths} strengths, "
                    f"{num_weaknesses} weaknesses, {num_improvements} improvements)"
                )

        if summary_parts:
            logger.info(f"[{node_name}] Outputs: {', '.join(summary_parts)}")

    except Exception as e:
        # Don't fail on logging errors
        logger.debug(f"Could not log output summary: {e}")


class NodeLogger:
    """
    Context manager for logging node execution with automatic timing.

    Example:
        with NodeLogger("Voice Analysis", state):
            # ... perform analysis
            pass
        # Automatically logs timing and context
    """

    def __init__(self, node_name: str, state: Dict[str, Any]):
        """
        Initialize node logger context.

        Args:
            node_name: Name of the node being executed
            state: Current state (to extract interview_id)
        """
        self.node_name = node_name
        self.interview_id = state.get('interview_id', 'unknown')
        self.logger = logging.getLogger(__name__)
        self.start_time = None

    def __enter__(self):
        """Start node execution logging."""
        self.start_time = time.time()
        self.logger.info(
            f"[{self.node_name}] Starting execution for interview_id={self.interview_id}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End node execution logging."""
        execution_time = time.time() - self.start_time

        if exc_type is None:
            self.logger.info(
                f"[{self.node_name}] Completed successfully in {execution_time:.2f}s "
                f"for interview_id={self.interview_id}"
            )
        else:
            self.logger.error(
                f"[{self.node_name}] Failed after {execution_time:.2f}s "
                f"for interview_id={self.interview_id}: {exc_val}",
                exc_info=True
            )

        return False  # Don't suppress exceptions
