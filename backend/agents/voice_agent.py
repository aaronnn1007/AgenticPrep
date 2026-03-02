"""
Voice Agent
===========
Signal extraction agent for objective voice metrics from audio input.

This is NOT a semantic reasoning agent.
It extracts deterministic, reproducible voice metrics using audio signal processing.

Architecture:
- Transcription: Whisper (local, no LLM)
- Speech Rate: Word count / duration (WPM)
- Filler Detection: Pattern matching against known filler words
- Clarity: Signal-to-noise, spectral quality, consistency
- Tone: Rule-based classification from metrics

All metrics are algorithmic and reproducible.
No LLM is used for metric extraction.
"""

import logging
import os
import re
from typing import Dict, Any, Optional
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field, validator

from backend.models.state import InterviewState, VoiceAnalysisModel
from backend.config import settings

logger = logging.getLogger(__name__)


# Common filler words to detect
FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "basically",
    "actually", "literally", "sort of", "kind of", "i mean",
    "well", "so", "okay", "right", "hmm", "you see"
}

# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}


class VoiceAnalysisInput(BaseModel):
    """Input validation model for voice analysis."""
    audio_file_path: str = Field(..., description="Path to audio file")

    @validator("audio_file_path")
    def validate_audio_path(cls, v):
        """Validate audio file exists and is accessible."""
        if not v:
            raise ValueError("audio_file_path cannot be empty")

        path = Path(v)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {v}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")

        # Check file extension
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )

        return v


class VoiceAnalysisOutput(BaseModel):
    """Output model for voice analysis."""
    voice_analysis: VoiceAnalysisModel


class VoiceAgent:
    """
    Signal extraction agent for objective voice metrics.

    Pipeline:
    1. Audio Validation & Loading
    2. Transcription (Whisper local model)
    3. Speech Rate Calculation (WPM)
    4. Filler Word Detection (pattern matching)
    5. Clarity Analysis (signal processing)
    6. Tone Classification (rule-based)

    All metrics are deterministic and reproducible.
    No LLM is used for metric extraction (only Whisper for transcription).
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model and audio processing tools.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.model_size = model_size
        self.whisper_model = self._load_whisper()
        logger.info(f"VoiceAgent initialized with model size: {model_size}")

    def _load_whisper(self) -> WhisperModel:
        """Load Whisper model with configured settings."""
        model_size = getattr(settings, 'WHISPER_MODEL_SIZE', self.model_size)
        device = getattr(settings, 'WHISPER_DEVICE', 'cpu')
        compute_type = getattr(settings, 'WHISPER_COMPUTE_TYPE', 'int8')

        logger.info(
            f"Loading Whisper model: {model_size} (device={device}, compute_type={compute_type})")

        return WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

    def _transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file using Whisper.

        Returns:
            Full transcript as string
        """
        logger.info(f"Transcribing audio: {audio_path}")

        try:
            segments, info = self.whisper_model.transcribe(
                audio_path,
                beam_size=5,
                language="en"
            )

            # Combine all segments
            transcript = " ".join([segment.text for segment in segments])

            logger.info(
                f"Transcription complete: {len(transcript)} characters")
            return transcript.strip()

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def _calculate_speech_rate(self, transcript: str, duration: float) -> float:
        """
        Calculate words per minute.

        Args:
            transcript: Full transcript text
            duration: Audio duration in seconds

        Returns:
            Words per minute (WPM)
        """
        if duration <= 0:
            return 0.0

        # Count words (split by whitespace)
        word_count = len(transcript.split())

        # Calculate WPM
        duration_minutes = duration / 60.0
        wpm = word_count / duration_minutes if duration_minutes > 0 else 0.0

        return round(wpm, 2)

    def _calculate_filler_ratio(self, transcript: str) -> float:
        """
        Calculate ratio of filler words to total words.

        Args:
            transcript: Full transcript text

        Returns:
            Filler ratio (0.0 to 1.0)
        """
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', transcript.lower())

        if not words:
            return 0.0

        # Count filler words
        filler_count = sum(1 for word in words if word in FILLER_WORDS)

        # Also check for multi-word fillers
        text_lower = transcript.lower()
        for filler in ["you know", "i mean", "sort of", "kind of"]:
            filler_count += text_lower.count(filler)

        # Calculate ratio
        total_words = len(words)
        ratio = filler_count / total_words if total_words > 0 else 0.0

        return round(ratio, 4)

    def _analyze_clarity(self, audio_path: str) -> float:
        """
        Analyze speech clarity using audio signal processing.

        Metrics:
        - Signal-to-noise ratio
        - Spectral clarity
        - Consistency of amplitude

        Returns:
            Clarity score (0.0 to 1.0)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)

            # Calculate spectral centroid (frequency clarity indicator)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            centroid_mean = np.mean(spectral_centroid)

            # Calculate zero crossing rate (voice quality indicator)
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)

            # Calculate RMS energy (amplitude consistency)
            rms = librosa.feature.rms(y=y)
            rms_std = np.std(rms)

            # Normalize and combine metrics
            # Higher spectral centroid = clearer speech (typically 200-3000 Hz)
            centroid_score = min(centroid_mean / 3000.0, 1.0)

            # Low ZCR variation = more consistent voice quality
            zcr_score = 1.0 - min(zcr_mean * 10, 1.0)

            # Low RMS std = consistent amplitude
            rms_score = 1.0 - min(rms_std * 5, 1.0)

            # Weighted combination
            clarity = (centroid_score * 0.5) + \
                (zcr_score * 0.3) + (rms_score * 0.2)

            return round(float(clarity), 3)

        except Exception as e:
            logger.error(f"Clarity analysis failed: {e}")
            return 0.5  # Default moderate clarity

    def _analyze_tone(
        self,
        speech_rate_wpm: float,
        filler_ratio: float
    ) -> str:
        """
        Classify tone using rule-based logic.

        Classification rules:
        - hesitant: filler_ratio > 0.15
        - uncertain: speech_rate_wpm < 110
        - confident: speech_rate_wpm between 120-180 AND filler_ratio < 0.05
        - neutral: everything else

        Args:
            speech_rate_wpm: Words per minute
            filler_ratio: Ratio of filler words (0.0 to 1.0)

        Returns:
            Tone label: "confident", "neutral", "hesitant", or "uncertain"
        """
        # Rule-based classification (deterministic, no LLM)
        if filler_ratio > 0.15:
            return "hesitant"
        elif speech_rate_wpm < 110:
            return "uncertain"
        elif 120 <= speech_rate_wpm <= 180 and filler_ratio < 0.05:
            return "confident"
        else:
            return "neutral"

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return float(duration)
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0

    def _validate_audio(self, audio_path: str) -> None:
        """
        Validate audio file.

        Checks:
        - File exists
        - File is readable
        - Format is supported
        - Audio is not empty or silent

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid or unsupported format
        """
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {audio_path}")

        # Check extension
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {path.suffix}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )

        # Check if file is readable
        try:
            y, sr = librosa.load(str(path), sr=None, duration=1.0)
        except Exception as e:
            raise ValueError(f"Cannot read audio file: {e}")

        logger.info(f"Audio file validated: {audio_path}")

    def analyze(self, audio_file_path: str) -> VoiceAnalysisModel:
        """
        Main analysis method - extracts objective voice metrics from audio.

        This is the primary public API for the VoiceAgent.

        Pipeline:
        1. Validate audio file
        2. Load audio and get duration
        3. Transcribe using Whisper
        4. Calculate speech rate (WPM)
        5. Detect filler words
        6. Analyze clarity
        7. Classify tone

        Args:
            audio_file_path: Path to audio file (.wav, .mp3, etc.)

        Returns:
            VoiceAnalysisModel with all metrics

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is invalid or unsupported format
        """
        logger.info(f"Starting voice analysis: {audio_file_path}")

        # Step 1: Validate
        self._validate_audio(audio_file_path)

        # Step 2: Get duration
        duration = self._get_audio_duration(audio_file_path)

        if duration <= 0:
            logger.warning(
                "Audio duration is 0 or negative - returning default metrics")
            return VoiceAnalysisModel(
                speech_rate_wpm=0.0,
                filler_ratio=0.0,
                clarity=0.0,
                tone="neutral"
            )

        # Step 3: Transcribe
        transcript = self._transcribe_audio(audio_file_path)

        if not transcript or len(transcript.strip()) == 0:
            logger.warning("Empty transcript - audio may be silent")
            return VoiceAnalysisModel(
                speech_rate_wpm=0.0,
                filler_ratio=0.0,
                clarity=0.0,
                tone="uncertain"
            )

        # Step 4: Calculate speech rate
        speech_rate_wpm = self._calculate_speech_rate(transcript, duration)

        # Step 5: Calculate filler ratio
        filler_ratio = self._calculate_filler_ratio(transcript)

        # Step 6: Analyze clarity
        clarity = self._analyze_clarity(audio_file_path)

        # Step 7: Classify tone (rule-based)
        tone = self._analyze_tone(speech_rate_wpm, filler_ratio)

        # Create and validate output
        voice_analysis = VoiceAnalysisModel(
            speech_rate_wpm=speech_rate_wpm,
            filler_ratio=filler_ratio,
            clarity=clarity,
            tone=tone
        )

        logger.info(
            f"Voice analysis complete: "
            f"WPM={speech_rate_wpm:.2f}, "
            f"filler={filler_ratio:.4f}, "
            f"clarity={clarity:.3f}, "
            f"tone={tone}"
        )

        return voice_analysis

    def execute(self, state: InterviewState) -> InterviewState:
        """
        Execute voice analysis pipeline for LangGraph integration.

        Supports three modes:
        1. Pre-computed: voice_analysis already in state (WebRTC w/ Redis metrics)
        2. File-based: audio_path provided, transcribe from file
        3. WebRTC/Live: transcript already provided, compute metrics from it

        Steps:
        1. Check if voice_analysis already computed (early return)
        2. Check if audio_path exists (file-based flow)
        3. If yes: Run full analyze() pipeline
        4. If no: Use existing transcript and compute metrics
        5. Update state with results

        Args:
            state: Current interview state (must have audio_path OR transcript)

        Returns:
            Updated state with transcript and voice_analysis populated

        Raises:
            ValueError: If both audio_path and transcript are missing
            FileNotFoundError: If audio file doesn't exist (file-based mode)
        """
        # Mode 0: Pre-computed metrics (WebRTC with Redis)
        # If voice_analysis already exists with real metrics, just return state
        if state.voice_analysis:
            # Check if it has meaningful metrics (not all zeros/defaults)
            va = state.voice_analysis
            if va.speech_rate_wpm > 0 or va.clarity > 0 or va.filler_ratio > 0:
                logger.info(
                    f"Voice analysis already computed - using existing metrics: "
                    f"WPM={va.speech_rate_wpm:.2f}, clarity={va.clarity:.3f}"
                )
                return state

        updated_state = state.model_copy(deep=True)

        # Mode 1: File-based flow (audio_path provided)
        if state.audio_path:
            audio_path = state.audio_path
            logger.info(
                f"File-based flow: Executing voice analysis from audio file: {audio_path}")

            # Run main analysis
            voice_analysis = self.analyze(audio_path)

            # Get transcript separately for state
            transcript = self._transcribe_audio(audio_path)

            # Update state
            updated_state.transcript = transcript
            updated_state.voice_analysis = voice_analysis

            logger.info(
                "Voice analysis execution complete (file-based) - state updated")
            return updated_state

        # Mode 2: WebRTC/Live flow (transcript already provided)
        elif state.transcript:
            transcript = state.transcript
            logger.info(
                f"WebRTC/Live flow: Executing voice analysis from existing transcript ({len(transcript)} chars)")

            # For WebRTC flow, we need duration for speech rate
            # Try multiple sources to get actual duration
            duration = 0.0

            # Option 1: Check if duration is in metadata (if available)
            if hasattr(state, 'metadata') and isinstance(state.metadata, dict):
                duration = state.metadata.get('duration', 0.0)
                if duration > 0:
                    logger.info(
                        f"Using duration from metadata: {duration:.2f}s")

            # Option 2: Check if video_path exists and we can get duration from it
            if duration <= 0 and state.video_path and Path(state.video_path).exists():
                try:
                    import cv2
                    cap = cv2.VideoCapture(state.video_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        if fps > 0:
                            duration = frame_count / fps
                            logger.info(
                                f"Extracted duration from video file: {duration:.2f}s")
                        cap.release()
                except Exception as e:
                    logger.warning(
                        f"Could not extract duration from video: {e}")

            # Option 3: Estimate duration from transcript (fallback)
            if duration <= 0:
                word_count = len(transcript.split())
                estimated_wpm = 150.0  # Average speaking rate
                duration = (word_count / estimated_wpm) * \
                    60.0 if word_count > 0 else 1.0
                logger.info(
                    f"Estimated duration: {duration:.2f}s from {word_count} words (fallback)")

            # Try to preserve existing metrics if already computed
            existing_clarity = 0.5  # Default moderate clarity for WebRTC

            if state.voice_analysis:
                # Keep existing clarity if already computed
                if state.voice_analysis.clarity > 0:
                    existing_clarity = state.voice_analysis.clarity
                    logger.info(
                        f"Using existing clarity from state: {existing_clarity:.3f}")

            # Calculate metrics from transcript
            speech_rate_wpm = self._calculate_speech_rate(transcript, duration)
            filler_ratio = self._calculate_filler_ratio(transcript)

            # For WebRTC, use existing clarity or default (can't analyze audio file)
            clarity = existing_clarity

            # Classify tone (rule-based)
            tone = self._analyze_tone(speech_rate_wpm, filler_ratio)

            # Create voice analysis model
            voice_analysis = VoiceAnalysisModel(
                speech_rate_wpm=speech_rate_wpm,
                filler_ratio=filler_ratio,
                clarity=clarity,
                tone=tone
            )

            # Update state
            updated_state.voice_analysis = voice_analysis

            logger.info(
                f"Voice analysis complete (WebRTC): "
                f"WPM={speech_rate_wpm:.2f}, "
                f"filler={filler_ratio:.4f}, "
                f"clarity={clarity:.3f}, "
                f"tone={tone}"
            )

            return updated_state

        # Mode 3: Empty transcript - use fallback metrics
        else:
            logger.warning(
                "Empty or missing transcript - using fallback voice metrics")
            updated_state.voice_analysis = VoiceAnalysisModel(
                speech_rate_wpm=0.0,
                filler_ratio=0.0,
                clarity=0.5,
                tone="neutral"
            )
            return updated_state


# LangGraph node wrapper
def voice_agent_node(state: InterviewState) -> InterviewState:
    """LangGraph node wrapper for VoiceAgent."""
    agent = VoiceAgent()
    updated_state = agent.execute(state)
    return {"voice_analysis": updated_state.voice_analysis}
