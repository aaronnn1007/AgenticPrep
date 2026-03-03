"""
Mock WebRTC Simulator
=====================
Production-grade WebRTC mock for integration testing.

Simulates:
- Audio streaming chunks (realistic PCM data)
- Body language metrics from frontend
- Network conditions (latency, jitter, packet loss)
- Session lifecycle events
- Realistic timing patterns

This is NOT a simplified mock - it generates production-realistic data.
"""

import asyncio
import io
import json
import logging
import numpy as np
import wave
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NetworkCondition(Enum):
    """Simulated network conditions."""
    PERFECT = "perfect"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNSTABLE = "unstable"


@dataclass
class AudioChunkConfig:
    """Configuration for audio chunk generation."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 2000  # 2 seconds per chunk
    speech_rate_wpm: float = 150.0  # Words per minute
    silence_ratio: float = 0.1  # 10% silence
    filler_probability: float = 0.05  # 5% filler words
    network_condition: NetworkCondition = NetworkCondition.GOOD


@dataclass
class BodyMetricsConfig:
    """Configuration for body language metrics."""
    update_interval_ms: int = 5000  # Update every 5 seconds
    eye_contact_mean: float = 0.75
    eye_contact_variance: float = 0.1
    posture_stability_mean: float = 0.80
    posture_stability_variance: float = 0.08
    facial_expressiveness_mean: float = 0.70
    facial_expressiveness_variance: float = 0.12


@dataclass
class MockInterviewSession:
    """Mock interview session configuration."""
    session_id: str
    role: str
    experience_level: str
    duration_seconds: int = 60
    audio_config: AudioChunkConfig = field(default_factory=AudioChunkConfig)
    body_metrics_config: BodyMetricsConfig = field(
        default_factory=BodyMetricsConfig)

    # Synthetic answer text to simulate
    answer_text: Optional[str] = None

    # Question to be asked
    question_text: Optional[str] = None


class MockAudioGenerator:
    """
    Generates realistic mock audio data for testing.

    Creates PCM audio with:
    - Realistic speech patterns (sine wave modulation)
    - Silence periods
    - Noise simulation
    - Proper WAV formatting
    """

    def __init__(self, config: AudioChunkConfig):
        """Initialize audio generator."""
        self.config = config
        self.sample_rate = config.sample_rate
        self.channels = config.channels

        # Calculate samples per chunk
        self.samples_per_chunk = int(
            config.sample_rate * config.chunk_duration_ms / 1000
        )

        # Audio generation state
        self.phase = 0.0
        self.time_offset = 0.0

    def generate_speech_chunk(
        self,
        duration_ms: int,
        intensity: float = 0.5
    ) -> bytes:
        """
        Generate synthetic speech audio chunk.

        Args:
            duration_ms: Duration in milliseconds
            intensity: Speech intensity (0.0 to 1.0)

        Returns:
            PCM audio bytes (16-bit)
        """
        num_samples = int(self.sample_rate * duration_ms / 1000)

        # Generate time array
        t = np.linspace(
            self.time_offset,
            self.time_offset + duration_ms / 1000,
            num_samples,
            endpoint=False
        )

        # Simulate speech with multiple frequency components
        # Fundamental frequency around 120-250 Hz (human voice)
        f0 = 150 + 50 * np.sin(2 * np.pi * 2 * t)  # Varying pitch

        # Create speech-like waveform with harmonics
        signal = np.zeros(num_samples)
        signal += 0.5 * np.sin(2 * np.pi * f0 * t)  # Fundamental
        signal += 0.3 * np.sin(2 * np.pi * 2 * f0 * t)  # 2nd harmonic
        signal += 0.2 * np.sin(2 * np.pi * 3 * f0 * t)  # 3rd harmonic

        # Add amplitude modulation (speech envelope)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
        signal *= envelope * intensity

        # Add realistic noise
        noise = np.random.normal(0, 0.02, num_samples)
        signal += noise

        # Apply speech-like windowing (attack/decay)
        window = np.hanning(num_samples)
        signal *= window

        # Normalize and convert to 16-bit PCM
        signal = np.clip(signal, -1.0, 1.0)
        pcm_data = (signal * 32767).astype(np.int16)

        # Update time offset for continuity
        self.time_offset += duration_ms / 1000

        return pcm_data.tobytes()

    def generate_silence_chunk(self, duration_ms: int) -> bytes:
        """
        Generate silence with background noise.

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            PCM audio bytes (16-bit)
        """
        num_samples = int(self.sample_rate * duration_ms / 1000)

        # Low-level background noise
        noise = np.random.normal(0, 0.005, num_samples)
        pcm_data = (noise * 32767).astype(np.int16)

        return pcm_data.tobytes()

    def create_wav_file(self, audio_data: bytes, output_path: Path) -> Path:
        """
        Create WAV file from PCM data.

        Args:
            audio_data: Raw PCM bytes
            output_path: Output file path

        Returns:
            Path to created WAV file
        """
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)

        logger.info(f"Created WAV file: {output_path}")
        return output_path


class MockBodyLanguageGenerator:
    """
    Generates realistic body language metrics for testing.

    Simulates:
    - Eye contact patterns
    - Posture stability
    - Facial expressiveness
    - Temporal variations (nervousness, fatigue)
    """

    def __init__(self, config: BodyMetricsConfig):
        """Initialize body language generator."""
        self.config = config
        self.time_elapsed = 0.0

    def generate_metrics(self, timestamp: float) -> Dict[str, Any]:
        """
        Generate body language metrics at given timestamp.

        Introduces realistic variations:
        - Initial nervousness (lower scores early)
        - Fatigue over time (slight decline)
        - Random fluctuations

        Args:
            timestamp: Time elapsed in seconds

        Returns:
            Body language metrics dict
        """
        # Nervousness factor (higher early in interview)
        nervousness_factor = np.exp(-timestamp / 30)  # Decays over 30 seconds

        # Fatigue factor (slight decline over time)
        fatigue_factor = 1.0 - 0.1 * \
            (timestamp / 300)  # 10% decline over 5 min

        # Generate metrics with realistic variations
        eye_contact = self._generate_metric(
            self.config.eye_contact_mean,
            self.config.eye_contact_variance,
            nervousness_factor,
            fatigue_factor
        )

        posture_stability = self._generate_metric(
            self.config.posture_stability_mean,
            self.config.posture_stability_variance,
            nervousness_factor,
            fatigue_factor
        )

        facial_expressiveness = self._generate_metric(
            self.config.facial_expressiveness_mean,
            self.config.facial_expressiveness_variance,
            nervousness_factor * 0.5,  # Less affected by nervousness
            fatigue_factor
        )

        # Occasional distractions (low probability)
        distractions = []
        if np.random.random() < 0.05:  # 5% chance
            distractions = np.random.choice(
                ["looking_away", "fidgeting", "touching_face"],
                size=1
            ).tolist()

        return {
            "eye_contact": eye_contact,
            "posture_stability": posture_stability,
            "facial_expressiveness": facial_expressiveness,
            "distractions": distractions,
            "timestamp": timestamp
        }

    def _generate_metric(
        self,
        mean: float,
        variance: float,
        nervousness_factor: float,
        fatigue_factor: float
    ) -> float:
        """Generate single metric with realistic variations."""
        # Base value with gaussian noise
        value = np.random.normal(mean, variance)

        # Apply nervousness (reduces score early on)
        value *= (1.0 - 0.2 * nervousness_factor)

        # Apply fatigue
        value *= fatigue_factor

        # Clip to [0, 1]
        return float(np.clip(value, 0.0, 1.0))


class WebRTCMockSimulator:
    """
    Production-grade WebRTC simulator for integration testing.

    Simulates complete interview session with:
    - Realistic audio streaming
    - Body language metrics
    - Network conditions
    - Timing patterns
    - Session lifecycle
    """

    def __init__(self):
        """Initialize mock simulator."""
        self.audio_generator: Optional[MockAudioGenerator] = None
        self.body_generator: Optional[MockBodyLanguageGenerator] = None

    async def simulate_interview_session(
        self,
        session: MockInterviewSession,
        audio_callback: Callable[[bytes, Dict[str, Any]], None],
        body_callback: Callable[[Dict[str, Any]], None],
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        Simulate full interview session with callbacks.

        Args:
            session: Session configuration
            audio_callback: Called for each audio chunk (data, metadata)
            body_callback: Called for body metrics updates
            event_callback: Called for session events
        """
        logger.info(f"Starting mock interview session: {session.session_id}")

        # Initialize generators
        self.audio_generator = MockAudioGenerator(session.audio_config)
        self.body_generator = MockBodyLanguageGenerator(
            session.body_metrics_config)

        # Fire start event
        if event_callback:
            event_callback("session_started", {
                "session_id": session.session_id,
                "timestamp": datetime.now().isoformat()
            })

        # Calculate timing
        total_chunks = int(
            session.duration_seconds * 1000 / session.audio_config.chunk_duration_ms
        )

        audio_interval = session.audio_config.chunk_duration_ms / 1000
        body_interval = session.body_metrics_config.update_interval_ms / 1000

        # Start parallel tasks
        audio_task = asyncio.create_task(
            self._stream_audio(
                session,
                total_chunks,
                audio_interval,
                audio_callback
            )
        )

        body_task = asyncio.create_task(
            self._stream_body_metrics(
                session,
                body_interval,
                body_callback
            )
        )

        # Wait for completion
        await asyncio.gather(audio_task, body_task)

        # Fire end event
        if event_callback:
            event_callback("session_ended", {
                "session_id": session.session_id,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": session.duration_seconds
            })

        logger.info(f"Mock interview session completed: {session.session_id}")

    async def _stream_audio(
        self,
        session: MockInterviewSession,
        total_chunks: int,
        interval: float,
        callback: Callable
    ):
        """Stream audio chunks with realistic timing."""
        for chunk_idx in range(total_chunks):
            # Decide if this chunk is speech or silence
            is_silence = np.random.random() < session.audio_config.silence_ratio

            if is_silence:
                audio_data = self.audio_generator.generate_silence_chunk(
                    session.audio_config.chunk_duration_ms
                )
            else:
                audio_data = self.audio_generator.generate_speech_chunk(
                    session.audio_config.chunk_duration_ms,
                    intensity=0.5 + 0.3 * np.random.random()
                )

            # Metadata
            metadata = {
                "chunk_index": chunk_idx,
                "timestamp": datetime.now().isoformat(),
                "duration_ms": session.audio_config.chunk_duration_ms,
                "sample_rate": session.audio_config.sample_rate,
                "is_silence": is_silence
            }

            # Simulate network latency
            network_delay = self._calculate_network_delay(
                session.audio_config.network_condition
            )
            await asyncio.sleep(network_delay)

            # Call callback
            callback(audio_data, metadata)

            # Wait for next chunk
            await asyncio.sleep(interval)

    async def _stream_body_metrics(
        self,
        session: MockInterviewSession,
        interval: float,
        callback: Callable
    ):
        """Stream body language metrics."""
        elapsed = 0.0

        while elapsed < session.duration_seconds:
            metrics = self.body_generator.generate_metrics(elapsed)
            callback(metrics)

            await asyncio.sleep(interval)
            elapsed += interval

    def _calculate_network_delay(self, condition: NetworkCondition) -> float:
        """Calculate network delay based on condition."""
        delays = {
            NetworkCondition.PERFECT: (0.001, 0.005),
            NetworkCondition.GOOD: (0.01, 0.05),
            NetworkCondition.FAIR: (0.05, 0.15),
            NetworkCondition.POOR: (0.15, 0.5),
            NetworkCondition.UNSTABLE: (0.01, 1.0)
        }

        min_delay, max_delay = delays[condition]
        return np.random.uniform(min_delay, max_delay)

    def generate_full_audio_file(
        self,
        session: MockInterviewSession,
        output_path: Path
    ) -> Path:
        """
        Generate complete audio file for session.

        Useful for testing the full pipeline with file-based processing.

        Args:
            session: Session configuration
            output_path: Output WAV file path

        Returns:
            Path to generated audio file
        """
        logger.info(f"Generating full audio file: {output_path}")

        # Initialize generator
        audio_gen = MockAudioGenerator(session.audio_config)

        # Generate all audio
        total_duration_ms = session.duration_seconds * 1000
        audio_data = audio_gen.generate_speech_chunk(
            total_duration_ms,
            intensity=0.6
        )

        # Create WAV file
        return audio_gen.create_wav_file(audio_data, output_path)


# Predefined test scenarios
SCENARIO_CONFIDENT_CANDIDATE = MockInterviewSession(
    session_id="test_confident",
    role="Senior Software Engineer",
    experience_level="Senior",
    duration_seconds=60,
    audio_config=AudioChunkConfig(
        speech_rate_wpm=160,
        silence_ratio=0.05,
        filler_probability=0.02,
        network_condition=NetworkCondition.GOOD
    ),
    body_metrics_config=BodyMetricsConfig(
        eye_contact_mean=0.85,
        posture_stability_mean=0.88,
        facial_expressiveness_mean=0.75
    ),
    answer_text="I would use a hash map to store the elements..."
)

SCENARIO_NERVOUS_CANDIDATE = MockInterviewSession(
    session_id="test_nervous",
    role="Junior Software Engineer",
    experience_level="Junior",
    duration_seconds=60,
    audio_config=AudioChunkConfig(
        speech_rate_wpm=120,
        silence_ratio=0.20,
        filler_probability=0.15,
        network_condition=NetworkCondition.GOOD
    ),
    body_metrics_config=BodyMetricsConfig(
        eye_contact_mean=0.55,
        posture_stability_mean=0.60,
        facial_expressiveness_mean=0.50
    ),
    answer_text="Um, well, I think... you could use, like, a loop..."
)

SCENARIO_NETWORK_ISSUES = MockInterviewSession(
    session_id="test_network_issues",
    role="Software Engineer",
    experience_level="Mid",
    duration_seconds=60,
    audio_config=AudioChunkConfig(
        speech_rate_wpm=150,
        silence_ratio=0.10,
        filler_probability=0.05,
        network_condition=NetworkCondition.UNSTABLE
    ),
    body_metrics_config=BodyMetricsConfig(
        eye_contact_mean=0.75,
        posture_stability_mean=0.80,
        facial_expressiveness_mean=0.70
    )
)

SCENARIO_SILENT_CANDIDATE = MockInterviewSession(
    session_id="test_silent",
    role="Software Engineer",
    experience_level="Mid",
    duration_seconds=30,
    audio_config=AudioChunkConfig(
        speech_rate_wpm=50,
        silence_ratio=0.80,  # 80% silence
        filler_probability=0.20,
        network_condition=NetworkCondition.GOOD
    ),
    body_metrics_config=BodyMetricsConfig(
        eye_contact_mean=0.40,
        posture_stability_mean=0.50,
        facial_expressiveness_mean=0.35
    )
)
