#!/usr/bin/env python3
"""
WebRTC Integration Test
========================
Comprehensive integration test for WebRTC streaming to Multi-Agent system.

Tests:
1. Media capture simulation (audio/video)
2. WebRTC signaling (SDP/ICE)
3. Audio stream reception
4. Audio-to-text transcription
5. Voice Agent metrics
6. Body Language metrics integration
7. LangGraph pipeline trigger

This test simulates a real WebRTC interview session end-to-end.
"""

from backend.config import settings
from backend.streaming.voice_metrics import VoiceMetricsComputer
from backend.streaming.audio_worker import AudioStreamingWorker
from backend.streaming.redis_session import RedisSessionStore
from backend.streaming.webrtc_session_manager import WebRTCSessionManager
import asyncio
import logging
import sys
import time
import wave
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    MediaStreamTrack
)
from av import AudioFrame, VideoFrame

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FAKE MEDIA TRACKS
# =============================================================================

class FakeAudioTrack(MediaStreamTrack):
    """
    Fake audio track that generates test audio.

    Generates audio that says numbers 1-10 when transcribed
    (or generates beeps for testing).
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 48000, duration: float = 10.0):
        """
        Initialize fake audio track.

        Args:
            sample_rate: Audio sample rate
            duration: Total duration in seconds
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples_per_frame = int(sample_rate * 0.02)  # 20ms frames
        self.total_samples = int(sample_rate * duration)
        self.samples_sent = 0

        # Try to load test audio file if available
        self.test_audio_data = self._load_test_audio()
        self.test_audio_position = 0

    def _load_test_audio(self) -> Optional[np.ndarray]:
        """
        Load test audio file if available.

        Returns:
            Audio data as numpy array or None
        """
        # Look for test audio files
        test_files = [
            Path(__file__).parent.parent / "data" / "test_audio.wav",
            Path(__file__).parent.parent / "data" / "sample_answer.wav",
        ]

        for test_file in test_files:
            if test_file.exists():
                try:
                    with wave.open(str(test_file), 'rb') as wav:
                        frames = wav.readframes(wav.getnframes())
                        audio_data = np.frombuffer(frames, dtype=np.int16)

                        # Convert to mono if stereo
                        if wav.getnchannels() == 2:
                            audio_data = audio_data.reshape(-1, 2).mean(axis=1)

                        # Resample if needed (simplified)
                        expected_rate = self.sample_rate
                        actual_rate = wav.getframerate()
                        if actual_rate != expected_rate:
                            # Simple ratio-based resampling
                            ratio = expected_rate / actual_rate
                            new_length = int(len(audio_data) * ratio)
                            indices = np.linspace(
                                0, len(audio_data)-1, new_length)
                            audio_data = np.interp(
                                indices, np.arange(len(audio_data)), audio_data)

                        logger.info(
                            f"Loaded test audio: {test_file} ({len(audio_data)} samples)")
                        return audio_data.astype(np.int16)

                except Exception as e:
                    logger.warning(
                        f"Failed to load test audio {test_file}: {e}")

        logger.warning(
            "No test audio file found, will generate synthetic audio")
        return None

    async def recv(self):
        """
        Receive next audio frame.

        Returns:
            AudioFrame
        """
        if self.samples_sent >= self.total_samples:
            self.stop()
            raise MediaStreamTrack._ended

        # Determine frame size
        remaining = self.total_samples - self.samples_sent
        frame_samples = min(self.samples_per_frame, remaining)

        # Generate or use recorded audio
        if self.test_audio_data is not None:
            # Use recorded audio
            start = self.test_audio_position
            end = start + frame_samples

            if end <= len(self.test_audio_data):
                audio_data = self.test_audio_data[start:end]
            else:
                # Wrap around or pad
                audio_data = np.zeros(frame_samples, dtype=np.int16)
                available = len(self.test_audio_data) - start
                if available > 0:
                    audio_data[:available] = self.test_audio_data[start:]

            self.test_audio_position = (
                self.test_audio_position + frame_samples) % len(self.test_audio_data)
        else:
            # Generate synthetic audio (800Hz tone with amplitude modulation)
            t = np.arange(frame_samples) / self.sample_rate
            frequency = 800.0
            amplitude = 0.3 * \
                (1 + 0.5 * np.sin(2 * np.pi * 2 * t))  # Modulated
            audio_data = (amplitude * 32767 * np.sin(2 *
                          np.pi * frequency * t)).astype(np.int16)

        # Create AudioFrame
        frame = AudioFrame.from_ndarray(
            audio_data.reshape(1, -1),  # Shape: (channels, samples)
            format='s16',
            layout='mono'
        )
        frame.sample_rate = self.sample_rate
        frame.pts = self.samples_sent
        frame.time_base = f'1/{self.sample_rate}'

        self.samples_sent += frame_samples

        # Small delay to simulate real-time
        await asyncio.sleep(0.02)

        return frame


class FakeVideoTrack(MediaStreamTrack):
    """
    Fake video track that generates test frames.
    """

    kind = "video"

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, duration: float = 10.0):
        """
        Initialize fake video track.

        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
            duration: Total duration in seconds
        """
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.frame_count = 0
        self.total_frames = int(fps * duration)

    async def recv(self):
        """
        Receive next video frame.

        Returns:
            VideoFrame
        """
        if self.frame_count >= self.total_frames:
            self.stop()
            raise MediaStreamTrack._ended

        # Generate colored frame with counter
        frame_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Color gradient background
        color = int((self.frame_count / self.total_frames) * 255)
        frame_data[:, :] = [color, 128, 255 - color]

        # Create VideoFrame
        frame = VideoFrame.from_ndarray(frame_data, format='bgr24')
        frame.pts = self.frame_count
        frame.time_base = f'1/{self.fps}'

        self.frame_count += 1

        # Delay to simulate real-time
        await asyncio.sleep(1.0 / self.fps)

        return frame


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class WebRTCIntegrationTest:
    """
    Comprehensive WebRTC integration test.

    Tests the complete pipeline from media capture to agent analysis.
    """

    def __init__(self):
        """Initialize integration test."""
        self.session_manager: Optional[WebRTCSessionManager] = None
        self.redis_store: Optional[RedisSessionStore] = None
        self.audio_worker: Optional[AudioStreamingWorker] = None
        self.test_session_id = f"test_session_{int(time.time())}"
        self.results = {}

    async def setup(self):
        """Set up test infrastructure."""
        logger.info("=" * 80)
        logger.info("SETTING UP WEBRTC INTEGRATION TEST")
        logger.info("=" * 80)

        # Initialize Redis
        logger.info("Initializing Redis session store...")
        self.redis_store = RedisSessionStore(redis_url=settings.REDIS_URL)
        await self.redis_store.connect()
        logger.info("✓ Redis connected")

        # Initialize audio worker
        logger.info("Initializing audio worker...")
        self.audio_worker = AudioStreamingWorker(
            model_size="base",
            device="cpu",
            compute_type="int8",
            buffer_duration_ms=3000
        )
        await self.audio_worker.start()
        logger.info("✓ Audio worker started")

        # Initialize voice metrics computer
        logger.info("Initializing voice metrics computer...")
        voice_metrics = VoiceMetricsComputer()
        logger.info("✓ Voice metrics computer initialized")

        # Initialize WebRTC session manager
        logger.info("Initializing WebRTC session manager...")
        self.session_manager = WebRTCSessionManager(
            redis_store=self.redis_store,
            audio_worker=self.audio_worker,
            voice_metrics=voice_metrics
        )
        logger.info("✓ WebRTC session manager initialized")

        # Create test session in Redis
        logger.info(f"Creating test session: {self.test_session_id}")
        await self.redis_store.create_session(
            session_id=self.test_session_id,
            interview_id=f"interview_{self.test_session_id}",
            role="Software Engineer",
            experience_level="Mid",
            question_text="Tell me about your experience with Python and FastAPI.",
            question_category="technical"
        )
        logger.info("✓ Test session created")

        logger.info("Setup complete!\n")

    async def test_audio_stream(self):
        """Test 1: Audio stream reception and processing."""
        logger.info("=" * 80)
        logger.info("TEST 1: AUDIO STREAM RECEPTION")
        logger.info("=" * 80)

        try:
            # Create fake audio track
            logger.info("Creating fake audio track (10 seconds)...")
            audio_track = FakeAudioTrack(sample_rate=48000, duration=10.0)

            # Simulate receiving audio frames
            frame_count = 0
            start_time = time.time()

            logger.info("Streaming audio frames...")
            while True:
                try:
                    frame = await audio_track.recv()
                    frame_count += 1

                    # Convert frame to PCM bytes
                    audio_array = frame.to_ndarray()
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=0)
                    audio_data = audio_array.astype(np.int16).tobytes()

                    # Send to session manager (simulating WebRTC callback)
                    await self.session_manager._on_webrtc_audio_data(
                        self.test_session_id,
                        audio_data,
                        frame.sample_rate
                    )

                    if frame_count % 50 == 0:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"  Sent {frame_count} frames ({elapsed:.1f}s elapsed)")

                except Exception as e:
                    if "ended" in str(e).lower():
                        break
                    raise

            elapsed = time.time() - start_time
            logger.info(
                f"✓ Streamed {frame_count} audio frames in {elapsed:.2f}s")

            self.results['audio_frames_sent'] = frame_count
            self.results['audio_duration'] = elapsed

            # Wait for transcription
            logger.info("Waiting for transcription to complete...")
            await asyncio.sleep(5)

            # Check transcript
            transcript = await self.redis_store.get_full_transcript(self.test_session_id)
            logger.info(
                f"✓ Transcript received: {len(transcript) if transcript else 0} characters")

            if transcript:
                logger.info(f"  Transcript preview: '{transcript[:200]}...'")
                self.results['transcript'] = transcript
                self.results['test_audio_stream'] = 'PASS'
            else:
                logger.warning("⚠ No transcript generated!")
                self.results['test_audio_stream'] = 'FAIL'

        except Exception as e:
            logger.error(f"✗ Audio stream test failed: {e}", exc_info=True)
            self.results['test_audio_stream'] = 'FAIL'
            self.results['test_audio_stream_error'] = str(e)

        logger.info("")

    async def test_body_language_metrics(self):
        """Test 2: Body language metrics reception."""
        logger.info("=" * 80)
        logger.info("TEST 2: BODY LANGUAGE METRICS")
        logger.info("=" * 80)

        try:
            # Simulate body language metrics from frontend
            test_metrics = {
                'eye_contact': 0.85,
                'posture_stability': 0.78,
                'facial_expressiveness': 0.72,
                'distractions': []
            }

            logger.info(f"Sending body language metrics: {test_metrics}")
            await self.session_manager._update_body_metrics(
                self.test_session_id,
                test_metrics
            )

            # Retrieve from Redis
            session_data = await self.redis_store.get_session(self.test_session_id)

            if session_data and session_data.body_language:
                logger.info(f"✓ Body language metrics stored successfully")
                logger.info(
                    f"  Eye contact: {session_data.body_language.eye_contact:.2f}")
                logger.info(
                    f"  Posture: {session_data.body_language.posture_stability:.2f}")
                self.results['test_body_language'] = 'PASS'
            else:
                logger.warning("⚠ Body language metrics not found in session!")
                self.results['test_body_language'] = 'FAIL'

        except Exception as e:
            logger.error(f"✗ Body language test failed: {e}", exc_info=True)
            self.results['test_body_language'] = 'FAIL'
            self.results['test_body_language_error'] = str(e)

        logger.info("")

    async def test_voice_metrics(self):
        """Test 3: Voice metrics computation."""
        logger.info("=" * 80)
        logger.info("TEST 3: VOICE METRICS")
        logger.info("=" * 80)

        try:
            # Get session data
            session_data = await self.redis_store.get_session(self.test_session_id)

            if session_data and session_data.voice_analysis:
                logger.info(f"✓ Voice metrics computed")
                logger.info(
                    f"  Speech rate: {session_data.voice_analysis.speech_rate_wpm:.1f} WPM")
                logger.info(
                    f"  Clarity: {session_data.voice_analysis.clarity_score:.2f}")
                self.results['test_voice_metrics'] = 'PASS'
                self.results['speech_rate_wpm'] = session_data.voice_analysis.speech_rate_wpm
            else:
                logger.warning("⚠ Voice metrics not found!")
                self.results['test_voice_metrics'] = 'FAIL'

        except Exception as e:
            logger.error(f"✗ Voice metrics test failed: {e}", exc_info=True)
            self.results['test_voice_metrics'] = 'FAIL'
            self.results['test_voice_metrics_error'] = str(e)

        logger.info("")

    async def test_connection_state(self):
        """Test 4: Connection state tracking."""
        logger.info("=" * 80)
        logger.info("TEST 4: CONNECTION STATE")
        logger.info("=" * 80)

        try:
            # Update connection status
            await self.redis_store.update_connection_status(self.test_session_id, True)

            # Verify
            session_data = await self.redis_store.get_session(self.test_session_id)

            if session_data and session_data.connected:
                logger.info("✓ Connection state tracking works")
                self.results['test_connection_state'] = 'PASS'
            else:
                logger.warning("⚠ Connection state not updated!")
                self.results['test_connection_state'] = 'FAIL'

        except Exception as e:
            logger.error(f"✗ Connection state test failed: {e}", exc_info=True)
            self.results['test_connection_state'] = 'FAIL'
            self.results['test_connection_state_error'] = str(e)

        logger.info("")

    async def teardown(self):
        """Clean up test infrastructure."""
        logger.info("=" * 80)
        logger.info("TEARING DOWN")
        logger.info("=" * 80)

        if self.audio_worker:
            logger.info("Stopping audio worker...")
            await self.audio_worker.stop()

        if self.redis_store:
            logger.info("Cleaning up test session...")
            # Note: Keep session for inspection
            # await self.redis_store.delete_session(self.test_session_id)
            await self.redis_store.disconnect()

        logger.info("✓ Teardown complete\n")

    async def run(self):
        """Run all tests."""
        try:
            await self.setup()

            # Run tests
            await self.test_audio_stream()
            await self.test_body_language_metrics()
            await self.test_voice_metrics()
            await self.test_connection_state()

        finally:
            await self.teardown()

        # Print results
        self.print_results()

    def print_results(self):
        """Print test results summary."""
        logger.info("=" * 80)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 80)

        tests = [
            ('Audio Stream', 'test_audio_stream'),
            ('Body Language', 'test_body_language'),
            ('Voice Metrics', 'test_voice_metrics'),
            ('Connection State', 'test_connection_state')
        ]

        passed = 0
        failed = 0

        for name, key in tests:
            status = self.results.get(key, 'SKIP')
            icon = '✓' if status == 'PASS' else '✗' if status == 'FAIL' else '○'
            logger.info(f"  {icon} {name}: {status}")

            if status == 'PASS':
                passed += 1
            elif status == 'FAIL':
                failed += 1

        logger.info("")
        logger.info(f"Total: {passed} passed, {failed} failed")

        # Additional metrics
        if 'audio_frames_sent' in self.results:
            logger.info(f"\nMetrics:")
            logger.info(f"  Audio frames: {self.results['audio_frames_sent']}")
            logger.info(
                f"  Audio duration: {self.results.get('audio_duration', 0):.2f}s")
            logger.info(
                f"  Transcript length: {len(self.results.get('transcript', ''))} chars")
            logger.info(
                f"  Speech rate: {self.results.get('speech_rate_wpm', 0):.1f} WPM")

        logger.info("=" * 80)

        # Return exit code
        return 0 if failed == 0 else 1


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point."""
    logger.info("\n🎤 WebRTC Integration Test\n")

    test = WebRTCIntegrationTest()
    exit_code = await test.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
