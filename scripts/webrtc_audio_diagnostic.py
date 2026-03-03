#!/usr/bin/env python3
"""
WebRTC Audio Diagnostic Tool
=============================
Diagnoses audio capture and streaming issues.

Tests:
1. Audio device availability
2. Sample rate conversion
3. Buffer processing
4. Transcription model loading
5. End-to-end audio pipeline

Usage:
    python webrtc_audio_diagnostic.py [--test-audio path/to/audio.wav]
"""

from datetime import datetime
from backend.exceptions import TranscriptionError
from backend.streaming.rtp_audio_receiver import RTPAudioReceiver
from backend.streaming.audio_worker import AudioStreamingWorker, AudioChunk
import asyncio
import logging
import sys
import wave
from pathlib import Path
from typing import Optional

import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class AudioDiagnostic:
    """Audio pipeline diagnostic tool."""

    def __init__(self, test_audio_path: Optional[str] = None):
        """
        Initialize audio diagnostic.

        Args:
            test_audio_path: Path to test audio file
        """
        self.test_audio_path = test_audio_path
        self.results = {}

    def test_audio_file_loading(self):
        """Test 1: Load and validate audio file."""
        print("=" * 80)
        print("TEST 1: AUDIO FILE LOADING")
        print("=" * 80)

        if not self.test_audio_path:
            print("⚠ No test audio file provided")
            print("  Use --test-audio path/to/audio.wav to test with real audio")
            self.results['test_file_loading'] = 'SKIP'
            print()
            return None

        try:
            audio_path = Path(self.test_audio_path)
            if not audio_path.exists():
                print(f"✗ File not found: {self.test_audio_path}")
                self.results['test_file_loading'] = 'FAIL'
                return None

            # Load WAV file
            with wave.open(str(audio_path), 'rb') as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                framerate = wav.getframerate()
                nframes = wav.getnframes()
                duration = nframes / framerate

                print(f"✓ Loaded audio file: {audio_path.name}")
                print(f"  Channels: {channels}")
                print(f"  Sample width: {sample_width} bytes")
                print(f"  Frame rate: {framerate} Hz")
                print(f"  Frames: {nframes}")
                print(f"  Duration: {duration:.2f}s")

                # Read audio data
                frames = wav.readframes(nframes)
                audio_data = np.frombuffer(frames, dtype=np.int16)

                if channels == 2:
                    audio_data = audio_data.reshape(-1,
                                                    2).mean(axis=1).astype(np.int16)
                    print(f"  Converted stereo to mono")

                self.results['test_file_loading'] = 'PASS'
                self.results['audio_duration'] = duration
                print()
                return audio_data, framerate

        except Exception as e:
            print(f"✗ Failed to load audio: {e}")
            self.results['test_file_loading'] = 'FAIL'
            print()
            return None

    def test_audio_generation(self):
        """Test 2: Generate synthetic audio."""
        print("=" * 80)
        print("TEST 2: SYNTHETIC AUDIO GENERATION")
        print("=" * 80)

        try:
            # Generate 3 seconds of audio
            sample_rate = 16000
            duration = 3.0
            samples = int(sample_rate * duration)

            # Generate sine wave at 440 Hz (A4 note)
            t = np.arange(samples) / sample_rate
            frequency = 440.0
            audio_data = (0.3 * 32767 * np.sin(2 * np.pi *
                          frequency * t)).astype(np.int16)

            print(f"✓ Generated {duration}s of audio")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Samples: {samples}")
            print(f"  Frequency: {frequency} Hz")
            print(f"  Data size: {len(audio_data) * 2} bytes")

            # Calculate audio level
            rms = np.sqrt(np.mean(audio_data.astype(float) ** 2))
            db = 20 * np.log10(rms / 32767) if rms > 0 else -np.inf
            print(f"  RMS level: {db:.1f} dB")

            self.results['test_audio_generation'] = 'PASS'
            print()
            return audio_data, sample_rate

        except Exception as e:
            print(f"✗ Failed to generate audio: {e}")
            self.results['test_audio_generation'] = 'FAIL'
            print()
            return None

    async def test_audio_resampling(self, audio_data, from_rate):
        """Test 3: Audio resampling."""
        print("=" * 80)
        print("TEST 3: AUDIO RESAMPLING")
        print("=" * 80)

        if audio_data is None:
            print("⚠ No audio data available")
            self.results['test_resampling'] = 'SKIP'
            print()
            return None

        try:
            target_rate = 16000

            if from_rate == target_rate:
                print(f"✓ Audio already at target rate: {target_rate} Hz")
                self.results['test_resampling'] = 'PASS'
                print()
                return audio_data

            # Resample
            from scipy import signal as scipy_signal

            ratio = target_rate / from_rate
            new_length = int(len(audio_data) * ratio)
            resampled = scipy_signal.resample(
                audio_data, new_length).astype(np.int16)

            print(f"✓ Resampled audio: {from_rate} Hz → {target_rate} Hz")
            print(f"  Original samples: {len(audio_data)}")
            print(f"  Resampled samples: {len(resampled)}")
            print(f"  Ratio: {ratio:.3f}")

            self.results['test_resampling'] = 'PASS'
            print()
            return resampled

        except Exception as e:
            print(f"✗ Resampling failed: {e}")
            self.results['test_resampling'] = 'FAIL'
            print()
            return None

    async def test_audio_worker(self, audio_data, sample_rate):
        """Test 4: Audio worker and transcription."""
        print("=" * 80)
        print("TEST 4: AUDIO WORKER & TRANSCRIPTION")
        print("=" * 80)

        if audio_data is None:
            print("⚠ No audio data available")
            self.results['test_audio_worker'] = 'SKIP'
            print()
            return

        try:
            # Initialize audio worker
            print("Initializing audio worker...")
            worker = AudioStreamingWorker(
                model_size="base",
                device="cpu",
                compute_type="int8",
                buffer_duration_ms=3000
            )

            # Set up transcript callback
            transcripts = []

            def on_transcript(result):
                transcripts.append(result)
                print(
                    f"  📝 Transcript: '{result.text}' (confidence: {result.confidence:.2f})")

            worker.set_transcript_callback(on_transcript)

            print("Starting audio worker...")
            await worker.start()

            # Send audio in chunks
            chunk_size_ms = 500
            chunk_samples = (chunk_size_ms * sample_rate) // 1000
            session_id = "diagnostic_test"

            print(f"Sending audio chunks ({chunk_size_ms}ms chunks)...")
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i:i + chunk_samples]
                if len(chunk) == 0:
                    break

                audio_chunk = AudioChunk(
                    session_id=session_id,
                    data=chunk.tobytes(),
                    timestamp=datetime.now(),
                    sample_rate=sample_rate,
                    channels=1
                )

                await worker.process_audio_chunk(audio_chunk)

                if (i // chunk_samples) % 5 == 0:
                    print(f"  Sent chunk {i // chunk_samples}")

            print("Waiting for transcription...")
            await asyncio.sleep(5)

            # Finalize
            await worker.finalize_session(session_id, sample_rate)
            await asyncio.sleep(2)

            print(f"✓ Audio worker test completed")
            print(f"  Transcripts received: {len(transcripts)}")

            if len(transcripts) > 0:
                full_text = " ".join(t.text for t in transcripts)
                print(f"  Full transcript: '{full_text[:200]}...'")
                self.results['test_audio_worker'] = 'PASS'
                self.results['transcripts'] = len(transcripts)
            else:
                print(f"  ⚠ No transcripts generated")
                self.results['test_audio_worker'] = 'FAIL'

            # Cleanup
            await worker.stop()

        except TranscriptionError as e:
            # Transcription failed with a known, recoverable error — not a bug
            # in the worker itself.  Callers should implement retry logic.
            print(f"✗ Transcription failed: {e}")
            logger.error(
                "Transcription error (TranscriptionError)", exc_info=True)
            self.results['test_audio_worker'] = 'FAIL'
        except Exception as e:
            print(f"✗ Audio worker test failed: {e}")
            logger.error("Audio worker error", exc_info=True)
            self.results['test_audio_worker'] = 'FAIL'

        print()

    async def test_buffer_processing(self):
        """Test 5: RTP audio receiver buffer processing."""
        print("=" * 80)
        print("TEST 5: BUFFER PROCESSING")
        print("=" * 80)

        try:
            chunks_received = []

            def on_chunk(session_id, data, rate):
                chunks_received.append((session_id, len(data), rate))
                print(f"  📦 Chunk: {len(data)} bytes @ {rate} Hz")

            # Create RTP receiver
            receiver = RTPAudioReceiver(
                target_sample_rate=16000,
                target_channels=1,
                buffer_size_ms=3000,
                chunk_size_ms=500,
                on_chunk_ready=on_chunk
            )

            # Generate test audio
            sample_rate = 48000
            duration = 2.0
            samples = int(sample_rate * duration)
            audio_data = (0.3 * 32767 * np.sin(2 * np.pi * 440 *
                          np.arange(samples) / sample_rate)).astype(np.int16)

            print(f"Sending audio to receiver...")
            session_id = "buffer_test"

            # Send in chunks
            chunk_size = 960  # 20ms at 48kHz
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size].tobytes()
                await receiver.receive_audio(session_id, chunk, sample_rate, 1)

            await asyncio.sleep(1)

            print(f"✓ Buffer processing test completed")
            print(f"  Chunks emitted: {len(chunks_received)}")

            if len(chunks_received) > 0:
                self.results['test_buffer_processing'] = 'PASS'
            else:
                print(f"  ⚠ No chunks emitted")
                self.results['test_buffer_processing'] = 'FAIL'

        except Exception as e:
            print(f"✗ Buffer processing test failed: {e}")
            logger.error("Buffer processing error", exc_info=True)
            self.results['test_buffer_processing'] = 'FAIL'

        print()

    async def run(self):
        """Run all diagnostic tests."""
        print("\n🔍 WebRTC Audio Diagnostic Tool\n")

        # Test 1: Load audio file (if provided)
        audio_result = self.test_audio_file_loading()
        if audio_result:
            audio_data, sample_rate = audio_result
        else:
            # Test 2: Generate synthetic audio
            audio_result = self.test_audio_generation()
            if audio_result:
                audio_data, sample_rate = audio_result
            else:
                print("✗ Cannot continue without audio data")
                return 1

        # Test 3: Resampling
        audio_data = await self.test_audio_resampling(audio_data, sample_rate)
        if audio_data is not None:
            sample_rate = 16000  # After resampling

        # Test 4: Audio worker
        await self.test_audio_worker(audio_data, sample_rate)

        # Test 5: Buffer processing
        await self.test_buffer_processing()

        # Print summary
        self.print_summary()

        # Return exit code
        failed = sum(1 for r in self.results.values() if r == 'FAIL')
        return 0 if failed == 0 else 1

    def print_summary(self):
        """Print test summary."""
        print("=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)

        tests = [
            'test_file_loading',
            'test_audio_generation',
            'test_resampling',
            'test_audio_worker',
            'test_buffer_processing'
        ]

        passed = 0
        failed = 0
        skipped = 0

        for test in tests:
            status = self.results.get(test, 'SKIP')
            icon = '✓' if status == 'PASS' else '✗' if status == 'FAIL' else '○'
            print(f"  {icon} {test}: {status}")

            if status == 'PASS':
                passed += 1
            elif status == 'FAIL':
                failed += 1
            elif status == 'SKIP':
                skipped += 1

        print()
        print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
        print("=" * 80)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WebRTC Audio Diagnostic Tool"
    )
    parser.add_argument(
        '--test-audio',
        help='Path to test audio file (WAV format)'
    )

    args = parser.parse_args()

    diagnostic = AudioDiagnostic(test_audio_path=args.test_audio)
    exit_code = await diagnostic.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
