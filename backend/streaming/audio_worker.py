"""
Audio Streaming Worker
======================
Buffers real-time audio chunks from WebRTC, runs Whisper transcription on
completed buffer windows, and fires a transcript callback.

Error contract
--------------
Transcription failures raise ``TranscriptionError`` (never return "").
The caller (API layer or LangGraph node) is responsible for retry / fallback.
"""

import asyncio
import logging
import tempfile
import wave
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np

from backend.exceptions import TranscriptionError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AudioChunk:
    """A single chunk of raw PCM audio received from a WebRTC session."""
    session_id: str
    data: bytes                    # raw PCM int16 LE
    timestamp: datetime
    sample_rate: int               # Hz of the incoming chunk
    channels: int = 1


@dataclass
class TranscriptResult:
    """Result produced by the Whisper transcription of one buffer window."""
    session_id: str
    text: str
    confidence: float              # 0.0–1.0, average segment probability
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class AudioStreamingWorker:
    """
    Buffers AudioChunk objects per session, transcribes completed windows,
    and delivers TranscriptResult objects via a registered callback.

    Transcription failures raise ``TranscriptionError`` — they are never
    swallowed and never returned as empty strings.

    Chunk-level timeout
    -------------------
    Each transcription call is wrapped in ``asyncio.wait_for()`` using
    ``transcription_timeout_s`` (default 2.0 s).  If a chunk exceeds that
    budget, a warning is logged and the chunk is **skipped** so the rest of
    the stream continues unblocked.
    """

    TARGET_SAMPLE_RATE = 16_000   # Whisper expects 16 kHz mono
    # 30 s gives a CPU-based Whisper 'small' model enough headroom to finish.
    # The previous 2 s budget caused every chunk to time out and be silently
    # skipped, producing no transcripts in the live-streaming path.
    CHUNK_TRANSCRIPTION_TIMEOUT_S: float = 30.0   # default per-chunk deadline

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        buffer_duration_ms: int = 3_000,
        transcription_timeout_s: float = CHUNK_TRANSCRIPTION_TIMEOUT_S,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.buffer_duration_ms = buffer_duration_ms
        self.transcription_timeout_s = transcription_timeout_s

        self._model = None                    # loaded lazily on start()
        self._buffers: Dict[str, List[np.ndarray]] = {}
        self._callback: Optional[Callable[[TranscriptResult], None]] = None
        self._running = False
        self._process_queue: asyncio.Queue = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_transcript_callback(self, callback: Callable[[TranscriptResult], None]) -> None:
        """Register the function called with every TranscriptResult."""
        self._callback = callback

    async def start(self) -> None:
        """Load the Whisper model and start the internal processing loop."""
        logger.info(
            f"AudioStreamingWorker starting — model={self.model_size}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )
        self._process_queue = asyncio.Queue()
        self._running = True

        # Load faster-whisper model (may take a few seconds)
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            self._running = False
            raise TranscriptionError(
                f"Failed to load Whisper model '{self.model_size}': {e}",
                cause=e,
            ) from e

        # Start background consumer
        asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Signal the processing loop to stop and wait for it to drain."""
        self._running = False
        if self._process_queue:
            await self._process_queue.put(None)   # sentinel
        logger.info("AudioStreamingWorker stopped")

    async def process_audio_chunk(self, chunk: AudioChunk) -> None:
        """
        Accept one PCM chunk and enqueue it for buffering.

        Does *not* block on transcription; the result arrives asynchronously
        via the registered callback.
        """
        if not self._running:
            raise RuntimeError("Worker is not running. Call start() first.")
        await self._process_queue.put(chunk)

    async def finalize_session(self, session_id: str, sample_rate: int) -> None:
        """
        Flush any remaining buffered audio for *session_id* and transcribe it.

        Raises:
            TranscriptionError: If the flush transcription fails.
        """
        logger.info(f"Finalizing session: {session_id}")
        buffered = self._buffers.pop(session_id, [])
        if buffered:
            combined = np.concatenate(buffered)
            await self._transcribe_and_dispatch(session_id, combined, sample_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _process_loop(self) -> None:
        """Background task: drain the queue and buffer chunks."""
        while self._running:
            try:
                item = await asyncio.wait_for(self._process_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if item is None:               # stop sentinel
                break
            await self._handle_chunk(item)

    async def _handle_chunk(self, chunk: AudioChunk) -> None:
        """Add chunk to the session buffer; transcribe when window is full."""
        # Convert bytes → int16 numpy, then resample to TARGET_SAMPLE_RATE
        pcm = np.frombuffer(chunk.data, dtype=np.int16).astype(np.float32)
        if chunk.channels == 2:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
        if chunk.sample_rate != self.TARGET_SAMPLE_RATE:
            try:
                from scipy.signal import resample
                ratio = self.TARGET_SAMPLE_RATE / chunk.sample_rate
                pcm = resample(pcm, int(len(pcm) * ratio)).astype(np.float32)
            except Exception as e:
                logger.warning(f"Resampling failed, using raw audio: {e}")

        buf = self._buffers.setdefault(chunk.session_id, [])
        buf.append(pcm)

        # Check if we have reached the buffer window
        total_samples = sum(len(a) for a in buf)
        window_samples = int(self.TARGET_SAMPLE_RATE *
                             self.buffer_duration_ms / 1_000)

        if total_samples >= window_samples:
            combined = np.concatenate(buf)
            self._buffers[chunk.session_id] = []   # reset buffer
            await self._transcribe_and_dispatch(
                chunk.session_id, combined, self.TARGET_SAMPLE_RATE
            )

    async def _transcribe_and_dispatch(
        self, session_id: str, audio: np.ndarray, sample_rate: int
    ) -> None:
        """
        Write audio to a temp file, run Whisper with a per-chunk timeout,
        then fire the callback.

        Timeout behaviour
        -----------------
        If the Whisper call does not complete within ``self.transcription_timeout_s``
        the chunk is **skipped** (a warning is logged) so the live stream remains
        unblocked.  This differs from a model error, which raises
        ``TranscriptionError`` and propagates to the caller.

        Raises:
            TranscriptionError: On Whisper model / I/O errors (not timeout).
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write WAV file for Whisper
            audio_int16 = (
                audio * (32767 / max(np.max(np.abs(audio)), 1e-9))).astype(np.int16)
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

            # Run transcription in thread pool to avoid blocking the event loop,
            # bounded by the per-chunk timeout.
            loop = asyncio.get_event_loop()
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, self._run_whisper, tmp_path),
                    timeout=self.transcription_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Transcription timeout ({self.transcription_timeout_s}s) "
                    f"exceeded for session '{session_id}' — chunk skipped to "
                    f"keep stream unblocked."
                )
                return   # skip this chunk; do NOT raise
            except TranscriptionError:
                raise   # already wrapped — propagate as-is
            except Exception as e:
                raise TranscriptionError(
                    f"Transcription executor failed for session '{session_id}': {e}",
                    audio_path=tmp_path,
                    cause=e,
                ) from e

            if self._callback and result.text:
                result.session_id = session_id
                self._callback(result)

        finally:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _run_whisper(self, audio_path: str) -> "TranscriptResult":
        """
        Synchronous Whisper call — runs in a thread pool executor.

        Raises:
            TranscriptionError: On any error from the Whisper model.
        """
        try:
            segments, _info = self._model.transcribe(
                audio_path, beam_size=5, language="en"
            )
            seg_list = list(segments)
            text = " ".join(s.text for s in seg_list).strip()
            avg_prob = (
                float(np.mean([s.avg_logprob for s in seg_list]))
                if seg_list else 0.0
            )
            # Convert log-prob to a rough 0–1 confidence
            confidence = float(np.clip(np.exp(avg_prob), 0.0, 1.0))
            return TranscriptResult(
                session_id="",   # filled in by caller
                text=text,
                confidence=confidence,
            )
        except Exception as e:
            logger.error(f"Whisper failed on '{audio_path}': {e}")
            raise TranscriptionError(
                f"Whisper model error: {e}",
                audio_path=audio_path,
                cause=e,
            ) from e
