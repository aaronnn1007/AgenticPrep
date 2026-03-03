"""
RTP Audio Receiver
==================
Receives raw PCM audio from a WebRTC/RTP source, converts it to the target
sample-rate and channel layout, and emits ready-to-process chunks via a
callback.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Signature: (session_id: str, data: bytes, sample_rate: int) -> None
ChunkReadyCallback = Callable[[str, bytes, int], None]


class RTPAudioReceiver:
    """
    Accepts arbitrary-rate, arbitrary-channel PCM audio, resamples to
    ``target_sample_rate`` / ``target_channels``, buffers it, and fires
    ``on_chunk_ready`` whenever a full ``chunk_size_ms`` window is available.

    The receiver is deliberately simple: it does not manage networking —
    callers push data via ``receive_audio``.
    """

    def __init__(
        self,
        target_sample_rate: int = 16_000,
        target_channels: int = 1,
        buffer_size_ms: int = 3_000,
        chunk_size_ms: int = 500,
        on_chunk_ready: Optional[ChunkReadyCallback] = None,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.buffer_size_ms = buffer_size_ms
        self.chunk_size_ms = chunk_size_ms
        self.on_chunk_ready = on_chunk_ready

        self._chunk_samples = int(target_sample_rate * chunk_size_ms / 1_000)
        self._buffers: Dict[str, List[np.ndarray]] = {}

    async def receive_audio(
        self,
        session_id: str,
        data: bytes,
        sample_rate: int,
        channels: int,
    ) -> None:
        """
        Accept raw PCM int16 audio and enqueue it for chunked delivery.

        Args:
            session_id: Identifier for the RTP session.
            data:        Raw PCM bytes (int16, interleaved if stereo).
            sample_rate: Sample rate of the incoming data (Hz).
            channels:    Number of channels in the incoming data.
        """
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32)

        # Downmix to mono
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)

        # Resample to target rate
        if sample_rate != self.target_sample_rate:
            try:
                from scipy.signal import resample
                ratio = self.target_sample_rate / sample_rate
                pcm = resample(pcm, int(len(pcm) * ratio)).astype(np.float32)
            except Exception as e:
                logger.warning(
                    f"RTPAudioReceiver: resampling {sample_rate}→"
                    f"{self.target_sample_rate} failed, using raw: {e}"
                )

        buf = self._buffers.setdefault(session_id, [])
        buf.append(pcm)

        # Emit full chunks
        combined = np.concatenate(buf)
        while len(combined) >= self._chunk_samples:
            chunk_pcm = combined[: self._chunk_samples]
            combined = combined[self._chunk_samples:]
            self._emit(session_id, chunk_pcm)

        self._buffers[session_id] = [combined] if len(combined) > 0 else []

    async def flush_session(self, session_id: str) -> None:
        """Emit any remaining buffered audio for the given session."""
        remaining = self._buffers.pop(session_id, [])
        if remaining:
            combined = np.concatenate(remaining)
            if len(combined) > 0:
                self._emit(session_id, combined)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, session_id: str, pcm: np.ndarray) -> None:
        """Convert float32 PCM back to int16 bytes and fire the callback."""
        if self.on_chunk_ready is None:
            return
        int16 = np.clip(pcm, -32768, 32767).astype(np.int16)
        self.on_chunk_ready(session_id, int16.tobytes(),
                            self.target_sample_rate)
