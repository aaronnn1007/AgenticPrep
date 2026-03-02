"""
Audio Utilities
================
Helper functions for audio processing and validation.

Provides reusable audio operations for the Voice Agent.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3",
                           ".m4a", ".ogg", ".flac", ".aac", ".wma"}
DEFAULT_SAMPLE_RATE = 16000  # Whisper's native sample rate


def load_audio(
    file_path: str,
    sr: Optional[int] = None,
    mono: bool = True,
    duration: Optional[float] = None
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with optional resampling and mono conversion.

    Args:
        file_path: Path to audio file
        sr: Target sample rate (None = preserve original)
        mono: Convert to mono if True
        duration: Load only first N seconds (None = full audio)

    Returns:
        Tuple of (audio_data, sample_rate)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be loaded
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        audio_data, sample_rate = librosa.load(
            file_path,
            sr=sr,
            mono=mono,
            duration=duration
        )

        logger.debug(
            f"Loaded audio: {file_path}, "
            f"duration={len(audio_data)/sample_rate:.2f}s, "
            f"sr={sample_rate}Hz"
        )

        return audio_data, sample_rate

    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")


def get_audio_duration(file_path: str) -> float:
    """
    Get audio duration in seconds without loading full file.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds

    Raises:
        ValueError: If duration cannot be determined
    """
    try:
        # Use soundfile for fast duration extraction
        info = sf.info(file_path)
        duration = info.duration

        logger.debug(f"Audio duration: {duration:.2f}s for {file_path}")

        return float(duration)

    except Exception as e:
        # Fallback to librosa
        try:
            duration = librosa.get_duration(path=file_path)
            return float(duration)
        except Exception as e2:
            raise ValueError(f"Failed to get audio duration: {e2}")


def validate_audio_file(file_path: str) -> None:
    """
    Validate that audio file exists and is readable.

    Args:
        file_path: Path to audio file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid or unsupported format
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    # Check extension
    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )

    # Try to read metadata
    try:
        info = sf.info(str(path))
        if info.frames == 0:
            raise ValueError("Audio file is empty (0 frames)")

        logger.debug(
            f"Validated audio: {file_path} "
            f"({info.duration:.2f}s, {info.samplerate}Hz, {info.channels}ch)"
        )

    except Exception as e:
        raise ValueError(f"Cannot read audio file: {e}")


def is_silent(audio_data: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Check if audio is silent or nearly silent.

    Args:
        audio_data: Audio signal as numpy array
        threshold: RMS threshold below which audio is considered silent

    Returns:
        True if audio is silent
    """
    rms = np.sqrt(np.mean(audio_data ** 2))
    is_quiet = rms < threshold

    if is_quiet:
        logger.warning(f"Audio is silent (RMS={rms:.6f} < {threshold})")

    return is_quiet


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1.0, 1.0] range.

    Args:
        audio_data: Audio signal as numpy array

    Returns:
        Normalized audio
    """
    max_val = np.abs(audio_data).max()

    if max_val > 0:
        normalized = audio_data / max_val
        logger.debug(f"Audio normalized: max={max_val:.6f}")
        return normalized
    else:
        logger.warning("Cannot normalize silent audio")
        return audio_data


def convert_to_mono(audio_data: np.ndarray) -> np.ndarray:
    """
    Convert stereo or multi-channel audio to mono.

    Args:
        audio_data: Audio signal (can be multi-channel)

    Returns:
        Mono audio signal
    """
    if audio_data.ndim == 1:
        return audio_data  # Already mono

    # Average across channels
    mono = np.mean(audio_data, axis=0)
    logger.debug(
        f"Converted audio to mono: {audio_data.shape} -> {mono.shape}")

    return mono


def trim_silence(
    audio_data: np.ndarray,
    sr: int,
    top_db: float = 20.0
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Args:
        audio_data: Audio signal
        sr: Sample rate
        top_db: Threshold in dB below reference to consider as silence

    Returns:
        Trimmed audio
    """
    trimmed, _ = librosa.effects.trim(audio_data, top_db=top_db)

    original_duration = len(audio_data) / sr
    trimmed_duration = len(trimmed) / sr

    logger.debug(
        f"Trimmed silence: {original_duration:.2f}s -> {trimmed_duration:.2f}s "
        f"(removed {original_duration - trimmed_duration:.2f}s)"
    )

    return trimmed


def calculate_rms_energy(audio_data: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) energy of audio signal.

    Args:
        audio_data: Audio signal

    Returns:
        RMS energy value
    """
    rms = np.sqrt(np.mean(audio_data ** 2))
    return float(rms)


def calculate_zero_crossing_rate(audio_data: np.ndarray) -> float:
    """
    Calculate zero crossing rate (ZCR) - indicates signal noisiness.

    Args:
        audio_data: Audio signal

    Returns:
        ZCR value (higher = more noisy/fricative sounds)
    """
    zcr = librosa.feature.zero_crossing_rate(audio_data)
    return float(np.mean(zcr))


def extract_spectral_features(
    audio_data: np.ndarray,
    sr: int
) -> dict:
    """
    Extract spectral features for clarity analysis.

    Args:
        audio_data: Audio signal
        sr: Sample rate

    Returns:
        Dictionary with spectral features
    """
    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)

    # Spectral rolloff (frequency below which X% of energy is contained)
    rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio_data)

    features = {
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std": float(np.std(rolloff)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_std": float(np.std(bandwidth)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr))
    }

    logger.debug(f"Extracted spectral features: {features}")

    return features


def save_audio(
    audio_data: np.ndarray,
    sr: int,
    output_path: str
) -> None:
    """
    Save audio to file.

    Args:
        audio_data: Audio signal
        sr: Sample rate
        output_path: Path to save audio file
    """
    sf.write(output_path, audio_data, sr)
    logger.info(f"Saved audio to: {output_path}")


def get_audio_info(file_path: str) -> dict:
    """
    Get comprehensive audio file information.

    Args:
        file_path: Path to audio file

    Returns:
        Dictionary with audio metadata
    """
    try:
        info = sf.info(file_path)

        return {
            "path": file_path,
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype
        }

    except Exception as e:
        logger.error(f"Failed to get audio info: {e}")
        return {}
