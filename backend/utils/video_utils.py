"""
Video Utilities
================
Helper functions for video processing and validation.

Provides reusable video operations for the Body Language Agent.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def validate_video_file(file_path: str) -> Path:
    """
    Validate that a video file exists and has a supported format.

    Args:
        file_path: Path to video file

    Returns:
        Path object of the validated file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        raise ValueError(
            f"Unsupported video format: {path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )

    return path


def load_video_frames(
    file_path: str,
    sample_rate: int = 5,
    max_frames: Optional[int] = None
) -> Tuple[List[np.ndarray], dict]:
    """
    Load and sample frames from a video file.

    Args:
        file_path: Path to video file
        sample_rate: Extract every Nth frame (default: 5)
        max_frames: Maximum number of frames to extract (None = all)

    Returns:
        Tuple of (frames_list, video_info_dict)
        - frames_list: List of numpy arrays (BGR format)
        - video_info_dict: Contains fps, total_frames, duration, width, height

    Raises:
        ValueError: If video cannot be opened or read
    """
    cap = cv2.VideoCapture(str(file_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {file_path}")

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        video_info = {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": width,
            "height": height
        }

        if total_frames == 0:
            logger.warning(f"Video has 0 frames: {file_path}")
            return [], video_info

        frames = []
        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Sample every Nth frame
            if frame_count % sample_rate == 0:
                frames.append(frame)
                extracted_count += 1

                # Check max_frames limit
                if max_frames and extracted_count >= max_frames:
                    break

            frame_count += 1

        logger.info(
            f"Extracted {len(frames)} frames from {total_frames} total frames "
            f"(sample_rate={sample_rate})"
        )

        return frames, video_info

    finally:
        cap.release()


def get_video_info(file_path: str) -> dict:
    """
    Get basic information about a video file without loading frames.

    Args:
        file_path: Path to video file

    Returns:
        Dictionary with video metadata (fps, total_frames, duration, width, height)

    Raises:
        ValueError: If video cannot be opened
    """
    cap = cv2.VideoCapture(str(file_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {file_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": width,
            "height": height
        }
    finally:
        cap.release()


def calculate_variance_normalized(values: List[float]) -> float:
    """
    Calculate variance of a list of values and normalize to 0-1 range.

    Args:
        values: List of numeric values

    Returns:
        Normalized variance (0 = no variance, 1 = high variance)
    """
    if not values or len(values) < 2:
        return 0.0

    arr = np.array(values)
    variance = np.var(arr)

    # Normalize variance using a reasonable scaling factor
    # Assumes variance typically ranges from 0 to 1000 for angles/positions
    normalized = min(variance / 1000.0, 1.0)

    return float(normalized)


def calculate_stability_from_variance(variance: float) -> float:
    """
    Convert variance to stability score (inverse relationship).

    Args:
        variance: Normalized variance value (0-1)

    Returns:
        Stability score (0-1, where 1 = very stable)
    """
    return 1.0 - variance
