from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import ffmpeg

from state import MediaState


def _to_posix_path_str(p: Path, trailing_slash: bool = False) -> str:
    s = p.as_posix()
    if trailing_slash and not s.endswith("/"):
        s += "/"
    return s


def _ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_dir(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)

def _clear_existing_frames(frames_dir: Path) -> None:
    if not frames_dir.exists():
        return
    if not frames_dir.is_dir():
        raise RuntimeError(f"Frames output path exists but is not a directory: {frames_dir}")
    for p in frames_dir.glob("frame_*.jpg"):
        try:
            p.unlink()
        except Exception as e:
            raise RuntimeError(f"Failed to remove existing frame file: {p}") from e
    for p in frames_dir.glob("frame_*.png"):
        try:
            p.unlink()
        except Exception as e:
            raise RuntimeError(f"Failed to remove existing frame file: {p}") from e


def _extract_audio_ffmpeg(video_path: Path, audio_path: Path) -> None:
    _ensure_parent_dir(audio_path)
    try:
        (
            ffmpeg.input(str(video_path))
            .output(
                str(audio_path),
                format="wav",
                ac=1,  # mono
                ar=16000,  # 16kHz
                loglevel="error",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg binary not found. Install ffmpeg and ensure it is on PATH."
        ) from e
    except ffmpeg.Error as e:
        stderr = ""
        try:
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        except Exception:
            stderr = "<unavailable>"
        raise RuntimeError(f"Audio extraction failed via ffmpeg. Details: {stderr}") from e


def _read_video_metadata(cap: cv2.VideoCapture) -> tuple[float, float]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        raise RuntimeError("Unable to read FPS from video (CAP_PROP_FPS <= 0).")

    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration_seconds = 0.0

    if frame_count > 0.0:
        duration_seconds = frame_count / fps
    else:
        # Fallback: seek to end and read timestamp
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
        pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if pos_msec > 0.0:
            duration_seconds = pos_msec / 1000.0
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)

    if duration_seconds <= 0.0:
        raise RuntimeError("Unable to determine video duration (duration_seconds <= 0).")

    return fps, duration_seconds


def _extract_frames(
    cap: cv2.VideoCapture,
    frames_dir: Path,
    interval_seconds: int,
    duration_seconds: float,
) -> int:
    _ensure_dir(frames_dir)
    _clear_existing_frames(frames_dir)

    total_saved = 0
    t = 0.0
    max_t = float(duration_seconds)

    while t <= max_t + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        total_saved += 1
        frame_path = frames_dir / f"frame_{total_saved:04d}.jpg"
        ok_write = cv2.imwrite(
            str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        if not ok_write:
            raise RuntimeError(f"Failed to write frame to disk: {frame_path}")

        t += float(interval_seconds)

    if total_saved <= 0:
        raise RuntimeError("Frame extraction produced zero frames.")

    return total_saved


def _validate_input(state: dict) -> tuple[str, int]:
    if not isinstance(state, dict):
        raise TypeError("Media Processor input must be a JSON object (dict).")

    if "video_path" not in state:
        raise KeyError('Missing required key "video_path".')
    if "frame_interval_seconds" not in state:
        raise KeyError('Missing required key "frame_interval_seconds".')

    video_path = state["video_path"]
    frame_interval_seconds = state["frame_interval_seconds"]

    if not isinstance(video_path, str) or not video_path.strip():
        raise TypeError('"video_path" must be a non-empty string.')

    if not isinstance(frame_interval_seconds, int):
        raise TypeError('"frame_interval_seconds" must be an integer.')
    if frame_interval_seconds <= 0:
        raise ValueError('"frame_interval_seconds" must be >= 1.')

    return video_path, frame_interval_seconds


def media_processor(state: MediaState) -> MediaState:
    if "video_path" not in state:
        raise KeyError('Missing required key "video_path".')
    video_path_str = state["video_path"]
    _, frame_interval_seconds = _validate_input(state)

    project_root = Path(os.getcwd())
    video_path_in = Path(video_path_str)
    video_path = video_path_in if video_path_in.is_absolute() else (project_root / video_path_in)
    if not video_path.exists():
        raise FileNotFoundError(
            f'Video file not found: "{video_path_str}" (resolved: "{video_path}")'
        )
    if not video_path.is_file():
        raise FileNotFoundError(
            f'Video path is not a file: "{video_path_str}" (resolved: "{video_path}")'
        )

    stem = video_path_in.stem
    audio_path_rel = Path("data") / "audio" / f"{stem}.wav"
    frames_dir_rel = Path("data") / "frames" / stem
    audio_path = project_root / audio_path_rel
    frames_dir = project_root / frames_dir_rel

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path_str}")

    try:
        fps, duration_seconds = _read_video_metadata(cap)

        _extract_audio_ffmpeg(video_path=video_path, audio_path=audio_path)

        # Reset capture to deterministic position before extraction
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _extract_frames(
            cap=cap,
            frames_dir=frames_dir,
            interval_seconds=frame_interval_seconds,
            duration_seconds=duration_seconds,
        )
    finally:
        cap.release()

    out: dict[str, Any] = dict(state)
    out["video_path"] = video_path_str
    out["frame_interval_seconds"] = frame_interval_seconds
    out["audio_path"] = _to_posix_path_str(audio_path_rel, trailing_slash=False)
    out["frames_dir"] = _to_posix_path_str(frames_dir_rel, trailing_slash=True)
    out["fps"] = float(fps)
    out["duration_seconds"] = float(duration_seconds)

    return out  # type: ignore[return-value]

