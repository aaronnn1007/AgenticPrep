from __future__ import annotations

from typing import NotRequired, TypedDict


class MediaProcessorState(TypedDict):
    # Required input contract
    video_path: str
    frame_interval_seconds: int

    # Required output contract
    audio_path: NotRequired[str]
    frames_dir: NotRequired[str]
    fps: NotRequired[float]
    duration_seconds: NotRequired[float]


# Compatibility alias for local absolute imports
MediaState = MediaProcessorState


def validate_media_processor_input(state: dict) -> tuple[str, int]:
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

