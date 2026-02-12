from __future__ import annotations

import json
import sys

from graph import build_media_graph


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python main.py <session_id> <video_path> [frame_interval_seconds]"
        )

    session_id = sys.argv[1]
    video_path = sys.argv[2]
    frame_interval_seconds = int(sys.argv[3]) if len(sys.argv) >= 4 else 1

    graph = build_media_graph()

    result = graph.invoke(
        {
            "session_id": session_id,
            "video_path": video_path,
            "frame_interval_seconds": frame_interval_seconds,
        }
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

