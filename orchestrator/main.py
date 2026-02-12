from __future__ import annotations

import json
import sys
from typing import Any, Dict

from orchestrator.graph import build_orchestrator_graph


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint for the InterviewPA orchestrator.

    Usage:
      python -m orchestrator.main <session_id> <video_path>
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 2:
        print(
            "Usage: python -m orchestrator.main <session_id> <video_path>",
            file=sys.stderr,
        )
        return 1

    session_id = argv[0]
    video_path = argv[1]

    initial_state: Dict[str, Any] = {
        "session_id": session_id,
        "video_path": video_path,
        "frame_interval_seconds": 1,
    }

    graph = build_orchestrator_graph()
    final_state = graph.invoke(initial_state)

    print(json.dumps(final_state, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
