from __future__ import annotations

from langgraph.graph import END, StateGraph

from node import media_processor
from state import MediaState


def build_media_graph():
    graph = StateGraph(MediaState)
    graph.add_node("media_processor", media_processor)
    graph.set_entry_point("media_processor")
    graph.add_edge("media_processor", END)
    return graph.compile()

