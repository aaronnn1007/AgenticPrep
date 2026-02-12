from langgraph.graph import StateGraph, END

from state import AnswerSegmentationState
from node import answer_segmentation_node


def build_graph():
    """Build the answer segmentation graph."""
    graph = StateGraph(AnswerSegmentationState)

    graph.add_node("answer_segmentation_agent", answer_segmentation_node)

    graph.set_entry_point("answer_segmentation_agent")

    graph.add_edge("answer_segmentation_agent", END)

    return graph.compile()
