from langgraph.graph import StateGraph
from state import FeedbackAgentState
from node import feedback_agent


def build_feedback_graph():
    """
    Build the LangGraph for the Feedback Agent.
    Single-node graph: Entry → feedback_agent → END
    """
    graph = StateGraph(FeedbackAgentState)

    # Add the feedback agent node
    graph.add_node("feedback_agent", feedback_agent)

    # Set entry point
    graph.set_entry_point("feedback_agent")

    # Set end point
    graph.add_edge("feedback_agent", "__end__")

    # Compile the graph
    compiled_graph = graph.compile()

    return compiled_graph
