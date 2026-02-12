from langgraph.graph import StateGraph, END
from state import RelevancyAgentState
from node import relevancy_agent


def create_relevancy_graph():
    """
    Create and return the Relevancy Agent LangGraph.
    Single-node graph: Entry -> relevancy_agent -> END
    """
    graph = StateGraph(RelevancyAgentState)

    graph.add_node("relevancy_agent", relevancy_agent)
    graph.set_entry_point("relevancy_agent")
    graph.add_edge("relevancy_agent", END)

    return graph.compile()
