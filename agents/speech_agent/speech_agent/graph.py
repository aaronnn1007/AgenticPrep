from langgraph.graph import StateGraph
from state import SpeechAgentState
from node import speech_agent


def create_speech_agent_graph():
    graph = StateGraph(SpeechAgentState)
    graph.add_node("speech_agent", speech_agent)
    graph.set_entry_point("speech_agent")
    graph.set_finish_point("speech_agent")
    return graph.compile()
