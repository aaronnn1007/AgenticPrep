from typing import Dict, Any
from state import ScoringState
from node import ScoringNode

try:
    # Prefer using LangGraph if available for orchestration
    from langgraph import Graph, Node  # type: ignore

    class ScoringGraph:
        def __init__(self) -> None:
            self.graph = Graph()
            # Node wrapper for langgraph
            class Wrapper(Node):
                def __init__(self, impl: ScoringNode):
                    super().__init__()
                    self.impl = impl

                def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
                    s = ScoringState(
                        speech_metrics=state.get("speech_metrics", {}),
                        relevancy_results=state.get("relevancy_results", []),
                        segmented_answers=state.get("segmented_answers", []),
                    )
                    s = self.impl.run(s)
                    return s.as_dict()

            self.wrapper = Wrapper(ScoringNode())
            self.graph.add_node("scoring_agent", self.wrapper)

        def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
            # Run single node graph; return the node output
            result = self.graph.run(initial_state)
            # If langgraph returns a mapping of node outputs, normalize
            if isinstance(result, dict) and "scoring_agent" in result:
                return result["scoring_agent"]
            return result

except Exception:
    # Fallback minimal graph implementation if LangGraph is not installed
    class ScoringGraph:
        def __init__(self) -> None:
            self.node = ScoringNode()

        def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
            s = ScoringState(
                speech_metrics=initial_state.get("speech_metrics", {}),
                relevancy_results=initial_state.get("relevancy_results", []),
                segmented_answers=initial_state.get("segmented_answers", []),
            )
            s = self.node.run(s)
            return s.as_dict()
