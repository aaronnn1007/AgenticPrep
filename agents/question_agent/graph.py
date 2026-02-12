from typing import Dict, Any
try:
	from node import question_agent
except Exception:
	from question_agent.node import question_agent

try:
	from langgraph.graph import StateGraph
except Exception:
	# Lightweight fallback for environments without langgraph.
	class StateGraph:
		def __init__(self, _type=None):
			self._nodes = {}
			self._entry = None
			self._finish = None

		def add_node(self, name, func):
			self._nodes[name] = func

		def set_entry_point(self, name):
			self._entry = name

		def set_finish_point(self, name):
			self._finish = name

		def compile(self):
			# Return a simple compiled_graph with an invoke(state) method
			class Compiled:
				def __init__(self, nodes, entry):
					self._nodes = nodes
					self._entry = entry

				def invoke(self, state: Dict[str, Any]):
					if not self._entry or self._entry not in self._nodes:
						raise RuntimeError("No entry node defined")
					fn = self._nodes[self._entry]
					return fn(state)

			return Compiled(self._nodes, self._entry)


# Create the graph
graph = StateGraph(Dict[str, Any])

# Add the single node
graph.add_node("question_agent", question_agent)

# Set entry and finish points
graph.set_entry_point("question_agent")
graph.set_finish_point("question_agent")

# Compile the graph
compiled_graph = graph.compile()