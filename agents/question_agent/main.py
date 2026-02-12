from graph import compiled_graph
import json

# Sample input
input_state = {
    "role": "Software Engineer",
    "experience_level": "Fresher",
    "question_index": 1,
    "previous_answers": []
}

# Invoke the graph
result = compiled_graph.invoke(input_state)

# Print the output JSON
print(json.dumps(result, indent=2))