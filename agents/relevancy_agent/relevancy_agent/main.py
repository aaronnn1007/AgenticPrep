import json
from graph import create_relevancy_graph


def main():
    """
    Create the Relevancy Agent graph, provide sample input, and invoke it.
    Prints the final JSON output.
    """
    # Create the graph
    graph = create_relevancy_graph()

    # Sample input state
    input_state = {
        "question_id": "q1",
        "question_text": "What is the difference between an API and an SDK?",
        "transcript": "An API is an Application Programming Interface that allows different pieces of software to communicate with each other. An SDK is a Software Development Kit that provides tools and libraries for developers to build applications.",
        "relevance": "",
        "relevance_score": 0.0,
        "key_points_covered": [],
        "missing_points": []
    }

    # Invoke the graph
    output_state = graph.invoke(input_state)

    # Extract and print JSON output
    json_output = {
        "question_id": output_state["question_id"],
        "relevance": output_state["relevance"],
        "relevance_score": output_state["relevance_score"],
        "key_points_covered": output_state["key_points_covered"],
        "missing_points": output_state["missing_points"]
    }

    print(json.dumps(json_output, indent=2))


if __name__ == "__main__":
    main()
