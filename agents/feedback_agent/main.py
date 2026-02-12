import json
from graph import build_feedback_graph
from state import FeedbackAgentState


def main():
    """
    Main entry point for the Feedback Agent.
    Creates the graph, provides sample input, and prints JSON output.
    """

    # Build the graph
    feedback_graph = build_feedback_graph()

    # Sample input data
    sample_input: FeedbackAgentState = {
        "final_score": 0.625,
        "score_label": "good",
        "speech_metrics": {
            "wpm": 163,
            "word_count": 137,
            "duration_seconds": 50.24,
            "filler_words": 0,
        },
        "relevancy_results": [
            {"question_id": "q1", "relevance": "high"},
            {"question_id": "q2", "relevance": "low"},
            {"question_id": "q3", "relevance": "high"},
        ],
        "breakdown": {
            "relevance_component": 0.5,
            "speech_component": 0.95,
            "penalties": [
                "missing_answer:q2:-0.1",
                "wpm_out_of_range:163.0:-0.05",
            ],
        },
        "overall_feedback": None,
        "positives": None,
        "areas_to_improve": None,
        "actionable_suggestions": None,
    }

    # Invoke the feedback graph
    result = feedback_graph.invoke(sample_input)

    # Extract feedback output
    feedback_output = {
        "overall_feedback": result.get("overall_feedback", ""),
        "positives": result.get("positives", []),
        "areas_to_improve": result.get("areas_to_improve", []),
        "actionable_suggestions": result.get("actionable_suggestions", []),
    }

    # Print JSON output
    print(json.dumps(feedback_output, indent=2))


if __name__ == "__main__":
    main()
