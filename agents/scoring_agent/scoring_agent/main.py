import json
from graph import ScoringGraph


def sample_input():
    return {
        "speech_metrics": {
            "wpm": 163,
            "word_count": 137,
            "duration_seconds": 50.24,
            "filler_words": 0,
        },
        "relevancy_results": [
            {"question_id": "q1", "relevance": "high", "relevance_score": 0.8},
            {"question_id": "q2", "relevance": "low", "relevance_score": 0.2},
            {"question_id": "q3", "relevance": "medium", "relevance_score": 0.5},
        ],
        "segmented_answers": [
            {"question_id": "q1", "answer_text": "I led a team on a project..."},
            {"question_id": "q2", "answer_text": ""},
            {"question_id": "q3", "answer_text": "We improved performance by 20%..."},
        ],
    }


def main():
    graph = ScoringGraph()
    initial = sample_input()
    result = graph.run(initial)
    # Ensure output matches required state schema keys
    output = {
        "speech_metrics": initial.get("speech_metrics", {}),
        "relevancy_results": initial.get("relevancy_results", []),
        "segmented_answers": initial.get("segmented_answers", []),
        "final_score": result.get("final_score"),
        "score_label": result.get("score_label"),
        "breakdown": result.get("breakdown"),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
