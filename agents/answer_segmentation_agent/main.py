import json

from graph import build_graph


def main():
    """Main entry point for answer segmentation agent."""
    sample_questions = [
        {
            "question_id": "q1",
            "question_text": "Tell me about your experience with Python programming.",
        },
        {
            "question_id": "q2",
            "question_text": "What is your approach to code testing?",
        },
        {
            "question_id": "q3",
            "question_text": "Describe a challenging project you worked on.",
        },
    ]

    sample_transcript = """Thank you for having me. I've been programming in Python for about 8 years now, 
    primarily in data science and backend development. I started with web scraping 
    and gradually moved into machine learning projects. Python is great for its 
    ecosystem and readability.
    
    As for testing, I believe in a multi-layered approach. I write unit tests for 
    individual functions, integration tests to ensure components work together, 
    and end-to-end tests for critical user flows. I use pytest for most projects 
    and aim for around 80% code coverage. Test-driven development has saved me 
    from many bugs.
    
    One challenging project was building a real-time data processing pipeline for 
    financial data. The challenge was handling millions of data points per second 
    while maintaining accuracy and low latency. We used Apache Kafka for streaming 
    and optimized our algorithms for performance. It taught me a lot about 
    distributed systems and optimization."""

    graph = build_graph()

    input_state = {
        "questions": sample_questions,
        "full_transcript": sample_transcript,
        "segmented_answers": [],
    }

    result = graph.invoke(input_state)

    output = {"segmented_answers": result["segmented_answers"]}

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
