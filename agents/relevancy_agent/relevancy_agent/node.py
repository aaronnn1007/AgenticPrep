import json
from langchain_community.llms import Ollama
from state import RelevancyAgentState


def relevancy_agent(state: RelevancyAgentState) -> dict:
    """
    Evaluate the relevance of a transcript to a question using LLaMA via Ollama.
    Returns a partial state update with relevance analysis.
    """
    llm = Ollama(model="llama3")

    prompt = f"""You are an expert interview analyst. Evaluate how relevant this answer is to the following question.

Question: {state['question_text']}

Answer: {state['transcript']}

Respond with ONLY valid JSON (no additional text):
{{
  "relevance_score": <float between 0.0 and 1.0>,
  "key_points_covered": ["point1", "point2"],
  "missing_points": ["point1", "point2"]
}}"""

    response = llm.invoke(prompt)

    # Parse JSON response
    try:
        analysis = json.loads(response)
    except json.JSONDecodeError:
        analysis = {
            "relevance_score": 0.0,
            "key_points_covered": [],
            "missing_points": []
        }

    relevance_score = float(analysis.get("relevance_score", 0.0))
    relevance_score = max(0.0, min(1.0, relevance_score))

    # Map score to relevance label
    if relevance_score >= 0.75:
        relevance = "high"
    elif relevance_score >= 0.4:
        relevance = "medium"
    else:
        relevance = "low"

    return {
        "relevance": relevance,
        "relevance_score": relevance_score,
        "key_points_covered": analysis.get("key_points_covered", []),
        "missing_points": analysis.get("missing_points", [])
    }
