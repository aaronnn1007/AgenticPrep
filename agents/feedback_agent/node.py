import json
from typing import Any, Dict
from langchain_community.llms import Ollama
from state import FeedbackAgentState


def feedback_agent(state: FeedbackAgentState) -> Dict[str, Any]:
    """
    Convert structured evaluation data into human-readable feedback.
    Calls LLaMA-3 via LangChain Ollama integration.
    """

    # Extract data from state
    final_score = state["final_score"]
    score_label = state["score_label"]
    speech_metrics = state["speech_metrics"]
    relevancy_results = state["relevancy_results"]
    breakdown = state["breakdown"]

    # Build context for LLM
    positives_context = []
    improvements_context = []

    # Identify positives from relevancy
    high_relevance_questions = [
        r["question_id"] for r in relevancy_results if r["relevance"] == "high"
    ]
    if high_relevance_questions:
        positives_context.append(
            f"High relevance answers: {', '.join(high_relevance_questions)}"
        )

    # Identify areas from penalties
    if breakdown["penalties"]:
        improvements_context.extend(breakdown["penalties"])

    # Identify low relevance questions
    low_relevance_questions = [
        r["question_id"] for r in relevancy_results if r["relevance"] == "low"
    ]
    if low_relevance_questions:
        improvements_context.append(
            f"Low relevance answers: {', '.join(low_relevance_questions)}"
        )

    # Speech metrics context
    wpm = speech_metrics["wpm"]
    filler_words = speech_metrics["filler_words"]
    word_count = speech_metrics["word_count"]
    duration = speech_metrics["duration_seconds"]

    # Build the prompt
    prompt = f"""You are an expert interview coach providing constructive feedback to an interviewee.

Interview Performance Data:
- Final Score: {final_score} ({score_label})
- Speech Metrics:
  - Words Per Minute (WPM): {wpm}
  - Total Words: {word_count}
  - Duration: {duration} seconds
  - Filler Words: {filler_words}
- Relevance Component Score: {breakdown['relevance_component']}
- Speech Component Score: {breakdown['speech_component']}
- Penalties Applied: {', '.join(breakdown['penalties']) if breakdown['penalties'] else 'None'}
- Relevance Results: {json.dumps(relevancy_results)}

Positives to highlight:
{chr(10).join(['- ' + p for p in positives_context]) if positives_context else '- No specific metrics highlight'}

Areas to improve:
{chr(10).join(['- ' + a for a in improvements_context]) if improvements_context else '- No specific areas identified'}

Generate ONLY a valid JSON object (no markdown, no explanation) with the following structure:
{{
  "overall_feedback": "A 2-3 sentence summary of the interview performance and key takeaway",
  "positives": ["List of 2-3 specific strengths demonstrated"],
  "areas_to_improve": ["List of 2-3 specific areas for improvement based on penalties and relevance scores"],
  "actionable_suggestions": ["List of 3-4 practical, specific steps to improve, directly addressing areas_to_improve"]
}}

Requirements:
- Keep tone professional and encouraging
- Base feedback only on provided data (no hallucinations)
- Do not mention numeric scores directly in prose text
- Positives must reference actual high-relevance answers or good speech metrics
- Areas to improve must reference actual low-relevance answers, missing answers, or penalties
- Suggestions must be actionable and specific
- Return ONLY the JSON object, nothing else"""

    # Call Ollama via LangChain
    try:
        llm = Ollama(model="llama3")
        response_text = llm.invoke(prompt)

        # Extract JSON from response (in case of extra text)
        response_text = response_text.strip()
        
        # Find JSON object boundaries
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end + 1]
        else:
            json_str = response_text
        
        # Parse JSON response
        feedback_data = json.loads(json_str)

        return {
            "overall_feedback": feedback_data.get("overall_feedback", ""),
            "positives": feedback_data.get("positives", []),
            "areas_to_improve": feedback_data.get("areas_to_improve", []),
            "actionable_suggestions": feedback_data.get("actionable_suggestions", []),
        }

    except (json.JSONDecodeError, ValueError, AttributeError):
        # Fallback if Ollama is not available or JSON parsing fails
        return _generate_fallback_feedback(
            score_label, positives_context, improvements_context, speech_metrics
        )


def _generate_fallback_feedback(
    score_label: str,
    positives_context: list,
    improvements_context: list,
    speech_metrics: dict,
) -> Dict[str, Any]:
    """
    Generate feedback without LLM (fallback).
    """
    wpm = speech_metrics["wpm"]
    filler_words = speech_metrics["filler_words"]

    positives = []
    if positives_context:
        positives.append("Demonstrated good relevance on key questions")
    if filler_words == 0:
        positives.append("Spoke with clear articulation and minimal filler words")
    if 130 <= wpm <= 180:
        positives.append("Maintained a natural speaking pace")

    if not positives:
        positives = ["Completed the interview"]

    areas = []
    if improvements_context:
        if any("relevance" in str(item).lower() for item in improvements_context):
            areas.append("Some answers may have strayed from the core question")
        if any("wpm" in str(item).lower() for item in improvements_context):
            areas.append("Speaking pace could be adjusted for clarity")
        if any("missing" in str(item).lower() for item in improvements_context):
            areas.append("Ensure all questions are addressed comprehensively")

    if not areas:
        areas = ["Continue working on answer relevance and clarity"]

    suggestions = [
        "Practice answering similar questions to build confidence",
        "Record yourself and listen for clarity and pacing",
        "Focus on directly addressing what is asked before elaborating",
        "Use pauses instead of filler words when collecting thoughts",
    ]

    overall = f"You demonstrated {score_label} interview performance. Focus on the areas below to further strengthen your responses."

    return {
        "overall_feedback": overall,
        "positives": positives,
        "areas_to_improve": areas,
        "actionable_suggestions": suggestions,
    }
