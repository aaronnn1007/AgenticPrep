import json
from typing import Any, Dict

from langchain_community.llms import Ollama

from state import AnswerSegmentationState


def answer_segmentation_node(state: AnswerSegmentationState) -> Dict[str, Any]:
    """
    Segment interview transcript into per-question answers using LLaMA-3 via Ollama.
    Returns partial state update with segmented_answers.
    """
    questions = state["questions"]
    full_transcript = state["full_transcript"]

    llm = Ollama(model="llama3")

    questions_json = json.dumps(questions, indent=2)

    prompt = f"""You are an expert at segmenting interview transcripts.
Given the following questions and the full interview transcript, extract the answer for each question.

QUESTIONS:
{questions_json}

FULL TRANSCRIPT:
{full_transcript}

Your task:
1. For each question_id, find the corresponding answer in the transcript
2. Extract only the relevant answer text
3. If the answer is not found or unclear, use an empty string
4. Output ONLY valid JSON, nothing else

Output must be valid JSON in this exact format:
{{
  "segmented_answers": [
    {{"question_id": "q1", "answer_text": "..."}},
    {{"question_id": "q2", "answer_text": "..."}}
  ]
}}

Do not include any explanation or commentary. Output only JSON."""

    response = llm.invoke(prompt)

    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            segmented_answers = result.get("segmented_answers", [])
        else:
            segmented_answers = []
    except json.JSONDecodeError:
        segmented_answers = []

    if not segmented_answers:
        segmented_answers = [
            {"question_id": q["question_id"], "answer_text": ""}
            for q in questions
        ]

    return {"segmented_answers": segmented_answers}
