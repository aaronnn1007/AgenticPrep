from typing import TypedDict


class RelevancyAgentState(TypedDict):
    question_id: str
    question_text: str
    transcript: str
    relevance: str
    relevance_score: float
    key_points_covered: list
    missing_points: list
