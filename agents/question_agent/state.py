from typing import TypedDict, List

class InputState(TypedDict):
    role: str
    experience_level: str  # Fresher | Junior | Mid | Senior
    question_index: int
    previous_answers: List[str]

class OutputState(TypedDict):
    question_id: str
    question_text: str
    question_type: str  # behavioral | technical | situational
    difficulty: str  # easy | medium | hard
    time_limit_seconds: int