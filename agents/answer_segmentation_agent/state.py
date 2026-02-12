from typing import TypedDict, List


class QuestionItem(TypedDict):
    question_id: str
    question_text: str


class SegmentedAnswer(TypedDict):
    question_id: str
    answer_text: str


class AnswerSegmentationState(TypedDict):
    questions: List[QuestionItem]
    full_transcript: str
    segmented_answers: List[SegmentedAnswer]
