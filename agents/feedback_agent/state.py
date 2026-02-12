from typing import TypedDict, List, Dict, Any, Optional


class SpeechMetrics(TypedDict):
    wpm: float
    word_count: int
    duration_seconds: float
    filler_words: int


class RelevancyResult(TypedDict):
    question_id: str
    relevance: str


class Breakdown(TypedDict):
    relevance_component: float
    speech_component: float
    penalties: List[str]


class FeedbackAgentState(TypedDict):
    final_score: float
    score_label: str
    speech_metrics: SpeechMetrics
    relevancy_results: List[RelevancyResult]
    breakdown: Breakdown
    overall_feedback: Optional[str]
    positives: Optional[List[str]]
    areas_to_improve: Optional[List[str]]
    actionable_suggestions: Optional[List[str]]
