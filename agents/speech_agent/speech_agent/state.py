from typing_extensions import TypedDict


class SpeechMetrics(TypedDict):
    word_count: int
    wpm: int
    duration_seconds: float
    filler_words: int


class SpeechAgentState(TypedDict):
    audio_path: str
    question_id: str
    transcript: str
    speech_metrics: SpeechMetrics
