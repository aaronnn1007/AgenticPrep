import os
import re
from pathlib import Path
from faster_whisper import WhisperModel
import librosa

from state import SpeechAgentState, SpeechMetrics


FILLER_WORDS = {"um", "uh", "like", "you know", "ah", "er"}


def speech_agent(state: SpeechAgentState) -> dict:
    audio_path = state["audio_path"]

    try:
        if not os.path.exists(audio_path):
            return {
                "transcript": "",
                "speech_metrics": {
                    "word_count": 0,
                    "wpm": 0,
                    "duration_seconds": 0.0,
                    "filler_words": 0,
                },
            }

        audio, sr = librosa.load(audio_path, sr=None)
        duration_seconds = float(librosa.get_duration(y=audio, sr=sr))

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, language="en")
        transcript = " ".join([segment.text for segment in segments]).strip()

        if not transcript:
            return {
                "transcript": "",
                "speech_metrics": {
                    "word_count": 0,
                    "wpm": 0,
                    "duration_seconds": duration_seconds,
                    "filler_words": 0,
                },
            }

        words = transcript.lower().split()
        word_count = len(words)

        filler_count = 0
        for filler in FILLER_WORDS:
            filler_count += sum(1 for word in words if word == filler.split()[0])

        text_lower = transcript.lower()
        filler_count = 0
        for filler in FILLER_WORDS:
            pattern = r"\b" + re.escape(filler) + r"\b"
            filler_count += len(re.findall(pattern, text_lower))

        wpm = int((word_count / duration_seconds) * 60) if duration_seconds > 0 else 0

        metrics: SpeechMetrics = {
            "word_count": word_count,
            "wpm": wpm,
            "duration_seconds": duration_seconds,
            "filler_words": filler_count,
        }

        return {
            "transcript": transcript,
            "speech_metrics": metrics,
        }

    except Exception:
        return {
            "transcript": "",
            "speech_metrics": {
                "word_count": 0,
                "wpm": 0,
                "duration_seconds": 0.0,
                "filler_words": 0,
            },
        }
