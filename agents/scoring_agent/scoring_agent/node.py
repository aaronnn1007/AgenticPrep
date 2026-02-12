from typing import Dict, Any, List
from state import ScoringState


class ScoringNode:
    """Deterministic scoring node implementing locked rules."""

    WPM_MIN = 120
    WPM_MAX = 160

    def run(self, state: ScoringState) -> ScoringState:
        speech = state.speech_metrics or {}
        relevancy = state.relevancy_results or []
        answers = state.segmented_answers or []

        penalties: List[str] = []

        # 1. Base score: average of relevance_score values
        relevance_scores = [
            float(r.get("relevance_score", 0.0)) for r in relevancy if r is not None
        ]
        if relevance_scores:
            relevance_component = sum(relevance_scores) / len(relevance_scores)
        else:
            relevance_component = 0.0

        # 2. Penalize unanswered questions
        missing_penalty_total = 0.0
        # Build a lookup of answers by question_id for determinism
        answer_map = {a.get("question_id"): a for a in answers}

        # Penalize any question referenced in relevancy_results that has missing or empty answer
        for r in relevancy:
            qid = r.get("question_id")
            a = answer_map.get(qid)
            text = "" if a is None else (a.get("answer_text") or "")
            if not text or str(text).strip() == "":
                missing_penalty_total += 0.1
                penalties.append(f"missing_answer:{qid}:-0.1")

        # Also penalize any segmented_answers entries that are empty but not in relevancy_results
        for a in answers:
            qid = a.get("question_id")
            text = a.get("answer_text") or ""
            if (not text or str(text).strip() == "") and all(
                (r.get("question_id") != qid) for r in relevancy
            ):
                missing_penalty_total += 0.1
                penalties.append(f"missing_answer:{qid}:-0.1")

        # 3 & 4 Speech quality adjustments
        speech_component = 1.0
        wpm = float(speech.get("wpm", 0.0)) if speech.get("wpm") is not None else 0.0
        filler_words = int(speech.get("filler_words", 0)) if speech.get("filler_words") is not None else 0

        if wpm < self.WPM_MIN or wpm > self.WPM_MAX:
            speech_component -= 0.05
            penalties.append(f"wpm_out_of_range:{wpm}:-0.05")

        if filler_words > 5:
            speech_component -= 0.05
            penalties.append(f"excess_filler_words:{filler_words}:-0.05")

        # Clamp speech_component
        speech_component = max(0.0, min(1.0, speech_component))

        # Combine relevance and speech: average of the two components
        combined = (relevance_component + speech_component) / 2.0

        # Apply missing answer penalties
        final = combined - missing_penalty_total

        # Clamp final score 0.0 - 1.0
        final = max(0.0, min(1.0, final))

        # Score label mapping
        if final >= 0.8:
            label = "excellent"
        elif final >= 0.6:
            label = "good"
        elif final >= 0.4:
            label = "average"
        else:
            label = "poor"

        state.final_score = round(final, 4)
        state.score_label = label
        state.breakdown = {
            "relevance_component": round(relevance_component, 4),
            "speech_component": round(speech_component, 4),
            "penalties": penalties,
        }

        return state
