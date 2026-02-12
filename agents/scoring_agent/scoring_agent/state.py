from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ScoringState:
    speech_metrics: Dict[str, Any] = field(default_factory=dict)
    relevancy_results: List[Dict[str, Any]] = field(default_factory=list)
    segmented_answers: List[Dict[str, Any]] = field(default_factory=list)
    final_score: float = 0.0
    score_label: str = ""
    breakdown: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "speech_metrics": self.speech_metrics,
            "relevancy_results": self.relevancy_results,
            "segmented_answers": self.segmented_answers,
            "final_score": self.final_score,
            "score_label": self.score_label,
            "breakdown": self.breakdown,
        }
