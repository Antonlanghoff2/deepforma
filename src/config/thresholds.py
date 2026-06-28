from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConfidenceLevel:
    label: str
    min_score: float
    max_score: float
    min_margin: float = 0.0


@dataclass(frozen=True)
class ClassificationThresholds:
    high_confidence: float = 0.80
    medium_confidence: float = 0.60
    low_confidence: float = 0.40
    rejection: float = 0.20
    uncertainty_margin: float = 0.10
    min_offers_for_conclusion: int = 5
    statistical_robustness_min: int = 20

    @property
    def confidence_levels(self) -> list[ConfidenceLevel]:
        return [
            ConfidenceLevel(label="forte", min_score=self.high_confidence, max_score=1.0),
            ConfidenceLevel(label="moyenne", min_score=self.medium_confidence, max_score=self.high_confidence),
            ConfidenceLevel(label="faible", min_score=self.low_confidence, max_score=self.medium_confidence),
            ConfidenceLevel(label="rejetée", min_score=self.rejection, max_score=self.low_confidence),
        ]

    def get_confidence_level(self, score: float) -> str:
        for level in self.confidence_levels:
            if level.min_score <= score <= level.max_score:
                return level.label
        return "rejetée"

    @property
    def classification_states(self) -> list[dict[str, Any]]:
        return [
            {"state": "fiable", "min_gap": 0.30, "min_score": 0.70, "description": "Classification fiable – écart suffisant entre les deux classes."},
            {"state": "probable", "min_gap": 0.15, "min_score": 0.55, "description": "Classification probable – écart modéré."},
            {"state": "incertaine", "min_gap": 0.0, "min_score": 0.40, "description": "Classification incertaine – les probabilités sont trop proches."},
            {"state": "insuffisant", "min_gap": -1.0, "min_score": -1.0, "description": "Données insuffisantes pour classer."},
        ]

    def get_classification_state(self, prob_ia: float, prob_non_ia: float) -> dict[str, Any]:
        gap = abs(prob_ia - prob_non_ia)
        max_prob = max(prob_ia, prob_non_ia)
        for state in self.classification_states:
            if gap >= state["min_gap"] and max_prob >= state["min_score"]:
                return {"state": state["state"], "description": state["description"], "gap": round(gap, 4)}
        return {"state": "insuffisant", "description": "Données insuffisantes pour classer.", "gap": round(gap, 4)}


THRESHOLDS = ClassificationThresholds()
