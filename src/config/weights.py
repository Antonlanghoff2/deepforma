from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SubScoreConfig:
    label: str
    weight: float
    description: str


@dataclass(frozen=True)
class ScoringWeights:
    sub_scores: tuple[SubScoreConfig, ...] = (
        SubScoreConfig(label="couverture_competences", weight=0.25, description="Proportion des compétences du marché présentes dans la formation."),
        SubScoreConfig(label="pertinence_metier", weight=0.20, description="Adéquation aux métiers dominants du territoire."),
        SubScoreConfig(label="adequation_territoriale", weight=0.20, description="Correspondance avec les besoins spécifiques du territoire."),
        SubScoreConfig(label="niveau_experience", weight=0.10, description="Alignement des niveaux d'expérience avec les offres locales."),
        SubScoreConfig(label="employabilite", weight=0.15, description="Potentiel d'insertion professionnelle estimé."),
        SubScoreConfig(label="actualite_programme", weight=0.10, description="Prise en compte des compétences émergentes et tendances."),
    )

    @property
    def total_weight(self) -> float:
        return sum(s.weight for s in self.sub_scores)

    def get_sub_score(self, label: str) -> SubScoreConfig | None:
        for s in self.sub_scores:
            if s.label == label:
                return s
        return None

    def compute_global(self, sub_score_values: dict[str, float]) -> dict[str, Any]:
        scores = {}
        explanations = {}
        weighted_sum = 0.0
        for sub in self.sub_scores:
            value = sub_score_values.get(sub.label, 0.0)
            scores[sub.label] = value
            weighted_sum += value * sub.weight
            explanations[sub.label] = {
                "value": round(value, 2),
                "weight": sub.weight,
                "description": sub.description,
                "contribution": round(value * sub.weight, 2),
            }

        global_score = round(weighted_sum / self.total_weight, 2) if self.total_weight else 0.0
        return {
            "global_score": global_score,
            "sub_scores": scores,
            "explanations": explanations,
            "weights_config": {s.label: s.weight for s in self.sub_scores},
        }


SCORING_WEIGHTS = ScoringWeights()
