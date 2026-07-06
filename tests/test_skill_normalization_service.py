from __future__ import annotations

from services.skill_normalization import normalize_offer_skill_labels, normalize_skill_label, normalize_skill_labels


class FakeNormalizer:
    def normalize(self, candidate: str) -> tuple[str | None, float, str | None]:
        value = candidate.strip()
        if not value:
            return None, 0.0, None
        return value.title(), 1.0, f"id::{value.lower()}"


def test_normalize_skill_label_uses_injected_normalizer() -> None:
    normalizer = FakeNormalizer()

    assert normalize_skill_label("  machine learning  ", normalizer=normalizer) == "Machine Learning"
    assert normalize_skill_label("   ", normalizer=normalizer) is None


def test_normalize_skill_labels_deduplicates_equivalent_labels() -> None:
    normalizer = FakeNormalizer()

    labels = normalize_skill_labels(
        [
            "python",
            {"label": "Python"},
            {"canonical_label": "SQL"},
            {"name": "sql"},
        ],
        normalizer=normalizer,
    )

    assert labels == ["Python", "Sql"]


def test_normalize_offer_skill_labels_flattens_multiple_sources() -> None:
    normalizer = FakeNormalizer()

    offer = {
        "normalized_skills": [{"canonical_label": "Docker"}, {"label": "docker"}],
        "structured_skills": [{"name": "Kubernetes"}],
        "model_skills": ["kubernetes"],
    }

    labels = normalize_offer_skill_labels(offer, normalizer=normalizer)

    assert labels == ["Docker", "Kubernetes"]
