"""Manager for referential-based skill matching and enrichment.

Supports matching extracted skills against known referentials:
- ROME (Répertoire Opérationnel des Métiers et des Emplois)
- NSF (Nomenclature des Spécialités de Formation)
- Custom enterprise referentials
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReferentialEntry:
    id: str = ""
    label: str = ""
    source: str = ""  # "rome", "nsf", "custom"
    domain: str = ""
    synonyms: list[str] = field(default_factory=list)
    rome_code: str = ""
    nsf_code: str = ""


# ---- Built-in minimal referential for common IA/data skills ----
BUILTIN_REFERENTIAL: list[ReferentialEntry] = [
    ReferentialEntry(id="SK-IA-001", label="Programmation Python", source="rome",
                     domain="informatique", synonyms=["Python", "développement Python",
                     "langage Python"], rome_code="M1805"),
    ReferentialEntry(id="SK-IA-002", label="Programmation R", source="rome",
                     domain="informatique", synonyms=["R", "langage R", "statistiques R"],
                     rome_code="M1805"),
    ReferentialEntry(id="SK-IA-003", label="Machine Learning", source="rome",
                     domain="intelligence_artificielle", synonyms=["apprentissage automatique",
                     "ML", "machine learning"]),
    ReferentialEntry(id="SK-IA-004", label="Deep Learning", source="rome",
                     domain="intelligence_artificielle", synonyms=["apprentissage profond",
                     "réseaux de neurones", "deep learning"]),
    ReferentialEntry(id="SK-IA-005", label="SQL", source="rome",
                     domain="base_de_données", synonyms=["langage SQL",
                     "requêtes SQL", "bases de données relationnelles"]),
    ReferentialEntry(id="SK-IA-006", label="Statistiques", source="rome",
                     domain="mathématiques", synonyms=["statistique", "analyse statistique",
                     "statistiques descriptives", "statistiques inférentielles"]),
    ReferentialEntry(id="SK-IA-007", label="Data Engineering", source="rome",
                     domain="informatique", synonyms=["ingénierie des données",
                     "data pipeline", "ETL", "data warehouse"]),
    ReferentialEntry(id="SK-IA-008", label="Visualisation de données", source="rome",
                     domain="informatique", synonyms=["data viz", "visualisation",
                     "dataviz", "graphiques"]),
    ReferentialEntry(id="SK-IA-009", label="NLP", source="rome",
                     domain="intelligence_artificielle",
                     synonyms=["traitement du langage naturel",
                     "natural language processing",
                     "traitement automatique du langage"]),
    ReferentialEntry(id="SK-IA-010", label="Computer Vision", source="rome",
                     domain="intelligence_artificielle",
                     synonyms=["vision par ordinateur", "traitement d'images",
                     "image processing"]),
    ReferentialEntry(id="SK-IA-011", label="Gestion de projet", source="rome",
                     domain="management", synonyms=["project management",
                     "gestion de projet agile", "chef de projet"]),
    ReferentialEntry(id="SK-IA-012", label="Cloud Computing", source="rome",
                     domain="infrastructure", synonyms=["cloud", "AWS", "Azure",
                     "GCP", "cloud public"]),
    ReferentialEntry(id="SK-IA-013", label="DevOps", source="rome",
                     domain="informatique", synonyms=["CI/CD", "intégration continue",
                     "déploiement continu", "Docker", "Kubernetes"]),
    ReferentialEntry(id="SK-IA-014", label="Big Data", source="rome",
                     domain="informatique", synonyms=["données massives",
                     "Spark", "Hadoop", "data lake"]),
    ReferentialEntry(id="SK-IA-015", label="IA Générative", source="rome",
                     domain="intelligence_artificielle",
                     synonyms=["IA générative", "generative AI", "LLM",
                     "grand modèle de langage", "GPT"]),
    ReferentialEntry(id="SK-NSF-201", label="Mathématiques", source="nsf",
                     domain="mathématiques", nsf_code="201"),
    ReferentialEntry(id="SK-NSF-326", label="Programmation informatique", source="nsf",
                     domain="informatique", synonyms=["code", "codage",
                     "développement logiciel"], nsf_code="326"),
    ReferentialEntry(id="SK-NSF-114", label="Langues", source="nsf",
                     domain="langues", nsf_code="114"),
]


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def _tokenize(text: str) -> set[str]:
    return set(_normalize(t) for t in text.split() if len(t) > 2)


def _overlap_ratio(extracted_tokens: set[str], ref_tokens: set[str]) -> float:
    if not extracted_tokens or not ref_tokens:
        return 0.0
    intersection = extracted_tokens & ref_tokens
    union = extracted_tokens | ref_tokens
    return len(intersection) / len(union) if union else 0.0


def match_referential(
    extracted_label: str,
    referential: list[ReferentialEntry] | None = None,
    threshold: float = 0.50,
) -> list[tuple[ReferentialEntry, float]]:
    """Match an extracted skill label against a referential.

    Returns a list of (entry, match_score) tuples above the threshold.
    """
    if referential is None:
        referential = BUILTIN_REFERENTIAL

    extracted_tokens = _tokenize(extracted_label)
    if not extracted_tokens:
        return []

    matches: list[tuple[ReferentialEntry, float]] = []

    for entry in referential:
        max_score = 0.0

        # Check label
        label_score = _overlap_ratio(extracted_tokens, _tokenize(entry.label))
        max_score = max(max_score, label_score)

        # Check synonyms
        for syn in entry.synonyms:
            syn_score = _overlap_ratio(extracted_tokens, _tokenize(syn))
            max_score = max(max_score, syn_score)

        # Direct substring match (higher weight)
        norm_extracted = _normalize(extracted_label)
        norm_label = _normalize(entry.label)
        if norm_extracted in norm_label or norm_label in norm_extracted:
            max_score = max(max_score, 0.80)

        if max_score >= threshold:
            matches.append((entry, max_score))

    matches.sort(key=lambda x: -x[1])
    return matches
