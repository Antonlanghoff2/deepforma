from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable
import re
import unicodedata

from common.text import clean_text, normalize_for_match

TAXONOMY_VERSION = "2026-07-05"

ENTITY_TYPES = ("SKILL", "METHOD", "TOOL", "DOMAIN", "SOFT_SKILL")
FAMILIES = ("Machine Learning", "Deep Learning", "NLP", "MLOps", "Other")

FAMILY_HIERARCHY: dict[str, str | None] = {
    "Machine Learning": None,
    "Deep Learning": "Machine Learning",
    "NLP": None,
    "MLOps": None,
    "Other": None,
}

FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "Machine Learning": ("ml", "machine learning", "apprentissage automatique", "apprentissage machine"),
    "Deep Learning": ("dl", "deep learning", "apprentissage profond", "deep neural network"),
    "NLP": ("nlp", "natural language processing", "traitement du langage naturel", "traitement automatique du langage"),
    "MLOps": ("mlops", "machine learning operations", "m.l. ops", "m l ops"),
    "Other": ("other", "autre", "misc"),
}

SKILL_TAXONOMY: list[dict[str, Any]] = [
    {
        "family": "Machine Learning",
        "id": "linear_regression",
        "label": "Régression linéaire",
        "type": "METHOD",
        "aliases": ["regression lineaire", "linear regression", "régression linéaire"],
    },
    {
        "family": "Machine Learning",
        "id": "logistic_regression",
        "label": "Régression logistique",
        "type": "METHOD",
        "aliases": ["regression logistique", "logistic regression", "régression logistique"],
    },
    {
        "family": "Machine Learning",
        "id": "svm",
        "label": "SVM",
        "type": "METHOD",
        "aliases": ["svm", "support vector machine", "support vector machines"],
    },
    {
        "family": "Machine Learning",
        "id": "decision_tree",
        "label": "Arbres de décision",
        "type": "METHOD",
        "aliases": ["arbre de decision", "arbre de décision", "decision tree", "decision trees"],
    },
    {
        "family": "Machine Learning",
        "id": "random_forest",
        "label": "Random Forest",
        "type": "METHOD",
        "aliases": ["random forest", "randomforest"],
    },
    {
        "family": "Machine Learning",
        "id": "xgboost",
        "label": "XGBoost",
        "type": "METHOD",
        "aliases": ["xgboost", "xg boost", "extreme gradient boosting"],
    },
    {
        "family": "Machine Learning",
        "id": "cross_validation",
        "label": "Validation croisée",
        "type": "METHOD",
        "aliases": ["validation croisee", "cross validation", "cross-validation"],
    },
    {
        "family": "Machine Learning",
        "id": "kmeans",
        "label": "K-Means",
        "type": "METHOD",
        "aliases": ["k means", "k-means", "kmeans"],
    },
    {
        "family": "Machine Learning",
        "id": "dbscan",
        "label": "DBSCAN",
        "type": "METHOD",
        "aliases": ["dbscan"],
    },
    {
        "family": "Machine Learning",
        "id": "pca",
        "label": "PCA",
        "type": "METHOD",
        "aliases": ["pca", "principal component analysis", "analyse en composantes principales"],
    },
    {
        "family": "Deep Learning",
        "id": "neural_networks",
        "label": "Réseaux de neurones",
        "type": "METHOD",
        "aliases": ["reseaux de neurones", "réseaux neuronaux", "network neural", "neural network", "neural networks"],
    },
    {
        "family": "Deep Learning",
        "id": "cnn",
        "label": "CNN",
        "type": "METHOD",
        "aliases": ["cnn", "convolutional neural network", "convolutional neural networks"],
    },
    {
        "family": "Deep Learning",
        "id": "rnn",
        "label": "RNN",
        "type": "METHOD",
        "aliases": ["rnn", "recurrent neural network", "recurrent neural networks"],
    },
    {
        "family": "Deep Learning",
        "id": "lstm",
        "label": "LSTM",
        "type": "METHOD",
        "aliases": ["lstm", "long short-term memory", "long short term memory"],
    },
    {
        "family": "Deep Learning",
        "id": "transformers",
        "label": "Transformers",
        "type": "METHOD",
        "aliases": ["transformer", "transformers", "transformer architecture"],
    },
    {
        "family": "Deep Learning",
        "id": "transfer_learning",
        "label": "Transfer Learning",
        "type": "METHOD",
        "aliases": ["transfer learning", "apprentissage par transfert", "transfert learning"],
    },
    {
        "family": "Deep Learning",
        "id": "tensorflow",
        "label": "TensorFlow",
        "type": "TOOL",
        "aliases": ["tensorflow", "tensor flow"],
    },
    {
        "family": "Deep Learning",
        "id": "keras",
        "label": "Keras",
        "type": "TOOL",
        "aliases": ["keras"],
    },
    {
        "family": "Deep Learning",
        "id": "pytorch",
        "label": "PyTorch",
        "type": "TOOL",
        "aliases": ["pytorch", "py torch"],
    },
    {
        "family": "NLP",
        "id": "tokenisation",
        "label": "Tokenisation",
        "type": "METHOD",
        "aliases": ["tokenisation", "tokenization", "tokeniser", "tokenizer"],
    },
    {
        "family": "NLP",
        "id": "embeddings",
        "label": "Embeddings",
        "type": "SKILL",
        "aliases": ["embedding", "embeddings", "représentations vectorielles"],
    },
    {
        "family": "NLP",
        "id": "word2vec",
        "label": "Word2Vec",
        "type": "METHOD",
        "aliases": ["word2vec", "word2vec model"],
    },
    {
        "family": "NLP",
        "id": "bert",
        "label": "BERT",
        "type": "METHOD",
        "aliases": ["bert", "bidirectional encoder representations from transformers"],
    },
    {
        "family": "NLP",
        "id": "gpt",
        "label": "GPT",
        "type": "METHOD",
        "aliases": ["gpt", "generative pre-trained transformer", "generative pretrained transformer"],
    },
    {
        "family": "NLP",
        "id": "text_classification",
        "label": "Classification de texte",
        "type": "METHOD",
        "aliases": ["classification de texte", "text classification", "classification textuelle"],
    },
    {
        "family": "NLP",
        "id": "sentiment_analysis",
        "label": "Analyse de sentiment",
        "type": "METHOD",
        "aliases": ["analyse de sentiment", "sentiment analysis", "analyse des sentiments"],
    },
    {
        "family": "MLOps",
        "id": "git",
        "label": "Git",
        "type": "TOOL",
        "aliases": ["git"],
    },
    {
        "family": "MLOps",
        "id": "docker",
        "label": "Docker",
        "type": "TOOL",
        "aliases": ["docker"],
    },
    {
        "family": "MLOps",
        "id": "mlflow",
        "label": "MLflow",
        "type": "TOOL",
        "aliases": ["mlflow", "ml flow"],
    },
    {
        "family": "MLOps",
        "id": "fastapi",
        "label": "FastAPI",
        "type": "TOOL",
        "aliases": ["fastapi", "fast api"],
    },
    {
        "family": "MLOps",
        "id": "cicd",
        "label": "CI/CD",
        "type": "TOOL",
        "aliases": ["ci/cd", "cicd", "continuous integration", "continuous delivery"],
    },
    {
        "family": "MLOps",
        "id": "aws_sagemaker",
        "label": "AWS SageMaker",
        "type": "TOOL",
        "aliases": ["aws sagemaker", "sagemaker", "amazon sagemaker"],
    },
    {
        "family": "MLOps",
        "id": "gcp_vertex_ai",
        "label": "GCP Vertex AI",
        "type": "TOOL",
        "aliases": ["gcp vertex ai", "vertex ai", "google vertex ai"],
    },
]

NEGATIVE_HINTS = (
    "prix",
    "tarif",
    "durée",
    "duree",
    "public",
    "certification",
    "organisme",
    "footer",
    "copyright",
    "marketing",
    "inscription",
    "taux de satisfaction",
)


def _normalized_alias(value: Any) -> str:
    return normalize_for_match(clean_text(value))


def _normalize_with_mapping(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    mapping: list[int] = []
    for index, char in enumerate(clean_text(text)):
        decomposed = unicodedata.normalize("NFKD", char)
        stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
        for item in stripped.lower():
            if item.isalnum():
                normalized_chars.append(item)
                mapping.append(index)
            else:
                if normalized_chars and normalized_chars[-1] != " ":
                    normalized_chars.append(" ")
                    mapping.append(index)
    normalized = re.sub(r"\s+", " ", "".join(normalized_chars)).strip()
    compact_mapping: list[int] = []
    normalized_index = 0
    for char in normalized:
        if char == " ":
            compact_mapping.append(mapping[min(normalized_index, len(mapping) - 1)] if mapping else 0)
        else:
            compact_mapping.append(mapping[min(normalized_index, len(mapping) - 1)] if mapping else 0)
        normalized_index += 1
    return normalized, compact_mapping


def canonical_entity_type(value: str | None) -> str:
    label = clean_text(value).upper()
    return label if label in ENTITY_TYPES else "OTHER"


def canonical_family_label(value: str | None) -> str:
    normalized = _normalized_alias(value)
    if not normalized:
        return "Other"
    for family, aliases in FAMILY_ALIASES.items():
        if normalized == _normalized_alias(family):
            return family
        for alias in aliases:
            if normalized == _normalized_alias(alias):
                return family
    return "Other"


@lru_cache(maxsize=1)
def family_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        index[family] = {
            "label": family,
            "parent": FAMILY_HIERARCHY[family],
            "aliases": list(FAMILY_ALIASES.get(family, ())),
            "skills": [],
        }
    for item in SKILL_TAXONOMY:
        family = item["family"]
        index[family]["skills"].append(
            {
                "id": item["id"],
                "label": item["label"],
                "type": item["type"],
                "aliases": list(item.get("aliases", [])),
                "canonical_family": family,
                "active": True,
            }
        )
    return index


def build_taxonomy() -> dict[str, Any]:
    families = []
    for family in FAMILIES:
        family_entry = family_index()[family]
        families.append(
            {
                "id": _normalized_alias(family).replace(" ", "_") or family.lower().replace(" ", "_"),
                "label": family,
                "parent_id": FAMILY_HIERARCHY[family].lower().replace(" ", "_") if FAMILY_HIERARCHY[family] else None,
                "aliases": family_entry["aliases"],
                "skills": family_entry["skills"],
            }
        )
    return {
        "version": TAXONOMY_VERSION,
        "families": families,
        "entity_types": list(ENTITY_TYPES),
        "meta": {
            "source_18_labels_mapping": {
                "Machine Learning": "Machine Learning",
                "Deep Learning": "Deep Learning",
                "NLP": "NLP",
                "MLOps": "MLOps",
                "Other": "Other",
            },
            "activation_criteria": {
                "approved_annotations_required": True,
                "document_split": "70/15/15",
                "human_validation_required": True,
            },
        },
    }


def canonicalize_term(value: str | None) -> tuple[str, str, str]:
    normalized = _normalized_alias(value)
    if not normalized:
        return "", "Other", "OTHER"
    for item in SKILL_TAXONOMY:
        label_norm = _normalized_alias(item["label"])
        if normalized == label_norm:
            return item["label"], item["family"], item["type"]
        for alias in item.get("aliases", []):
            if normalized == _normalized_alias(alias):
                return item["label"], item["family"], item["type"]
    family = canonical_family_label(value)
    if family != "Other":
        return family, family, "DOMAIN"
    return clean_text(value), "Other", "OTHER"


def alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for family, aliases in FAMILY_ALIASES.items():
        mapping[_normalized_alias(family)] = family
        for alias in aliases:
            mapping[_normalized_alias(alias)] = family
    for item in SKILL_TAXONOMY:
        mapping[_normalized_alias(item["label"])] = item["label"]
        for alias in item.get("aliases", []):
            mapping[_normalized_alias(alias)] = item["label"]
    return mapping


def iter_all_terms() -> Iterable[dict[str, Any]]:
    for family in FAMILIES:
        parent = FAMILY_HIERARCHY[family]
        for item in SKILL_TAXONOMY:
            if item["family"] == family:
                yield {
                    "family": family,
                    "parent_family": parent,
                    **item,
                }


def dedupe_mentions(mentions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for mention in mentions:
        start = int(mention.get("start", 0))
        end = int(mention.get("end", 0))
        canonical = normalize_for_match(mention.get("canonical_name") or mention.get("text") or "")
        if not canonical or end <= start:
            continue
        key = (start, end, canonical)
        current = best_by_key.get(key)
        if current is None or float(mention.get("confidence", 0.0) or 0.0) > float(current.get("confidence", 0.0) or 0.0):
            best_by_key[key] = dict(mention)
    return sorted(best_by_key.values(), key=lambda item: (int(item.get("start", 0)), -(int(item.get("end", 0)) - int(item.get("start", 0))), clean_text(item.get("canonical_name") or "")))


def find_mentions(text: str) -> list[dict[str, Any]]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    normalized, mapping = _normalize_with_mapping(cleaned)
    if not normalized:
        return []
    mentions: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    family_items = []
    for family in FAMILIES:
        family_items.append({
            'family': family,
            'parent_family': FAMILY_HIERARCHY[family],
            'label': family,
            'aliases': list(FAMILY_ALIASES.get(family, ())),
            'type': 'DOMAIN',
        })
    items = sorted([*family_items, *list(iter_all_terms())], key=lambda item: (-len(_normalized_alias(item["label"])), item["label"]))
    for item in items:
        aliases = [item["label"], *item.get("aliases", [])]
        for alias in aliases:
            alias_norm = _normalized_alias(alias)
            if not alias_norm:
                continue
            start_idx = 0
            while True:
                idx = normalized.find(alias_norm, start_idx)
                if idx < 0:
                    break
                end_idx = idx + len(alias_norm)
                before = normalized[idx - 1] if idx > 0 else ' '
                after = normalized[end_idx] if end_idx < len(normalized) else ' '
                if before.isalnum() or after.isalnum():
                    start_idx = idx + 1
                    continue
                if idx >= len(mapping) or end_idx <= 0:
                    start_idx = idx + 1
                    continue
                start = mapping[idx]
                end = mapping[min(end_idx - 1, len(mapping) - 1)] + 1
                key = (start, end, item["label"])
                if key not in seen:
                    seen.add(key)
                    mentions.append(
                        {
                            "start": start,
                            "end": end,
                            "text": cleaned[start:end],
                            "canonical_name": item["label"],
                            "family": item["family"],
                            "entity_type": item["type"],
                            "parent_family": item["parent_family"],
                            "alias": alias,
                        }
                    )
                start_idx = idx + len(alias_norm)
    return dedupe_mentions(mentions)


def infer_families(text: str) -> list[str]:
    families: list[str] = []
    for mention in find_mentions(text):
        family = mention["family"]
        if family not in families:
            families.append(family)
        parent = mention.get("parent_family")
        if parent and parent not in families:
            families.append(parent)
    if not families and any(hint in normalize_for_match(text) for hint in NEGATIVE_HINTS):
        return ["Other"]
    return families or ["Other"]


def negative_hint_score(text: str) -> int:
    normalized = normalize_for_match(text)
    return sum(1 for hint in NEGATIVE_HINTS if hint in normalized)


def section_for_text(text: str) -> str:
    normalized = normalize_for_match(text)
    if not normalized:
        return "OTHER"
    if any(token in normalized for token in ("programme", "module", "contenu", "curriculum")):
        return "PROGRAM"
    if any(token in normalized for token in ("competence", "compétence", "skills", "outils", "technologies")):
        return "SKILLS"
    if any(token in normalized for token in ("objectif", "objectifs")):
        return "OBJECTIVES"
    if any(token in normalized for token in ("prix", "tarif", "duree", "durée")):
        return "PRICE"
    return "OTHER"


def normalize_canonical_name(value: str | None) -> str:
    canonical, _, _ = canonicalize_term(value)
    return canonical or clean_text(value)

