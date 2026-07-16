from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from common.text import clean_text, normalize_for_match
from deepforma.cpf.embeddings import build_encoder

from .models import AIRecommendationCategoryMapping, AIRecommendationRuleCategory
from .normalizer import normalize_ai_keyword

TAXONOMY_PATH = Path(__file__).resolve().parents[2] / 'data' / 'referentials' / 'ai_skill_taxonomy.json'
DEFAULT_EMBEDDING_MODEL = Path(__file__).resolve().parents[2] / 'models' / 'cpf-recommender' / 'final'


@lru_cache(maxsize=4)
def load_taxonomy(path: str | Path = TAXONOMY_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Taxonomie IA invalide: {path}')
    return payload


@lru_cache(maxsize=1)
def _taxonomy_labels(path: str | Path = TAXONOMY_PATH) -> list[dict[str, Any]]:
    taxonomy = load_taxonomy(path)
    labels = taxonomy.get('labels') or []
    result: list[dict[str, Any]] = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        aliases = [clean_text(alias) for alias in item.get('aliases', []) if clean_text(alias)]
        result.append({
            'id': clean_text(item.get('id')),
            'label': clean_text(item.get('label')),
            'description': clean_text(item.get('description')),
            'aliases': aliases,
            'normalized': normalize_for_match(clean_text(item.get('label'))),
            'normalized_aliases': [normalize_for_match(alias) for alias in aliases],
        })
    return result


def _load_encoder(embedding_model: str | Path | None = None):
    model_path = Path(embedding_model or DEFAULT_EMBEDDING_MODEL)
    if not model_path.exists():
        return None
    try:
        return build_encoder(str(model_path))
    except Exception:
        return None


def _semantic_scores(text: str, *, embedding_model: str | Path | None = None) -> list[tuple[dict[str, Any], float]]:
    encoder = _load_encoder(embedding_model)
    labels = _taxonomy_labels()
    if encoder is None or not labels:
        return []
    try:
        texts = [text, *[f"{label['label']} {label['description']} {' '.join(label['aliases'])}".strip() for label in labels]]
        embeddings = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        query = np.asarray(embeddings[0], dtype=float)
        label_vectors = np.asarray(embeddings[1:], dtype=float)
        sims = label_vectors @ query
        ranked = sorted(zip(labels, sims.tolist()), key=lambda item: item[1], reverse=True)
        return [(label, float(score)) for label, score in ranked]
    except Exception:
        return []


def _rule_category_from_label(label: dict[str, Any], *, score: float, method: str, status: str) -> AIRecommendationRuleCategory:
    return AIRecommendationRuleCategory(label=label['label'], score=round(float(score), 4), method=method, status=status)


def map_rule_categories(keyword: str, recommendation: str, *, embedding_model: str | Path | None = None) -> list[AIRecommendationRuleCategory]:
    text = f"{clean_text(keyword)} {clean_text(recommendation)}".strip()
    if not text:
        return []
    normalized = normalize_ai_keyword(text)
    labels = _taxonomy_labels()
    matched: list[AIRecommendationRuleCategory] = []
    seen: set[str] = set()

    for label in labels:
        if label['normalized'] and label['normalized'] in normalized:
            matched.append(_rule_category_from_label(label, score=1.0, method='exact', status='accepted'))
            seen.add(label['label'])
            continue
        for alias in label['normalized_aliases']:
            if alias and alias in normalized:
                matched.append(_rule_category_from_label(label, score=0.96, method='alias', status='accepted'))
                seen.add(label['label'])
                break

    if matched:
        return matched

    semantic = _semantic_scores(text, embedding_model=embedding_model)
    for label, score in semantic[:3]:
        if not label.get('label') or label['label'] in seen:
            continue
        if score >= 0.88:
            status = 'accepted'
        elif score >= 0.82:
            status = 'to_review'
        else:
            status = 'rejected'
        matched.append(_rule_category_from_label(label, score=score, method='semantic', status=status))
    return matched
