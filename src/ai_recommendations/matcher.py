from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from common.text import clean_text, normalize_for_match, split_multi_values

from .fusion import AIRecommendationSourceScore, fuse_ai_recommendation_scores
from .loader import DEFAULT_RULE_TEXT, SOURCE_FILENAME, load_ai_recommendation_rules
from .models import AIRecommendationRuleMatch, AIRecommendationRuleMatchEvidence
from .normalizer import normalize_ai_keyword

WORD_BOUNDARY_CHARS = r'[^\wÀ-ÿ]'
DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / 'data' / 'indexes' / 'ai_recommendations'
DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / 'data' / 'referentials' / 'ai_recommendation_rules.json'
DEFAULT_EMBEDDING_MODEL = Path(__file__).resolve().parents[2] / 'models' / 'cpf-recommender' / 'final'
DEFAULT_RECOMMENDATION = "Acculturation à l'IA et découverte de l'IA agentique"
DEFAULT_RULE_KEYWORD = '(aucune mention IA dans le référentiel)'
DEFAULT_RULE_NORMALIZED = normalize_ai_keyword(DEFAULT_RULE_KEYWORD)


def _token_set(text: str) -> set[str]:
    return {token for token in normalize_for_match(text).split() if token}


def _phrase_match(needle: str, haystack: str) -> bool:
    if not needle or not haystack:
        return False
    n = normalize_for_match(needle)
    h = normalize_for_match(haystack)
    if not n or not h:
        return False
    if n == h:
        return True
    return f' {n} ' in f' {h} '


def _load_rules(rules_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(rules_path or DEFAULT_RULES_PATH)
    if path.suffix.lower() == '.json' and path.exists():
        return load_ai_recommendation_rules(path)
    if path.exists():
        from .loader import load_ai_recommendation_rules_csv
        rules, _, _ = load_ai_recommendation_rules_csv(path)
        return rules
    return []


def _score_to_status(score: float) -> str:
    if score >= 0.88:
        return 'reliable'
    if score >= 0.82:
        return 'to_review'
    return 'rejected'


def _rule_evidence(rule: dict[str, Any], text: str, score: float) -> list[AIRecommendationRuleMatchEvidence]:
    matched = clean_text(text)
    return [AIRecommendationRuleMatchEvidence(text=matched[:120], rule_id=str(rule.get('id', '')), similarity=round(score, 4))]


def _prepare_text(*parts: Any) -> str:
    values: list[str] = []
    for part in parts:
        if isinstance(part, (list, tuple, set)):
            values.extend(clean_text(item) for item in part if clean_text(item))
        else:
            item = clean_text(part)
            if item:
                values.append(item)
    return ' '.join(values)


def _semantic_similarity(query: str, rule_text: str, *, embedding_model: str | Path | None = None) -> float:
    model_path = Path(embedding_model or DEFAULT_EMBEDDING_MODEL)
    if not model_path.exists():
        return 0.0
    try:
        from deepforma.cpf.embeddings import build_encoder
        encoder = build_encoder(str(model_path))
        vectors = encoder.encode([query, rule_text], normalize_embeddings=True, convert_to_numpy=True)
        return float(np.asarray(vectors[0], dtype=float) @ np.asarray(vectors[1], dtype=float))
    except Exception:
        return 0.0


def _best_rule_scores(query: str, rules: list[dict[str, Any]], *, embedding_model: str | Path | None = None) -> list[dict[str, Any]]:
    query_norm = normalize_ai_keyword(query)
    query_tokens = _token_set(query)
    ranked: list[dict[str, Any]] = []
    for rule in rules:
        if not rule.get('enabled', True):
            continue
        kw = clean_text(rule.get('keyword', ''))
        kw_norm = clean_text(rule.get('normalized_keyword', '')) or normalize_ai_keyword(kw)
        if not kw:
            continue
        exact = query_norm == kw_norm or _phrase_match(kw_norm, query_norm)
        lexical = False
        if not exact:
            kw_tokens = _token_set(kw)
            overlap = len(query_tokens & kw_tokens)
            lexical = bool(overlap and overlap / max(len(kw_tokens), 1) >= 0.6)
        semantic = 0.0
        if not exact and not lexical:
            semantic = _semantic_similarity(query, f"{kw} {clean_text(rule.get('recommendation', ''))}", embedding_model=embedding_model)
        if exact or lexical or semantic >= 0.82:
            score = 1.0 if exact else (0.92 if lexical else semantic)
            ranked.append({
                'rule': rule,
                'score': round(float(score), 4),
                'match_method': 'exact' if exact else 'lexical' if lexical else 'semantic',
                'matched_text': kw if exact else query[:120],
                'status': _score_to_status(score),
                'sources': ['exact_rule'] if exact else ['lexical_rule'] if lexical else ['semantic_rule'],
                'evidence': _rule_evidence(rule, query, score),
            })
    ranked.sort(key=lambda item: item['score'], reverse=True)
    return ranked


def match_ai_recommendations(
    *,
    referential_title: str | None = None,
    activities: list[str] | None = None,
    official_skills: list[str] | None = None,
    subskills: list[str] | None = None,
    full_text: str | None = None,
    rules_path: str | Path | None = None,
    embedding_model: str | Path | None = None,
    model_score_std: float | None = None,
    model_mean_score: float | None = None,
    model_non_discriminant: bool = False,
) -> dict[str, Any]:
    rules = _load_rules(rules_path)
    if not rules:
        return {'input_text': clean_text(full_text or referential_title or ''), 'detected_categories': [], 'recommendations': [], 'default_recommendation_applied': False}

    query = _prepare_text(referential_title, activities or [], official_skills or [], subskills or [], full_text)
    if not query:
        query = clean_text(referential_title or '')

    candidate_scores = _best_rule_scores(query, rules, embedding_model=embedding_model)
    recommendations: list[dict[str, Any]] = []
    category_scores: dict[str, list[AIRecommendationSourceScore]] = {}
    categories_by_rule: dict[str, list[dict[str, Any]]] = {}
    rule_by_id = {str(rule.get('id', '')): rule for rule in rules}

    for item in candidate_scores[:10]:
        rule = item['rule']
        rule_id = str(rule.get('id', ''))
        rule_categories = [cat for cat in (rule.get('categories') or []) if isinstance(cat, dict)]
        categories_by_rule[rule_id] = rule_categories
        recommendations.append({
            'rule_id': rule_id,
            'keyword': rule.get('keyword', ''),
            'recommendation': rule.get('recommendation', ''),
            'score': item['score'],
            'match_method': item['match_method'],
            'matched_text': item['matched_text'],
            'sources': item['sources'],
            'evidence': [asdict(ev) for ev in item['evidence']],
            'status': item['status'],
        })
        for category in rule_categories:
            label = clean_text(category.get('label') or '')
            if not label:
                continue
            category_scores.setdefault(label, []).append(
                AIRecommendationSourceScore(source=item['sources'][0], score=float(category.get('score', item['score'])), details={'rule_id': rule_id})
            )

    detected_categories: list[dict[str, Any]] = []
    for label, scores in category_scores.items():
        fused = fuse_ai_recommendation_scores(scores, model_score_std=model_score_std, model_mean_score=model_mean_score, model_non_discriminant=model_non_discriminant)
        status = 'reliable' if fused['score'] >= 0.88 else 'to_review' if fused['score'] >= 0.82 else 'rejected'
        if status != 'rejected':
            evidence = []
            for rec in recommendations:
                rule_categories = categories_by_rule.get(rec['rule_id'], [])
                if any(clean_text(cat.get('label') or '') == label for cat in rule_categories):
                    evidence.append({'text': rec['matched_text'], 'rule_id': rec['rule_id'], 'similarity': rec['score']})
            detected_categories.append({
                'label': label,
                'score': fused['score'],
                'status': status,
                'sources': [item['source'] for item in fused['contributions']],
                'evidence': evidence,
            })

    default_applied = False
    if not detected_categories and not recommendations:
        default_rule = next((rule for rule in rules if rule.get('is_default')), None)
        if default_rule:
            default_applied = True
            recommendations.append({
                'rule_id': default_rule.get('id', ''),
                'keyword': default_rule.get('keyword', DEFAULT_RULE_KEYWORD),
                'recommendation': default_rule.get('recommendation', DEFAULT_RECOMMENDATION),
                'score': 0.4,
                'match_method': 'default',
                'matched_text': query[:120],
                'sources': ['manually_validated_rule'],
                'evidence': [],
                'status': 'to_review',
            })

    return {
        'input_text': query,
        'detected_categories': detected_categories,
        'recommendations': recommendations,
        'default_recommendation_applied': default_applied,
    }
