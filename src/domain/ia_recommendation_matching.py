from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.text import normalize_for_match
from data_sources.ia_recommendations import normalize_recommendation_keyword
from domain.models import IARecommendation, IARecommendationMatch

CONFIDENCE_HIGH = 'HIGH'
CONFIDENCE_MEDIUM = 'MEDIUM'
CONFIDENCE_LOW = 'LOW'
CONFIDENCE_DEFAULT = 'DEFAULT'

METHOD_EXACT = 'EXACT'
METHOD_ALIAS = 'ALIAS'
METHOD_INCLUSION = 'INCLUSION'
METHOD_EMBEDDING = 'EMBEDDING'
METHOD_CLUSTER = 'CLUSTER'
METHOD_DEFAULT = 'DEFAULT'

FRENCH_STOP_WORDS = frozenset({
    'le', 'la', 'les', 'de', 'des', 'du', 'et', 'un', 'une', 'en', 'au', 'aux',
    'ce', 'ces', 'cet', 'cette', 'par', 'pour', 'sur', 'dans', 'avec', 'est',
    'sont', 'que', 'qui', 'pas', 'ne', 'ni', 'ou', 'mais', 'se', 'sa', 'son',
    'ses', 'leur', 'leurs', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
    'plus', 'tres', 'peu', 'tout', 'tous', 'toute', 'toutes', 'comme', 'faire',
    'fait', 'peut', 'avoir', 'etre', 'apres', 'avant', 'sans', 'sous',
})


def _match_method_label(method: str) -> str:
    labels = {
        'EXACT': 'Correspondance exacte',
        'ALIAS': 'Correspondance par alias',
        'INCLUSION': 'Inclusion prudente',
        'EMBEDDING': 'Similarite semantique',
        'CLUSTER': 'Cluster',
        'DEFAULT': 'Regle par defaut',
    }
    return labels.get(method, method)


def _is_significant(w: str) -> bool:
    return len(w) >= 3 and w not in FRENCH_STOP_WORDS


def _phrase_in_text(phrase: str, text: str) -> bool:
    return f' {phrase} ' in f' {text} '


def _get_significant_words(text: str) -> list[str]:
    return [w for w in text.split() if _is_significant(w)]


def _significant_word_overlap_ratio(keyword_words: list[str], skill_words: list[str]) -> float:
    if not keyword_words or not skill_words:
        return 0.0
    kw_set = set(keyword_words)
    skill_set = set(skill_words)
    overlap = len(kw_set & skill_set)
    return overlap / len(kw_set)


@dataclass
class RecommendationIndex:
    phrase_index: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    significant_word_index: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    default_rules: list[dict[str, Any]] = field(default_factory=list)
    all_recommendations: list[dict[str, Any]] = field(default_factory=list)
    keywords_with_aliases: list[dict[str, Any]] = field(default_factory=list)
    embedding_model: Any = None
    keyword_embeddings: dict[str, list[float]] = field(default_factory=dict)


def build_recommendation_index(
    recommendations: list[dict[str, Any]],
    *,
    embedding_model=None,
) -> RecommendationIndex:
    index = RecommendationIndex()
    for rec in recommendations:
        if not rec.get('is_active', True):
            continue
        if rec.get('is_default'):
            index.default_rules.append(rec)
            continue
        norm = rec.get('keyword_normalized', '')
        if not norm:
            continue

        item = {
            'keyword': rec.get('keyword', ''),
            'keyword_normalized': norm,
            'aliases': rec.get('aliases', []),
            'aliases_normalized': rec.get('aliases_normalized', []),
            'recommendation': rec.get('recommendation', ''),
            'recommendation_id': rec.get('recommendation_id', ''),
        }

        index.phrase_index.setdefault(norm, []).append(item)

        for alias_norm in item['aliases_normalized']:
            if alias_norm:
                index.phrase_index.setdefault(alias_norm, []).append(item)

        sig_words = _get_significant_words(norm)
        for sw in sig_words:
            index.significant_word_index.setdefault(sw, []).append(item)

        index.keywords_with_aliases.append(item)
        index.all_recommendations.append(rec)

    if embedding_model is not None:
        texts = [rec.get('keyword_normalized', '') for rec in index.all_recommendations if rec.get('keyword_normalized')]
        if texts:
            import numpy as np
            embeddings = embedding_model.encode(texts, convert_to_numpy=True)
            for rec, emb in zip(index.all_recommendations, embeddings):
                kw = rec.get('keyword_normalized', '')
                if kw:
                    index.keyword_embeddings[kw] = emb.tolist()
    return index


def match_ia_recommendations(
    skills: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    *,
    embedding_model=None,
    exact_threshold: float = 1.0,
    semantic_threshold: float = 0.82,
    review_threshold: float = 0.70,
    max_recommendations_per_skill: int = 3,
) -> list[IARecommendationMatch]:
    if not skills or not recommendations:
        return []
    index = build_recommendation_index(recommendations, embedding_model=embedding_model)
    seen_recs: set[str] = set()
    matches: list[IARecommendationMatch] = []

    for skill in skills:
        skill_orig = skill.get('name') or skill.get('text') or ''
        skill_norm_str = skill.get('normalized_name') or normalize_for_match(skill_orig) or ''
        skill_norm = normalize_recommendation_keyword(skill_orig)
        skill_matches: list[IARecommendationMatch] = []
        used_recs: set[str] = set()

        def add_match(matched_kw: str, rec: dict, score: float, method: str):
            dedup_key = f'{skill_norm}::{rec.get("recommendation", "")}'
            if dedup_key in seen_recs:
                return
            rec_id = rec.get('recommendation_id', '') + '::' + matched_kw
            if rec_id in used_recs:
                return
            if score >= 1.0:
                conf = CONFIDENCE_HIGH
            elif score >= semantic_threshold:
                conf = CONFIDENCE_MEDIUM
            elif score >= review_threshold:
                conf = CONFIDENCE_LOW
            else:
                conf = CONFIDENCE_DEFAULT
            match = IARecommendationMatch(
                skill_original=skill_orig,
                skill_normalized=skill_norm_str,
                matched_keyword=matched_kw,
                recommendation=rec.get('recommendation', ''),
                score=round(score, 4),
                match_method=method,
                confidence_label=conf,
            )
            skill_matches.append(match)
            used_recs.add(rec_id)

        # 1. Exact / alias phrase match
        if not skill_matches:
            for keyword_norm, items in index.phrase_index.items():
                if _phrase_in_text(keyword_norm, skill_norm):
                    for item in items:
                        score = 1.0 if keyword_norm == item['keyword_normalized'] else 0.85
                        method = METHOD_EXACT if keyword_norm == item['keyword_normalized'] else METHOD_ALIAS
                        add_match(keyword_norm, item, score, method)

        # 2. Inclusion match on significant words
        if not skill_matches:
            skill_sig = _get_significant_words(skill_norm)
            if skill_sig:
                candidates: dict[str, list[dict]] = {}
                for item in index.keywords_with_aliases:
                    kw_sig = _get_significant_words(item['keyword_normalized'])
                    ratio = _significant_word_overlap_ratio(kw_sig, skill_sig)
                    if ratio >= 0.5:
                        kw = item['keyword_normalized']
                        if kw not in candidates:
                            candidates[kw] = []
                        candidates[kw].append(item)

                for kw, items in candidates.items():
                    for item in items:
                        kw_sig = _get_significant_words(item['keyword_normalized'])
                        overlap_count = len(set(kw_sig) & set(skill_sig))
                        total_kw = max(len(kw_sig), 1)
                        score = 0.80 + 0.05 * (overlap_count / total_kw)
                        add_match(kw, item, min(score, 0.95), METHOD_INCLUSION)

        # 3. Embedding match
        if not skill_matches and embedding_model is not None and index.keyword_embeddings:
            try:
                import numpy as np
                from sklearn.metrics.pairwise import cosine_similarity
                skill_emb = embedding_model.encode([skill_norm], convert_to_numpy=True)
                best_score = 0.0
                best_items = []
                for kw, kw_emb in index.keyword_embeddings.items():
                    kw_emb_arr = np.array(kw_emb).reshape(1, -1)
                    sim = float(cosine_similarity(skill_emb, kw_emb_arr)[0][0])
                    if sim > best_score:
                        best_score = sim
                        best_items = [kw]
                    elif sim == best_score and sim > 0:
                        best_items.append(kw)
                if best_items and best_score >= review_threshold:
                    for kw in best_items:
                        items = index.phrase_index.get(kw, [])
                        for item in items:
                            add_match(kw, item, best_score, METHOD_EMBEDDING)
            except Exception:
                pass

        # 4. Default rule
        if not skill_matches and index.default_rules:
            for rec in index.default_rules:
                add_match('(regle par defaut)', rec, 0.0, METHOD_DEFAULT)

        skill_matches.sort(key=lambda m: m.score, reverse=True)
        matches.extend(skill_matches[:max_recommendations_per_skill])

        for m in skill_matches[:max_recommendations_per_skill]:
            seen_recs.add(f'{skill_norm}::{m.recommendation}')

    return matches
