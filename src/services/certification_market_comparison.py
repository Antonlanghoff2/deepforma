from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match, stable_hash
from france_travail.client import FranceTravailClient, SearchCriteria
from france_travail.normalizer import normalize_offer
from referentials.ai_certification_referential import AICertificationReferential
from referential_learning.ai_certification_taxonomy import infer_skill_taxonomy, normalize_market_alias
from skill_extraction.ai_certification_extractor import AICertificationSkillExtractor
from skills.merge_offer_skills import extract_skills_from_text, merge_offer_skills

DEFAULT_REFERENTIAL_PATH = Path(
    __import__('os').getenv('AI_CERTIFICATION_REFERENTIAL_PATH', 'data/referentials/ai_engineer_certification_2025.json')
)
DEFAULT_EMBEDDING_MODEL = __import__('os').getenv('AI_CERT_SKILL_EMBEDDING_MODEL', '').strip() or None
DEFAULT_JOB_TITLES = [
    'ingénieur intelligence artificielle',
    'AI Engineer',
    'Machine Learning Engineer',
    'Data Scientist',
    'MLOps Engineer',
    'ingénieur Machine Learning',
    'ingénieur NLP',
    'ingénieur Deep Learning',
    'ingénieur IA générative',
    'Data Engineer IA',
    'chef de projet IA',
]
DEFAULT_ROME_CODES = ['M1805']
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+|[•·;]+')

SOURCE_WEIGHTS = {
    'structured': 1.0,
    'france_travail_structured': 1.0,
    'ai_cert_extractor': 0.95,
    'text_explicit': 0.8,
    'general_text': 0.75,
    'model_prediction': 0.65,
    'competences': 0.9,
}
PRIORITY = {'exact': 4, 'alias': 3, 'semantic': 2, 'implicit': 1, 'unmapped': 0}


@dataclass(frozen=True, slots=True)
class SkillMarketRow:
    referential_id: str = ''
    block: str = ''
    block_name: str = ''
    activity: str = ''
    code: str = ''
    label: str = ''
    official_description: str = ''
    normalized_label: str = ''
    category: str = ''
    subcategory: str = ''
    technical_keywords: list[str] = field(default_factory=list)
    source_page: int = 0
    offer_count: int = 0
    weighted_score: float = 0.0
    share_percent: float = 0.0
    coverage_score: float = 0.0
    match_methods: dict[str, int] = field(default_factory=dict)
    evidence_examples: list[str] = field(default_factory=list)
    example_offers: list[str] = field(default_factory=list)
    status: str = 'missing'

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['technical_keywords'] = list(self.technical_keywords)
        payload['evidence_examples'] = list(self.evidence_examples)
        payload['example_offers'] = list(self.example_offers)
        payload['match_methods'] = dict(self.match_methods)
        return payload


@dataclass(frozen=True, slots=True)
class BlockCoverageRow:
    block: str
    block_name: str
    skill_count: int
    covered_skill_count: int
    total_weight: float
    covered_weight: float
    market_share_percent: float
    average_skill_share_percent: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OfferExampleRow:
    offer_id: str
    title: str
    location_label: str
    contract_label: str
    rome_code: str
    rome_label: str
    date: str
    matches: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['matches'] = list(self.matches)
        payload['unmatched'] = list(self.unmatched)
        return payload


@dataclass
class CertificationMarketComparisonReport:
    referential_id: str
    referential_title: str
    referential_version: str
    territory: str
    radius_km: int | None
    period: str
    job_titles: list[str]
    rome_codes: list[str]
    date_min: str | None
    date_max: str | None
    offer_count: int
    analyzed_offer_count: int
    covered_offer_count: int
    total_market_weight: float
    covered_weight: float
    global_coverage_score: float
    block_summaries: list[BlockCoverageRow] = field(default_factory=list)
    skill_rows: list[SkillMarketRow] = field(default_factory=list)
    top_demanded_skills: list[SkillMarketRow] = field(default_factory=list)
    covered_skills: list[SkillMarketRow] = field(default_factory=list)
    missing_skills: list[SkillMarketRow] = field(default_factory=list)
    low_demand_skills: list[SkillMarketRow] = field(default_factory=list)
    unmapped_market_skills: list[SkillMarketRow] = field(default_factory=list)
    common_skills: list[SkillMarketRow] = field(default_factory=list)
    example_offers: list[OfferExampleRow] = field(default_factory=list)
    source_queries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'referential_id': self.referential_id,
            'referential_title': self.referential_title,
            'referential_version': self.referential_version,
            'territory': self.territory,
            'radius_km': self.radius_km,
            'period': self.period,
            'job_titles': self.job_titles,
            'rome_codes': self.rome_codes,
            'date_min': self.date_min,
            'date_max': self.date_max,
            'offer_count': self.offer_count,
            'analyzed_offer_count': self.analyzed_offer_count,
            'covered_offer_count': self.covered_offer_count,
            'total_market_weight': round(self.total_market_weight, 4),
            'covered_weight': round(self.covered_weight, 4),
            'global_coverage_score': round(self.global_coverage_score, 2),
            'block_summaries': [row.to_dict() for row in self.block_summaries],
            'skill_rows': [row.to_dict() for row in self.skill_rows],
            'top_demanded_skills': [row.to_dict() for row in self.top_demanded_skills],
            'covered_skills': [row.to_dict() for row in self.covered_skills],
            'missing_skills': [row.to_dict() for row in self.missing_skills],
            'low_demand_skills': [row.to_dict() for row in self.low_demand_skills],
            'unmapped_market_skills': [row.to_dict() for row in self.unmapped_market_skills],
            'common_skills': [row.to_dict() for row in self.common_skills],
            'example_offers': [row.to_dict() for row in self.example_offers],
            'source_queries': self.source_queries,
            'warnings': self.warnings,
        }



def _split_values(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r'[\n,;|]+', value)
        return [clean_text(part) for part in parts if clean_text(part)]
    result: list[str] = []
    for item in value:
        text = clean_text(item)
        if text:
            result.append(text)
    return result



def _sentence_candidates(text: str) -> list[str]:
    sentences: list[str] = []
    for part in SENTENCE_SPLIT_RE.split(clean_text(text)):
        cleaned = clean_text(part)
        if cleaned:
            sentences.append(cleaned)
    return sentences



def _best_sentence(text: str, phrase: str) -> str:
    sentences = _sentence_candidates(text)
    if not sentences:
        return clean_text(text)[:200]
    norm_phrase = normalize_for_match(phrase)
    if not norm_phrase:
        return sentences[0]
    for sentence in sentences:
        if norm_phrase in normalize_for_match(sentence):
            return sentence
    phrase_tokens = set(norm_phrase.split())
    best_sentence = sentences[0]
    best_score = -1.0
    for sentence in sentences:
        sentence_tokens = set(normalize_for_match(sentence).split())
        if not sentence_tokens:
            continue
        score = len(sentence_tokens & phrase_tokens) / max(len(sentence_tokens | phrase_tokens), 1)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence



def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = clean_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        return None



def _recency_coefficient(creation_date: str | None, reference_date: datetime | None) -> float:
    parsed = _parse_date(creation_date)
    if parsed is None or reference_date is None:
        return 1.0
    age_days = max((reference_date - parsed).days, 0)
    return round(max(0.35, 1.0 / (1.0 + age_days / 90.0)), 4)



def _obligation_coefficient(requirement: str | None, evidence: str | None) -> float:
    text = normalize_for_match(' '.join(part for part in (clean_text(requirement), clean_text(evidence)) if part))
    if not text:
        return 1.0
    if any(token in text for token in ('oblig', 'required', 'must', 'indispens', 'essentiel')):
        return 1.2
    if any(token in text for token in ('souhait', 'optionnel', 'bonus', 'nice to have', 'facultatif')):
        return 0.8
    return 1.0



def _source_weight(source: str | None, match_type: str | None) -> float:
    source_key = clean_text(source).lower() or 'general_text'
    weight = SOURCE_WEIGHTS.get(source_key, SOURCE_WEIGHTS['general_text'])
    if clean_text(match_type) == 'semantic':
        weight *= 0.95
    if clean_text(match_type) == 'implicit':
        weight *= 0.85
    return weight



def _skill_priority(match_type: str | None) -> int:
    return PRIORITY.get(clean_text(match_type).lower(), 0)



def _normalize_candidate_label(label: str | None) -> str:
    text = clean_text(label)
    if not text:
        return ''
    text = normalize_market_alias(text)
    return clean_text(text)



def _extract_offer_title(offer: dict[str, Any]) -> str:
    return clean_text(offer.get('title') or offer.get('intitule') or offer.get('titre') or offer.get('label') or '')



def _extract_offer_description(offer: dict[str, Any]) -> str:
    return clean_text(offer.get('description') or offer.get('offer_text') or offer.get('description_original') or offer.get('content') or '')



def _extract_offer_text(offer: dict[str, Any]) -> str:
    title = _extract_offer_title(offer)
    description = _extract_offer_description(offer)
    return '\n'.join(part for part in (title, description) if part)



def _candidate_entries(offer: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def add(label: str | None, *, source: str, confidence: float = 0.0, evidence: str | None = None, requirement: str | None = None, referential_id: str | None = None, match_type: str | None = None) -> None:
        cleaned = clean_text(label)
        if not cleaned:
            return
        entries.append({
            'label': cleaned,
            'source': source,
            'confidence': confidence,
            'evidence': clean_text(evidence) or None,
            'requirement': clean_text(requirement) or None,
            'referential_id': clean_text(referential_id) or None,
            'match_type': clean_text(match_type) or None,
        })

    for item in offer.get('structured_skills') or []:
        if isinstance(item, dict):
            add(
                item.get('canonical_label') or item.get('label') or item.get('canonical_name'),
                source='structured',
                confidence=float(item.get('confidence', 1.0) or 1.0),
                evidence=item.get('label') or item.get('canonical_label'),
                requirement=item.get('requirement'),
            )
        else:
            add(item, source='structured', confidence=1.0)

    for item in offer.get('normalized_skills') or []:
        if isinstance(item, dict):
            add(
                item.get('canonical_label') or item.get('label') or item.get('name'),
                source='general_text',
                confidence=float(item.get('confidence', 0.75) or 0.75),
                evidence=item.get('source_text') or item.get('label'),
            )
        else:
            add(item, source='general_text', confidence=0.75)

    for item in offer.get('merged_skills') or []:
        if isinstance(item, dict):
            add(
                item.get('canonical_label') or item.get('label') or item.get('canonical_name'),
                source='general_text',
                confidence=float(item.get('confidence', 0.75) or 0.75),
                evidence=item.get('surface_form') or item.get('label'),
            )
        else:
            add(item, source='general_text', confidence=0.75)

    for item in offer.get('model_skills') or []:
        if isinstance(item, dict):
            add(
                item.get('label') or item.get('canonical_label') or item.get('canonical_name'),
                source='model_prediction',
                confidence=float(item.get('confidence', item.get('probability', 0.65)) or 0.65),
                evidence=item.get('surface_form') or item.get('label'),
            )
        else:
            add(item, source='model_prediction', confidence=0.65)

    for item in offer.get('competences') or []:
        if isinstance(item, dict):
            add(
                item.get('libelle_officiel') or item.get('canonical_label') or item.get('libelle') or item.get('label'),
                source='competences',
                confidence=float(item.get('confidence', 0.9) or 0.9),
                evidence=item.get('evidence') or item.get('libelle') or item.get('libelle_officiel'),
                referential_id=item.get('referential_id'),
                match_type=item.get('match_type'),
            )
        else:
            add(item, source='competences', confidence=0.9)

    text = _extract_offer_text(offer)
    for item in extract_skills_from_text(text):
        if isinstance(item, dict):
            add(
                item.get('canonical_label') or item.get('label'),
                source='text_explicit',
                confidence=float(item.get('confidence', 0.8) or 0.8),
                evidence=text,
            )

    return entries



def _build_query_rows(
    *,
    commune: str | None,
    departement: str | None,
    distance_km: int | None,
    date_min: str | None,
    date_max: str | None,
    job_titles: Iterable[str] | None,
    rome_codes: Iterable[str] | None,
) -> list[SearchCriteria]:
    titles = [title for title in _split_values(job_titles) if title]
    romes = [code for code in _split_values(rome_codes) if code]
    if not titles and not romes:
        titles = DEFAULT_JOB_TITLES
    rows: list[SearchCriteria] = []
    location_kwargs = {
        'commune': clean_text(commune) or None,
        'departement': clean_text(departement) or None,
        'distance_km': distance_km,
        'date_min': clean_text(date_min) or None,
        'date_max': clean_text(date_max) or None,
    }
    for title in titles:
        rows.append(SearchCriteria(keywords=title, **location_kwargs))
    for code in romes:
        rows.append(SearchCriteria(rome_code=code, **location_kwargs))
    if not rows:
        rows.append(SearchCriteria(**location_kwargs))
    return rows



def collect_market_offers(
    client: FranceTravailClient,
    *,
    commune: str | None = None,
    departement: str | None = None,
    distance_km: int | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
    job_titles: Iterable[str] | None = None,
    rome_codes: Iterable[str] | None = None,
    max_pages: int = 3,
    max_offers: int | None = 200,
    page_size: int = 20,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _build_query_rows(
        commune=commune,
        departement=departement,
        distance_km=distance_km,
        date_min=date_min,
        date_max=date_max,
        job_titles=job_titles,
        rome_codes=rome_codes,
    )
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    source_queries: list[str] = []
    for criteria in rows:
        query_label = criteria.keywords or criteria.rome_code or criteria.departement or criteria.commune or 'default'
        source_queries.append(query_label)
        for offer in client.iter_offers(criteria, max_pages=max_pages, max_offers=max_offers, page_size=page_size):
            normalized_offer = normalize_offer(offer)
            key = (normalized_offer.offer_id, normalized_offer.creation_date or normalized_offer.update_date or '')
            if key in seen:
                continue
            seen.add(key)
            merged_skills = merge_offer_skills(
                structured_skills=normalized_offer.structured_skills,
                explicit_skills=extract_skills_from_text(normalized_offer.offer_text),
                model_skills=normalized_offer.model_skills,
                rome_skills=[],
            )
            normalized_dict = normalized_offer.to_dict()
            normalized_dict['merged_skills'] = merged_skills
            normalized_dict['normalized_skills'] = [item['canonical_label'] for item in merged_skills if item.get('canonical_label')]
            normalized_dict['source_query'] = query_label
            normalized.append(normalized_dict)
            if max_offers and len(normalized) >= max_offers:
                return normalized, source_queries
    return normalized, source_queries


class CertificationMarketComparator:
    def __init__(
        self,
        referential_path: str | Path | None = None,
        *,
        embedding_model: str | None = None,
        semantic_threshold: float = 0.72,
    ) -> None:
        self.referential = AICertificationReferential(referential_path or DEFAULT_REFERENTIAL_PATH, embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL)
        self.semantic_threshold = semantic_threshold
        self.extractor = AICertificationSkillExtractor(
            referential=self.referential,
            semantic_threshold=semantic_threshold,
            implicit_threshold=max(semantic_threshold, 0.8),
            embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        )

    def _match_skill(self, label: str) -> tuple[dict[str, Any] | None, str, float, str]:
        candidate = clean_text(label)
        if not candidate:
            return None, '', 0.0, ''
        normalized = _normalize_candidate_label(candidate)
        exact = self.referential.search_exact(candidate) or self.referential.search_exact(normalized)
        if exact:
            return exact, 'exact', 1.0, exact.get('label', '')
        alias = self.referential.search_alias(candidate) or self.referential.search_alias(normalized)
        if alias:
            return alias, 'alias', 0.92, alias.get('label', '')
        semantic_hits = self.referential.search_semantic(candidate, top_k=1)
        if semantic_hits:
            hit = semantic_hits[0]
            if hit.score >= self.semantic_threshold:
                match_type = 'semantic' if hit.score < 0.8 else 'implicit'
                skill = self.referential.get_skill_by_id(hit.referential_id)
                if skill:
                    return skill, match_type, float(hit.score), skill.get('label', '')
        return None, '', 0.0, ''

    def _match_offer(self, offer: dict[str, Any], reference_date: datetime | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        text = _extract_offer_text(offer)
        title = _extract_offer_title(offer)
        creation_date = offer.get('creation_date') or offer.get('date_publication') or offer.get('published_at')
        recency = _recency_coefficient(clean_text(creation_date), reference_date)
        candidates = _candidate_entries(offer)
        text_hits = self.extractor.extract(title=title, description=_extract_offer_description(offer)).get('competences', [])
        for item in text_hits:
            if isinstance(item, dict):
                candidates.append({
                    'label': clean_text(item.get('libelle_officiel') or item.get('libelle') or item.get('canonical_label') or ''),
                    'source': 'ai_cert_extractor',
                    'confidence': float(item.get('confidence', 0.9) or 0.9),
                    'evidence': clean_text(item.get('evidence') or ''),
                    'requirement': None,
                    'referential_id': clean_text(item.get('referential_id') or ''),
                    'match_type': clean_text(item.get('match_type') or ''),
                })

        referential_hits: dict[str, dict[str, Any]] = {}
        unmapped_hits: dict[str, dict[str, Any]] = {}
        matched_labels: list[str] = []
        unmatched_labels: list[str] = []

        for candidate in candidates:
            label = clean_text(candidate.get('label') or '')
            if not label:
                continue
            source = clean_text(candidate.get('source') or '')
            confidence = float(candidate.get('confidence') or 0.0)
            evidence = clean_text(candidate.get('evidence') or '') or _best_sentence(text, label)
            requirement = candidate.get('requirement')
            direct_skill = None
            match_type = clean_text(candidate.get('match_type') or '')
            if candidate.get('referential_id'):
                direct_skill = self.referential.get_skill_by_id(candidate['referential_id'])
                if direct_skill:
                    match_type = match_type or 'exact'
                    confidence = max(confidence, 0.9)
            if direct_skill is None:
                direct_skill, match_type, semantic_score, _ = self._match_skill(label)
                if direct_skill and semantic_score:
                    confidence = max(confidence, semantic_score)
            if direct_skill:
                skill_id = direct_skill['id']
                weight = confidence * recency * _obligation_coefficient(requirement, evidence) * _source_weight(source, match_type)
                current = referential_hits.get(skill_id)
                skill_entry = {
                    'referential_id': skill_id,
                    'block': clean_text(direct_skill.get('block') or ''),
                    'block_name': clean_text(direct_skill.get('block_name') or f"Bloc {clean_text(direct_skill.get('block') or '')[1:]}") if clean_text(direct_skill.get('block') or '').startswith('B') else clean_text(direct_skill.get('block_name') or ''),
                    'activity': clean_text(direct_skill.get('activity') or ''),
                    'code': clean_text(direct_skill.get('code') or ''),
                    'label': clean_text(direct_skill.get('label') or label),
                    'official_description': clean_text(direct_skill.get('official_description') or ''),
                    'normalized_label': clean_text(direct_skill.get('normalized_label') or normalize_for_match(direct_skill.get('label') or label)),
                    'category': clean_text(direct_skill.get('category') or infer_skill_taxonomy(label, evidence, direct_skill.get('aliases', []), block=direct_skill.get('block'), activity=direct_skill.get('activity'), origin_document=direct_skill.get('origin_document')).category),
                    'subcategory': clean_text(direct_skill.get('subcategory') or infer_skill_taxonomy(label, evidence, direct_skill.get('aliases', []), block=direct_skill.get('block'), activity=direct_skill.get('activity'), origin_document=direct_skill.get('origin_document')).subcategory),
                    'technical_keywords': list(direct_skill.get('technical_keywords') or infer_skill_taxonomy(label, evidence, direct_skill.get('aliases', []), block=direct_skill.get('block'), activity=direct_skill.get('activity'), origin_document=direct_skill.get('origin_document')).technical_keywords),
                    'source_page': int(direct_skill.get('source_page') or 0),
                    'offer_count': 1,
                    'weighted_score': round(weight, 4),
                    'share_percent': 0.0,
                    'coverage_score': 0.0,
                    'match_methods': {match_type or 'exact': 1},
                    'evidence_examples': [evidence],
                    'example_offers': [title] if title else [],
                    'status': 'covered',
                }
                if current is None or (weight, _skill_priority(match_type)) > (current['weighted_score'], _skill_priority(max(current['match_methods'], key=current['match_methods'].get, default=''))):
                    referential_hits[skill_id] = skill_entry
                matched_labels.append(skill_entry['label'])
                continue

            normalized_label = _normalize_candidate_label(label)
            if not normalized_label:
                continue
            weight = confidence * recency * _obligation_coefficient(requirement, evidence) * _source_weight(source, match_type)
            current = unmapped_hits.get(normalized_label)
            taxonomy = infer_skill_taxonomy(label, evidence, [], block=None, activity=None, origin_document=None)
            skill_entry = {
                'referential_id': '',
                'block': '',
                'block_name': '',
                'activity': '',
                'code': '',
                'label': label,
                'official_description': '',
                'normalized_label': normalized_label,
                'category': taxonomy.category,
                'subcategory': taxonomy.subcategory,
                'technical_keywords': taxonomy.technical_keywords,
                'source_page': 0,
                'offer_count': 1,
                'weighted_score': round(weight, 4),
                'share_percent': 0.0,
                'coverage_score': 0.0,
                'match_methods': {'unmapped': 1},
                'evidence_examples': [evidence],
                'example_offers': [title] if title else [],
                'status': 'unmapped',
            }
            if current is None or weight > float(current['weighted_score']):
                unmapped_hits[normalized_label] = skill_entry
            unmatched_labels.append(label)

        location = offer.get('lieuTravail') if isinstance(offer.get('lieuTravail'), dict) else {}
        rome = offer.get('rome') if isinstance(offer.get('rome'), dict) else {}
        offer_summary = {
            'offer_id': clean_text(offer.get('offer_id') or offer.get('id') or normalized_offer_id(offer)),
            'title': title,
            'location_label': clean_text(offer.get('location_label') or location.get('libelle') or ''),
            'contract_label': clean_text(offer.get('contract_label') or offer.get('contract_type') or offer.get('typeContrat') or ''),
            'rome_code': clean_text(offer.get('rome_code') or rome.get('code') or ''),
            'rome_label': clean_text(offer.get('rome_label') or rome.get('libelle') or ''),
            'date': clean_text(creation_date or offer.get('update_date') or offer.get('date_publication') or ''),
            'matches': matched_labels,
            'unmatched': unmatched_labels,
        }
        return offer_summary, list(referential_hits.values()), list(unmapped_hits.values()), matched_labels

    def compare(
        self,
        offers: list[dict[str, Any]],
        *,
        territory: str,
        radius_km: int | None = None,
        date_min: str | None = None,
        date_max: str | None = None,
        job_titles: Iterable[str] | None = None,
        rome_codes: Iterable[str] | None = None,
        source_queries: Iterable[str] | None = None,
    ) -> CertificationMarketComparisonReport:
        normalized_offers = list(offers)
        if not normalized_offers:
            return CertificationMarketComparisonReport(
                referential_id=clean_text(self.referential.payload.get('referential_id') or 'ingenieur_ia_2025'),
                referential_title=clean_text(self.referential.payload.get('title') or 'Ingénieur en intelligence artificielle'),
                referential_version=clean_text(self.referential.payload.get('version') or ''),
                territory=clean_text(territory),
                radius_km=radius_km,
                period=f"{date_min or ''} / {date_max or ''}".strip(' /'),
                job_titles=[clean_text(item) for item in _split_values(job_titles)],
                rome_codes=[clean_text(item) for item in _split_values(rome_codes)],
                date_min=date_min,
                date_max=date_max,
                offer_count=0,
                analyzed_offer_count=0,
                covered_offer_count=0,
                total_market_weight=0.0,
                covered_weight=0.0,
                global_coverage_score=0.0,
                warnings=['Aucune offre à comparer.'],
            )

        period_end = _parse_date(date_max) or max((_parse_date(offer.get('creation_date') or offer.get('update_date') or offer.get('date_publication')) for offer in normalized_offers if _parse_date(offer.get('creation_date') or offer.get('update_date') or offer.get('date_publication'))), default=None)
        if period_end is None:
            period_end = datetime.now(timezone.utc)

        aggregated: dict[str, dict[str, Any]] = {}
        unmapped: dict[str, dict[str, Any]] = {}
        example_offers: list[OfferExampleRow] = []
        warnings: list[str] = []
        covered_offer_ids: set[str] = set()
        source_queries_list = [clean_text(item) for item in source_queries or [] if clean_text(item)]

        for offer in normalized_offers:
            offer_summary, referential_hits, unmapped_hits, matched_labels = self._match_offer(offer, period_end)
            if matched_labels:
                covered_offer_ids.add(offer_summary['offer_id'])
            if len(example_offers) < 8:
                example_offers.append(OfferExampleRow(
                    offer_id=offer_summary['offer_id'],
                    title=offer_summary['title'],
                    location_label=offer_summary['location_label'],
                    contract_label=offer_summary['contract_label'],
                    rome_code=offer_summary['rome_code'],
                    rome_label=offer_summary['rome_label'],
                    date=offer_summary['date'],
                    matches=matched_labels[:8],
                    unmatched=[row['label'] for row in unmapped_hits[:4]],
                ))
            for row in referential_hits:
                key = row['referential_id']
                current = aggregated.get(key)
                if current is None:
                    current = aggregated[key] = row
                else:
                    current['offer_count'] += row['offer_count']
                    current['weighted_score'] += row['weighted_score']
                    current['match_methods'].update(row['match_methods'])
                    current['evidence_examples'] = list(dict.fromkeys([*current['evidence_examples'], *row['evidence_examples']]))[:3]
                    current['example_offers'] = list(dict.fromkeys([*current['example_offers'], *row['example_offers']]))[:3]
                    current['status'] = 'covered'
            for row in unmapped_hits:
                key = row['normalized_label']
                current = unmapped.get(key)
                if current is None:
                    unmapped[key] = row
                else:
                    current['offer_count'] += row['offer_count']
                    current['weighted_score'] += row['weighted_score']
                    current['match_methods'].update(row['match_methods'])
                    current['evidence_examples'] = list(dict.fromkeys([*current['evidence_examples'], *row['evidence_examples']]))[:3]
                    current['example_offers'] = list(dict.fromkeys([*current['example_offers'], *row['example_offers']]))[:3]

        referential_skills = self.referential.get_all_skills()
        total_market_weight = sum(float(row['weighted_score']) for row in aggregated.values()) + sum(float(row['weighted_score']) for row in unmapped.values())
        covered_weight = sum(float(row['weighted_score']) for row in aggregated.values())
        global_coverage_score = (covered_weight / total_market_weight * 100.0) if total_market_weight else 0.0

        referential_by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for skill in referential_skills:
            block = clean_text(skill.get('block') or 'Other') or 'Other'
            referential_by_block[block].append(skill)

        skill_rows: list[SkillMarketRow] = []
        for skill in referential_skills:
            skill_id = clean_text(skill.get('id') or '')
            row = aggregated.get(skill_id)
            if row is None:
                skill_rows.append(SkillMarketRow(
                    referential_id=skill_id,
                    block=clean_text(skill.get('block') or ''),
                    block_name=clean_text(skill.get('block_name') or f"Bloc {clean_text(skill.get('block') or '')[1:]}") if clean_text(skill.get('block') or '').startswith('B') else clean_text(skill.get('block_name') or ''),
                    activity=clean_text(skill.get('activity') or ''),
                    code=clean_text(skill.get('code') or ''),
                    label=clean_text(skill.get('label') or ''),
                    official_description=clean_text(skill.get('official_description') or ''),
                    normalized_label=clean_text(skill.get('normalized_label') or normalize_for_match(skill.get('label') or '')),
                    category=clean_text(skill.get('category') or ''),
                    subcategory=clean_text(skill.get('subcategory') or ''),
                    technical_keywords=list(skill.get('technical_keywords') or []),
                    source_page=int(skill.get('source_page') or 0),
                    offer_count=0,
                    weighted_score=0.0,
                    share_percent=0.0,
                    coverage_score=0.0,
                    match_methods={},
                    evidence_examples=[],
                    example_offers=[],
                    status='missing',
                ))
                continue
            skill_row = SkillMarketRow(
                referential_id=skill_id,
                block=clean_text(skill.get('block') or ''),
                block_name=clean_text(skill.get('block_name') or f"Bloc {clean_text(skill.get('block') or '')[1:]}") if clean_text(skill.get('block') or '').startswith('B') else clean_text(skill.get('block_name') or ''),
                activity=clean_text(skill.get('activity') or ''),
                code=clean_text(skill.get('code') or ''),
                label=clean_text(skill.get('label') or ''),
                official_description=clean_text(skill.get('official_description') or ''),
                normalized_label=clean_text(skill.get('normalized_label') or normalize_for_match(skill.get('label') or '')),
                category=clean_text(skill.get('category') or ''),
                subcategory=clean_text(skill.get('subcategory') or ''),
                technical_keywords=list(skill.get('technical_keywords') or []),
                source_page=int(skill.get('source_page') or 0),
                offer_count=int(row['offer_count']),
                weighted_score=float(row['weighted_score']),
                share_percent=round((row['offer_count'] / len(normalized_offers)) * 100.0, 2) if normalized_offers else 0.0,
                coverage_score=round((float(row['weighted_score']) / total_market_weight) * 100.0, 2) if total_market_weight else 0.0,
                match_methods=dict(row['match_methods']),
                evidence_examples=list(row['evidence_examples']),
                example_offers=list(row['example_offers']),
                status='covered',
            )
            skill_rows.append(skill_row)

        top_demanded_skills = sorted(skill_rows, key=lambda row: (-row.weighted_score, -row.offer_count, row.label))[:15]
        covered_skills = [row for row in skill_rows if row.status == 'covered']
        missing_skills = [row for row in skill_rows if row.status == 'missing']
        low_demand_skills = sorted(skill_rows, key=lambda row: (row.weighted_score, -row.offer_count, row.label))[:15]
        common_skills = sorted((row for row in skill_rows if row.status == 'covered'), key=lambda row: (-row.offer_count, -row.weighted_score, row.label))[:15]
        unmapped_market_rows = []
        for row in sorted(unmapped.values(), key=lambda item: (-float(item['weighted_score']), -int(item['offer_count']), item['label'])):
            taxonomy = infer_skill_taxonomy(row['label'], ' '.join(row.get('evidence_examples', [])), [], origin_document=None)
            unmapped_market_rows.append(SkillMarketRow(
                referential_id='',
                block='',
                block_name='',
                activity='',
                code='',
                label=row['label'],
                official_description='',
                normalized_label=row['normalized_label'],
                category=taxonomy.category,
                subcategory=taxonomy.subcategory,
                technical_keywords=taxonomy.technical_keywords,
                source_page=0,
                offer_count=int(row['offer_count']),
                weighted_score=float(row['weighted_score']),
                share_percent=round((row['offer_count'] / len(normalized_offers)) * 100.0, 2) if normalized_offers else 0.0,
                coverage_score=0.0,
                match_methods=dict(row['match_methods']),
                evidence_examples=list(row['evidence_examples']),
                example_offers=list(row['example_offers']),
                status='unmapped',
            ))
        block_summaries: list[BlockCoverageRow] = []
        for block, skills in sorted(referential_by_block.items(), key=lambda item: item[0]):
            total_block_weight = sum(row.weighted_score for row in skill_rows if row.block == block)
            covered_block_weight = sum(row.weighted_score for row in skill_rows if row.block == block and row.status == 'covered')
            covered_block_count = sum(1 for row in skill_rows if row.block == block and row.status == 'covered')
            average_skill_share = (sum(row.share_percent for row in skill_rows if row.block == block) / len(skills)) if skills else 0.0
            block_summaries.append(BlockCoverageRow(
                block=block,
                block_name=clean_text(skills[0].get('block_name') or f"Bloc {block[1:]}") if skills else block,
                skill_count=len(skills),
                covered_skill_count=covered_block_count,
                total_weight=round(total_block_weight, 4),
                covered_weight=round(covered_block_weight, 4),
                market_share_percent=round((total_block_weight / total_market_weight) * 100.0, 2) if total_market_weight else 0.0,
                average_skill_share_percent=round(average_skill_share, 2),
            ))

        return CertificationMarketComparisonReport(
            referential_id=clean_text(self.referential.payload.get('referential_id') or 'ingenieur_ia_2025'),
            referential_title=clean_text(self.referential.payload.get('title') or 'Ingénieur en intelligence artificielle'),
            referential_version=clean_text(self.referential.payload.get('version') or ''),
            territory=clean_text(territory),
            radius_km=radius_km,
            period=f"{date_min or ''} / {date_max or ''}".strip(' /'),
            job_titles=[clean_text(item) for item in _split_values(job_titles)],
            rome_codes=[clean_text(item) for item in _split_values(rome_codes)],
            date_min=date_min,
            date_max=date_max,
            offer_count=len(normalized_offers),
            analyzed_offer_count=len(normalized_offers),
            covered_offer_count=len(covered_offer_ids),
            total_market_weight=round(total_market_weight, 4),
            covered_weight=round(covered_weight, 4),
            global_coverage_score=round(global_coverage_score, 2),
            block_summaries=block_summaries,
            skill_rows=skill_rows,
            top_demanded_skills=top_demanded_skills,
            covered_skills=covered_skills,
            missing_skills=missing_skills,
            low_demand_skills=low_demand_skills,
            unmapped_market_skills=unmapped_market_rows,
            common_skills=common_skills,
            example_offers=example_offers,
            source_queries=source_queries_list,
            warnings=warnings,
        )



def normalized_offer_id(offer: dict[str, Any]) -> str:
    value = clean_text(offer.get('offer_id') or offer.get('id') or offer.get('reference') or '')
    if value:
        return value
    return stable_hash(_extract_offer_text(offer))



def write_comparison_outputs(report: CertificationMarketComparisonReport, output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / 'ai_certification_market_comparison.json'
    validation_csv = directory / 'ai_certification_market_validation.csv'
    gaps_csv = directory / 'ai_certification_market_gaps.csv'

    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    validation_rows: list[dict[str, Any]] = []
    for row in report.skill_rows:
        validation_rows.append({
            'referential_id': row.referential_id,
            'block': row.block,
            'block_name': row.block_name,
            'activity': row.activity,
            'code': row.code,
            'label': row.label,
            'category': row.category,
            'subcategory': row.subcategory,
            'technical_keywords': ' | '.join(row.technical_keywords),
            'offer_count': row.offer_count,
            'weighted_score': round(row.weighted_score, 4),
            'share_percent': row.share_percent,
            'coverage_score': row.coverage_score,
            'status': row.status,
            'match_methods': json.dumps(row.match_methods, ensure_ascii=False),
            'evidence_examples': ' || '.join(row.evidence_examples),
            'example_offers': ' || '.join(row.example_offers),
        })
    with validation_csv.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(validation_rows[0].keys()) if validation_rows else ['referential_id'])
        writer.writeheader()
        if validation_rows:
            writer.writerows(validation_rows)

    gap_rows: list[dict[str, Any]] = []
    for row in report.missing_skills:
        gap_rows.append({
            'type': 'missing',
            'label': row.label,
            'block': row.block,
            'category': row.category,
            'subcategory': row.subcategory,
            'offer_count': row.offer_count,
            'weighted_score': round(row.weighted_score, 4),
        })
    for row in report.unmapped_market_skills:
        gap_rows.append({
            'type': 'unmapped',
            'label': row.label,
            'block': row.block,
            'category': row.category,
            'subcategory': row.subcategory,
            'offer_count': row.offer_count,
            'weighted_score': round(row.weighted_score, 4),
        })
    with gaps_csv.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(gap_rows[0].keys()) if gap_rows else ['type'])
        writer.writeheader()
        if gap_rows:
            writer.writerows(gap_rows)

    return {
        'json': json_path,
        'validation_csv': validation_csv,
        'gaps_csv': gaps_csv,
    }
