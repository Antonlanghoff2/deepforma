
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from common.text import clean_text, normalize_for_match
from domain.models import JobOffer, Skill, Territory
from france_travail.client import FranceTravailClient, ROME_CODE_RE, count_returned_rome_codes, normalize_rome_code
from france_travail.normalizer import normalize_offer


class UnexpectedRomeOfferError(ValueError):
    def __init__(self, offer_id: str, expected: str, actual: str | None) -> None:
        self.offer_id = offer_id
        self.expected = expected
        self.actual = actual
        super().__init__(f"Unexpected ROME offer: offer_id={offer_id} expected={expected} actual={actual or 'MISSING_ROME_CODE'}")


class OfferCollection(list[dict[str, Any]]):
    def __init__(self, offers: list[dict[str, Any]] | None = None, *, audit: dict[str, Any] | None = None) -> None:
        super().__init__(offers or [])
        self.audit = audit or {}


@dataclass(frozen=True, slots=True)
class RomeOfferRejection:
    offer_id: str
    reason: str
    expected_rome_code: str
    actual_rome_code: str | None
    title: str | None = None
    source_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'offer_id': self.offer_id,
            'reason': self.reason,
            'expected_rome_code': self.expected_rome_code,
            'actual_rome_code': self.actual_rome_code,
            'title': self.title,
            'source_query': self.source_query,
        }


def _skill_from_value(value: Any) -> Skill:
    if isinstance(value, dict):
        candidate = value.get('normalized_label') or value.get('canonical_label') or value.get('label') or value.get('name')
    else:
        candidate = value
    label = clean_text(candidate)
    return Skill(name=label or str(candidate or ''), normalized_name=label or None, source='web_app')


def _offer_from_normalized_offer(offer: dict[str, Any]) -> JobOffer:
    raw_skills = offer.get('normalized_skills') or offer.get('structured_skills') or offer.get('model_skills') or []
    skills = [_skill_from_value(item) for item in raw_skills if clean_text(item.get('normalized_label') if isinstance(item, dict) else item)]
    offer_id = clean_text(offer.get('offer_id') or offer.get('id') or offer.get('reference') or '')
    if not offer_id:
        from common.text import stable_hash

        offer_id = stable_hash(offer.get('title', ''), offer.get('description', ''), offer.get('rome_code', ''), length=24)
    title = clean_text(offer.get('title') or offer.get('intitule') or '')
    description = clean_text(offer.get('description') or offer.get('body') or '')
    rome_code = normalize_rome_code(offer.get('rome_code') or offer.get('romeCode') or offer.get('rome') or '') or None
    location = clean_text(offer.get('location') or offer.get('city') or offer.get('commune') or offer.get('department') or '') or None
    return JobOffer(offer_id=offer_id, title=title, description=description, skills=skills, rome_code=rome_code, location=location)


def _territory_query_kwargs(territory: Territory) -> dict[str, str | None]:
    code = clean_text(territory.code or territory.department_code or '')
    if not code:
        return {'departement': None, 'commune': None}
    if code.isdigit() and len(code) == 5:
        return {'departement': None, 'commune': code}
    if (code.isdigit() and len(code) in {2, 3}) or code in {'2A', '2B'}:
        return {'departement': code, 'commune': None}
    if territory.type == 'commune':
        return {'departement': None, 'commune': code}
    return {'departement': code, 'commune': None}


def _rome_code_from_offer(offer: dict[str, Any]) -> str:
    if not isinstance(offer, dict):
        return ''
    candidate = offer.get('romeCode') or offer.get('rome_code') or offer.get('code_rome')
    normalized = normalize_rome_code(candidate if isinstance(candidate, str) else str(candidate or ''))
    if normalized:
        return normalized
    rome = offer.get('rome')
    if isinstance(rome, dict):
        return normalize_rome_code(rome.get('romeCode') or rome.get('code'))
    return ''


def filter_offers_by_exact_rome(
    offers: list[dict[str, Any]],
    requested_rome_code: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_requested = normalize_rome_code(requested_rome_code)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for offer in offers:
        if not isinstance(offer, dict):
            rejected.append({
                'reason': 'MALFORMED_OFFER',
                'expected_rome_code': normalized_requested,
                'actual_rome_code': None,
            })
            continue
        actual = _rome_code_from_offer(offer)
        if not actual:
            rejected.append({
                **offer,
                'reason': 'MISSING_ROME_CODE',
                'expected_rome_code': normalized_requested,
                'actual_rome_code': None,
            })
            continue
        if not ROME_CODE_RE.match(actual):
            rejected.append({
                **offer,
                'reason': 'INVALID_ROME_CODE',
                'expected_rome_code': normalized_requested,
                'actual_rome_code': actual,
            })
            continue
        if actual != normalized_requested:
            rejected.append({
                **offer,
                'reason': 'ROME_MISMATCH',
                'expected_rome_code': normalized_requested,
                'actual_rome_code': actual,
            })
            continue
        accepted.append(offer)
    return accepted, rejected


def assert_offers_match_rome(
    offers: list[JobOffer],
    requested_rome_code: str,
) -> None:
    normalized_requested = normalize_rome_code(requested_rome_code)
    for offer in offers:
        actual = normalize_rome_code(offer.rome_code)
        if actual != normalized_requested:
            raise UnexpectedRomeOfferError(offer.offer_id, normalized_requested, actual or None)


def fetch_offers_by_rome(
    client: FranceTravailClient,
    rome_code: str,
    territory: Territory,
    *,
    radius_km: int | None = None,
    contract_types: list[str] | None = None,
    max_results: int = 500,
) -> OfferCollection:
    requested_rome_code = normalize_rome_code(rome_code)
    location = _territory_query_kwargs(territory)
    contract_filters = [clean_text(item) for item in (contract_types or []) if clean_text(item)] or [None]
    collected: list[dict[str, Any]] = []
    accepted_ids: set[str] = set()
    rejected_entries: list[dict[str, Any]] = []
    returned_raw_offers: list[dict[str, Any]] = []
    raw_count = 0
    for contract_type in contract_filters:
        from services.certification_market_comparison import collect_market_offers

        normalized_offers, _ = collect_market_offers(
            client,
            commune=location['commune'],
            departement=location['departement'],
            distance_km=radius_km or territory.radius_km,
            job_titles=None,
            rome_codes=[requested_rome_code],
            max_offers=max_results,
            page_size=min(20, max_results) if max_results else 20,
        )
        raw_count += len(normalized_offers)
        returned_raw_offers.extend(
            [offer.get('raw_offer') if isinstance(offer.get('raw_offer'), dict) else offer for offer in normalized_offers if isinstance(offer, dict)]
        )
        accepted, rejected = filter_offers_by_exact_rome(normalized_offers, requested_rome_code)
        for rejected_offer in rejected:
            title = clean_text(rejected_offer.get('title') or rejected_offer.get('intitule') or '') or None
            rejected_entries.append(
                {
                    'offer_id': clean_text(rejected_offer.get('offer_id') or rejected_offer.get('id') or rejected_offer.get('reference') or '') or None,
                    'reason': rejected_offer.get('reason') or 'ROME_MISMATCH',
                    'expected_rome_code': requested_rome_code,
                    'actual_rome_code': rejected_offer.get('actual_rome_code'),
                    'title': title,
                    'source_query': clean_text(rejected_offer.get('source_query') or requested_rome_code) or requested_rome_code,
                }
            )
        for offer in accepted:
            if contract_type:
                offer_contract = clean_text(offer.get('contract_type') or offer.get('contract_label') or offer.get('contract') or '')
                if offer_contract and contract_type not in offer_contract:
                    rejected_entries.append(
                        {
                            'offer_id': clean_text(offer.get('offer_id') or offer.get('id') or '') or None,
                            'reason': 'CONTRACT_MISMATCH',
                            'expected_rome_code': requested_rome_code,
                            'actual_rome_code': normalize_rome_code(offer.get('rome_code') or offer.get('romeCode')) or None,
                            'title': clean_text(offer.get('title') or '') or None,
                            'source_query': clean_text(offer.get('source_query') or requested_rome_code) or requested_rome_code,
                        }
                    )
                    continue
            offer_id = clean_text(offer.get('offer_id') or offer.get('id') or '')
            if offer_id and offer_id in accepted_ids:
                continue
            if offer_id:
                accepted_ids.add(offer_id)
            requested_codes = list(offer.get('matched_requested_rome_codes') or [])
            if requested_rome_code not in requested_codes:
                requested_codes.append(requested_rome_code)
            offer['matched_requested_rome_codes'] = requested_codes
            offer['source_query'] = clean_text(offer.get('source_query') or requested_rome_code) or requested_rome_code
            collected.append(offer)
            if max_results and len(collected) >= max_results:
                break
        if max_results and len(collected) >= max_results:
            break
    audit = {
        'query': {
            'rome_code': requested_rome_code,
            'territory_type': clean_text(territory.type or '') or ('departement' if location['departement'] else 'commune' if location['commune'] else ''),
            'territory_code': clean_text(territory.code or territory.department_code or '') or None,
            'radius_km': radius_km or territory.radius_km,
            'contracts': [item for item in contract_filters if item],
            'query_version': 2,
        },
        'raw_count': raw_count,
        'accepted_count': len(collected),
        'rejected_count': len(rejected_entries),
        'rejected_reasons': dict(Counter(entry['reason'] for entry in rejected_entries)),
        'rejections': rejected_entries[:25],
        'rome_distribution': count_returned_rome_codes(returned_raw_offers),
    }
    return OfferCollection(collected, audit=audit)

def build_market_context(*, skill_extraction: Any, normalized_offers: list[dict[str, Any]], departement: str, recommendation_service: Any) -> dict[str, Any]:
    from analytics.territorial_skills import compute_territorial_stats
    from services.market_analysis import analyze_market_fit

    extracted_labels = [s.normalized_label for s in skill_extraction.skills if getattr(s, 'normalized_label', None)]
    extracted_labels += [s.normalized_label for s in skill_extraction.tools if getattr(s, 'normalized_label', None)]
    recommendation = recommendation_service.compare(extracted_labels, normalized_offers)
    market_analysis = analyze_market_fit(
        [_skill_from_value(label) for label in extracted_labels],
        Territory(code=departement or None, label=departement or None, department_code=departement or None, region_code=None, remote_allowed=True),
        [_offer_from_normalized_offer(offer) for offer in normalized_offers],
    )
    territorial_stats = compute_territorial_stats(normalized_offers, territory_key=departement)
    return {
        'recommendation': recommendation,
        'market_analysis': market_analysis,
        'territorial_stats': territorial_stats,
        'extracted_labels': extracted_labels,
        'offer_count': len(normalized_offers),
        'market_offers_preview': normalized_offers[:10],
        'market_offers_more_count': max(len(normalized_offers) - min(len(normalized_offers), 10), 0),
    }


@dataclass(frozen=True, slots=True)
class RejectedOffer:
    offer_id: str | None
    reason: str
    expected_rome_code: str
    actual_rome_code: str | None = None
    title: str | None = None
    source_query: str | None = None


@dataclass(frozen=True, slots=True)
class RomeSearchStats:
    rome_code: str
    rome_label: str
    raw_count: int
    accepted_count: int
    rejected_count: int
    pages_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MultiRomeSearchResult:
    requested_rome_codes: list[str]
    offers: list[dict[str, Any]]
    stats_by_rome: list[RomeSearchStats]
    rejected_offers: list[RejectedOffer]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RomeMarketComparison:
    rome_code: str
    rome_label: str
    offers_count: int
    common_skills: list[Any]
    missing_skills: list[Any]
    coverage_score: float


@dataclass(frozen=True, slots=True)
class MultiRomeMarketComparison:
    requested_occupations: list[Any]
    territory: Territory | None
    unique_offers_count: int
    global_comparison: Any
    comparisons_by_rome: list[RomeMarketComparison]
    warnings: list[str] = field(default_factory=list)


def merge_offers_by_id(offers_by_rome: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for rome_code, offers in offers_by_rome.items():
        for offer in offers:
            if not isinstance(offer, dict):
                continue
            offer_id = clean_text(offer.get('offer_id') or offer.get('id') or offer.get('reference') or '')
            if not offer_id:
                from common.text import stable_hash
                offer_id = stable_hash(offer.get('title', ''), offer.get('description', ''), rome_code, length=24)
            current = merged.get(offer_id)
            incoming_codes = list(offer.get('matched_requested_rome_codes') or [])
            if rome_code not in incoming_codes:
                incoming_codes.append(rome_code)
            if current is None:
                clone = dict(offer)
                clone['offer_id'] = offer_id
                clone['matched_requested_rome_codes'] = list(dict.fromkeys([clean_text(code).upper() for code in incoming_codes if clean_text(code)]))
                merged[offer_id] = clone
                order.append(offer_id)
                continue
            existing_codes = list(current.get('matched_requested_rome_codes') or [])
            existing_codes.extend(incoming_codes)
            current['matched_requested_rome_codes'] = list(dict.fromkeys([clean_text(code).upper() for code in existing_codes if clean_text(code)]))
            current_skills = list(current.get('skills') or [])
            incoming_skills = list(offer.get('skills') or [])
            current['skills'] = current_skills + [skill for skill in incoming_skills if skill not in current_skills]
            current_sources = list(current.get('sources') or [])
            incoming_sources = list(offer.get('sources') or [])
            if not incoming_sources:
                incoming_sources = [{'query_rome_code': rome_code, 'accepted': True}]
            current_sources.extend(incoming_sources)
            current['sources'] = current_sources
    return [merged[offer_id] for offer_id in order]


def fetch_offers_by_rome_codes(rome_codes: list[str], territory: Territory, *, radius_km: int | None = None, contract_types: list[str] | None = None, max_results_per_code: int = 500, client: FranceTravailClient | None = None) -> MultiRomeSearchResult:
    from referentials.rome_referential import validate_rome_codes

    active_client = client or FranceTravailClient()
    validated_codes = validate_rome_codes(rome_codes, service=None)
    offers_by_rome: dict[str, list[dict[str, Any]]] = {}
    rejected_offers: list[RejectedOffer] = []
    stats_by_rome: list[RomeSearchStats] = []
    warnings: list[str] = []
    for rome_code in validated_codes:
        try:
            result = fetch_offers_by_rome(active_client, rome_code, territory, radius_km=radius_km, contract_types=contract_types, max_results=max_results_per_code)
        except Exception as exc:
            stats_by_rome.append(RomeSearchStats(rome_code=rome_code, rome_label=rome_code, raw_count=0, accepted_count=0, rejected_count=0, pages_count=0, error=str(exc)))
            warnings.append(f'{rome_code}: {exc}')
            continue
        offers_by_rome[rome_code] = list(result)
        audit = getattr(result, 'audit', {}) or {}
        stats_by_rome.append(RomeSearchStats(rome_code=rome_code, rome_label=rome_code, raw_count=int(audit.get('raw_count', len(result)) or len(result)), accepted_count=int(audit.get('accepted_count', len(result)) or len(result)), rejected_count=int(audit.get('rejected_count', 0) or 0), pages_count=int(audit.get('pages_count', 1) or 1), error=None))
        for rejected in audit.get('rejections', []) or []:
            rejected_offers.append(RejectedOffer(offer_id=rejected.get('offer_id'), reason=rejected.get('reason') or 'ROME_MISMATCH', expected_rome_code=rome_code, actual_rome_code=rejected.get('actual_rome_code'), title=rejected.get('title'), source_query=rejected.get('source_query')))
    merged = merge_offers_by_id(offers_by_rome)
    return MultiRomeSearchResult(requested_rome_codes=validated_codes, offers=merged, stats_by_rome=stats_by_rome, rejected_offers=rejected_offers, warnings=warnings)
