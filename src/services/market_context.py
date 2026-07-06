from __future__ import annotations

from typing import Any

from analytics.territorial_skills import compute_territorial_stats
from france_travail.client import FranceTravailClient
from services.certification_market_comparison import collect_market_offers
from common.text import clean_text, normalize_for_match
from domain.models import JobOffer, Skill, Territory
from services.market_analysis import analyze_market_fit
from services.recommendation_service import RecommendationService


def _skill_from_value(value: Any) -> Skill:
    if isinstance(value, dict):
        candidate = value.get("normalized_label") or value.get("canonical_label") or value.get("label") or value.get("name")
    else:
        candidate = value
    label = clean_text(candidate)
    return Skill(name=label or str(candidate or ""), normalized_name=label or None, source="web_app")


def _offer_from_normalized_offer(offer: dict[str, Any]) -> JobOffer:
    raw_skills = offer.get("normalized_skills") or offer.get("structured_skills") or offer.get("model_skills") or []
    skills = [_skill_from_value(item) for item in raw_skills if clean_text(item.get("normalized_label") if isinstance(item, dict) else item)]
    offer_id = clean_text(offer.get("offer_id") or offer.get("id") or offer.get("reference") or "")
    if not offer_id:
        from common.text import stable_hash

        offer_id = stable_hash(offer.get("title", ""), offer.get("description", ""), offer.get("rome_code", ""), length=24)
    title = clean_text(offer.get("title") or offer.get("intitule") or "")
    description = clean_text(offer.get("description") or offer.get("body") or "")
    rome_code = clean_text(offer.get("rome_code") or offer.get("rome") or "") or None
    location = clean_text(offer.get("location") or offer.get("city") or offer.get("commune") or offer.get("department") or "") or None
    return JobOffer(offer_id=offer_id, title=title, description=description, skills=skills, rome_code=rome_code, location=location)




def _territory_query_kwargs(territory: Territory) -> dict[str, str | None]:
    code = clean_text(territory.code or territory.department_code or '')
    if not code:
        return {'departement': None, 'commune': None}
    if code.isdigit() and len(code) == 5:
        return {'departement': None, 'commune': code}
    if code.isdigit() and len(code) in {2, 3} or code in {'2A', '2B'}:
        return {'departement': code, 'commune': None}
    if territory.type == 'commune':
        return {'departement': None, 'commune': code}
    return {'departement': code, 'commune': None}


def fetch_offers_by_rome(
    client: FranceTravailClient,
    rome_code: str,
    territory: Territory,
    *,
    radius_km: int | None = None,
    contract_types: list[str] | None = None,
    max_results: int = 500,
) -> list[dict[str, Any]]:
    location = _territory_query_kwargs(territory)
    contract_filters = [clean_text(item) for item in (contract_types or []) if clean_text(item)] or [None]
    collected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for contract_type in contract_filters:
        normalized_offers, _ = collect_market_offers(
            client,
            commune=location['commune'],
            departement=location['departement'],
            distance_km=radius_km or territory.radius_km,
            job_titles=None,
            rome_codes=[rome_code],
            max_offers=max_results,
            page_size=min(20, max_results) if max_results else 20,
        )
        for offer in normalized_offers:
            if contract_type:
                offer_contract = clean_text(offer.get('contract_type') or offer.get('contract_label') or offer.get('contract') or '')
                if offer_contract and contract_type not in offer_contract:
                    continue
            offer_id = clean_text(offer.get('offer_id') or offer.get('id') or '')
            if offer_id and offer_id in seen_ids:
                continue
            if offer_id:
                seen_ids.add(offer_id)
            collected.append(offer)
            if max_results and len(collected) >= max_results:
                return collected
    return collected


def build_market_context(*, skill_extraction: Any, normalized_offers: list[dict[str, Any]], departement: str, recommendation_service: RecommendationService) -> dict[str, Any]:
    extracted_labels = [s.normalized_label for s in skill_extraction.skills if getattr(s, "normalized_label", None)]
    extracted_labels += [s.normalized_label for s in skill_extraction.tools if getattr(s, "normalized_label", None)]
    recommendation = recommendation_service.compare(extracted_labels, normalized_offers)
    market_analysis = analyze_market_fit(
        [_skill_from_value(label) for label in extracted_labels],
        Territory(code=departement or None, label=departement or None, department_code=departement or None, region_code=None, remote_allowed=True),
        [_offer_from_normalized_offer(offer) for offer in normalized_offers],
    )
    territorial_stats = compute_territorial_stats(normalized_offers, territory_key=departement)
    return {
        "recommendation": recommendation,
        "market_analysis": market_analysis,
        "territorial_stats": territorial_stats,
        "extracted_labels": extracted_labels,
        "offer_count": len(normalized_offers),
        "market_offers_preview": normalized_offers[:10],
        "market_offers_more_count": max(len(normalized_offers) - min(len(normalized_offers), 10), 0),
    }
