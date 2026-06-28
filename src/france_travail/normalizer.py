from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from common.text import clean_text, normalize_for_match, stable_hash
from france_travail.skill_extractor import extract_structured_skills


@dataclass
class NormalizedOffer:
    offer_id: str
    title: str
    description: str
    offer_text: str
    rome_code: str | None
    rome_label: str | None
    department_code: str | None
    commune_code: str | None
    postal_code: str | None
    location_label: str | None
    latitude: float | None
    longitude: float | None
    contract_type: str | None
    contract_label: str | None
    creation_date: str | None
    update_date: str | None
    structured_skills: list
    model_skills: list
    normalized_skills: list
    raw_offer: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["raw_offer"] = self.raw_offer
        return data


def _pick(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = raw.get(key)
        if value not in (None, "", []):
            return value
    return None


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, "", "null"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_offer_text(title: str, description: str) -> str:
    parts = []
    title = clean_text(title)
    description = clean_text(description)
    if title:
        parts.append(f"Titre : {title}")
    if description:
        parts.append(f"Description : {description}")
    return "\n".join(parts)


def normalize_offer(raw_offer: dict[str, Any], *, model_skills: list[dict[str, Any]] | None = None) -> NormalizedOffer:
    title = clean_text(_pick(raw_offer, "title", "intitule", "titre", "libelle"))
    description = clean_text(_pick(raw_offer, "description", "text", "content"))
    offer_text = build_offer_text(title, description)
    structured_skills = extract_structured_skills(raw_offer)
    model_skills = model_skills or []
    normalized_skills = [
        item["label"]
        for item in structured_skills
        if item.get("label")
    ]
    normalized_skills.extend(
        item["label"]
        for item in model_skills
        if isinstance(item, dict) and item.get("label")
    )
    normalized_skills = list(dict.fromkeys(clean_text(skill) for skill in normalized_skills if clean_text(skill)))

    location = raw_offer.get("lieuTravail") or raw_offer.get("location") or {}
    if isinstance(location, dict):
        location_label = clean_text(_pick(location, "libelle", "label", "name"))
        commune_code = clean_text(_pick(location, "codeInseeCommune", "communeCode"))
        postal_code = clean_text(_pick(location, "codePostal", "postalCode"))
        latitude = _to_float(_pick(location, "latitude"))
        longitude = _to_float(_pick(location, "longitude"))
    else:
        location_label = clean_text(location)
        commune_code = postal_code = None
        latitude = longitude = None

    rome = raw_offer.get("rome") or {}
    if isinstance(rome, dict):
        rome_code = clean_text(_pick(rome, "code", "romeCode"))
        rome_label = clean_text(_pick(rome, "libelle", "label"))
    else:
        rome_code = clean_text(raw_offer.get("romeCode"))
        rome_label = clean_text(raw_offer.get("romeLabel"))

    contract = raw_offer.get("contract") or raw_offer.get("contrat") or {}
    if isinstance(contract, dict):
        contract_type = clean_text(_pick(contract, "type", "code"))
        contract_label = clean_text(_pick(contract, "label", "libelle"))
    else:
        contract_type = clean_text(raw_offer.get("contractType"))
        contract_label = clean_text(raw_offer.get("contractLabel"))

    offer_id = clean_text(_pick(raw_offer, "id", "idOffre", "reference")) or stable_hash(title, description, rome_code, location_label)
    department_code = clean_text(_pick(raw_offer, "departmentCode", "codeDepartement", "departement"))

    return NormalizedOffer(
        offer_id=offer_id,
        title=title,
        description=description,
        offer_text=offer_text,
        rome_code=rome_code or None,
        rome_label=rome_label or None,
        department_code=department_code or None,
        commune_code=commune_code or None,
        postal_code=postal_code or None,
        location_label=location_label or None,
        latitude=latitude,
        longitude=longitude,
        contract_type=contract_type or None,
        contract_label=contract_label or None,
        creation_date=clean_text(_pick(raw_offer, "creationDate", "dateCreation")) or None,
        update_date=clean_text(_pick(raw_offer, "updateDate", "dateMiseAJour")) or None,
        structured_skills=structured_skills,
        model_skills=model_skills,
        normalized_skills=normalized_skills,
        raw_offer=raw_offer,
    )

