from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analytics.territorial_skills import compute_territorial_stats, stats_to_dataframe
from common.text import clean_text, normalize_for_match, stable_hash
from continual_learning.store import ContinualLearningStore
from france_travail.client import FranceTravailClient, SearchCriteria
from france_travail.normalizer import normalize_offer
from inference.skill_model import SkillModel
from skills.merge_offer_skills import extract_skills_from_text, merge_offer_skills


LOGGER = logging.getLogger(__name__)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _flatten_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flattened_rows = [{key: _flatten_value(value) for key, value in row.items()} for row in rows]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flattened_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flattened_rows)


def deduplicate_offers(offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for offer in offers:
        offer_id = str(offer.get("offer_id") or offer.get("id") or offer.get("raw_offer", {}).get("id") or "")
        content_version = str(offer.get("content_version") or offer.get("content_hash") or "")
        key = (offer_id, content_version)
        if not offer_id or key in seen:
            continue
        seen.add(key)
        unique.append(offer)
    return unique


def _skill_provenance(label: str, text: str) -> str:
    label_norm = normalize_for_match(label)
    text_norm = normalize_for_match(text)
    if label_norm and label_norm in text_norm:
        return "exact_reference_match"
    return "semantic_match"


def _structured_skill_payload(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in skills:
        label = clean_text(item.get("label"))
        if not label:
            continue
        payload.append(
            {
                "code": clean_text(item.get("code")) or None,
                "label": label,
                "canonical_name": label,
                "requirement": clean_text(item.get("requirement")) or None,
                "source": "france_travail_api",
                "provenance": "france_travail_api",
                "confidence": 1.0,
            }
        )
    return payload


def _explicit_skill_payload(skills: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in skills:
        label = clean_text(item.get("canonical_label") or item.get("label"))
        if not label:
            continue
        payload.append(
            {
                "label": label,
                "canonical_name": label,
                "surface_form": clean_text(item.get("source_label") or label),
                "start": item.get("start"),
                "end": item.get("end"),
                "source": item.get("sources", ["text_explicit"])[0],
                "provenance": _skill_provenance(label, text),
                "confidence": float(item.get("confidence", 0.0)),
                "is_explicit": True,
            }
        )
    return payload


def _model_skill_payload(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in skills:
        label = clean_text(item.get("label"))
        if not label:
            continue
        payload.append(
            {
                "label": label,
                "canonical_name": label,
                "surface_form": None,
                "start": None,
                "end": None,
                "source": item.get("source", "camembert_multilabel"),
                "provenance": "model_prediction",
                "confidence": float(item.get("probability", item.get("confidence", 0.0))),
                "is_explicit": False,
            }
        )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect France Travail offers and enrich them with Deepforma models.")
    parser.add_argument("--departement", default=None)
    parser.add_argument("--commune", default=None)
    parser.add_argument("--rome-code", default=None)
    parser.add_argument("--keywords", default=None)
    parser.add_argument("--contract", default=None)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--max-offers", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--pause-seconds", type=float, default=0.2)
    parser.add_argument("--save-prefix", type=str, default=None)
    return parser


def run_collection(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    france_travail_root = project_root / "data" / "france_travail"
    raw_dir = france_travail_root / "raw"
    normalized_dir = france_travail_root / "normalized"
    reports_dir = france_travail_root / "reports"
    continual_store = ContinualLearningStore(project_root / "data" / "continual_learning" / "continual_learning.sqlite3")

    output_path: Path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Le fichier de sortie existe déjà: {output_path}. Utiliser --overwrite pour l'écraser."
        )

    client = FranceTravailClient()
    model = SkillModel() if args.run_model else None

    criteria = SearchCriteria(
        keywords=args.keywords,
        rome_code=args.rome_code,
        commune=args.commune,
        departement=args.departement,
        contract_type=args.contract,
        size=args.page_size,
    )

    raw_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []
    review_events = 0

    for offer in client.iter_offers(
        criteria,
        max_pages=args.max_pages,
        max_offers=args.max_offers,
        page_size=args.page_size,
        pause_seconds=args.pause_seconds,
    ):
        collected_at = datetime.now(timezone.utc).isoformat()
        raw_rows.append(
            {
                "collected_at": collected_at,
                "offer": offer,
            }
        )
        model_skills: list[dict[str, Any]] = []
        if model is not None:
            model_skills = model.predict_offer(
                offer.get("title") or offer.get("intitule"),
                offer.get("description"),
                structured_skills=offer.get("competences_structured", []),
            )
        normalized = normalize_offer(offer, model_skills=model_skills)
        explicit_skills = extract_skills_from_text(normalized.offer_text)
        merged_skills = merge_offer_skills(
            structured_skills=normalized.structured_skills,
            explicit_skills=explicit_skills,
            model_skills=normalized.model_skills,
            rome_skills=[],
        )
        normalized_dict = normalized.to_dict()
        normalized_dict["collected_at"] = collected_at
        normalized_dict["content_version"] = continual_store.normalize_content_version(
            normalized.title,
            normalized.description,
            normalized.structured_skills,
        )
        normalized_dict["merged_skills"] = merged_skills
        normalized_rows.append(normalized_dict)
        enriched_rows.append(
            {
                **normalized_dict,
                "skills_flat": " | ".join(item["canonical_label"] for item in merged_skills),
                "skills_sources": " | ".join(",".join(item["sources"]) for item in merged_skills),
            }
        )

        structured_payload = _structured_skill_payload(normalized.structured_skills)
        explicit_payload = _explicit_skill_payload(explicit_skills, normalized.offer_text)
        model_payload = _model_skill_payload(model_skills)
        predicted_payload = [
            {
                "label": item.get("label"),
                "canonical_name": item.get("canonical_name") or item.get("label"),
                "confidence": item.get("confidence", item.get("probability", 0.0)),
                "source": item.get("source", "camembert_multilabel"),
                "provenance": item.get("provenance", "model_prediction"),
                "surface_form": item.get("surface_form"),
                "start": item.get("start"),
                "end": item.get("end"),
                "is_explicit": bool(item.get("is_explicit", False)),
            }
            for item in [*structured_payload, *explicit_payload, *model_payload]
        ]
        offsets = [item for item in predicted_payload if item.get("start") is not None and item.get("end") is not None]
        sources = [
            {
                "label": item.get("label"),
                "source": item.get("source"),
                "provenance": item.get("provenance"),
                "confidence": item.get("confidence", 0.0),
            }
            for item in predicted_payload
        ]
        offer_result = continual_store.upsert_offer(
            offer_id=normalized.offer_id,
            title=normalized.title,
            description_original=normalized.description,
            collected_at=collected_at,
            location_label=normalized.location_label,
            territory=normalized.department_code or normalized.commune_code or normalized.rome_code,
            job_family=normalized.rome_label,
            structured_skills=structured_payload,
            predicted_skills=predicted_payload,
            detected_forms=explicit_payload,
            offsets=offsets,
            confidence={
                "structured": 1.0 if structured_payload else 0.0,
                "explicit": max((float(item.get("confidence", 0.0)) for item in explicit_payload), default=0.0),
                "model": max((float(item.get("confidence", 0.0)) for item in model_payload), default=0.0),
            },
            sources=sources,
            model_version=str(model.model_dir) if model is not None else None,
            validation_status="pending",
            raw_payload=normalized_dict,
        )
        for item in structured_payload:
            continual_store.upsert_annotation(
                offer_row_id=offer_result.offer_row_id,
                offer_id=normalized.offer_id,
                content_version=offer_result.content_version,
                canonical_name=item["canonical_name"],
                surface_form=item["label"],
                normalized_name=item["canonical_name"],
                label="SKILL",
                start=None,
                end=None,
                confidence=float(item.get("confidence", 1.0)),
                source="france_travail_api",
                provenance="france_travail_api",
                is_explicit=False,
                text_sentence=None,
                referential_code=item.get("code"),
                referential_label=item.get("label"),
                validation_status="pending",
            )
        for item in explicit_payload:
            continual_store.upsert_annotation(
                offer_row_id=offer_result.offer_row_id,
                offer_id=normalized.offer_id,
                content_version=offer_result.content_version,
                canonical_name=item["canonical_name"],
                surface_form=item["surface_form"],
                normalized_name=item["canonical_name"],
                label="SKILL",
                start=item.get("start"),
                end=item.get("end"),
                confidence=float(item.get("confidence", 0.0)),
                source=item.get("source", "text_explicit"),
                provenance=item.get("provenance", "semantic_match"),
                is_explicit=True,
                text_sentence=normalized.offer_text,
                validation_status="pending",
            )
            review_events += 1
        for item in model_payload:
            continual_store.upsert_annotation(
                offer_row_id=offer_result.offer_row_id,
                offer_id=normalized.offer_id,
                content_version=offer_result.content_version,
                canonical_name=item["canonical_name"],
                surface_form=item["surface_form"] or item["canonical_name"],
                normalized_name=item["canonical_name"],
                label="SKILL",
                start=None,
                end=None,
                confidence=float(item.get("confidence", 0.0)),
                source=item.get("source", "camembert_multilabel"),
                provenance="model_prediction",
                is_explicit=False,
                text_sentence=None,
                validation_status="pending",
            )

    normalized_rows = deduplicate_offers(normalized_rows)
    enriched_rows = deduplicate_offers(enriched_rows)

    normalized_jsonl = output_path
    save_jsonl(normalized_jsonl, normalized_rows)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    normalized_copy_path = normalized_dir / output_path.name
    if normalized_copy_path != normalized_jsonl:
        save_jsonl(normalized_copy_path, normalized_rows)

    csv_path = normalized_dir / f"{output_path.stem}.csv"
    save_csv(csv_path, enriched_rows)

    raw_path = raw_dir / f"{output_path.stem}_raw.jsonl"
    save_jsonl(raw_path, raw_rows)

    report = {
        "territory": {
            "departement": args.departement,
            "commune": args.commune,
            "rome_code": args.rome_code,
        },
        "query": {
            "keywords": args.keywords,
            "contract": args.contract,
            "max_pages": args.max_pages,
            "max_offers": args.max_offers,
            "page_size": args.page_size,
        },
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "offers_received": len(raw_rows),
        "offers_deduplicated": len(normalized_rows),
        "offers_with_structured_skills": sum(1 for row in normalized_rows if row.get("structured_skills")),
        "offers_with_model_skills": sum(1 for row in normalized_rows if row.get("model_skills")),
        "unique_structured_skills": len(
            {
                item["label"]
                for row in normalized_rows
                for item in row.get("structured_skills", [])
                if item.get("label")
            }
        ),
        "unique_normalized_skills": len(
            {
                skill
                for row in normalized_rows
                for skill in row.get("normalized_skills", [])
                if skill
            }
        ),
        "errors": [],
        "review_candidates_created": review_events,
    }
    report_path = reports_dir / f"{output_path.stem}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if normalized_rows:
        stats = compute_territorial_stats(normalized_rows, territory_key=args.departement or args.commune or args.rome_code or "unknown")
        stats_to_dataframe(stats).to_csv(reports_dir / f"{output_path.stem}_territorial_stats.csv", index=False, encoding="utf-8")

    LOGGER.info("Collecte terminée: %s offres normalisées", len(normalized_rows))
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()
    run_collection(args)


if __name__ == "__main__":
    main()
