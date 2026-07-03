from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match, stable_hash


@dataclass(frozen=True)
class ExportedExample:
    id: str
    text: str
    entities: list[dict[str, Any]]
    document_skills: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "entities": self.entities,
            "document_skills": self.document_skills,
            "metadata": self.metadata,
        }


def example_id(offer_id: str, content_version: str) -> str:
    return stable_hash(offer_id, content_version, length=24)


def _annotation_to_entity(annotation: dict[str, Any]) -> dict[str, Any] | None:
    start = annotation.get("start")
    end = annotation.get("end")
    if start is None or end is None:
        return None
    try:
        start = int(start)
        end = int(end)
    except Exception:
        return None
    if start < 0 or end <= start:
        return None
    return {
        "start": start,
        "end": end,
        "label": annotation.get("label", "SKILL"),
        "canonical_name": clean_text(annotation.get("canonical_name")),
        "surface_form": clean_text(annotation.get("surface_form")),
        "provenance": annotation.get("provenance"),
    }


def build_export_record(offer: dict[str, Any], annotations: Iterable[dict[str, Any]]) -> ExportedExample:
    text = clean_text(offer.get("description_original"))
    entities: list[dict[str, Any]] = []
    document_skills: list[dict[str, Any]] = []

    for annotation in annotations:
        entity = _annotation_to_entity(annotation)
        if entity is None:
            document_skills.append(
                {
                    "label": annotation.get("label", "SKILL"),
                    "canonical_name": clean_text(annotation.get("canonical_name")),
                    "surface_form": clean_text(annotation.get("surface_form")),
                    "provenance": annotation.get("provenance"),
                    "source": annotation.get("source"),
                    "validation_status": annotation.get("validation_status"),
                }
            )
        else:
            entities.append(entity)

    metadata = {
        "territory": offer.get("territory"),
        "job_family": offer.get("job_family"),
        "offer_date": offer.get("collected_at"),
    }
    return ExportedExample(
        id=example_id(str(offer["offer_id"]), str(offer["content_version"])),
        text=text,
        entities=entities,
        document_skills=document_skills,
        metadata=metadata,
    )


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(f"{record}\n")

