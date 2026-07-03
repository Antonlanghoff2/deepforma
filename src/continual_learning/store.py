from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from common.text import clean_text, normalize_for_match, stable_hash
from continual_learning.schema import DETAILED_SCHEMA_SQL, SCHEMA_VERSION


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def ensure_db_path(path: str | Path | None = None) -> Path:
    if path is None:
        path = Path("data/continual_learning/continual_learning.sqlite3")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class OfferUpsertResult:
    offer_row_id: int
    inserted: bool
    content_version: str


class ContinualLearningStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = ensure_db_path(db_path)
        self._ensure_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(DETAILED_SCHEMA_SQL)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def normalize_content_version(title: str, description: str, structured_skills: Iterable[dict[str, Any]] | None = None) -> str:
        skill_bits = []
        for item in structured_skills or []:
            label = clean_text(item.get("canonical_name") or item.get("label") or "")
            code = clean_text(item.get("code") or item.get("referential_code") or "")
            if label or code:
                skill_bits.append(f"{label}:{code}")
        payload = "||".join(
            [
                normalize_for_match(title),
                normalize_for_match(description),
                "|".join(sorted(dict.fromkeys(skill_bits))),
            ]
        )
        return stable_hash(payload, length=24)

    def upsert_offer(
        self,
        *,
        offer_id: str,
        title: str,
        description_original: str,
        collected_at: str,
        location_label: str | None,
        territory: str | None,
        job_family: str | None,
        structured_skills: list[dict[str, Any]],
        predicted_skills: list[dict[str, Any]],
        detected_forms: list[dict[str, Any]],
        offsets: list[dict[str, Any]],
        confidence: dict[str, Any],
        sources: list[dict[str, Any]],
        model_version: str | None,
        validation_status: str = "pending",
        validation_actor: str | None = None,
        validation_date: str | None = None,
        validation_note: str | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> OfferUpsertResult:
        title = clean_text(title)
        description_original = clean_text(description_original)
        content_version = self.normalize_content_version(title, description_original, structured_skills)
        content_hash = stable_hash(title, description_original, content_version, length=32)
        now = utcnow()
        raw_payload = raw_payload or {}

        with self.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM continual_learning_offers WHERE offer_id = ? AND content_version = ?",
                (offer_id, content_version),
            ).fetchone()
            payload = (
                offer_id,
                content_version,
                content_hash,
                title,
                description_original,
                collected_at,
                location_label,
                territory,
                job_family,
                _json(structured_skills),
                _json(predicted_skills),
                _json(detected_forms),
                _json(offsets),
                _json(confidence),
                _json(sources),
                model_version,
                validation_status,
                validation_actor,
                validation_date,
                validation_note,
                now,
                now,
                _json(raw_payload),
            )
            if existing:
                conn.execute(
                    """
                    UPDATE continual_learning_offers
                    SET content_hash = ?, title = ?, description_original = ?, collected_at = ?,
                        location_label = ?, territory = ?, job_family = ?, structured_skills_json = ?,
                        predicted_skills_json = ?, detected_forms_json = ?, offsets_json = ?,
                        confidence_json = ?, sources_json = ?, model_version = ?, validation_status = ?,
                        validation_actor = ?, validation_date = ?, validation_note = ?, updated_at = ?,
                        raw_payload_json = ?
                    WHERE id = ?
                    """,
                    (
                        content_hash,
                        title,
                        description_original,
                        collected_at,
                        location_label,
                        territory,
                        job_family,
                        _json(structured_skills),
                        _json(predicted_skills),
                        _json(detected_forms),
                        _json(offsets),
                        _json(confidence),
                        _json(sources),
                        model_version,
                        validation_status,
                        validation_actor,
                        validation_date,
                        validation_note,
                        now,
                        _json(raw_payload),
                        existing["id"],
                    ),
                )
                offer_row_id = int(existing["id"])
                inserted = False
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO continual_learning_offers (
                        offer_id, content_version, content_hash, title, description_original,
                        collected_at, location_label, territory, job_family, structured_skills_json,
                        predicted_skills_json, detected_forms_json, offsets_json, confidence_json,
                        sources_json, model_version, validation_status, validation_actor,
                        validation_date, validation_note, created_at, updated_at, raw_payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
                offer_row_id = int(cursor.lastrowid)
                inserted = True

        return OfferUpsertResult(offer_row_id=offer_row_id, inserted=inserted, content_version=content_version)

    def upsert_annotation(
        self,
        *,
        offer_row_id: int,
        offer_id: str,
        content_version: str,
        canonical_name: str,
        surface_form: str,
        normalized_name: str,
        label: str,
        start: int | None,
        end: int | None,
        confidence: float,
        source: str,
        provenance: str,
        is_explicit: bool,
        text_sentence: str | None = None,
        referential_code: str | None = None,
        referential_label: str | None = None,
        validation_status: str = "pending",
        correction: dict[str, Any] | None = None,
        rejected_reason: str | None = None,
        validated_at: str | None = None,
        validated_by: str | None = None,
    ) -> None:
        now = utcnow()
        correction = correction or {}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO continual_learning_annotations (
                    offer_row_id, offer_id, content_version, canonical_name, surface_form,
                    normalized_name, label, start, end, start_key, end_key, confidence, source, provenance,
                    is_explicit, text_sentence, referential_code, referential_label,
                    validation_status, correction_json, rejected_reason, validated_at,
                    validated_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(offer_id, content_version, normalized_name, start_key, end_key, provenance, source)
                DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    surface_form = excluded.surface_form,
                    label = excluded.label,
                    confidence = excluded.confidence,
                    is_explicit = excluded.is_explicit,
                    text_sentence = excluded.text_sentence,
                    referential_code = excluded.referential_code,
                    referential_label = excluded.referential_label,
                    validation_status = excluded.validation_status,
                    correction_json = excluded.correction_json,
                    rejected_reason = excluded.rejected_reason,
                    validated_at = excluded.validated_at,
                    validated_by = excluded.validated_by,
                    updated_at = excluded.updated_at
                """,
                (
                    offer_row_id,
                    offer_id,
                    content_version,
                    clean_text(canonical_name),
                    clean_text(surface_form),
                    normalize_for_match(normalized_name) or normalize_for_match(canonical_name),
                    clean_text(label) or "SKILL",
                    start,
                    end,
                    -1 if start is None else int(start),
                    -1 if end is None else int(end),
                    float(confidence),
                    clean_text(source),
                    provenance,
                    1 if is_explicit else 0,
                    clean_text(text_sentence),
                    clean_text(referential_code),
                    clean_text(referential_label),
                    validation_status,
                    _json(correction),
                    clean_text(rejected_reason),
                    validated_at,
                    validated_by,
                    now,
                    now,
                ),
            )

    def list_offers(self, where: str = "1=1", params: tuple[Any, ...] = (), limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM continual_learning_offers WHERE " + where + " ORDER BY collected_at DESC, id DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def list_annotations(self, where: str = "1=1", params: tuple[Any, ...] = (), limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM continual_learning_annotations WHERE " + where + " ORDER BY id DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def get_offer(self, offer_row_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM continual_learning_offers WHERE id = ?", (offer_row_id,)).fetchone()
            return dict(row) if row else None

    def get_offer_with_annotations(self, offer_row_id: int) -> dict[str, Any] | None:
        offer = self.get_offer(offer_row_id)
        if not offer:
            return None
        annotations = self.list_annotations("offer_row_id = ?", (offer_row_id,))
        offer["annotations"] = annotations
        return offer

    def get_annotation(self, annotation_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM continual_learning_annotations WHERE id = ?", (annotation_id,)).fetchone()
            return dict(row) if row else None

    def update_annotation_fields(self, annotation_id: int, **fields: Any) -> None:
        if not fields:
            return
        now = utcnow()
        allowed = {
            "canonical_name",
            "surface_form",
            "normalized_name",
            "label",
            "start",
            "end",
            "start_key",
            "end_key",
            "confidence",
            "source",
            "provenance",
            "is_explicit",
            "text_sentence",
            "referential_code",
            "referential_label",
            "validation_status",
            "correction_json",
            "rejected_reason",
            "validated_at",
            "validated_by",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = now
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = []
        for key in updates:
            value = updates[key]
            if key in {"canonical_name", "surface_form", "normalized_name", "label", "source", "provenance", "text_sentence", "referential_code", "referential_label", "rejected_reason", "validated_by"} and value is not None:
                value = clean_text(value)
            if key == "correction_json":
                value = _json(value if value is not None else {})
            values.append(value)
        with self.connect() as conn:
            conn.execute(f"UPDATE continual_learning_annotations SET {assignments} WHERE id = ?", tuple(values) + (annotation_id,))

    def mark_offer_status(
        self,
        offer_row_id: int,
        validation_status: str,
        *,
        validation_actor: str | None = None,
        validation_note: str | None = None,
    ) -> None:
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE continual_learning_offers
                SET validation_status = ?, validation_actor = ?, validation_note = ?, validation_date = ?, updated_at = ?
                WHERE id = ?
                """,
                (validation_status, validation_actor, validation_note, now, now, offer_row_id),
            )

    def add_validation_event(self, offer_row_id: int, actor: str, event_type: str, details: dict[str, Any] | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO continual_learning_validation_events (offer_row_id, actor, event_type, details_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (offer_row_id, actor, event_type, _json(details or {}), utcnow()),
            )

    def register_model_version(
        self,
        *,
        version: str,
        date: str,
        git_hash: str,
        base_model: str | None,
        dataset_hashes: dict[str, str],
        example_count: int,
        metrics: dict[str, Any],
        taxonomy: dict[str, Any],
        referential: dict[str, Any],
        state: str,
        previous_version: str | None,
        model_dir: str,
    ) -> None:
        now = utcnow()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO continual_learning_model_registry (
                    version, date, git_hash, base_model, dataset_hashes_json, example_count,
                    metrics_json, taxonomy_json, referential_json, state, previous_version,
                    model_dir, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version) DO UPDATE SET
                    date = excluded.date,
                    git_hash = excluded.git_hash,
                    base_model = excluded.base_model,
                    dataset_hashes_json = excluded.dataset_hashes_json,
                    example_count = excluded.example_count,
                    metrics_json = excluded.metrics_json,
                    taxonomy_json = excluded.taxonomy_json,
                    referential_json = excluded.referential_json,
                    state = excluded.state,
                    previous_version = excluded.previous_version,
                    model_dir = excluded.model_dir,
                    updated_at = excluded.updated_at
                """,
                (
                    version,
                    date,
                    git_hash,
                    base_model,
                    _json(dataset_hashes),
                    example_count,
                    _json(metrics),
                    _json(taxonomy),
                    _json(referential),
                    state,
                    previous_version,
                    model_dir,
                    now,
                    now,
                ),
            )

    def get_model_version(self, version: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM continual_learning_model_registry WHERE version = ?", (version,)).fetchone()
            if not row:
                return None
            item = dict(row)
            for key in ("dataset_hashes_json", "metrics_json", "taxonomy_json", "referential_json"):
                item[key[:-5]] = _loads(item.pop(key), {})
            return item

    def list_model_versions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM continual_learning_model_registry ORDER BY datetime(created_at) DESC, id DESC").fetchall()
            result = []
            for row in rows:
                item = dict(row)
                for key in ("dataset_hashes_json", "metrics_json", "taxonomy_json", "referential_json"):
                    item[key[:-5]] = _loads(item.pop(key), {})
                result.append(item)
            return result

