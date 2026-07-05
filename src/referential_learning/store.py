from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_ANNOTATION_PATH = Path('data/annotation/referential_candidates.jsonl')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class AnnotationStore:
    path: Path = DEFAULT_ANNOTATION_PATH

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

    def save(self, records: Iterable[dict[str, Any]]) -> Path:
        data = [dict(record) for record in records]
        self.path.write_text('\n'.join(json.dumps(record, ensure_ascii=False) for record in data) + ('\n' if data else ''), encoding='utf-8')
        return self.path

    def upsert(self, record: dict[str, Any]) -> None:
        rows = self.load()
        document_id = record.get('document_id')
        if not document_id:
            raise ValueError('document_id manquant')
        replaced = False
        for index, existing in enumerate(rows):
            if existing.get('document_id') == document_id:
                rows[index] = record
                replaced = True
                break
        if not replaced:
            rows.append(record)
        self.save(rows)

    def by_status(self, status: str = 'pending') -> list[dict[str, Any]]:
        return [row for row in self.load() if row.get('status') == status]

    def get(self, document_id: str) -> dict[str, Any] | None:
        for row in self.load():
            if row.get('document_id') == document_id:
                return row
        return None

    def update_status(self, document_id: str, status: str, *, validated_by: str | None = None) -> dict[str, Any]:
        rows = self.load()
        updated: dict[str, Any] | None = None
        for row in rows:
            if row.get('document_id') == document_id:
                row['status'] = status
                row['updated_at'] = _now()
                if validated_by:
                    row['validated_by'] = validated_by
                updated = row
                break
        if updated is None:
            raise KeyError(document_id)
        self.save(rows)
        return updated
