from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from common.text import clean_text

from .models import DerivedSkill, EvaluationCriterion, ImportIssue, ImportReport, OfficialCompetency, ReferentialActivity, ReferentialBlock, ReferentialDocument

DEFAULT_DB_PATH = Path("data/referentials/referential_imports.sqlite3")

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS imports (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    importer_version TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    review_status TEXT NOT NULL,
    created_at TEXT,
    validated_at TEXT,
    validated_by TEXT,
    document_json TEXT NOT NULL,
    report_json TEXT NOT NULL,
    UNIQUE(sha256, importer_version)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_imports_raw_hash_version
ON imports(sha256, importer_version);
CREATE TABLE IF NOT EXISTS blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_sha256 TEXT NOT NULL,
    code TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(document_sha256, code)
);
CREATE TABLE IF NOT EXISTS activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_sha256 TEXT NOT NULL,
    code TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(document_sha256, code)
);
CREATE TABLE IF NOT EXISTS competencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_sha256 TEXT NOT NULL,
    code TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(document_sha256, code)
);
CREATE TABLE IF NOT EXISTS criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_sha256 TEXT NOT NULL,
    code TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(document_sha256, code)
);
CREATE TABLE IF NOT EXISTS derived_skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_sha256 TEXT NOT NULL,
    code TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_sha256 TEXT NOT NULL,
    severity TEXT NOT NULL,
    code TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""


class ReferentialImportStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("PRAGMA journal_mode=WAL;")
            existing = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            conn.executescript(_CREATE_TABLES_SQL)
            if "referential_imports" in existing and "imports" not in existing:
                conn.executescript(
                    """
                    INSERT OR IGNORE INTO imports
                        (id, source_path, file_name, sha256, schema_version,
                         importer_version, page_count, review_status,
                         created_at, validated_at, validated_by,
                         document_json, report_json)
                    SELECT
                        id, source_path, file_name, sha256, schema_version,
                        importer_version, page_count, review_status,
                        created_at, validated_at, validated_by,
                        document_json, report_json
                    FROM referential_imports;
                    """
                )
            conn.commit()
        self._initialized = True

    def has_document(self, sha256: str, importer_version: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM imports WHERE sha256 = ? AND importer_version = ?",
                (sha256, importer_version),
            ).fetchone()
            return row is not None

    def list_imports(self, *, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM imports"
        params: tuple[Any, ...] = ()
        if status:
            query += " WHERE review_status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def list_annotations(self, query: str = "1=1", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM competencies WHERE {query}", params).fetchall()
        return [dict(row) for row in rows]

    def save_import(
        self,
        document: ReferentialDocument,
        report: ImportReport,
        blocks: list[ReferentialBlock],
        activities: list[ReferentialActivity],
        competencies: list[OfficialCompetency],
        criteria: list[EvaluationCriterion],
        derived_skills: list[DerivedSkill],
        issues: list[ImportIssue],
        *,
        review_status: str = "pending",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO imports
                (id, source_path, file_name, sha256, schema_version, importer_version, page_count, review_status, created_at, validated_at, validated_by, document_json, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.source_path,
                    document.file_name,
                    document.sha256,
                    document.schema_version,
                    document.importer_version,
                    document.page_count,
                    review_status,
                    document.collected_at,
                    document.validated_at,
                    document.validated_by,
                    json.dumps(document.to_dict(), ensure_ascii=False),
                    json.dumps(report.to_dict(), ensure_ascii=False),
                ),
            )
            for table, items in (
                ("blocks", blocks),
                ("activities", activities),
                ("competencies", competencies),
                ("criteria", criteria),
            ):
                conn.execute(f"DELETE FROM {table} WHERE document_sha256 = ?", (document.sha256,))
                for item in items:
                    code = getattr(item, "code", "")
                    conn.execute(
                        f"INSERT OR REPLACE INTO {table} (document_sha256, code, payload_json) VALUES (?, ?, ?)",
                        (document.sha256, code, json.dumps(item.to_dict(), ensure_ascii=False)),
                    )
            conn.execute("DELETE FROM derived_skills WHERE document_sha256 = ?", (document.sha256,))
            for index, item in enumerate(derived_skills):
                code = f"{item.source_code}:{index}"
                conn.execute(
                    "INSERT INTO derived_skills (document_sha256, code, payload_json) VALUES (?, ?, ?)",
                    (document.sha256, code, json.dumps(item.to_dict(), ensure_ascii=False)),
                )
            conn.execute("DELETE FROM issues WHERE document_sha256 = ?", (document.sha256,))
            for issue in issues:
                conn.execute(
                    "INSERT INTO issues (document_sha256, severity, code, payload_json) VALUES (?, ?, ?, ?)",
                    (document.sha256, issue.severity, issue.code, json.dumps(issue.to_dict(), ensure_ascii=False)),
                )

    def load_import_payload(self, sha256: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT document_json, report_json FROM imports WHERE sha256 = ?", (sha256,)).fetchone()
            if not row:
                return None
            return {"document": json.loads(row["document_json"]), "report": json.loads(row["report_json"])}

