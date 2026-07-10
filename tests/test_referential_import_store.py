"""Tests de non-régression pour ReferentialImportStore."""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from referential_import.store import ReferentialImportStore


def test_store_initialization_no_recursion():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store1 = ReferentialImportStore(db_path)
        assert db_path.exists()

        store2 = ReferentialImportStore(db_path)
        assert db_path.exists()


def test_store_initialization_creates_parent_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "subdir" / "nested" / "test.sqlite3"
        assert not db_path.parent.exists()
        store = ReferentialImportStore(db_path)
        assert db_path.parent.exists()
        assert db_path.exists()


def test_store_tables_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        ReferentialImportStore(db_path)
        conn = sqlite3.connect(db_path)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        expected = {"imports", "blocks", "activities", "competencies",
                     "criteria", "derived_skills", "issues"}
        assert expected.issubset(tables), f"Missing: {expected - tables}"


def test_store_multiple_instances_same_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        s1 = ReferentialImportStore(db_path)
        s2 = ReferentialImportStore(db_path)
        s3 = ReferentialImportStore(db_path)
        for s in (s1, s2, s3):
            assert isinstance(s.list_imports(), list)


def test_store_has_document_returns_false_on_new_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store = ReferentialImportStore(db_path)
        assert store.has_document("fake_hash", "v1.0") is False


def test_store_has_document_method():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store = ReferentialImportStore(db_path)
        result = store.has_document("fake_hash", "v1.0")
        assert result is False


def test_store_list_imports_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store = ReferentialImportStore(db_path)
        assert store.list_imports() == []


def test_store_initialized_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store = ReferentialImportStore(db_path)
        assert store._initialized is True
        store._init_db()
        assert store._initialized is True


def test_non_regression_no_recursion_noOperationalError():
    with tempfile.TemporaryDirectory() as tmp:
        store = ReferentialImportStore(Path(tmp) / "db" / "imports.sqlite3")
        assert store.has_document("abc", "test-version") is False


def test_has_document_after_insert():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store = ReferentialImportStore(db_path)
        sha256 = "deadbeef" * 8
        version = "1.0"
        assert store.has_document(sha256, version) is False

        with store._connect() as conn:
            conn.execute(
                """INSERT INTO imports
                   (id, source_path, file_name, sha256, schema_version,
                    importer_version, page_count, review_status,
                    created_at, validated_at, validated_by,
                    document_json, report_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("id1", "/tmp/test.pdf", "test.pdf", sha256,
                 "1.0", version, 1, "imported", "2025-01-01",
                 None, None, "{}", "{}"),
            )
            conn.commit()
        assert store.has_document(sha256, version) is True
        assert store.has_document(sha256, "other") is False


def test_existing_empty_db_migrated():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        db_path.touch()
        assert db_path.stat().st_size == 0
        store = ReferentialImportStore(db_path)
        assert db_path.stat().st_size > 0
        assert store.has_document("x", "y") is False


def test_existing_referential_imports_table_migrated():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE referential_imports (
                id TEXT PRIMARY KEY,
                source_path TEXT,
                file_name TEXT,
                sha256 TEXT,
                schema_version TEXT,
                importer_version TEXT,
                page_count INTEGER,
                review_status TEXT,
                created_at TEXT,
                validated_at TEXT,
                validated_by TEXT,
                document_json TEXT,
                report_json TEXT
            );
            INSERT INTO referential_imports
                (id, source_path, file_name, sha256, schema_version,
                 importer_version, page_count, review_status, created_at,
                 document_json, report_json)
            VALUES
                ('mig1', '/a.pdf', 'a.pdf', 'hash_aaa', '1.0', 'v1', 1,
                 'imported', '2025-01-01', '{}', '{}');
            """
        )
        conn.close()

        store = ReferentialImportStore(db_path)
        assert store.has_document("hash_aaa", "v1") is True
        imports = store.list_imports()
        assert len(imports) == 1
        assert imports[0]["id"] == "mig1"


def test_connect_never_calls_init_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.sqlite3"
        store = ReferentialImportStore(db_path)
        store._initialized = False
        with store._connect() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "imports" in tables
