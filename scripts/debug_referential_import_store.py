#!/usr/bin/env python3
"""Debug script for ReferentialImportStore – verify DB initialisation and tables."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from referential_import.store import ReferentialImportStore


def main(db_path: str | None = None) -> None:
    from referential_import.store import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    print(f"DB path      : {path}")
    print(f"DB exists    : {path.exists()}")
    if path.exists():
        print(f"DB size      : {path.stat().st_size} bytes")

    store = ReferentialImportStore(path)

    with store._connect() as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        print(f"Tables       : {tables}")

        if "imports" in tables:
            cols = [
                row[1]
                for row in conn.execute("PRAGMA table_info(imports)").fetchall()
            ]
            print(f"imports cols : {cols}")
            count = conn.execute("SELECT count(*) FROM imports").fetchone()[0]
            print(f"imports rows : {count}")
        else:
            print("ERROR: table 'imports' does NOT exist")

    result = store.has_document("__debug_nonexistent_hash__", "__debug_version__")
    print(f"has_document : {result}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
