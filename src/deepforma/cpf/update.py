from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from deepforma.cpf.io import json_dump


@dataclass(frozen=True)
class UpdateReport:
    """Rapport de mise à jour du catalogue CPF."""

    corpus_hash: str | None
    previous_corpus_hash: str | None
    added: list[str]
    modified: list[str]
    removed: list[str]
    needs_embedding_rebuild: bool


def _read_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def diff_metadata(old_df: pd.DataFrame, new_df: pd.DataFrame) -> dict[str, list[str]]:
    """Calcule les différences entre deux métadonnées CPF."""

    if old_df.empty:
        return {
            "added": new_df.get("formation_uid", pd.Series(dtype=str)).astype(str).tolist(),
            "modified": [],
            "removed": [],
        }
    old = old_df.set_index("formation_uid")
    new = new_df.set_index("formation_uid")
    added = sorted(set(new.index) - set(old.index))
    removed = sorted(set(old.index) - set(new.index))
    modified = sorted(
        uid
        for uid in set(old.index) & set(new.index)
        if str(old.loc[uid, "row_hash"]) != str(new.loc[uid, "row_hash"])
    )
    return {"added": added, "modified": modified, "removed": removed}


def build_update_report(
    *,
    new_metadata_path: Path,
    previous_metadata_path: Path,
    new_manifest_path: Path | None = None,
    previous_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Construit un rapport de diff pour une mise à jour CPF."""

    new_df = _read_metadata(new_metadata_path)
    old_df = _read_metadata(previous_metadata_path)
    diff = diff_metadata(old_df, new_df)
    needs_embedding_rebuild = bool(diff["added"] or diff["modified"] or diff["removed"])
    previous_corpus_hash = None
    if previous_manifest_path and previous_manifest_path.exists():
        import json

        previous_corpus_hash = json.loads(previous_manifest_path.read_text(encoding="utf-8")).get("corpus_hash")
    corpus_hash = None
    if new_manifest_path and new_manifest_path.exists():
        import json

        corpus_hash = json.loads(new_manifest_path.read_text(encoding="utf-8")).get("corpus_hash")
    return {
        "corpus_hash": corpus_hash,
        "previous_corpus_hash": previous_corpus_hash,
        "diff": diff,
        "needs_embedding_rebuild": needs_embedding_rebuild,
    }

