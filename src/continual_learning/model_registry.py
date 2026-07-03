from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.text import stable_hash


def utc_date() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ModelVersionInfo:
    version: str
    model_dir: Path
    state: str
    previous_version: str | None


def registry_root(root: str | Path = "models/skill-extractor") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "versions").mkdir(parents=True, exist_ok=True)
    (root / "candidates").mkdir(parents=True, exist_ok=True)
    return root


def load_registry(path: str | Path = "models/skill-extractor/registry.json") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"versions": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(payload: dict[str, Any], path: str | Path = "models/skill-extractor/registry.json") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def model_version_name(prefix: str, git_hash: str, dataset_hashes: dict[str, str], seed: int) -> str:
    return f"{prefix}-{stable_hash(git_hash, json.dumps(dataset_hashes, sort_keys=True), seed, length=10)}"


def update_registry(
    *,
    version: str,
    git_hash: str,
    base_model: str | None,
    dataset_hashes: dict[str, str],
    example_count: int,
    metrics: dict[str, Any],
    taxonomy: dict[str, Any],
    referential: dict[str, Any],
    state: str,
    previous_version: str | None,
    model_dir: str | Path,
    registry_path: str | Path = "models/skill-extractor/registry.json",
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    registry["versions"] = [item for item in registry.get("versions", []) if item.get("version") != version]
    item = {
        "version": version,
        "date": utc_date(),
        "git_hash": git_hash,
        "base_model": base_model,
        "dataset_hashes": dataset_hashes,
        "example_count": example_count,
        "metrics": metrics,
        "taxonomy": taxonomy,
        "referential": referential,
        "state": state,
        "previous_version": previous_version,
        "model_dir": str(model_dir),
    }
    registry["versions"].append(item)
    save_registry(registry, registry_path)
    return item


def promote_model_version(
    *,
    version_dir: str | Path,
    production_link: str | Path = "models/skill-extractor/production",
    backup_link: str | Path | None = None,
) -> Path:
    version_dir = Path(version_dir)
    production_link = Path(production_link)
    production_link.parent.mkdir(parents=True, exist_ok=True)
    if backup_link is not None:
        backup_link = Path(backup_link)
        if production_link.is_symlink() or production_link.exists():
            if backup_link.exists() or backup_link.is_symlink():
                backup_link.unlink()
            shutil.copytree(production_link, backup_link, symlinks=True)
    tmp_link = production_link.parent / ".production.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(version_dir.resolve(), target_is_directory=True)
    tmp_link.replace(production_link)
    return production_link

