from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from common.text import clean_text, normalize_for_match, stable_hash


LOGGER = logging.getLogger(__name__)

DEFAULT_INPUTS = [
    Path("data/processed/dataset_entrainement.csv"),
    Path("data/processed/dataset_formations_nettoye.csv"),
    Path("data/processed/dataset_a_verifier.csv"),
]

TEXT_FIELDS = [
    "texte_modele",
    "intitule",
    "intitulé",
    "titre",
    "description",
    "objectifs",
    "programme",
    "contenu",
    "competences",
    "competences_ia",
    "competences_ia_suggerees",
    "tags",
    "metiers_cibles",
    "certification",
    "code_certification",
    "code_rncp",
    "code_rs",
    "organisme",
    "niveau",
]

LABEL_FIELDS = [
    "est_lie_ia",
    "is_ai",
    "label",
    "classe",
    "target",
    "cible",
    "statut_annotation",
    "ai_label",
]

GROUP_FIELDS = [
    "formation_group_id",
    "group_id",
    "source_group_id",
    "formation_id",
    "source_row_id",
    "source_row",
    "code_certification",
    "certification_code",
    "code_rncp",
    "code_rs",
]


@dataclass(frozen=True, slots=True)
class BinaryAIDatasetAudit:
    source_files: list[str]
    source_fingerprint: str
    rows_read: int
    rows_kept: int
    rows_dropped_empty_text: int
    rows_dropped_missing_label: int
    exact_duplicates: int
    near_duplicates: int
    conflicts: int
    positives: int
    negatives: int
    positive_rate: float
    unique_groups: int
    duplicate_texts: int
    duplicate_group_ids: int
    label_convention: str
    class_distribution: dict[str, int]
    column_selection: dict[str, list[str]]
    warnings: list[str]
    generated_at: str


@dataclass(frozen=True, slots=True)
class BinaryAISplitManifest:
    seed: int
    generated_at: str
    source_fingerprint: str
    sizes: dict[str, int]
    class_distribution: dict[str, dict[str, int]]
    group_count: int
    group_ids_hash: str
    group_ids: list[str]
    target_ratios: dict[str, float]


def _normalize_column_name(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return normalize_for_match(text).replace(" ", "_")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [_stringify(item) for item in value]
        return " | ".join(part for part in parts if part)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = value.tolist()
            if converted is not value:
                return _stringify(converted)
        except Exception:  # pragma: no cover - defensive
            pass
    text = clean_text(value)
    return text


def _load_jsonl(path: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                payload["_source_line"] = line_no
                records.append(payload)
            elif isinstance(payload, list):
                for offset, item in enumerate(payload, start=1):
                    if isinstance(item, dict):
                        item = dict(item)
                        item["_source_line"] = f"{line_no}:{offset}"
                        records.append(item)
    return pd.DataFrame.from_records(records)


def load_tabular_file(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source, dtype=object, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source, dtype=object)
    if suffix == ".jsonl":
        return _load_jsonl(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(f"Format non supporte: {source.suffix}")


def discover_default_inputs(base_dir: str | Path = "data") -> list[Path]:
    root = Path(base_dir)
    candidates: list[Path] = []
    for pattern in ("**/*.csv", "**/*.xlsx", "**/*.xls", "**/*.jsonl", "**/*.parquet"):
        candidates.extend(root.glob(pattern))
    selected: list[Path] = []
    for candidate in candidates:
        if candidate.name.startswith("."):
            continue
        try:
            frame = load_tabular_file(candidate)
        except Exception:
            continue
        normalized = {_normalize_column_name(column) for column in frame.columns}
        if any(name in normalized for name in {"est_lie_ia", "statut_annotation", "is_ai", "ai_label"}):
            selected.append(candidate)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in sorted(selected):
        key = str(candidate.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _first_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized_lookup = {_normalize_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized_lookup:
            return normalized_lookup[key]
    return None


def _row_values(row: pd.Series, columns: Iterable[str]) -> list[str]:
    values: list[str] = []
    for column in columns:
        if column not in row.index:
            continue
        text = _stringify(row.get(column))
        if text:
            values.append(text)
    return values


def build_text(row: pd.Series) -> str:
    values = []
    for column in TEXT_FIELDS:
        if column not in row.index:
            continue
        text = _stringify(row.get(column))
        if text:
            values.append(f"{column.upper()}: {text}")
    if not values:
        for column in row.index:
            text = _stringify(row.get(column))
            if text:
                values.append(text)
    text = "\n".join(values)
    text = " ".join(text.split())
    return text[:2500].strip()


def normalize_binary_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return 1 if value == 1 else 0 if value == 0 else None
    text = _stringify(value)
    if not text:
        return None
    normalized = normalize_for_match(text)
    if normalized in {"1", "true", "yes", "oui", "ia", "ai", "ia confirmee", "ai confirmee", "ia confirmee oui"}:
        return 1
    if normalized in {"0", "false", "no", "non", "non ia", "non ai", "non ia confirmee", "non ai confirmee"}:
        return 0
    if "ia_confirmee" in normalized or normalized.endswith(" ia"):
        return 1
    if "non ia" in normalized or "non_ai" in normalized:
        return 0
    try:
        numeric = int(float(normalized))
    except Exception:
        return None
    return 1 if numeric == 1 else 0 if numeric == 0 else None


def _label_source(row: pd.Series, label_column: str | None) -> tuple[int | None, str]:
    if label_column is None:
        return None, ""
    value = row.get(label_column)
    if label_column == "statut_annotation":
        normalized = normalize_for_match(value)
        if "ia confirme" in normalized:
            return 1, label_column
        if "non ia confirme" in normalized:
            return 0, label_column
    label = normalize_binary_label(value)
    return label, label_column


def build_group_id(row: pd.Series) -> str:
    for column in GROUP_FIELDS:
        if column not in row.index:
            continue
        value = _stringify(row.get(column))
        if value:
            normalized = normalize_for_match(value)
            if normalized:
                return f"{column}:{normalized}"
    payload = [
        _stringify(row.get(column))
        for column in ("intitule", "titre", "description", "objectifs", "programme", "contenu")
    ]
    return f"hash:{stable_hash(*payload, length=24)}"


def _source_fingerprint(paths: Iterable[Path]) -> str:
    parts = []
    for path in sorted({Path(p) for p in paths}):
        stat = path.stat()
        parts.append(f"{path.resolve()}::{stat.st_size}::{int(stat.st_mtime)}")
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()


def _stable_datetime() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_binary_ai_dataset(inputs: Iterable[str | Path]) -> tuple[pd.DataFrame, BinaryAIDatasetAudit, pd.DataFrame, pd.DataFrame]:
    source_paths = [Path(path) for path in inputs]
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    column_selection: dict[str, list[str]] = {}
    rows_read = 0
    rows_dropped_empty_text = 0
    rows_dropped_missing_label = 0

    for source_path in source_paths:
        frame = load_tabular_file(source_path)
        rows_read += len(frame)
        normalized_columns = {_normalize_column_name(column): column for column in frame.columns}
        label_column = None
        for candidate in LABEL_FIELDS:
            if _normalize_column_name(candidate) in normalized_columns:
                label_column = normalized_columns[_normalize_column_name(candidate)]
                break
        if label_column is None:
            warnings.append(f"{source_path.name}: aucune colonne de label binaire detectee")
            continue
        selected_columns = []
        for candidate in TEXT_FIELDS:
            normalized = _normalize_column_name(candidate)
            if normalized in normalized_columns:
                selected_columns.append(normalized_columns[normalized])
        column_selection[source_path.name] = sorted(set(selected_columns + [label_column]))
        for idx, row in frame.iterrows():
            label, label_source = _label_source(row, label_column)
            text = build_text(row)
            if not text:
                rows_dropped_empty_text += 1
                continue
            if label is None:
                rows_dropped_missing_label += 1
                continue
            group_id = build_group_id(row)
            normalized_text = normalize_for_match(text)
            records.append(
                {
                    "source_file": source_path.name,
                    "source_path": str(source_path),
                    "source_line": int(row.get("_source_line", idx + 2)) if not isinstance(row.get("_source_line"), str) else str(row.get("_source_line")),
                    "label_source": label_source,
                    "is_ai": int(label),
                    "text": text[:2500].strip(),
                    "text_normalized": normalized_text,
                    "text_hash": stable_hash(normalized_text, length=24),
                    "group_id": group_id,
                    "formation_id": _stringify(row.get("formation_id") or row.get("source_row_id") or row.get("source_row")),
                    "intitule": _stringify(row.get("intitule") or row.get("intitulé") or row.get("titre")),
                    "description": _stringify(row.get("description")),
                    "objectifs": _stringify(row.get("objectifs")),
                    "programme": _stringify(row.get("programme")),
                    "raw_label": _stringify(row.get(label_column)),
                }
            )

    if not records:
        empty_frame = pd.DataFrame(
            columns=[
                "source_file",
                "source_path",
                "source_line",
                "label_source",
                "is_ai",
                "text",
                "text_normalized",
                "text_hash",
                "group_id",
                "formation_id",
                "intitule",
                "description",
                "objectifs",
                "programme",
                "raw_label",
            ]
        )
        audit = BinaryAIDatasetAudit(
            source_files=[str(path) for path in source_paths],
            source_fingerprint=_source_fingerprint(source_paths),
            rows_read=rows_read,
            rows_kept=0,
            rows_dropped_empty_text=rows_dropped_empty_text,
            rows_dropped_missing_label=rows_dropped_missing_label,
            exact_duplicates=0,
            near_duplicates=0,
            conflicts=0,
            positives=0,
            negatives=0,
            positive_rate=0.0,
            unique_groups=0,
            duplicate_texts=0,
            duplicate_group_ids=0,
            label_convention="0 = non_ia, 1 = ia",
            class_distribution={"non_ia": 0, "ia": 0},
            column_selection=column_selection,
            warnings=warnings,
            generated_at=_stable_datetime(),
        )
        return empty_frame, audit, empty_frame.copy(), empty_frame.copy()

    frame = pd.DataFrame.from_records(records)
    frame["text_normalized"] = frame["text_normalized"].fillna("").astype(str).str.strip()
    frame["group_id"] = frame["group_id"].fillna("").astype(str).str.strip()
    frame["is_ai"] = frame["is_ai"].astype(int)

    conflict_keys: set[str] = set()
    conflicts: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []

    grouped = frame.groupby("text_normalized", dropna=False, sort=False)
    kept_indices: list[int] = []
    seen_texts: set[str] = set()
    seen_groups: set[str] = set()

    for text_key, group in grouped:
        labels = sorted(set(int(v) for v in group["is_ai"].tolist()))
        if len(labels) > 1:
            conflict_keys.add(text_key)
            for _, row in group.iterrows():
                conflicts.append(
                    {
                        **row.to_dict(),
                        "conflict_type": "same_text_different_label",
                        "conflict_key": text_key,
                        "conflicting_labels": labels,
                    }
                )

    conflict_group_ids = set()
    for group_id, group in frame.groupby("group_id", dropna=False, sort=False):
        labels = sorted(set(int(v) for v in group["is_ai"].tolist()))
        if len(labels) > 1:
            conflict_group_ids.add(group_id)
            if group_id not in conflict_keys:
                for _, row in group.iterrows():
                    conflicts.append(
                        {
                            **row.to_dict(),
                            "conflict_type": "same_group_different_label",
                            "conflict_key": group_id,
                            "conflicting_labels": labels,
                        }
                    )

    conflict_mask = frame["text_normalized"].isin(conflict_keys) | frame["group_id"].isin(conflict_group_ids)
    frame = frame.loc[~conflict_mask].copy()

    # Keep the first occurrence of each normalized text, then deduplicate group ids.
    for idx, row in frame.iterrows():
        text_key = row["text_normalized"]
        group_id = row["group_id"]
        if text_key in seen_texts or group_id in seen_groups:
            duplicate_rows.append(
                {
                    **row.to_dict(),
                    "duplicate_type": "near_duplicate" if text_key in seen_texts else "group_duplicate",
                    "duplicate_of": text_key if text_key in seen_texts else group_id,
                }
            )
            continue
        seen_texts.add(text_key)
        seen_groups.add(group_id)
        kept_indices.append(idx)

    clean_frame = frame.loc[kept_indices].copy()
    clean_frame = clean_frame.sort_values(["is_ai", "group_id", "text_normalized"]).reset_index(drop=True)

    duplicate_frame = pd.DataFrame.from_records(duplicate_rows)
    conflict_frame = pd.DataFrame.from_records(conflicts)

    positives = int(clean_frame["is_ai"].sum())
    negatives = int(len(clean_frame) - positives)
    audit = BinaryAIDatasetAudit(
        source_files=[str(path) for path in source_paths],
        source_fingerprint=_source_fingerprint(source_paths),
        rows_read=rows_read,
        rows_kept=int(len(clean_frame)),
        rows_dropped_empty_text=rows_dropped_empty_text,
        rows_dropped_missing_label=rows_dropped_missing_label,
        exact_duplicates=int(len(duplicate_frame)),
        near_duplicates=int(len(duplicate_frame)),
        conflicts=int(len(conflict_frame)),
        positives=positives,
        negatives=negatives,
        positive_rate=float(positives / len(clean_frame)) if len(clean_frame) else 0.0,
        unique_groups=int(clean_frame["group_id"].nunique()) if not clean_frame.empty else 0,
        duplicate_texts=int(duplicate_frame["text_normalized"].nunique()) if not duplicate_frame.empty and "text_normalized" in duplicate_frame else 0,
        duplicate_group_ids=int(duplicate_frame["group_id"].nunique()) if not duplicate_frame.empty and "group_id" in duplicate_frame else 0,
        label_convention="0 = non_ia, 1 = ia",
        class_distribution={"non_ia": negatives, "ia": positives},
        column_selection=column_selection,
        warnings=warnings,
        generated_at=_stable_datetime(),
    )
    return clean_frame, audit, duplicate_frame, conflict_frame


def _split_objective(current_rows: int, current_pos: int, total_rows: int, total_pos: int, ratio: float) -> float:
    desired_rows = total_rows * ratio
    desired_pos = total_pos * ratio
    row_term = ((current_rows - desired_rows) / max(desired_rows, 1.0)) ** 2
    pos_term = ((current_pos - desired_pos) / max(desired_pos, 1.0)) ** 2
    return row_term + pos_term


def group_stratified_split(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> tuple[dict[str, pd.DataFrame], BinaryAISplitManifest]:
    if ratios is None:
        ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    if not abs(sum(ratios.values()) - 1.0) < 1e-6:
        raise ValueError(f"Les ratios doivent sommer a 1.0, recu: {ratios}")
    if frame.empty:
        empty = {name: frame.copy() for name in ratios}
        manifest = BinaryAISplitManifest(
            seed=seed,
            generated_at=_stable_datetime(),
            source_fingerprint="",
            sizes={name: 0 for name in ratios},
            class_distribution={name: {"non_ia": 0, "ia": 0} for name in ratios},
            group_count=0,
            group_ids_hash=hashlib.sha1(b"").hexdigest(),
            group_ids=[],
            target_ratios=ratios,
        )
        return empty, manifest

    groups = []
    for group_id, group in frame.groupby("group_id", sort=False):
        groups.append(
            {
                "group_id": str(group_id),
                "indices": group.index.tolist(),
                "size": int(len(group)),
                "positives": int(group["is_ai"].sum()),
                "negatives": int(len(group) - group["is_ai"].sum()),
            }
        )

    rng = random.Random(seed)
    groups.sort(key=lambda item: (-item["size"], -abs(item["positives"] - item["negatives"]), item["group_id"]))
    # deterministic jitter to avoid tie-pathologies
    for group in groups:
        group["_tie"] = rng.random()
    groups.sort(key=lambda item: (-item["size"], -abs(item["positives"] - item["negatives"]), item["_tie"], item["group_id"]))

    total_rows = len(frame)
    total_pos = int(frame["is_ai"].sum())
    total_neg = int(total_rows - total_pos)
    state = {
        name: {"rows": 0, "pos": 0, "neg": 0, "groups": []}
        for name in ratios
    }

    split_names = list(ratios)
    # Seed each split with one of the first groups when possible.
    for split_name, group in zip(split_names, groups[: len(split_names)]):
        state[split_name]["rows"] += group["size"]
        state[split_name]["pos"] += group["positives"]
        state[split_name]["neg"] += group["negatives"]
        state[split_name]["groups"].append(group["group_id"])
        group["_assigned"] = split_name

    for group in groups[len(split_names) :]:
        best_split = None
        best_score = None
        for split_name, ratio in ratios.items():
            candidate_rows = state[split_name]["rows"] + group["size"]
            candidate_pos = state[split_name]["pos"] + group["positives"]
            score = _split_objective(candidate_rows, candidate_pos, total_rows, total_pos, ratio)
            # Encourage staying close to the target negative count too.
            candidate_neg = state[split_name]["neg"] + group["negatives"]
            desired_neg = total_neg * ratio
            score += ((candidate_neg - desired_neg) / max(desired_neg, 1.0)) ** 2
            if best_score is None or score < best_score or (score == best_score and split_name < str(best_split)):
                best_score = score
                best_split = split_name
        assert best_split is not None
        state[best_split]["rows"] += group["size"]
        state[best_split]["pos"] += group["positives"]
        state[best_split]["neg"] += group["negatives"]
        state[best_split]["groups"].append(group["group_id"])
        group["_assigned"] = best_split

    splits: dict[str, pd.DataFrame] = {}
    for split_name in split_names:
        group_ids = state[split_name]["groups"]
        subset = frame[frame["group_id"].isin(group_ids)].copy()
        subset = subset.sort_values(["group_id", "text_normalized"]).reset_index(drop=True)
        splits[split_name] = subset

    # Ensure no leakage.
    group_sets = [set(df["group_id"]) for df in splits.values()]
    for idx, left in enumerate(group_sets):
        for right in group_sets[idx + 1 :]:
            overlap = left & right
            if overlap:
                raise ValueError(f"Fuite detectee entre splits: {sorted(list(overlap))[:3]}")

    group_ids = sorted(frame["group_id"].astype(str).unique().tolist())
    manifest = BinaryAISplitManifest(
        seed=seed,
        generated_at=_stable_datetime(),
        source_fingerprint=hashlib.sha1("||".join(sorted(frame["source_file"].astype(str).unique().tolist())).encode("utf-8")).hexdigest(),
        sizes={name: int(len(subset)) for name, subset in splits.items()},
        class_distribution={
            name: {
                "non_ia": int((subset["is_ai"] == 0).sum()),
                "ia": int((subset["is_ai"] == 1).sum()),
            }
            for name, subset in splits.items()
        },
        group_count=int(len(group_ids)),
        group_ids_hash=hashlib.sha1("||".join(group_ids).encode("utf-8")).hexdigest(),
        group_ids=group_ids,
        target_ratios=ratios,
    )
    return splits, manifest


def write_dataset_outputs(
    frame: pd.DataFrame,
    audit: BinaryAIDatasetAudit,
    duplicates: pd.DataFrame,
    conflicts: pd.DataFrame,
    *,
    dataset_path: str | Path,
    audit_path: str | Path,
    duplicates_path: str | Path,
    conflicts_path: str | Path,
) -> None:
    dataset_path = Path(dataset_path)
    audit_path = Path(audit_path)
    duplicates_path = Path(duplicates_path)
    conflicts_path = Path(conflicts_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    duplicates_path.parent.mkdir(parents=True, exist_ok=True)
    conflicts_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(dataset_path, index=False)
    duplicates.to_csv(duplicates_path, index=False, encoding="utf-8")
    conflicts.to_csv(conflicts_path, index=False, encoding="utf-8")
    audit_path.write_text(json.dumps(asdict(audit), ensure_ascii=False, indent=2), encoding="utf-8")


def write_split_outputs(
    splits: dict[str, pd.DataFrame],
    manifest: BinaryAISplitManifest,
    *,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        subset.to_parquet(output_dir / f"{name}.parquet", index=False)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

