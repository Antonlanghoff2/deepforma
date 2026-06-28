"""Prepare multi-label dataset with multi-hot vectors from taxonomy."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TAXONOMY_PATH = ROOT / "data" / "referentials" / "ai_skill_taxonomy.json"
CSV_PATH = ROOT / "data" / "processed" / "dataset_entrainement.csv"
OUTPUT_DIR = ROOT / "data" / "multilabel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVATION_THRESHOLDS = {
    "actif": 50,
    "experimental": 10,
    "inactif": 0,
}


def load_taxonomy(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_active_skill_ids(taxonomy: dict, min_frequency: int = 0) -> list[str]:
    """Return stable ordered list of active skill IDs.
    
    Only includes skills marked active=True in the taxonomy,
    plus an optional minimum frequency filter.
    """
    ids: list[str] = []
    for family in taxonomy["families"]:
        for skill in family["skills"]:
            if not skill.get("active", False):
                continue
            if min_frequency > 0 and skill.get("frequency_dataset", 0) < min_frequency:
                continue
            ids.append(skill["id"])
    return ids


def taxonomy_hash(taxonomy: dict) -> str:
    return hashlib.sha256(
        json.dumps(taxonomy, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def build_label_mapping(taxonomy: dict) -> dict[str, dict]:
    """Build id2label and label2id from taxonomy."""
    active_ids = get_active_skill_ids(taxonomy)
    id2label = {str(i): sid for i, sid in enumerate(active_ids)}
    label2id = {sid: i for i, sid in enumerate(active_ids)}
    return {
        "id2label": id2label,
        "label2id": label2id,
        "num_labels": len(active_ids),
        "active_ids": active_ids,
    }


def map_18_label_to_taxonomy(label_18: str, taxonomy: dict) -> list[str]:
    """Map an 18-label name to taxonomy IDs."""
    for family in taxonomy["families"]:
        for skill in family["skills"]:
            if skill.get("source_18_label") == label_18:
                return [skill["id"]]
            for alias in skill.get("aliases", []):
                if alias.lower() == label_18.lower():
                    return [skill["id"]]
    return []


def text_based_label_proposals(
    text: str,
    taxonomy: dict,
    label2id: dict[str, int],
    extractor_results: list[str] | None = None,
) -> list[str]:
    """Propose taxonomy labels based on text content."""
    from skills.open_extractor import extract_skills

    proposed: set[str] = set()
    text_lower = text.lower()

    if extractor_results:
        for ext_label in extractor_results:
            for family in taxonomy["families"]:
                for skill in family["skills"]:
                    if not skill.get("active", False):
                        continue
                    ext_lower = ext_label.lower()
                    for alias in skill.get("aliases", []):
                        if alias.lower() in ext_lower or ext_lower in alias.lower():
                            proposed.add(skill["id"])
                            break
                    if skill["id"] in proposed:
                        break

    # Direct text matching on aliases
    for family in taxonomy["families"]:
        for skill in family["skills"]:
            if not skill.get("active", False):
                continue
            for alias in skill.get("aliases", []):
                if alias.lower() in text_lower:
                    proposed.add(skill["id"])
                    break

    return [l for l in proposed if l in label2id]


def build_multi_hot(
    label_ids: list[str],
    all_ids: list[str],
) -> list[int]:
    return [1 if sid in label_ids else 0 for sid in all_ids]


def prepare_dataset(
    csv_path: Path = CSV_PATH,
    taxonomy_path: Path = TAXONOMY_PATH,
    output_dir: Path = OUTPUT_DIR,
    min_label_frequency: int = 0,
) -> dict[str, Any]:
    taxonomy = load_taxonomy(taxonomy_path)
    mapping = build_label_mapping(taxonomy)
    active_ids = mapping["active_ids"]

    if min_label_frequency > 0:
        active_ids = [
            sid for sid in active_ids
            if _get_label_frequency(sid, taxonomy) >= min_label_frequency
        ]
        mapping["active_ids"] = active_ids
        mapping["num_labels"] = len(active_ids)
        mapping["id2label"] = {str(i): sid for i, sid in enumerate(active_ids)}
        mapping["label2id"] = {sid: i for i, sid in enumerate(active_ids)}

    df = pd.read_csv(csv_path)
    ia = df[df["statut_annotation"] == "ia_confirmee"].copy()

    rows: list[dict[str, Any]] = []
    label_pos_counts: Counter = Counter()
    label_neg_counts: Counter = Counter()

    for idx, row in ia.iterrows():
        text = str(row.get("texte_modele", "") or "")
        labels_18 = str(row.get("competences_ia", "") or "")

        # Map old 18 labels to taxonomy IDs
        taxonomy_ids: set[str] = set()
        for l18 in labels_18.split("|"):
            l18 = l18.strip()
            if l18:
                mapped = map_18_label_to_taxonomy(l18, taxonomy)
                taxonomy_ids.update(mapped)

        # Also try text-based proposals
        text_ids = text_based_label_proposals(text, taxonomy, mapping["label2id"])
        taxonomy_ids.update(text_ids)

        # Filter to active IDs only
        taxonomy_ids = {t for t in taxonomy_ids if t in mapping["label2id"]}

        multi_hot = build_multi_hot(list(taxonomy_ids), active_ids)

        for tid in taxonomy_ids:
            label_pos_counts[tid] += 1
        for tid in active_ids:
            if tid not in taxonomy_ids:
                label_neg_counts[tid] += 1

        rows.append({
            "formation_id": row.get("formation_id", f"row_{idx}"),
            "text": text,
            "intitule": row.get("intitule", ""),
            "labels_18": labels_18,
            "taxonomy_ids": "|".join(sorted(taxonomy_ids)),
            "multi_hot": multi_hot,
            "statut_annotation": row.get("statut_annotation", ""),
        })

    # Build pos_weight for BCEWithLogitsLoss
    pos_weights: list[float] = []
    pos_weight_max = float(
        json.loads(
            (TAXONOMY_PATH.parent / "taxonomy_config.json").read_text()
            if (TAXONOMY_PATH.parent / "taxonomy_config.json").exists() else "{}"
        ).get("pos_weight_max", 10.0)
    )

    for tid in active_ids:
        pos = label_pos_counts.get(tid, 0)
        neg = label_neg_counts.get(tid, 1)
        if pos == 0:
            pw = pos_weight_max
        else:
            pw = min(neg / pos, pos_weight_max)
        pos_weights.append(round(pw, 4))

    hash_val = taxonomy_hash(taxonomy)

    dataset_info = {
        "taxonomy_version": taxonomy.get("version", "N/A"),
        "taxonomy_hash": hash_val,
        "num_labels": len(active_ids),
        "num_samples": len(rows),
        "label_ids": active_ids,
        "id2label": mapping["id2label"],
        "label2id": mapping["label2id"],
        "label_pos_counts": {sid: label_pos_counts.get(sid, 0) for sid in active_ids},
        "label_neg_counts": {sid: label_neg_counts.get(sid, 0) for sid in active_ids},
        "pos_weight": pos_weights,
        "label_frequencies": {
            sid: round(label_pos_counts.get(sid, 0) / max(len(rows), 1) * 100, 2)
            for sid in active_ids
        },
    }

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "multilabel_dataset.csv", index=False)
    (output_dir / "multilabel_dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False)
    )

    print(f"Dataset genere: {output_dir / 'multilabel_dataset.csv'}")
    print(f"  Labels actifs: {len(active_ids)}")
    print(f"  Echantillons: {len(rows)}")
    print(f"  Taxonomy hash: {hash_val}")

    return dataset_info


def _get_label_frequency(skill_id: str, taxonomy: dict) -> int:
    for family in taxonomy["families"]:
        for skill in family["skills"]:
            if skill["id"] == skill_id:
                return skill.get("frequency_dataset", 0)
    return 0


if __name__ == "__main__":
    prepare_dataset()
