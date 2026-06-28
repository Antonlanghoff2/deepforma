"""Migrate 18 old labels to new taxonomy — produces dataset with evidence."""
from __future__ import annotations

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


def load_taxonomy(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_18_to_taxonomy_mapping(taxonomy: dict) -> dict[str, list[dict]]:
    """Map each 18-label name to one or more taxonomy entries."""
    mapping: dict[str, list[dict]] = {}
    for family in taxonomy["families"]:
        for skill in family["skills"]:
            src = skill.get("source_18_label")
            if src:
                if src not in mapping:
                    mapping[src] = []
                mapping[src].append({
                    "taxonomy_id": skill["id"],
                    "taxonomy_label": skill["label"],
                    "family": family["label"],
                    "active": skill.get("active", False),
                })
    return mapping


def migrate_row(
    labels_18: str,
    text: str,
    taxonomy: dict,
    mapping_18: dict[str, list[dict]],
) -> dict[str, Any]:
    """Migrate a single row from 18 labels to taxonomy."""
    from skills.open_extractor import extract_skills

    original_labels = [l.strip() for l in labels_18.split("|") if l.strip()]

    taxonomy_ids: set[str] = set()
    evidence: list[dict] = []
    low_confidence: list[dict] = []
    unmapped: list[str] = []

    # 1. Direct mapping from 18-label → taxonomy
    for l18 in original_labels:
        mapped = mapping_18.get(l18, [])
        if mapped:
            for m in mapped:
                taxonomy_ids.add(m["taxonomy_id"])
                evidence.append({
                    "source_label_18": l18,
                    "taxonomy_id": m["taxonomy_id"],
                    "taxonomy_label": m["taxonomy_label"],
                    "method": "direct_mapping",
                    "confidence": 1.0,
                })
        else:
            unmapped.append(l18)

    # 2. Text-based refinement via open extractor
    extracted = extract_skills(text)
    ext_labels = [e.normalized_label for e in extracted]

    for family in taxonomy["families"]:
        for skill in family["skills"]:
            sid = skill["id"]
            if sid in taxonomy_ids:
                continue
            text_lower = text.lower()
            for alias in skill.get("aliases", []):
                if alias.lower() in text_lower:
                    taxonomy_ids.add(sid)
                    low_confidence.append({
                        "taxonomy_id": sid,
                        "taxonomy_label": skill["label"],
                        "method": "text_alias_match",
                        "confidence": 0.7,
                        "evidence": alias,
                    })
                    break
            for ext in ext_labels:
                if any(skill["label"].lower() in ext.lower() or
                       ext.lower() in skill["label"].lower() for a in skill.get("aliases", [])):
                    if sid not in taxonomy_ids:
                        taxonomy_ids.add(sid)
                        low_confidence.append({
                            "taxonomy_id": sid,
                            "taxonomy_label": skill["label"],
                            "method": "open_extractor_match",
                            "confidence": 0.6,
                            "evidence": ext,
                        })
                    break

    return {
        "original_labels_18": original_labels,
        "taxonomy_ids": sorted(taxonomy_ids),
        "evidence": evidence,
        "low_confidence": low_confidence,
        "unmapped_labels": unmapped,
        "total_taxonomy_labels": len(taxonomy_ids),
        "total_high_confidence": len(evidence),
        "total_low_confidence": len(low_confidence),
        "total_unmapped": len(unmapped),
    }


def migrate_dataset(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    mapping_18 = get_18_to_taxonomy_mapping(taxonomy)

    df = pd.read_csv(csv_path)
    ia = df[df["statut_annotation"] == "ia_confirmee"].copy()

    rows: list[dict[str, Any]] = []
    stats_total_mapped = Counter()
    stats_total_evidence = 0
    stats_total_low = 0
    stats_total_unmapped = 0

    for idx, row in ia.iterrows():
        labels_18 = str(row.get("competences_ia", "") or "")
        text = str(row.get("texte_modele", "") or "")
        result = migrate_row(labels_18, text, taxonomy, mapping_18)

        rows.append({
            "formation_id": row.get("formation_id", f"row_{idx}"),
            "intitule": row.get("intitule", ""),
            "text": text,
            "original_labels_18": "|".join(result["original_labels_18"]),
            "taxonomy_ids": "|".join(result["taxonomy_ids"]),
            "evidence": json.dumps(result["evidence"], ensure_ascii=False),
            "low_confidence": json.dumps(result["low_confidence"], ensure_ascii=False),
            "unmapped_labels": "|".join(result["unmapped_labels"]),
            "statut_annotation": row.get("statut_annotation", ""),
        })

        for tid in result["taxonomy_ids"]:
            stats_total_mapped[tid] += 1
        stats_total_evidence += result["total_high_confidence"]
        stats_total_low += result["total_low_confidence"]
        stats_total_unmapped += result["total_unmapped"]

    result_df = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "migrated_18_to_taxonomy.csv"
    result_df.to_csv(output_path, index=False)

    # Stats
    print(f"Migration terminee: {len(rows)} lignes")
    print(f"  Labels haute confiance: {stats_total_evidence}")
    print(f"  Labels basse confiance: {stats_total_low}")
    print(f"  Labels non mappes: {stats_total_unmapped}")
    print(f"  Labels uniques dans le dataset migre: {len(stats_total_mapped)}")
    print(f"  Fichier: {output_path}")

    # Top mapped labels
    print("\nTop 20 labels migres:")
    for tid, count in stats_total_mapped.most_common(20):
        print(f"  {count:4d}  {tid}")

    return result_df


if __name__ == "__main__":
    migrate_dataset()
