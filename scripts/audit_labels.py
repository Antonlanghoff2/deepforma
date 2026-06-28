"""Audit complet des 18 labels actuels : fréquences, distribution, F1."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "data" / "processed" / "dataset_entrainement.csv"
TAXONOMY_PATH = ROOT / "data" / "referentials" / "ai_skill_taxonomy.json"
LABEL_CLASSES_PATH = ROOT / "models" / "multilabel_competences_v2" / "final" / "label_classes.json"
REPORT_PATH = ROOT / "reports" / "label_audit_report.json"


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_label_frequencies(df: pd.DataFrame) -> dict[str, dict]:
    ia = df[df["statut_annotation"] == "ia_confirmee"]
    non_ia = df[df["statut_annotation"] == "non_ia_confirmee"]

    # Frequences depuis la colonne competences_ia
    pos_counter = Counter()
    neg_counter: Counter = Counter()

    for labels_str in ia["competences_ia"].dropna():
        labels = [l.strip() for l in labels_str.split("|") if l.strip()]
        for l in labels:
            pos_counter[l] += 1

    # Négatifs : ceux qui ne sont PAS dans une ligne donnée
    all_labels_ordered = sorted(
        set(
            l.strip()
            for labels_str in ia["competences_ia"].dropna()
            for l in labels_str.split("|") if l.strip()
        )
    )

    for labels_str in ia["competences_ia"].dropna():
        present = {l.strip() for l in labels_str.split("|") if l.strip()}
        for l in all_labels_ordered:
            if l not in present:
                neg_counter[l] += 1

    total_pos = sum(pos_counter.values())
    total_rows = len(ia)

    results: dict[str, dict] = {}
    for label in all_labels_ordered:
        pos = pos_counter.get(label, 0)
        neg = neg_counter.get(label, 0)
        results[label] = {
            "label": label,
            "positives": pos,
            "negatives": neg,
            "frequency_pct": round(pos / total_rows * 100, 2) if total_rows else 0.0,
            "total_rows": total_rows,
            "activation_level": (
                "actif" if pos >= 50 else
                "experimental" if pos >= 10 else
                "inactif"
            ),
        }

    return results


def load_taxonomy(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_taxonomy_mapping(taxonomy: dict) -> dict[str, str]:
    mapping = {}
    for family in taxonomy["families"]:
        for skill in family["skills"]:
            src = skill.get("source_18_label")
            if src:
                mapping[src] = skill["id"]
    return mapping


def generate_report(csv_path: Path = CSV_PATH) -> dict:
    df = load_dataset(csv_path)
    label_stats = compute_label_frequencies(df)

    ia = df[df["statut_annotation"] == "ia_confirmee"]
    non_ia = df[df["statut_annotation"] == "non_ia_confirmee"]

    taxonomy = load_taxonomy(TAXONOMY_PATH) if TAXONOMY_PATH.exists() else {}
    mapping_18_to_taxonomy = compute_taxonomy_mapping(taxonomy) if taxonomy else {}
    label_classes = json.loads(LABEL_CLASSES_PATH.read_text()) if LABEL_CLASSES_PATH.exists() else []

    report = {
        "dataset": {
            "path": str(csv_path),
            "total_rows": len(df),
            "ia_confirmee": len(ia),
            "non_ia_confirmee": len(non_ia),
            "ia_pct": round(len(ia) / len(df) * 100, 2) if len(df) else 0.0,
            "label_classes_source": str(LABEL_CLASSES_PATH) if LABEL_CLASSES_PATH.exists() else "N/A",
            "taxonomy_version": taxonomy.get("version", "N/A") if taxonomy else "N/A",
        },
        "labels_18": {
            "count": len(label_classes),
            "order": label_classes,
            "id2label_status": (
                "GENERIQUE (LABEL_0..LABEL_17)"
                if not any(l.startswith("LABEL_") for l in label_classes)
                else "Nominal"
            ),
        },
        "per_label_statistics": [
            label_stats.get(l, {
                "label": l, "positives": 0, "negatives": 0,
                "frequency_pct": 0.0, "total_rows": len(ia),
                "activation_level": "inconnu",
            })
            for l in label_classes
        ],
        "taxonomy_mapping_18": mapping_18_to_taxonomy,
        "summary": {
            "total_labels_in_dataset": len(label_stats),
            "total_labels_in_taxonomy": sum(
                len(f["skills"]) for f in taxonomy.get("families", [])
            ) if taxonomy else 0,
            "active_labels": sum(
                1 for v in label_stats.values()
                if v["activation_level"] == "actif"
            ),
            "experimental_labels": sum(
                1 for v in label_stats.values()
                if v["activation_level"] == "experimental"
            ),
            "inactive_labels": sum(
                1 for v in label_stats.values()
                if v["activation_level"] == "inactif"
            ),
            "total_positives": sum(v["positives"] for v in label_stats.values()),
            "total_negatives": sum(v["negatives"] for v in label_stats.values()),
            "imbalance_ratio_max_min": (
                round(
                    max(v["positives"] for v in label_stats.values()) /
                    max(min(v["positives"] for v in label_stats.values()), 1),
                    2,
                )
                if label_stats else 0
            ),
        },
    }
    return report


def main():
    report = generate_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Rapport d'audit genere: {REPORT_PATH}")

    print("\n=== Statistiques par label ===")
    for s in report["per_label_statistics"]:
        print(
            f"  {s['label']:30s}  pos={s['positives']:4d}  "
            f"neg={s['negatives']:4d}  freq={s['frequency_pct']:5.1f}%  "
            f"niveau={s['activation_level']}"
        )

    print("\n=== Mapping 18 labels -> Taxonomie ===")
    for old, new in report["taxonomy_mapping_18"].items():
        print(f"  {old:30s} -> {new}")

    print(f"\n=== Resume ===")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
