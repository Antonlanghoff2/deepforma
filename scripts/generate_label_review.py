"""Generate human review file for proposed label assignments."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIGRATED_PATH = ROOT / "data" / "multilabel" / "migrated_18_to_taxonomy.csv"
TAXONOMY_PATH = ROOT / "data" / "referentials" / "ai_skill_taxonomy.json"
REVIEW_DIR = ROOT / "data" / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)


def load_taxonomy(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def taxonomy_id_to_label(taxonomy: dict) -> dict[str, str]:
    mapping = {}
    for family in taxonomy["families"]:
        for skill in family["skills"]:
            mapping[skill["id"]] = skill["label"]
    return mapping


def generate_review(csv_path: Path = MIGRATED_PATH) -> pd.DataFrame:
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    id_to_label = taxonomy_id_to_label(taxonomy)

    df = pd.read_csv(csv_path)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        formation_id = row.get("formation_id", "")
        title = row.get("intitule", "")
        text = str(row.get("text", "") or "")[:500]
        original_18 = str(row.get("original_labels_18", "") or "")
        taxonomy_ids = str(row.get("taxonomy_ids", "") or "")
        evidence_raw = row.get("evidence", "[]")
        low_conf_raw = row.get("low_confidence", "[]")

        # Parse evidence
        evidence_list = []
        try:
            evidence_list = json.loads(evidence_raw) if isinstance(evidence_raw, str) else []
        except (json.JSONDecodeError, TypeError):
            pass

        high_conf_labels = "; ".join(
            f"{e.get('taxonomy_label', '?')} ({e.get('method', '?')})"
            for e in evidence_list
        )

        low_conf_list = []
        try:
            low_conf_list = json.loads(low_conf_raw) if isinstance(low_conf_raw, str) else []
        except (json.JSONDecodeError, TypeError):
            pass

        low_conf_labels = "; ".join(
            f"{e.get('taxonomy_label', '?')} ({e.get('method', '?')}, conf={e.get('confidence', 0)})"
            for e in low_conf_list
        )

        # Extract open-extracted skills for reference
        extracted_skills = []
        if text:
            try:
                from skills.open_extractor import extract_skills
                extracted = extract_skills(text)
                extracted_skills = [e.source_label for e in extracted]
            except Exception:
                pass

        proposed_labels_verbose = "; ".join(
            id_to_label.get(tid, tid)
            for tid in taxonomy_ids.split("|") if tid
        )

        rows.append({
            "formation_id": formation_id,
            "title": title,
            "source_text": text,
            "extracted_skills": "; ".join(extracted_skills),
            "original_labels_18": original_18,
            "proposed_labels": proposed_labels_verbose,
            "high_confidence_evidence": high_conf_labels,
            "low_confidence_evidence": low_conf_labels,
            "human_validation": "",
            "human_labels": "",
            "comment": "",
        })

    review_df = pd.DataFrame(rows)
    output_path = REVIEW_DIR / "ai_skill_labels_review.csv"
    review_df.to_csv(output_path, index=False)
    print(f"Fichier de revue genere: {output_path}")
    print(f"  Lignes a revoir: {len(review_df)}")
    print(f"  Colonnes: {list(review_df.columns)}")
    return review_df


if __name__ == "__main__":
    generate_review()
