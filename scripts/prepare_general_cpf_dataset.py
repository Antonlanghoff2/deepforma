#!/usr/bin/env python3
"""Prepare le dataset generaliste CPF pour l'entrainement du recommender.

Lit Dataset_Generaliste_CPF_V4.xlsx, nettoie, normalise, et sauvegarde
au format Parquet et JSONL avec metadonnees.

Usage:
    python scripts/prepare_general_cpf_dataset.py \\
        --input data/raw/Dataset_Generaliste_CPF_V4.xlsx \\
        --output-dir data/processed/cpf
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_general_cpf_dataset")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    logger.error("Dependance manquante: %s", e)
    sys.exit(1)


TEXT_SOURCE_COL = "texte_source_competences"
SKILLS_COL = "competences_structurees"


def normalize_text(text: Any) -> str:
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_pipe_list(value: Any) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    parts = [s.strip() for s in str(value).split("|") if s.strip()]
    return [normalize_text(p) for p in parts if normalize_text(p)]


def build_formation_id(row: pd.Series) -> str:
    base = str(row.get("#", "")) or str(row.name)
    code = str(row.get("Code certification", "") or "")
    if code and code.lower() not in ("nan", "none", ""):
        code_part = code.replace("/", "_").replace(" ", "_")[:30]
        return f"CPF-{code_part}-{base}"
    return f"CPF-{base}"


def _get_title_col(row: pd.Series) -> str:
    """Return title column, trying accented and unaccented variants."""
    for variant in ("Intitulé de la formation", "Intitule de la formation"):
        val = row.get(variant)
        if val is not None and str(val).strip():
            return str(val)
    return ""


def build_group_id(row: pd.Series) -> str:
    code = str(row.get("Code certification", "") or "").strip()
    if code and code.lower() not in ("nan", "none", ""):
        return f"cert:{code}"
    title = _get_title_col(row).strip().lower()
    title = re.sub(r"\s+", " ", title)[:80]
    org = str(row.get("Organisme de formation", "") or "").strip()
    if org and title:
        return f"org:{org}|{title}"
    if title:
        return f"title:{title}"
    return f"idx:{row.get('#', row.name)}"


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Chargement: %s", input_path)
    df = pd.read_excel(input_path, sheet_name=args.sheet)
    total_raw = len(df)
    logger.info("Lignes brutes: %d", total_raw)

    removed: dict[str, int] = {}

    # Remove rows without title
    # Try accented variant first, fallback to unaccented
    possible_titles = ["Intitulé de la formation", "Intitule de la formation"]
    title_col = possible_titles[0] if possible_titles[0] in df.columns else possible_titles[1]
    before = len(df)
    df = df[df[title_col].notna() & df[title_col].astype(str).str.strip().ne("")].copy()
    removed["sans_titre"] = before - len(df)

    # Remove rows without source text AND without structured skills
    before = len(df)
    has_text = df[TEXT_SOURCE_COL].notna() & df[TEXT_SOURCE_COL].astype(str).str.strip().ne("")
    has_skills = df[SKILLS_COL].notna() & df[SKILLS_COL].astype(str).str.strip().ne("")
    df = df[has_text | has_skills].copy()
    removed["sans_texte_ni_competences"] = before - len(df)

    # Remove exact duplicates (text + skills)
    before = len(df)
    df = df.drop_duplicates(
        subset=[title_col, TEXT_SOURCE_COL, SKILLS_COL]
    ).copy()
    removed["doublons_exacts"] = before - len(df)

    logger.info(
        "Apres nettoyage: %d lignes (enleve: %s)",
        len(df), removed,
    )

    # Normalize
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        titre = normalize_text(row.get(title_col, ""))
        org = str(row.get("Organisme de formation", "") or "")
        secteur = str(row.get("Secteur", "") or "")
        cert_type = str(row.get("Type de certification", "") or "")
        code_cert = str(row.get("Code certification", "") or "")
        niveau = str(row.get("Niveau", "") or "")
        rome = parse_pipe_list(row.get("Codes ROME", ""))
        source_text = normalize_text(row.get(TEXT_SOURCE_COL, ""))
        skills = parse_pipe_list(row.get(SKILLS_COL, ""))
        modalite = str(row.get("Modalite", row.get("Modalité", "")) or "")
        duree = str(row.get("Duree", row.get("Durée", "")) or "")
        price_raw = row.get("Prix TTC (€)", None)
        price = None
        try:
            price = float(price_raw) if pd.notna(price_raw) else None
        except (ValueError, TypeError):
            price = None
        tags = parse_pipe_list(row.get("Tags", ""))

        records.append({
            "formation_id": build_formation_id(row),
            "group_id": build_group_id(row),
            "title": titre,
            "sector": secteur,
            "provider": org,
            "certification_type": cert_type,
            "certification_code": code_cert,
            "level": niveau,
            "rome_codes": rome,
            "source_text": source_text,
            "skills": skills,
            "modality": modalite,
            "duration": duree,
            "price": price,
            "tags": tags,
        })

    # Save Parquet
    parquet_path = output_dir / "formations_generalistes.parquet"
    pd.DataFrame(records).to_parquet(parquet_path, index=False)
    logger.info("Sauvegarde Parquet: %s (%d records)", parquet_path, len(records))

    # Save JSONL
    jsonl_path = output_dir / "formations_generalistes.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Sauvegarde JSONL: %s", jsonl_path)

    # Quality report
    label_counts = df[SKILLS_COL].dropna().str.split("|").explode().str.strip()
    label_freq = label_counts.value_counts().head(30).to_dict()
    label_freq = {str(k): int(v) for k, v in label_freq.items()}

    report = {
        "source_file": str(input_path),
        "total_raw": total_raw,
        "total_after_cleaning": len(records),
        "removed": removed,
        "fields": {
            "num_with_title": sum(1 for r in records if r["title"]),
            "num_with_source_text": sum(1 for r in records if r["source_text"]),
            "num_with_skills": sum(1 for r in records if r["skills"]),
            "num_with_rome_codes": sum(1 for r in records if r["rome_codes"]),
            "num_with_tags": sum(1 for r in records if r["tags"]),
            "num_with_price": sum(1 for r in records if r["price"] is not None),
        },
        "unique_certification_codes": int(df["Code certification"].nunique()),
        "unique_providers": int(df["Organisme de formation"].nunique()),
        "unique_sectors": int(df["Secteur"].nunique()),
        "top_skills": label_freq,
        "output_files": {
            "parquet": str(parquet_path),
            "jsonl": str(jsonl_path),
        },
    }

    report_path = Path("reports") / "cpf_generaliste_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Rapport qualite: %s", report_path)

    logger.info(
        "Preparation terminee: %d formations enregistrees.", len(records)
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preparation du dataset generaliste CPF"
    )
    p.add_argument("--input", type=str, default="data/raw/Dataset_Generaliste_CPF_V4.xlsx")
    p.add_argument("--output-dir", type=str, default="data/processed/cpf")
    p.add_argument("--sheet", type=str, default="Dataset_Generaliste")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main()
