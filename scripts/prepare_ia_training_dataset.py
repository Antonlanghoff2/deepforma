#!/usr/bin/env python3
"""Prepare le dataset d'entraînement pour le classifieur multilabel IA.

Lit Dataset_IA_V9_synth.xlsx, normalise les labels selon la taxonomie v2,
construit un decoupage groupe (train/val/test) sans fuite, et sauvegarde
les splits au format JSONL.

Usage:
    python scripts/prepare_ia_training_dataset.py \\
        --input data/raw/Dataset_IA_V9_synth.xlsx \\
        --output-dir data/processed \\
        --taxonomy config/ia_taxonomy_v2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_ia_training_dataset")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
except ImportError as e:
    logger.error("Dependance manquante: %s", e)
    sys.exit(1)


ALIAS_MAP: dict[str, str] = {
    "ethique ia & rgpd": "Ethique IA & RGPD",
    "ethique ia": "Ethique IA & RGPD",
    "ethique de l ia": "Ethique IA & RGPD",
    "ia generative": "IA Generative",
    "ia generative / llm": "IA Generative",
    "generative": "IA Generative",
    "langchain / agents rag": "LangChain / Agents RAG",
    "langchain agents rag": "LangChain / Agents RAG",
    "langchain": "LangChain / Agents RAG",
    "nlp / traitement du langage": "NLP / Traitement du langage",
    "nlp traitement du langage": "NLP / Traitement du langage",
    "nlp": "NLP / Traitement du langage",
    "traitement du langage": "NLP / Traitement du langage",
    "mlops / deploiement": "MLOps / Deploiement",
    "mlops deploiement": "MLOps / Deploiement",
    "mlops": "MLOps / Deploiement",
    "sql / data engineering": "SQL / Data Engineering",
    "sql data engineering": "SQL / Data Engineering",
    "series temporelles": "Series temporelles",
    "no-code / low-code": "No-code / Low-code",
    "no code / low code": "No-code / Low-code",
    "no-code/low-code": "No-code / Low-code",
    "gestion de projet ia": "Gestion de projet ia",
    "prompt engineering": "Prompt Engineering",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "computer vision": "Computer Vision",
    "data science": "Data Science",
    "data engineering": "Data Engineering",
    "big data": "Big Data",
    "automatisation": "Automatisation",
    "visualisation": "Visualisation",
    "reinforcement learning": "Reinforcement Learning",
    "python": "Python",
}


def load_taxonomy(taxonomy_path: str) -> dict[str, Any]:
    with open(taxonomy_path, encoding="utf-8") as f:
        return json.load(f)


def build_label_set(taxonomy: dict[str, Any]) -> set[str]:
    return set(taxonomy["labels"])


def normalize_label(raw: str, label_set: set[str]) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw in label_set:
        return raw
    key = raw.lower().strip()
    key = re.sub(r"\s+", " ", key)
    key = key.replace("'", " ").replace("’", " ")
    # Normalize common accented chars
    key = (
        key.replace("é", "e").replace("è", "e").replace("ê", "e")
        .replace("à", "a").replace("â", "a")
        .replace("ù", "u").replace("û", "u")
        .replace("ô", "o").replace("ö", "o")
        .replace("î", "i").replace("ï", "i")
        .replace("ç", "c")
    )
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    for candidate in label_set:
        if key == candidate.lower():
            return candidate
    logger.warning("Label non reconnu, ignore: '%s'", raw)
    return None


def normalize_labels(raw_skills: str, label_set: set[str]) -> list[str]:
    if pd.isna(raw_skills) or not str(raw_skills).strip():
        return []
    parts = [s.strip() for s in str(raw_skills).split("|") if s.strip()]
    normalized = []
    for part in parts:
        n = normalize_label(part, label_set)
        if n is not None:
            normalized.append(n)
    return sorted(set(normalized))


def build_text(row: pd.Series) -> str:
    parts = []
    titre = str(row.get("Intitule de la formation", row.get("Intitulé de la formation", "")) or "")
    if titre:
        parts.append(f"[TITRE] {titre}")
    secteur = str(row.get("Secteur", "") or "")
    if secteur:
        parts.append(f"[SECTEUR] {secteur}")
    org = str(row.get("Organisme de formation", "") or "")
    if org:
        parts.append(f"[ORGANISME] {org}")
    cert = str(row.get("Type de certification", "") or "")
    if cert:
        parts.append(f"[CERTIFICATION] {cert}")
    niveau = str(row.get("Niveau", "") or "")
    if niveau:
        parts.append(f"[NIVEAU] {niveau}")
    rome = str(row.get("Codes ROME", "") or "")
    if rome:
        parts.append(f"[ROME] {rome}")
    tags = str(row.get("Tags TrendRadar", "") or "")
    if tags:
        parts.append(f"[TAGS] {tags}")
    return " | ".join(parts)


def build_group_id(row: pd.Series) -> str:
    code = str(row.get("Code certification", "") or "").strip()
    if code and code.lower() not in ("nan", "none", ""):
        return f"cert:{code}"
    org = str(row.get("Organisme de formation", "") or "").strip()
    titre_key = str(
        row.get("Intitule de la formation", row.get("Intitulé de la formation", "")) or ""
    ).strip().lower()
    titre_key = re.sub(r"\s+", " ", titre_key)[:80]
    if org and titre_key:
        return f"org:{org}|{titre_key}"
    if titre_key:
        return f"title:{titre_key}"
    return f"idx:{row.get('#', row.name)}"


def verify_no_leak(
    splits: dict[str, pd.DataFrame], group_col: str = "_group_id"
) -> None:
    groups = {}
    for split_name, df in splits.items():
        gset = set(df[group_col].unique())
        for prev_name, prev_set in groups.items():
            overlap = gset & prev_set
            if overlap:
                raise ValueError(
                    f"FUITE detectee: {len(overlap)} groupe(s) present(s) dans "
                    f"'{prev_name}' et '{split_name}'. Ex: {list(overlap)[:3]}"
                )
        groups[split_name] = gset
    logger.info("Aucune fuite detectee entre les splits.")


def compute_pos_weights(
    y: np.ndarray, cap: float = 10.0
) -> list[float]:
    n_pos = y.sum(axis=0)
    n_neg = y.shape[0] - n_pos
    weights = np.where(n_pos > 0, n_neg / n_pos, cap)
    weights = np.clip(weights, None, cap)
    return weights.tolist()


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    taxonomy = load_taxonomy(args.taxonomy)
    label_set = build_label_set(taxonomy)
    labels_ordered = taxonomy["labels"]
    num_labels = len(labels_ordered)
    label2id = {lbl: i for i, lbl in enumerate(labels_ordered)}

    logger.info("Taxonomie: %s (%d labels)", taxonomy["taxonomy_version"], num_labels)
    logger.info("Fichier source: %s", args.input)

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    logger.info("Lignes brutes: %d", len(df))

    raw_skills_col = "Compétences IA extraites"
    title_col = "Intitulé de la formation"

    df = df[df[raw_skills_col].notna() & df[raw_skills_col].astype(str).str.strip().ne("")].copy()
    df = df[df[title_col].notna() & df[title_col].astype(str).str.strip().ne("")].copy()
    logger.info("Apres filtre lignes sans titre ou competences: %d", len(df))

    df["_text"] = df.apply(build_text, axis=1)
    df["_labels"] = df[raw_skills_col].apply(lambda x: normalize_labels(x, label_set))
    df["_label_set"] = df["_labels"].apply(lambda x: frozenset(x))

    empty_labels = df["_labels"].apply(len).eq(0).sum()
    if empty_labels:
        logger.warning("Lignes sans labels valides apres normalisation: %d", empty_labels)
        df = df[df["_labels"].apply(len) > 0].copy()
        logger.info("Apres filtre lignes sans labels valides: %d", len(df))

    df["_group_id"] = df.apply(build_group_id, axis=1)

    # Dedup: same group_id AND same label set
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["_group_id", "_label_set"]).copy()
    logger.info("Apres dedup (groupe + labels): %d (enleve %d)", len(df), before_dedup - len(df))

    # Build multi-hot matrix
    y = np.zeros((len(df), num_labels), dtype=np.float32)
    for i, labels in enumerate(df["_labels"]):
        for lbl in labels:
            y[i, label2id[lbl]] = 1.0

    df["_multi_hot"] = [y[i].tolist() for i in range(len(df))]

    logger.info("Distribution des labels (total dataset):")
    label_counts = y.sum(axis=0).astype(int)
    for i, lbl in enumerate(labels_ordered):
        pct = label_counts[i] / len(df) * 100
        logger.info("  %s: %d (%.1f%%)", lbl, label_counts[i], pct)

    # Grouped split
    groups = df["_group_id"].values
    gss = GroupShuffleSplit(n_splits=1, train_size=args.train_ratio, random_state=args.seed)
    train_idx, temp_idx = next(gss.split(df, groups=groups))

    temp_df = df.iloc[temp_idx].copy()
    temp_groups = temp_df["_group_id"].values
    val_ratio = args.val_ratio / (args.val_ratio + args.test_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio, random_state=args.seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_groups))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    split_dfs = {
        "train": df.iloc[train_idx].copy(),
        "validation": df.iloc[val_idx].copy(),
        "test": df.iloc[test_idx].copy(),
    }
    verify_no_leak(split_dfs, "_group_id")

    for split_name, sdf in split_dfs.items():
        logger.info(
            "  %s: %d echantillons, %d groupes",
            split_name, len(sdf), sdf["_group_id"].nunique(),
        )

    # Build metadata
    train_labels = y[train_idx]
    pos_weights = compute_pos_weights(train_labels, cap=args.pos_weight_cap)

    metadata = {
        "taxonomy_version": taxonomy["taxonomy_version"],
        "model_version": "1.0",
        "labels": labels_ordered,
        "label2id": label2id,
        "num_labels": num_labels,
        "training_date": "",
        "num_examples": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "label_distribution": {
            lbl: {
                "train": int(train_labels[:, i].sum()),
                "validation": int(y[val_idx][:, i].sum()),
                "test": int(y[test_idx][:, i].sum()),
            }
            for i, lbl in enumerate(labels_ordered)
        },
        "pos_weight": pos_weights,
        "thresholds": {lbl: 0.50 for lbl in labels_ordered},
        "metrics": {"validation": {}, "test": {}},
        "seed": args.seed,
        "source_file": str(args.input),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "validation", "test"):
        sdf = split_dfs[split_name]
        records = []
        for _, row in sdf.iterrows():
            records.append({
                "text": row["_text"],
                "labels": row["_labels"],
                "multi_hot": row["_multi_hot"],
                "group_id": row["_group_id"],
                "formation_id": str(row.get("#", "")),
            })
        split_path = output_dir / f"ia_multilabel_{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Sauvegarde: %s (%d enregistrements)", split_path, len(records))

    # Label distribution CSV
    dist_rows = []
    for i, lbl in enumerate(labels_ordered):
        dist_rows.append({
            "label": lbl,
            "train_count": int(train_labels[:, i].sum()),
            "train_pct": round(train_labels[:, i].sum() / len(train_idx) * 100, 1),
            "val_count": int(y[val_idx][:, i].sum()),
            "val_pct": round(y[val_idx][:, i].sum() / len(val_idx) * 100, 1),
            "test_count": int(y[test_idx][:, i].sum()),
            "test_pct": round(y[test_idx][:, i].sum() / len(test_idx) * 100, 1),
        })
    dist_df = pd.DataFrame(dist_rows)
    dist_path = output_dir / "ia_label_distribution.csv"
    dist_df.to_csv(dist_path, index=False)
    logger.info("Sauvegarde distribution: %s", dist_path)

    # Metadata JSON
    meta_path = output_dir / "ia_multilabel_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Sauvegarde metadonnees: %s", meta_path)

    logger.info("Preparation terminee.")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preparation du dataset d'entrainement classifieur multilabel IA"
    )
    p.add_argument("--input", type=str, default="data/raw/Dataset_IA_V9_synth.xlsx")
    p.add_argument("--output-dir", type=str, default="data/processed")
    p.add_argument("--taxonomy", type=str, default="config/ia_taxonomy_v2.json")
    p.add_argument("--sheet", type=str, default="Dataset_IA")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos-weight-cap", type=float, default=10.0)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main()
