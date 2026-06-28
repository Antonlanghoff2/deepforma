from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from common.text import normalize_for_match
from deepforma.skills.normalizer import SkillTaxonomyNormalizer


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construit des triplets d'entraînement pour le ranking CPF")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/training/cpf_recommendation_pairs.jsonl"))
    parser.add_argument("--output-review", type=Path, default=Path("data/training/cpf_recommendation_pairs_review.csv"))
    parser.add_argument("--limit", type=int, default=5_000)
    return parser


def _format_triplet(query: str, positive: str, negative: str, label_source: str) -> dict[str, Any]:
    return {
        "query": query,
        "positive": positive,
        "negative": negative,
        "label_source": label_source,
    }


def build_pairs_from_catalog(metadata: pd.DataFrame, limit: int = 5_000) -> list[dict[str, Any]]:
    """Construit des triplets heuristiques à partir du catalogue CPF."""

    normalizer = SkillTaxonomyNormalizer()
    rows = metadata.fillna("").to_dict(orient="records")
    pairs: list[dict[str, Any]] = []
    for row in rows:
        skills = [skill for skill in row.get("skills_normalized", []) if skill]
        query_parts = [
            row.get("title", ""),
            row.get("department_code", ""),
            " ".join(skills[:3]),
        ]
        query = " | ".join(part for part in query_parts if part)
        positive = row.get("search_text") or row.get("title") or ""
        same_skill_candidate = next(
            (other for other in rows if other.get("formation_uid") != row.get("formation_uid") and normalize_for_match(other.get("title")) == normalize_for_match(row.get("title"))),
            None,
        )
        negative = same_skill_candidate.get("search_text") if same_skill_candidate else "Formation hors domaine"
        pairs.append(_format_triplet(query, positive, negative, "heuristic"))
        if len(pairs) >= limit:
            break
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    metadata = pd.read_parquet(args.input)
    rows = build_pairs_from_catalog(metadata, limit=args.limit)
    write_jsonl(args.output_jsonl, rows)
    write_review_csv(args.output_review, rows)
    LOGGER.info("Paires d'entraînement écrites dans %s et %s", args.output_jsonl, args.output_review)


if __name__ == "__main__":
    main()

