#!/usr/bin/env python3
"""Construit les paires d'entrainement (positives et negatives difficiles)
pour le recommender de formations CPF generalistes.

Usage:
    python scripts/build_cpf_training_pairs.py \\
        --input data/processed/cpf/formations_generalistes.jsonl \\
        --output-dir data/processed/cpf \\
        --output-pairs pairs_generalistes.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common.text import clean_text, normalize_for_match

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_cpf_training_pairs")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit
except ImportError as e:
    logger.error("Dependance manquante: %s", e)
    sys.exit(1)


def load_formations(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_text_representation(rec: dict[str, Any]) -> str:
    parts = []
    if rec.get("title"):
        parts.append(f"[TITRE] {rec['title']}")
    if rec.get("sector"):
        parts.append(f"[SECTEUR] {rec['sector']}")
    if rec.get("source_text"):
        parts.append(f"[COMPETENCES] {rec['source_text']}")
    if rec.get("skills"):
        parts.append(f"[COMPETENCES_STRUCTUREES] {' | '.join(rec['skills'])}")
    if rec.get("tags"):
        parts.append(f"[TAGS] {' | '.join(rec['tags'])}")
    if rec.get("rome_codes"):
        parts.append(f"[ROME] {' | '.join(rec['rome_codes'])}")
    if rec.get("modality"):
        parts.append(f"[MODALITE] {rec['modality']}")
    return " | ".join(parts)


def _stable_formation_uid(rec: dict[str, Any]) -> str:
    formation_id = clean_text(rec.get('formation_id') or rec.get('formation_uid') or rec.get('uid'))
    if formation_id:
        return formation_id
    certification = clean_text(rec.get('certification_code') or rec.get('certification') or '')
    organization = clean_text(rec.get('organization') or rec.get('organisme') or '')
    title = normalize_for_match(clean_text(rec.get('title') or ''))
    if certification or organization or title:
        seed = '||'.join([normalize_for_match(certification), normalize_for_match(organization), title])
    else:
        seed = normalize_for_match(build_text_representation(rec))
    return hashlib.sha256(seed.encode('utf-8')).hexdigest()


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def build_pairs(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    records = load_formations(args.input)
    logger.info("Formations chargees: %d", len(records))

    # Build indices
    rec_by_idx = {i: rec for i, rec in enumerate(records)}
    uid_by_idx = {i: _stable_formation_uid(rec) for i, rec in rec_by_idx.items()}
    skill_to_recs: dict[str, list[int]] = defaultdict(list)
    rome_to_recs: dict[str, list[int]] = defaultdict(list)
    sector_to_recs: dict[str, list[int]] = defaultdict(list)
    group_to_recs: dict[str, list[int]] = defaultdict(list)

    for i, rec in enumerate(records):
        for skill in rec.get("skills", []):
            skill_to_recs[skill].append(i)
        for rome in rec.get("rome_codes", []):
            rome_to_recs[rome].append(i)
        sector = rec.get("sector", "")
        if sector:
            sector_to_recs[sector].append(i)
        group_id = rec.get("group_id", "")
        if group_id:
            group_to_recs[group_id].append(i)

    # Positive pairs: same group_id
    positive_pairs: list[dict[str, Any]] = []
    for group_id, indices in group_to_recs.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a, b = indices[i], indices[j]
                positive_pairs.append({
                    "anchor": a,
                    "positive": b,
                    "anchor_uid": uid_by_idx[a],
                    "positive_uid": uid_by_idx[b],
                    "group_id": group_id,
                    "type": "same_certification",
                    "anchor_text": build_text_representation(rec_by_idx[a]),
                    "positive_text": build_text_representation(rec_by_idx[b]),
                })
    logger.info("Paires positives (meme certification): %d", len(positive_pairs))

    # Additional positive pairs: same skills, same sector (different group)
    skill_groups = list(skill_to_recs.items())
    for skill, indices in skill_groups:
        if len(indices) < 2:
            continue
        for i in range(min(len(indices), 50)):
            for j in range(i + 1, min(len(indices), 50)):
                a, b = indices[i], indices[j]
                if rec_by_idx[a].get("group_id") == rec_by_idx[b].get("group_id"):
                    continue
                if jaccard_similarity(
                    set(rec_by_idx[a].get("skills", [])),
                    set(rec_by_idx[b].get("skills", [])),
                ) >= 0.3:
                    positive_pairs.append({
                        "anchor": a,
                        "positive": b,
                        "anchor_uid": uid_by_idx[a],
                        "positive_uid": uid_by_idx[b],
                        "group_id": f"skill:{skill}",
                        "type": "same_skills",
                        "anchor_text": build_text_representation(rec_by_idx[a]),
                        "positive_text": build_text_representation(rec_by_idx[b]),
                    })

    logger.info("Paires positives totales: %d", len(positive_pairs))

    # Hard negative pairs
    negative_pairs: list[dict[str, Any]] = []

    # 1. Same sector, different skills
    for sector, indices in sector_to_recs.items():
        if len(indices) < 5:
            continue
        sampled = random.sample(indices, min(len(indices), 200))
        for i in range(len(sampled)):
            a = sampled[i]
            a_skills = set(rec_by_idx[a].get("skills", []))
            for j in range(i + 1, len(sampled)):
                b = sampled[j]
                if rec_by_idx[a].get("group_id") == rec_by_idx[b].get("group_id"):
                    continue
                b_skills = set(rec_by_idx[b].get("skills", []))
                if jaccard_similarity(a_skills, b_skills) < 0.2:
                    if skill_overlap_ratio(a_skills, b_skills) < 0.3:
                        negative_pairs.append({
                            "anchor": a,
                            "negative": b,
                            "anchor_uid": uid_by_idx[a],
                            "negative_uid": uid_by_idx[b],
                            "type": "same_sector_diff_skills",
                            "anchor_text": build_text_representation(rec_by_idx[a]),
                            "negative_text": build_text_representation(rec_by_idx[b]),
                        })

    # 2. Same ROME code, different skills
    for rome, indices in rome_to_recs.items():
        if len(indices) < 3:
            continue
        sampled = random.sample(indices, min(len(indices), 100))
        for i in range(len(sampled)):
            a = sampled[i]
            a_skills = set(rec_by_idx[a].get("skills", []))
            for j in range(i + 1, len(sampled)):
                b = sampled[j]
                if rec_by_idx[a].get("group_id") == rec_by_idx[b].get("group_id"):
                    continue
                b_skills = set(rec_by_idx[b].get("skills", []))
                if jaccard_similarity(a_skills, b_skills) < 0.15:
                    negative_pairs.append({
                        "anchor": a,
                        "negative": b,
                        "anchor_uid": uid_by_idx[a],
                        "negative_uid": uid_by_idx[b],
                        "type": "same_rome_diff_skills",
                        "anchor_text": build_text_representation(rec_by_idx[a]),
                        "negative_text": build_text_representation(rec_by_idx[b]),
                    })

    # 3. Lexically similar titles, different skills
    from collections import Counter
    title_word_index: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        title = rec.get("title", "").lower()
        for word in set(title.split()):
            if len(word) > 3:
                title_word_index[word].append(i)

    for word, indices in list(title_word_index.items())[:500]:
        if len(indices) < 3:
            continue
        sampled = random.sample(indices, min(len(indices), 50))
        for i in range(len(sampled)):
            a = sampled[i]
            a_skills = set(rec_by_idx[a].get("skills", []))
            for j in range(i + 1, len(sampled)):
                b = sampled[j]
                if rec_by_idx[a].get("group_id") == rec_by_idx[b].get("group_id"):
                    continue
                b_skills = set(rec_by_idx[b].get("skills", []))
                if not a_skills or not b_skills:
                    continue
                if jaccard_similarity(a_skills, b_skills) < 0.1:
                    negative_pairs.append({
                        "anchor": a,
                        "negative": b,
                        "anchor_uid": uid_by_idx[a],
                        "negative_uid": uid_by_idx[b],
                        "type": "similar_title_diff_skills",
                        "anchor_text": build_text_representation(rec_by_idx[a]),
                        "negative_text": build_text_representation(rec_by_idx[b]),
                    })

    positive_before_dedup = len(positive_pairs)
    positive_pairs, positive_dedup_removed = _dedupe_pairs(positive_pairs, keep_type=False)
    positive_pairs = _limit_pairs_per_anchor(positive_pairs, max_pairs_per_formation=getattr(args, 'max_pairs_per_formation', 10), seed=args.seed)
    positive_removed = max(0, positive_before_dedup - len(positive_pairs))

    negative_before_dedup = len(negative_pairs)
    negative_pairs, negative_dedup_removed = _dedupe_pairs(negative_pairs, keep_type=True)
    negative_pairs = _limit_pairs_per_anchor(negative_pairs, max_pairs_per_formation=getattr(args, 'max_pairs_per_formation', 10), seed=args.seed)
    negative_removed = max(0, negative_before_dedup - len(negative_pairs))

    logger.info("Paires positives dedoublonnees: %d", len(positive_pairs))
    logger.info("Paires negatives dedoublonnees: %d", len(negative_pairs))
    logger.info("Paires positives supprimees: %d", positive_removed)
    logger.info("Paires negatives supprimees: %d", negative_removed)

    # Split: group pairs by their group_id to avoid leak
    all_pairs = positive_pairs + negative_pairs
    pair_groups = []
    for pair in all_pairs:
        if pair.get("type", "").startswith("same_certification"):
            pair_groups.append(pair.get("group_id", f"pair:{pair['anchor']}"))
        else:
            pair_groups.append(f"pair:{pair['anchor']}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.output_pairs
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info("Paires sauvegardees: %s (%d total)", output_path, len(all_pairs))

    # Summary
    summary = {
        "total_formations": len(records),
        "total_pairs": len(all_pairs),
        "positive_pairs": len(positive_pairs),
        "negative_pairs": len(negative_pairs),
        "positive_pairs_removed": positive_removed,
        "negative_pairs_removed": negative_removed,
        "negative_types": dict(
            sorted(
                Counter(p["type"] for p in negative_pairs).items()
            )
        ),
        "output_file": str(output_path),
    }

    summary_path = output_dir / "pairs_generalistes_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Resume: %s", summary_path)

    return summary


def skill_overlap_ratio(set_a: set[str], set_b: set[str]) -> float:
    if not set_a:
        return 0.0
    intersection = set_a & set_b
    return len(intersection) / len(set_a)


def _normalize_pair_text(value: Any) -> str:
    return normalize_for_match(clean_text(value))


def _pair_key(pair: dict[str, Any]) -> tuple[str, str, str]:
    anchor = _normalize_pair_text(pair.get('anchor_text') or pair.get('anchor') or pair.get('query') or pair.get('text_a') or '')
    positive = _normalize_pair_text(pair.get('positive_text') or pair.get('negative_text') or pair.get('positive') or pair.get('negative') or pair.get('candidate') or pair.get('text_b') or '')
    pair_type = clean_text(pair.get('type')) or ''
    return anchor, positive, pair_type


def _pair_anchor_key(pair: dict[str, Any]) -> str:
    return _normalize_pair_text(pair.get('anchor_text') or pair.get('anchor') or pair.get('query') or pair.get('text_a') or '')


def _dedupe_pairs(pairs: list[dict[str, Any]], *, keep_type: bool = False) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    removed = 0
    for pair in pairs:
        anchor, positive, pair_type = _pair_key(pair)
        key = (anchor, positive, pair_type if keep_type else '')
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(pair)
    return deduped, removed


def _limit_pairs_per_anchor(pairs: list[dict[str, Any]], *, max_pairs_per_formation: int | None, seed: int) -> list[dict[str, Any]]:
    if not max_pairs_per_formation or max_pairs_per_formation <= 0:
        return pairs
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        buckets[_pair_anchor_key(pair)].append(pair)
    ordered_anchors = sorted(buckets)
    rng = random.Random(seed)
    rng.shuffle(ordered_anchors)
    limited: list[dict[str, Any]] = []
    for anchor in ordered_anchors:
        bucket = sorted(
            buckets[anchor],
            key=lambda pair: (
                clean_text(pair.get('type')),
                _normalize_pair_text(pair.get('positive_text') or pair.get('negative_text') or pair.get('positive') or pair.get('negative') or pair.get('candidate') or ''),
                clean_text(pair.get('group_id') or ''),
            ),
        )
        limited.extend(bucket[:max_pairs_per_formation])
    return limited


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Construction des paires d'entrainement CPF"
    )
    p.add_argument("--input", type=str,
                   default="data/processed/cpf/formations_generalistes.jsonl")
    p.add_argument("--output-dir", type=str, default="data/processed/cpf")
    p.add_argument("--output-pairs", type=str, default="pairs_generalistes.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-pairs-per-formation", type=int, default=10)
    p.add_argument("--max-train-samples", type=int, default=None)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    build_pairs(args)


if __name__ == "__main__":
    main()
