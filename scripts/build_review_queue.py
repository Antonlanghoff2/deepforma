#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from continual_learning.review_selector import ReviewQueueSelector, build_review_candidates_from_offers
from continual_learning.store import ContinualLearningStore
from common.text import clean_text, normalize_for_match

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('build_review_queue')


def _load_json(value: str | None) -> Any:
    if not value:
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


def _skills_from_offer(offer: dict[str, Any], field: str) -> list[dict[str, Any]]:
    raw = offer.get(field)
    if isinstance(raw, str):
        parsed = _load_json(raw)
        return parsed if isinstance(parsed, list) else []
    if isinstance(raw, list):
        return raw
    return []


def _text_corpus(offfers: list[dict[str, Any]]) -> list[str]:
    corpus = []
    for offer in offfers:
        text = clean_text(offer.get('description_original'))
        if text:
            corpus.append(text)
    return corpus


def _compute_embedding_distance(reference_texts: list[str], current_text: str) -> float:
    if not reference_texts or not current_text:
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return 0.0
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(reference_texts + [current_text])
    sims = cosine_similarity(matrix[-1], matrix[:-1])
    if sims.size == 0:
        return 0.0
    best_sim = float(sims.max())
    return max(0.0, 1.0 - best_sim)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build the continual learning review queue')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('data/continual_learning/review_queue.jsonl'))
    parser.add_argument('--db-path', type=Path, default=Path('data/continual_learning/continual_learning.sqlite3'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weights-json', type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = ContinualLearningStore(args.db_path)
    if args.weights_json and args.weights_json.exists():
        weights = json.loads(args.weights_json.read_text(encoding='utf-8'))
    else:
        weights = None
    selector = ReviewQueueSelector(weights=weights)

    pending_offers = store.list_offers("validation_status IN ('pending', 'corrected')")
    approved_offers = store.list_offers("validation_status IN ('approved', 'used_for_training')")
    approved_annotations = store.list_annotations("validation_status IN ('approved', 'corrected', 'used_for_training')")

    recent_skill_counts: Counter[str] = Counter()
    historical_skill_counts: Counter[str] = Counter()
    for annotation in approved_annotations:
        key = normalize_for_match(annotation.get('canonical_name') or annotation.get('surface_form') or '')
        if key:
            historical_skill_counts[key] += 1
    for offer in pending_offers:
        for item in _skills_from_offer(offer, 'predicted_skills_json'):
            key = normalize_for_match(item.get('canonical_name') or item.get('label') or '')
            if key:
                recent_skill_counts[key] += 1

    seen_families = {clean_text(offer.get('job_family')) for offer in approved_offers if clean_text(offer.get('job_family'))}
    seen_territories = {clean_text(offer.get('territory')) for offer in approved_offers if clean_text(offer.get('territory'))}
    corpus = _text_corpus(approved_offers)

    candidates = build_review_candidates_from_offers(
        pending_offers,
        recent_skill_counts=dict(recent_skill_counts),
        historical_skill_counts=dict(historical_skill_counts),
        seen_families=seen_families,
        seen_territories=seen_territories,
    )

    enriched_candidates = []
    for candidate in candidates:
        offer = next((item for item in pending_offers if int(item['id']) == candidate.offer_row_id), None)
        if offer:
            embedding_distance = _compute_embedding_distance(corpus, clean_text(offer.get('description_original')))
            candidate = candidate.__class__(**{**candidate.__dict__, 'embedding_distance': embedding_distance})
        enriched_candidates.append(candidate)

    queue = selector.select(enriched_candidates, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as fh:
        for item in queue:
            fh.write(json.dumps(item, ensure_ascii=False) + '\n')

    summary = {
        'selected': len(queue),
        'pending_offers': len(pending_offers),
        'approved_offers': len(approved_offers),
        'approved_annotations': len(approved_annotations),
        'output': str(args.output),
    }
    (args.output.parent / 'review_queue_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('Review queue generated: %s', summary)


if __name__ == '__main__':
    main()
