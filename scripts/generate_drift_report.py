#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from common.text import clean_text, normalize_for_match
from continual_learning.store import ContinualLearningStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('generate_drift_report')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate drift report for continual learning')
    parser.add_argument('--db-path', type=Path, default=Path('data/continual_learning/continual_learning.sqlite3'))
    parser.add_argument('--output', type=Path, default=Path('reports/drift_report.json'))
    parser.add_argument('--recent-limit', type=int, default=200)
    return parser


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[\wÀ-ÿ'-]{3,}", clean_text(text).lower()) if len(t) >= 3]


def _skill_counts(offfers: list[dict[str, Any]], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for offer in offfers:
        try:
            items = json.loads(offer.get(field) or '[]')
        except Exception:
            items = []
        for item in items:
            name = normalize_for_match(item.get('canonical_name') or item.get('label') or '')
            if name:
                counts[name] += 1
    return counts


def main() -> None:
    args = build_parser().parse_args()
    store = ContinualLearningStore(args.db_path)
    recent_offers = store.list_offers(limit=args.recent_limit)
    approved_offers = store.list_offers("validation_status IN ('approved', 'corrected', 'used_for_training')")
    approved_annotations = store.list_annotations("validation_status IN ('approved', 'corrected', 'used_for_training')")

    recent_texts = [clean_text(item.get('description_original')) for item in recent_offers if clean_text(item.get('description_original'))]
    historical_texts = [clean_text(item.get('description_original')) for item in approved_offers if clean_text(item.get('description_original'))]
    if recent_texts and historical_texts:
        vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(historical_texts + recent_texts)
        distances = cosine_distances(matrix[len(historical_texts):], matrix[:len(historical_texts)])
        mean_distance = float(np.mean(np.min(distances, axis=1))) if distances.size else 0.0
    else:
        mean_distance = 0.0

    recent_tokens = Counter(token for text in recent_texts for token in _tokens(text))
    historical_tokens = Counter(token for text in historical_texts for token in _tokens(text))
    new_expressions = [
        {'token': token, 'recent_count': count, 'historical_count': historical_tokens.get(token, 0)}
        for token, count in recent_tokens.most_common(50)
        if count >= 3 and historical_tokens.get(token, 0) == 0
    ]

    recent_skill_counts = _skill_counts(recent_offers, 'predicted_skills_json')
    historical_skill_counts = Counter()
    for annotation in approved_annotations:
        name = normalize_for_match(annotation.get('canonical_name') or annotation.get('surface_form') or '')
        if name:
            historical_skill_counts[name] += 1

    unknown_skills = [
        {'skill': skill, 'recent_count': count}
        for skill, count in recent_skill_counts.most_common(50)
        if historical_skill_counts.get(skill, 0) == 0
    ]

    drift = {
        'recent_offers': len(recent_offers),
        'approved_offers': len(approved_offers),
        'approved_annotations': len(approved_annotations),
        'new_expressions': new_expressions[:25],
        'unknown_skills': unknown_skills[:25],
        'mean_embedding_distance': mean_distance,
        'recent_families': sorted({clean_text(item.get('job_family')) for item in recent_offers if clean_text(item.get('job_family'))}),
        'recent_territories': sorted({clean_text(item.get('territory')) for item in recent_offers if clean_text(item.get('territory'))}),
        'low_confidence_rate': sum(1 for offer in recent_offers if json.loads(offer.get('confidence_json') or '{}') and max(json.loads(offer.get('confidence_json') or '{}').values(), default=0.0) < 0.5) / max(len(recent_offers), 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(drift, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    logger.info('Drift report generated: %s', drift)


if __name__ == '__main__':
    main()
