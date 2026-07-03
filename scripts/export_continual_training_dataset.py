#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from continual_learning.dataset_export import build_export_record
from continual_learning.store import ContinualLearningStore
from common.text import clean_text, normalize_for_match

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('export_continual_training_dataset')

PROVENANCE_RANK = {
    'model_prediction': 0,
    'semantic_match': 1,
    'exact_reference_match': 2,
    'france_travail_api': 3,
    'imported_gold_dataset': 4,
    'human_review': 5,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Export approved continual learning dataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-provenance', default='semantic_match')
    parser.add_argument('--since', default=None)
    parser.add_argument('--exclude-model-only', action='store_true')
    parser.add_argument('--include-france-travail-api', action='store_true')
    parser.add_argument('--include-human-reviewed', action='store_true')
    parser.add_argument('--db-path', type=Path, default=Path('data/continual_learning/continual_learning.sqlite3'))
    parser.set_defaults(exclude_model_only=True)
    return parser


def _rank(name: str) -> int:
    return PROVENANCE_RANK.get(name, 0)


def _load_annotations(store: ContinualLearningStore, offer_row_id: int) -> list[dict[str, Any]]:
    return store.list_annotations('offer_row_id = ?', (offer_row_id,))


def _keep_annotation(annotation: dict[str, Any], minimum_provenance: str, include_ft: bool, include_human: bool, exclude_model_only: bool) -> bool:
    provenance = annotation.get('provenance') or 'model_prediction'
    if _rank(provenance) < _rank(minimum_provenance):
        return False
    if provenance == 'france_travail_api' and not include_ft:
        return False
    if provenance == 'human_review' and not include_human:
        return False
    if exclude_model_only and provenance == 'model_prediction':
        return False
    return annotation.get('validation_status') in {'approved', 'corrected', 'used_for_training'} or provenance == 'france_travail_api'


def main() -> None:
    args = build_parser().parse_args()
    store = ContinualLearningStore(args.db_path)
    minimum_provenance = args.minimum_provenance
    if minimum_provenance not in PROVENANCE_RANK:
        raise SystemExit(f'Unknown provenance threshold: {minimum_provenance}')

    offers = store.list_offers("validation_status IN ('approved', 'corrected', 'used_for_training')")
    if args.since:
        offers = [offer for offer in offers if clean_text(offer.get('collected_at')) >= args.since]

    output_records = []
    skipped_model_only = 0
    for offer in offers:
        annotations = _load_annotations(store, int(offer['id']))
        if args.exclude_model_only and annotations and all((ann.get('provenance') == 'model_prediction') for ann in annotations):
            skipped_model_only += 1
            continue
        filtered = [ann for ann in annotations if _keep_annotation(ann, minimum_provenance, args.include_france_travail_api, args.include_human_reviewed, args.exclude_model_only)]
        if not filtered:
            continue
        record = build_export_record(offer, filtered)
        output_records.append(record.to_dict())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as fh:
        for record in output_records:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')

    summary = {
        'output': str(args.output),
        'records': len(output_records),
        'offers_scanned': len(offers),
        'skipped_model_only': skipped_model_only,
        'minimum_provenance': minimum_provenance,
    }
    (args.output.parent / 'export_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('Export complete: %s', summary)


if __name__ == '__main__':
    main()
