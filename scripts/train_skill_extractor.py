#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.text import normalize_for_match
from scripts.train_continual_skill_extractor import train as train_continual

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('train_skill_extractor')

REJECT_REASONS = {
    'EMPTY_LINE': 'ligne vide',
    'INVALID_JSON': 'JSON invalide',
    'MISSING_TEXT': 'champ text absent',
    'EMPTY_TEXT': 'texte vide ou compose uniquement d espaces',
    'MISSING_LABELS': 'champ labels absent',
    'INVALID_LABEL_ENTRY': 'entree labels sans canonical_label ou evidence_text',
    'INVALID_ENTITY_RANGE': 'entite avec start >= end',
    'ENTITY_NOT_IN_TEXT': 'texte de l entite introuvable dans le texte de l offre',
    'TEXT_TOO_SHORT': 'texte trop court (< 10 caracteres)',
    'TEXT_TOO_LONG': 'texte trop long (> 100000 caracteres)',
    'DUPLICATE': 'doublon (hash identique apres normalisation)',
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Entraîne l extracteur de competences base sur le dataset RNCP/ROME.')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--base-model', default='camembert-base')
    parser.add_argument('--output-dir', type=Path, default=Path('models/skill-extractor'))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--validate-only', action='store_true', help='Valide les datasets sans lancer l entrainement.')
    return parser


def _text_hash(text: str) -> str:
    norm = normalize_for_match(text)
    if not norm:
        return ''
    return sha1(norm.encode('utf-8')).hexdigest()[:24]


def _validate_rows(path: Path) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    rejected_samples: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    total = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        total += 1
        raw = line.strip()
        if not raw:
            counts['EMPTY_LINE'] += 1
            _record_reject(rejected_samples, counts, 'EMPTY_LINE', line)
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            counts['INVALID_JSON'] += 1
            _record_reject(rejected_samples, counts, 'INVALID_JSON', raw[:200])
            continue
        if not isinstance(row, dict):
            counts['INVALID_JSON'] += 1
            _record_reject(rejected_samples, counts, 'INVALID_JSON', raw[:200])
            continue
        text = row.get('text')
        if text is None:
            counts['MISSING_TEXT'] += 1
            _record_reject(rejected_samples, counts, 'MISSING_TEXT', row)
            continue
        if not isinstance(text, str):
            counts['MISSING_TEXT'] += 1
            _record_reject(rejected_samples, counts, 'MISSING_TEXT', row)
            continue
        if not text.strip():
            counts['EMPTY_TEXT'] += 1
            _record_reject(rejected_samples, counts, 'EMPTY_TEXT', row)
            continue
        if len(text.strip()) < 10:
            counts['TEXT_TOO_SHORT'] += 1
            _record_reject(rejected_samples, counts, 'TEXT_TOO_SHORT', row)
            continue
        if len(text) > 100000:
            counts['TEXT_TOO_LONG'] += 1
            _record_reject(rejected_samples, counts, 'TEXT_TOO_LONG', row)
            continue
        dedup_key = _text_hash(text)
        if not dedup_key:
            counts['EMPTY_TEXT'] += 1
            _record_reject(rejected_samples, counts, 'EMPTY_TEXT', row)
            continue
        if dedup_key in seen_hashes:
            counts['DUPLICATE'] += 1
            _record_reject(rejected_samples, counts, 'DUPLICATE', row)
            continue
        seen_hashes.add(dedup_key)
        labels = row.get('labels')
        if labels is None:
            counts['MISSING_LABELS'] += 1
            _record_reject(rejected_samples, counts, 'MISSING_LABELS', row)
            continue
        if not isinstance(labels, list):
            counts['MISSING_LABELS'] += 1
            _record_reject(rejected_samples, counts, 'MISSING_LABELS', row)
            continue
        skip_row = False
        for li, label in enumerate(labels):
            if not isinstance(label, dict):
                counts['INVALID_LABEL_ENTRY'] += 1
                _record_reject(rejected_samples, counts, 'INVALID_LABEL_ENTRY', {'row_id': row.get('id'), 'label_index': li})
                skip_row = True
                break
            evidence_text = label.get('evidence_text')
            if not evidence_text or not isinstance(evidence_text, str) or not evidence_text.strip():
                counts['INVALID_LABEL_ENTRY'] += 1
                _record_reject(rejected_samples, counts, 'INVALID_LABEL_ENTRY', {'row_id': row.get('id'), 'label_index': li})
                skip_row = True
                break
            start = label.get('evidence_start')
            end = label.get('evidence_end')
            if start is None or end is None or not isinstance(start, int) or not isinstance(end, int):
                counts['INVALID_LABEL_ENTRY'] += 1
                _record_reject(rejected_samples, counts, 'INVALID_LABEL_ENTRY', {'row_id': row.get('id'), 'label_index': li})
                skip_row = True
                break
            if end <= start:
                counts['INVALID_ENTITY_RANGE'] += 1
                _record_reject(rejected_samples, counts, 'INVALID_ENTITY_RANGE', {'row_id': row.get('id'), 'label_index': li, 'start': start, 'end': end})
                skip_row = True
                break
            if start > len(text) or end > len(text):
                counts['INVALID_ENTITY_RANGE'] += 1
                _record_reject(rejected_samples, counts, 'INVALID_ENTITY_RANGE', {'row_id': row.get('id'), 'label_index': li, 'start': start, 'end': end, 'text_len': len(text)})
                skip_row = True
                break
            snippet = text[start:end]
            norm_evidence = normalize_for_match(evidence_text)
            norm_snippet = normalize_for_match(snippet)
            if norm_evidence and norm_snippet and norm_evidence != norm_snippet:
                counts['ENTITY_NOT_IN_TEXT'] += 1
                _record_reject(rejected_samples, counts, 'ENTITY_NOT_IN_TEXT', {'row_id': row.get('id'), 'label_index': li, 'start': start, 'end': end, 'expected': evidence_text, 'actual': snippet})
                skip_row = True
                break
        if skip_row:
            continue
        rows.append(row)

    counts['ACCEPTED'] = len(rows)
    counts['TOTAL_READ'] = total
    return rows, counts, rejected_samples


def _record_reject(rejected_samples: list[dict[str, Any]], counts: Counter[str], reason: str, detail: Any) -> None:
    if len(rejected_samples) >= 50:
        return
    if isinstance(detail, dict):
        entry = {'reason': reason, 'detail': {k: v for k, v in detail.items() if isinstance(v, (str, int, float, bool, list))}}
    else:
        entry = {'reason': reason, 'detail': str(detail)[:500]}
    rejected_samples.append(entry)


def _print_validation(name: str, rows: list[dict[str, Any]], counts: Counter[str]) -> None:
    print(f'\n=== {name} ===')
    print(f'  lignes lues                   : {counts["TOTAL_READ"]}')
    for reason in ['EMPTY_LINE', 'INVALID_JSON', 'MISSING_TEXT', 'EMPTY_TEXT', 'MISSING_LABELS',
                   'INVALID_LABEL_ENTRY', 'INVALID_ENTITY_RANGE', 'ENTITY_NOT_IN_TEXT',
                   'TEXT_TOO_SHORT', 'TEXT_TOO_LONG', 'DUPLICATE']:
        if counts[reason] > 0:
            print(f'  rejetes ({REJECT_REASONS[reason]:30s}): {counts[reason]}')
    print(f'  acceptes                      : {counts["ACCEPTED"]}')
    positive = sum(1 for r in rows if r.get('labels'))
    print(f'  dont exemples positifs         : {positive}')
    print(f'  dont exemples negatifs         : {len(rows) - positive}')
    all_labels = [lbl['canonical_label'] for r in rows for lbl in (r.get('labels') or []) if isinstance(lbl, dict)]
    if all_labels:
        dist = Counter(all_labels)
        print(f'  distribution des labels        : {dict(dist.most_common(10))}')
        print(f'  labels uniques                 : {len(dist)}')


def validate_only(train: Path, validation: Path, test: Path, report_path: Path) -> int:
    report: dict[str, Any] = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'files': {},
        'overall': {},
    }
    splits = {
        'train': train,
        'validation': validation,
        'test': test,
    }
    all_train_hashes: set[str] = set()
    all_val_hashes: set[str] = set()
    all_test_hashes: set[str] = set()

    for name, path in splits.items():
        rows, counts, rejected = _validate_rows(path)
        report['files'][name] = {
            'path': str(path),
            'total_read': counts['TOTAL_READ'],
            'accepted': counts['ACCEPTED'],
            'rejected_by_reason': {k: v for k, v in counts.items() if k not in ('TOTAL_READ', 'ACCEPTED') and v > 0},
            'positive_examples': sum(1 for r in rows if r.get('labels')),
            'negative_examples': sum(1 for r in rows if not r.get('labels')),
        }
        if rows:
            all_labels = [lbl['canonical_label'] for r in rows for lbl in (r.get('labels') or []) if isinstance(lbl, dict)]
            report['files'][name]['label_distribution'] = dict(Counter(all_labels).most_common(25))
            report['files'][name]['unique_labels'] = len(set(all_labels))
            report['files'][name]['sample_sizes'] = [len(r.get('text', '')) for r in rows[:5]]
        _print_validation(name, rows, counts)
        if name == 'train':
            all_train_hashes = {_text_hash(r.get('text', '')) for r in rows if r.get('text')}
        elif name == 'validation':
            all_val_hashes = {_text_hash(r.get('text', '')) for r in rows if r.get('text')}
        elif name == 'test':
            all_test_hashes = {_text_hash(r.get('text', '')) for r in rows if r.get('text')}

    train_val_overlap = all_train_hashes & all_val_hashes
    train_test_overlap = all_train_hashes & all_test_hashes
    val_test_overlap = all_val_hashes & all_test_hashes

    report['overall'] = {
        'train_val_overlap': len(train_val_overlap),
        'train_test_overlap': len(train_test_overlap),
        'val_test_overlap': len(val_test_overlap),
        'has_positive_examples': report['files'].get('train', {}).get('positive_examples', 0) > 0,
        'has_negative_examples': report['files'].get('train', {}).get('negative_examples', 0) > 0,
        'train_is_empty': report['files'].get('train', {}).get('accepted', 0) == 0,
    }

    print('\n=== Chevauchements entre splits ===')
    print(f'  train vs validation: {len(train_val_overlap)}')
    print(f'  train vs test      : {len(train_test_overlap)}')
    print(f'  validation vs test : {len(val_test_overlap)}')

    print('\n=== Verdict ===')
    train_acc = report['files'].get('train', {}).get('accepted', 0)
    val_acc = report['files'].get('validation', {}).get('accepted', 0)
    test_acc = report['files'].get('test', {}).get('accepted', 0)
    pos = report['files'].get('train', {}).get('positive_examples', 0)

    errors = []
    if train_acc == 0:
        errors.append('train est vide')
    if val_acc == 0:
        errors.append('validation est vide')
    if test_acc == 0:
        errors.append('test est vide')
    if pos == 0:
        errors.append('pas d exemples positifs dans train')
    if train_val_overlap:
        errors.append(f'chevauchenment train/validation: {len(train_val_overlap)}')
    if train_test_overlap:
        errors.append(f'chevauchenment train/test: {len(train_test_overlap)}')

    if errors:
        print(f'  ECHEC: {"; ".join(errors)}')
        report['overall']['verdict'] = 'FAIL'
        report['overall']['errors'] = errors
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
        print(f'\nRapport ecrit dans {report_path}')
        return 1
    else:
        print('  OK : le dataset est exploitable')
        report['overall']['verdict'] = 'PASS'
        report['overall']['errors'] = []
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
        print(f'\nRapport ecrit dans {report_path}')
        return 0


def _to_ner_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        text = row.get('text')
        if not isinstance(text, str) or not text.strip():
            continue
        entities = []
        for label in row.get('labels') or []:
            if not isinstance(label, dict):
                continue
            start = label.get('evidence_start')
            end = label.get('evidence_end')
            evidence_text = label.get('evidence_text')
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                continue
            if start > len(text) or end > len(text):
                continue
            entities.append({
                'start': start,
                'end': end,
                'text': evidence_text or '',
                'provenance': 'human_review',
            })
        if not entities:
            continue
        records.append({
            'id': row.get('id'),
            'text': text,
            'entities': entities,
            'document_skills': [],
            'metadata': {'title': row.get('title'), 'rome_code': row.get('rome_code'), 'rome_label': row.get('rome_label')},
            'source_path': 'derived',
        })
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ('\n' if rows else ''), encoding='utf-8')


def main() -> None:
    args = build_parser().parse_args()
    report_path = Path('data/training/skill_extraction/validation_report.json')

    if args.validate_only:
        sys.exit(validate_only(args.train, args.validation, args.test, report_path))

    train_rows = _read_jsonl(args.train)
    validation_rows = _read_jsonl(args.validation)
    test_rows = _read_jsonl(args.test)

    ner_train = _to_ner_records(train_rows)
    ner_validation = _to_ner_records(validation_rows)
    ner_test = _to_ner_records(test_rows)

    if not ner_train:
        logger.error('Aucun exemple d entrainement exploitable apres conversion NER.')
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix='skill_extractor_') as tmp:
        tmpdir = Path(tmp)
        base_path = tmpdir / 'base.jsonl'
        incremental_path = tmpdir / 'incremental.jsonl'
        validation_path = tmpdir / 'validation.jsonl'
        test_path = tmpdir / 'test.jsonl'
        _write_jsonl(base_path, ner_train)
        _write_jsonl(incremental_path, [])
        _write_jsonl(validation_path, ner_validation)
        _write_jsonl(test_path, ner_test)
        train_continual(SimpleNamespace(
            base_dataset=base_path,
            incremental_dataset=incremental_path,
            validation_dataset=validation_path,
            test_dataset=test_path,
            base_model=args.base_model,
            output_dir=args.output_dir,
            resume_from_model=None,
            max_samples=None,
            seed=args.seed,
            device=args.device,
            fp16=args.fp16,
            batch_size=args.batch_size,
            epochs=args.epochs,
        ))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


if __name__ == '__main__':
    main()
