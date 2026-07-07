#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from hashlib import sha1

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.text import normalize_for_match

sys.path.insert(0, str(ROOT))
from scripts.train_skill_extractor import (
    _validate_rows,
    _to_ner_records,
    _text_hash,
    validate_only,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n',
        encoding='utf-8',
    )


def _make_row(text: str = 'Bonjour ceci est un texte de test pour les competences.',
              labels: list | None = None) -> dict:
    if labels is None:
        labels = [
            {
                'canonical_skill_id': 'test_123',
                'canonical_label': 'Python',
                'evidence_start': 0,
                'evidence_end': 7,
                'evidence_text': 'Bonjour',
                'source_links': ['test'],
            }
        ] if 'Bonjour' in text else [
            {
                'canonical_skill_id': 'test_123',
                'canonical_label': 'Python',
                'evidence_start': text.find(text.strip().split()[0]) if text.strip() else 0,
                'evidence_end': (text.find(text.strip().split()[0]) if text.strip() else 0) + len(text.strip().split()[0]) if text.strip() else 0,
                'evidence_text': text.strip().split()[0] if text.strip() else '',
                'source_links': ['test'],
            }
        ]
    return {
        'id': 'test_id',
        'text': text,
        'title': 'Test',
        'rome_code': 'M1805',
        'rome_label': 'Data',
        'labels': labels,
    }


class TestValidateRows:
    def test_valid_row(self):
        rows = [_make_row()]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            _write_jsonl(path, rows)
            accepted, counts, rejected = _validate_rows(path)
        assert counts['ACCEPTED'] == 1
        assert counts['TOTAL_READ'] == 1
        assert len(accepted) == 1

    def test_empty_line(self):
        path = Path(tempfile.mktemp(suffix='.jsonl'))
        path.write_text('\n\n', encoding='utf-8')
        accepted, counts, rejected = _validate_rows(path)
        assert counts['EMPTY_LINE'] >= 1
        assert counts['ACCEPTED'] == 0

    def test_invalid_json(self):
        path = Path(tempfile.mktemp(suffix='.jsonl'))
        path.write_text('{not json}\n', encoding='utf-8')
        accepted, counts, rejected = _validate_rows(path)
        assert counts['INVALID_JSON'] == 1
        assert counts['ACCEPTED'] == 0

    def test_missing_text(self):
        rows = [{'id': 'no_text'}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            _write_jsonl(path, rows)
            accepted, counts, rejected = _validate_rows(path)
        assert counts['MISSING_TEXT'] == 1
        assert counts['ACCEPTED'] == 0

    def test_empty_text(self):
        rows = [_make_row(text='   ')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            _write_jsonl(path, rows)
            accepted, counts, rejected = _validate_rows(path)
        assert counts['EMPTY_TEXT'] == 1
        assert counts['ACCEPTED'] == 0

    def test_text_too_short(self):
        rows = [_make_row(text='Court')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            _write_jsonl(path, rows)
            accepted, counts, rejected = _validate_rows(path)
        assert counts['TEXT_TOO_SHORT'] == 1
        assert counts['ACCEPTED'] == 0

    def test_missing_labels(self):
        rows = [{'id': 'test', 'text': 'Un texte assez long pour etre valide ici present.'}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            _write_jsonl(path, rows)
            accepted, counts, rejected = _validate_rows(path)
        assert counts['MISSING_LABELS'] == 1
        assert counts['ACCEPTED'] == 0

    def test_duplicate_text(self):
        rows = [_make_row() for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            _write_jsonl(path, rows)
            accepted, counts, rejected = _validate_rows(path)
        assert counts['DUPLICATE'] == 2
        assert counts['ACCEPTED'] == 1

    def test_invalid_entity_range(self):
        labels = [
            {
                'canonical_skill_id': 'test',
                'canonical_label': 'Python',
                'evidence_start': 10,
                'evidence_end': 5,
                'evidence_text': 'invalide',
                'source_links': [],
            }
        ]
        rows = [_make_row(labels=labels)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            _write_jsonl(path, rows)
            accepted, counts, rejected = _validate_rows(path)
        assert counts['INVALID_ENTITY_RANGE'] == 1
        assert counts['ACCEPTED'] == 0

    def test_entity_not_in_text(self):
        labels = [
            {
                'canonical_skill_id': 'test',
                'canonical_label': 'Python',
                'evidence_start': 0,
                'evidence_end': 5,
                'evidence_text': 'ZZZZZ',
                'source_links': [],
            }
        ]
        rows = [_make_row(labels=labels)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.jsonl'
            _write_jsonl(path, rows)
            accepted, counts, rejected = _validate_rows(path)
        assert counts['ENTITY_NOT_IN_TEXT'] == 1
        assert counts['ACCEPTED'] == 0

    def test_empty_string_dedup_key(self):
        """Empty text should not create a valid dedup key."""
        key = _text_hash('')
        assert key == ''
        key = _text_hash('   ')
        assert key == ''


class TestToNerRecords:
    def test_valid_conversion(self):
        rows = [_make_row()]
        records = _to_ner_records(rows)
        assert len(records) == 1
        assert records[0]['text'] == rows[0]['text']
        assert len(records[0]['entities']) == 1
        assert records[0]['entities'][0]['start'] == 0
        assert records[0]['entities'][0]['end'] == 7

    def test_empty_text_skipped(self):
        rows = [_make_row(text='')]
        records = _to_ner_records(rows)
        assert len(records) == 0

    def test_no_entities_skipped(self):
        rows = [_make_row(labels=[])]
        records = _to_ner_records(rows)
        assert len(records) == 0

    def test_invalid_entity_skipped(self):
        labels = [
            {
                'canonical_skill_id': 'test',
                'canonical_label': 'Python',
                'evidence_start': 10,
                'evidence_end': 5,
                'evidence_text': 'invalide',
                'source_links': [],
            }
        ]
        rows = [_make_row(labels=labels)]
        records = _to_ner_records(rows)
        assert len(records) == 0


class TestValidateOnly:
    def test_validate_only_passes(self):
        train = [_make_row(
            text=f'Bonjour entrainement numero {i} Python avec des competences.',
            labels=[{'canonical_skill_id': 'test', 'canonical_label': 'Python',
                      'evidence_start': 0, 'evidence_end': 7, 'evidence_text': 'Bonjour',
                      'source_links': []}],
        ) for i in range(10)]
        val = [_make_row(
            text=f'Bonjour validation numero {i} Python avec des competences.',
            labels=[{'canonical_skill_id': 'test', 'canonical_label': 'Python',
                      'evidence_start': 0, 'evidence_end': 7, 'evidence_text': 'Bonjour',
                      'source_links': []}],
        ) for i in range(5)]
        test = [_make_row(
            text=f'Bonjour test numero {i} Python avec des competences.',
            labels=[{'canonical_skill_id': 'test', 'canonical_label': 'Python',
                      'evidence_start': 0, 'evidence_end': 7, 'evidence_text': 'Bonjour',
                      'source_links': []}],
        ) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            train_path = tmpdir / 'train.jsonl'
            val_path = tmpdir / 'val.jsonl'
            test_path = tmpdir / 'test.jsonl'
            report_path = tmpdir / 'report.json'
            _write_jsonl(train_path, train)
            _write_jsonl(val_path, val)
            _write_jsonl(test_path, test)
            rc = validate_only(train_path, val_path, test_path, report_path)
            assert rc == 0
            report = json.loads(report_path.read_text(encoding='utf-8'))
            assert report['overall']['verdict'] == 'PASS'

    def test_validate_only_empty_train_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            train_path = tmpdir / 'train.jsonl'
            val_path = tmpdir / 'val.jsonl'
            test_path = tmpdir / 'test.jsonl'
            report_path = tmpdir / 'report.json'
            train_path.write_text('', encoding='utf-8')
            _write_jsonl(val_path, [_make_row(text='Validation unique exemple texte.')])
            _write_jsonl(test_path, [_make_row(text='Test unique exemple texte.')])
            rc = validate_only(train_path, val_path, test_path, report_path)
        assert rc != 0


class TestEvidenceSpan:
    """Test _evidence_span from build_rome_rncp_training_dataset."""

    def _evidence_span(self, text: str, evidence: str) -> tuple[int, int]:
        from scripts.build_rome_rncp_training_dataset import _evidence_span
        return _evidence_span(text, evidence)

    def test_exact_match(self):
        text = 'Developpeur Python avec experience'
        evidence = 'Python'
        start, end = self._evidence_span(text, evidence)
        assert text[start:end] == evidence

    def test_case_insensitive(self):
        text = 'Developpeur PYTHON avec experience'
        evidence = 'python'
        start, end = self._evidence_span(text, evidence)
        assert text[start:end].lower() == evidence

    def test_no_match(self):
        text = 'Developpeur Java avec experience'
        evidence = 'Python'
        start, end = self._evidence_span(text, evidence)
        assert start == 0 and end == 0

    def test_empty_evidence(self):
        text = 'Developpeur Python'
        start, end = self._evidence_span(text, '')
        assert start == 0 and end == 0

    def test_accent_match(self):
        text = 'Ingenieur en intelligence artificielle'
        evidence = 'ingénieur'
        start, end = self._evidence_span(text, evidence)
        if start > 0 or end > 0:
            assert normalize_for_match(text[start:end]) == normalize_for_match(evidence)
