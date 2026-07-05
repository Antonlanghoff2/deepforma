from __future__ import annotations

import pytest

from referential_learning.pdf_loader import load_pdf_document
from referential_learning.pipeline import build_annotation_document
from referential_learning.section_labels import classify_section_label
from referential_learning.store import AnnotationStore


pytest.importorskip('fitz')


def _make_pdf(path: Path) -> None:
    import fitz  # type: ignore

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), 'Titre de la formation\nObjectifs\nPython\nLean Management', fontsize=14)
    doc.save(path)
    doc.close()


def test_loader_and_candidate_generation(tmp_path):
    pdf_path = tmp_path / 'sample.pdf'
    _make_pdf(pdf_path)

    document = load_pdf_document(pdf_path)
    assert document.page_count == 1
    assert document.pages[0].text
    assert document.pages[0].blocks

    annotation = build_annotation_document(document)
    assert annotation.document_id == document.document_id
    assert annotation.blocks
    assert annotation.entities

    title_match = classify_section_label('OBJECTIFS PÉDAGOGIQUES')
    assert title_match.label == 'OBJECTIVES'


def test_annotation_store_roundtrip(tmp_path):
    store = AnnotationStore(tmp_path / 'candidates.jsonl')
    record = {
        'document_id': 'doc-1',
        'source_file': 'sample.pdf',
        'pages': [],
        'blocks': [],
        'entities': [],
        'status': 'pending',
    }
    store.save([record])
    assert store.get('doc-1')['status'] == 'pending'
    updated = store.update_status('doc-1', 'validated', validated_by='tester')
    assert updated['status'] == 'validated'
    assert store.get('doc-1')['validated_by'] == 'tester'


def test_export_and_split_helpers():
    from scripts.export_approved_referential_annotations import _approved, _split_documents

    docs = [
        {'document_id': 'a', 'status': 'approved'},
        {'document_id': 'b', 'status': 'validated'},
        {'document_id': 'c', 'status': 'pending'},
    ]
    assert _approved(docs[0]) is True
    assert _approved(docs[2]) is False
    splits = _split_documents([docs[0], docs[1]], seed=42)
    assert sum(len(items) for items in splits.values()) == 2
    assert set(doc['document_id'] for items in splits.values() for doc in items) == {'a', 'b'}
