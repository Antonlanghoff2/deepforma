from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from common.text import clean_text, split_multi_values

from .ml_dl_taxonomy import canonical_entity_type, dedupe_mentions, find_mentions, infer_families, negative_hint_score
from .models import AnnotationBlock, AnnotationDocument, AnnotationEntity, AnnotationPage, PdfBlock, PdfDocument
from .section_labels import classify_section_label
from .skill_normalizer import SkillNormalizer

DEFAULT_SECTION_MODEL_DIR = Path(os.getenv('REFERENTIAL_SECTION_MODEL_DIR', 'models/referential-section-classifier/current'))
DEFAULT_NER_MODEL_DIR = Path(os.getenv('REFERENTIAL_NER_MODEL_DIR', 'models/referential-skill-ner/current'))
DEFAULT_MULTILABEL_MODEL_DIR = Path(os.getenv('REFERENTIAL_MULTILABEL_MODEL_DIR', 'models/referential-multilabel/current'))
ML_ENABLED_ENV = os.getenv('REFERENTIAL_ML_IMPORT_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'on'}
ML_SKILL_MODEL_ENABLED = os.getenv('REFERENTIAL_ML_SKILL_MODEL_ENABLED', 'false').lower() in {'1', 'true', 'yes', 'on'}

@lru_cache(maxsize=1)
def _load_section_model() -> tuple[Any | None, Any | None]:
    if not ML_ENABLED_ENV or not DEFAULT_SECTION_MODEL_DIR.exists():
        return None, None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        return (
            AutoTokenizer.from_pretrained(DEFAULT_SECTION_MODEL_DIR),
            AutoModelForSequenceClassification.from_pretrained(DEFAULT_SECTION_MODEL_DIR),
        )
    except Exception:
        return None, None

@lru_cache(maxsize=1)
def _load_ner_model() -> tuple[Any | None, Any | None]:
    if not (ML_ENABLED_ENV and ML_SKILL_MODEL_ENABLED and DEFAULT_NER_MODEL_DIR.exists()):
        return None, None
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore
        return (
            AutoTokenizer.from_pretrained(DEFAULT_NER_MODEL_DIR),
            AutoModelForTokenClassification.from_pretrained(DEFAULT_NER_MODEL_DIR),
        )
    except Exception:
        return None, None

@lru_cache(maxsize=1)
def _load_multilabel_model() -> tuple[Any | None, Any | None, dict[str, float] | None]:
    if not (ML_ENABLED_ENV and ML_SKILL_MODEL_ENABLED and DEFAULT_MULTILABEL_MODEL_DIR.exists()):
        return None, None, None
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        thresholds_path = DEFAULT_MULTILABEL_MODEL_DIR / 'thresholds.json'
        thresholds: dict[str, float] = {}
        if thresholds_path.exists():
            import json
            thresholds = json.loads(thresholds_path.read_text(encoding='utf-8'))
        return (
            AutoTokenizer.from_pretrained(DEFAULT_MULTILABEL_MODEL_DIR),
            AutoModelForSequenceClassification.from_pretrained(DEFAULT_MULTILABEL_MODEL_DIR),
            thresholds,
        )
    except Exception:
        return None, None, None


def _block_bbox(block: PdfBlock) -> list[float] | None:
    if block.bbox is None:
        return None
    return [float(item) for item in block.bbox]


def _section_from_block(block: PdfBlock) -> AnnotationBlock:
    match = classify_section_label(block.text)
    return AnnotationBlock(
        block_id=block.block_id,
        page=block.page,
        bbox=_block_bbox(block),
        text=block.text,
        predicted_section=match.label,
        confidence=match.confidence,
        approved_section=None,
        order=block.order,
        source_file=block.source_file,
        document_id=block.document_id,
    )


def _predict_section_label(text: str) -> tuple[str, float]:
    tokenizer, model = _load_section_model()
    if tokenizer is None or model is None:
        match = classify_section_label(text)
        return match.label, match.confidence
    try:
        import torch  # type: ignore
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        label = model.config.id2label.get(idx, 'OTHER')
        return label, float(probs[idx].item())
    except Exception:
        match = classify_section_label(text)
        return match.label, match.confidence


def _extract_entities_from_text(
    text: str,
    normalizer: SkillNormalizer,
    *,
    page_number: int,
    source_file: str,
    document_id: str,
) -> list[AnnotationEntity]:
    entities: list[AnnotationEntity] = []
    entity_index: dict[tuple[int, int, str], AnnotationEntity] = {}

    def add(
        label: str,
        start: int,
        end: int,
        surface: str,
        *,
        canonical: str = '',
        confidence: float = 0.82,
        referential_id: str | None = None,
        evidence: str = '',
    ) -> None:
        key = (start, end, clean_text(canonical or surface).lower())
        current = entity_index.get(key)
        if current is not None:
            if confidence > float(current.confidence):
                current.predicted_label = label
                current.approved_label = None
                current.canonical_name = canonical or current.canonical_name
                current.confidence = confidence
                current.referential_id = referential_id or current.referential_id
                current.evidence = evidence or current.evidence
            return
        entity = AnnotationEntity(
            entity_id=f'{label}:{start}:{end}:{clean_text(canonical or surface)}',
            start=start,
            end=end,
            text=surface,
            predicted_label=label,
            approved_label=None,
            canonical_name=canonical,
            confidence=confidence,
            page=page_number,
            source_file=source_file,
            document_id=document_id,
            evidence=evidence,
            referential_id=referential_id,
        )
        entity_index[key] = entity
        entities.append(entity)

    import re

    explicit_patterns = [
        ('REFERENCE', r'(?i)\b(?:référence|reference)\s*(?:[:\-–—])\s*([^\n\r;]+)'),
        ('PROVIDER', r'(?i)\b(?:organisme|provider|éditeur|editeur)\s*(?:[:\-–—])\s*([^\n\r;]+)'),
        ('DURATION', r'(?i)\b(?:durée|duree)\s*(?:[:\-–—])\s*([^\n\r;]+)'),
        ('PRICE', r'(?i)\b(?:prix|tarif)\s*(?:[:\-–—])\s*([^\n\r;]+)'),
        ('CERTIFICATION', r'(?i)\b(?:certification|titre rncp)\s*(?:[:\-–—])\s*([^\n\r;]+)'),
    ]
    for label, pattern in explicit_patterns:
        for match in re.finditer(pattern, text):
            surface = clean_text(match.group(1))
            if surface:
                add(label, match.start(1), match.end(1), surface, canonical=surface, confidence=0.95, evidence='explicit_field')

    for match in re.finditer(r'(?i)\b(?:cpf|éligible cpf|eligible cpf)\b', text):
        surface = clean_text(match.group(0))
        add('OTHER', match.start(), match.end(), surface, canonical=surface, confidence=0.72, evidence='cpf_marker')

    for mention in find_mentions(text):
        label = canonical_entity_type(mention['entity_type'])
        add(
            label,
            int(mention['start']),
            int(mention['end']),
            clean_text(mention['text']),
            canonical=mention['canonical_name'],
            confidence=0.97,
            evidence='taxonomy_alias',
        )

    try:
        from skills.open_extractor import extract_skills as extract_open_skills  # type: ignore
        for item in extract_open_skills(text):
            canonical = clean_text(item.normalized_label or item.source_label)
            if not canonical:
                continue
            if item.type in {'tool', 'tool_with_context'}:
                label = 'TOOL'
            elif item.type == 'knowledge':
                label = 'SKILL'
            elif item.type == 'soft_skill':
                label = 'SOFT_SKILL'
            elif item.type == 'method':
                label = 'METHOD'
            elif item.type == 'domain':
                label = 'DOMAIN'
            else:
                label = 'SKILL'
            add(
                label,
                item.start,
                item.end,
                clean_text(item.source_text or item.source_label),
                canonical=canonical,
                confidence=float(item.confidence or 0.8),
                evidence=item.method,
            )
    except Exception:
        for candidate in split_multi_values(text):
            result = normalizer.normalize(candidate)
            if result.accepted and result.canonical_name:
                start = text.find(candidate)
                if start >= 0:
                    add(
                        'SKILL',
                        start,
                        start + len(candidate),
                        candidate,
                        canonical=result.canonical_name,
                        confidence=result.confidence,
                        referential_id=result.referential_id,
                        evidence=result.provenance,
                    )
    return entities


def build_annotation_document(document: PdfDocument) -> AnnotationDocument:
    normalizer = SkillNormalizer()
    annotation_pages: list[AnnotationPage] = []
    all_blocks: list[AnnotationBlock] = []
    all_entities: list[AnnotationEntity] = []

    for page in document.pages:
        page_blocks = [_section_from_block(block) for block in page.blocks]
        page_entities = _extract_entities_from_text(
            page.text,
            normalizer,
            page_number=page.number,
            source_file=document.source_file,
            document_id=document.document_id,
        )
        annotation_pages.append(AnnotationPage(number=page.number, text=page.text, blocks=page_blocks, entities=page_entities))
        all_blocks.extend(page_blocks)
        all_entities.extend(page_entities)

    return AnnotationDocument(
        document_id=document.document_id,
        source_file=document.source_file,
        sha256=document.sha256,
        page_count=document.page_count,
        pages=annotation_pages,
        blocks=all_blocks,
        entities=all_entities,
        status='pending',
        needs_ocr=document.needs_ocr,
        notes='generated_candidates',
        metadata={'extraction_method': document.extraction_method},
    )


def enrich_with_ml_predictions(document: AnnotationDocument) -> AnnotationDocument:
    if not ML_ENABLED_ENV or not ML_SKILL_MODEL_ENABLED:
        return document
    try:
        multilabels: list[dict[str, Any]] = []
        for page in document.pages:
            page_labels = infer_families(page.text)
            if page_labels:
                multilabels.append({'page': page.number, 'labels': page_labels})
        document.metadata['ml_predictions'] = {'multilabels': multilabels}
    except Exception as exc:
        document.metadata['ml_predictions_error'] = str(exc)
    return document
