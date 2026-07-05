from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from common.text import clean_text

from .models import PdfBlock, PdfDocument, PdfPage

LOGGER = logging.getLogger(__name__)


def _load_fitz() -> Any | None:
    try:
        import fitz  # type: ignore
    except Exception:
        return None
    return fitz


def _block_from_line(*, document_id: str, source_file: str, page_number: int, block_index: int, line_index: int, line: dict[str, Any]) -> PdfBlock:
    spans = line.get('spans', []) or []
    text_parts: list[str] = []
    font_name = ''
    font_size: float | None = None
    bold = False
    for span in spans:
        text = clean_text(span.get('text'))
        if text:
            text_parts.append(text)
            font_name = font_name or clean_text(span.get('font'))
            try:
                font_size = max(font_size or 0.0, float(span.get('size', 0.0)))
            except Exception:
                pass
            bold = bold or 'bold' in clean_text(span.get('font')).lower()
    text = clean_text(' '.join(text_parts))
    bbox_values = line.get('bbox')
    bbox = None
    if bbox_values:
        try:
            bbox = tuple(float(value) for value in bbox_values)
        except Exception:
            bbox = None
    return PdfBlock(
        block_id=hashlib.sha1(f'{document_id}:{page_number}:{block_index}:{line_index}:{text}'.encode('utf-8')).hexdigest()[:16],
        page=page_number,
        order=block_index,
        text=text,
        bbox=bbox,
        font_size=font_size,
        font_name=font_name,
        bold=bold,
        block_type='text',
        line_count=1,
        source_file=source_file,
        document_id=document_id,
    )


def _extract_with_fitz(path: Path) -> PdfDocument:
    fitz = _load_fitz()
    if fitz is None:
        raise ImportError('PyMuPDF indisponible')
    doc = fitz.open(path)
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    document_id = hashlib.sha1(f'{path.name}|{sha256}'.encode('utf-8')).hexdigest()[:24]
    pages: list[PdfPage] = []
    for page_number, page in enumerate(doc, start=1):
        page_dict = page.get_text('dict')
        blocks: list[PdfBlock] = []
        order = 0
        for block_index, block in enumerate(page_dict.get('blocks', [])):
            if block.get('type', 0) != 0:
                continue
            for line_index, line in enumerate(block.get('lines', []) or [], start=1):
                pdf_block = _block_from_line(
                    document_id=document_id,
                    source_file=path.name,
                    page_number=page_number,
                    block_index=order,
                    line_index=line_index,
                    line=line,
                )
                if pdf_block.text:
                    blocks.append(pdf_block)
                    order += 1
        page_text = '\n'.join(block.text for block in blocks if block.text)
        text_length = len(page_text)
        area = max(float(page.rect.width) * float(page.rect.height), 1.0)
        density = text_length / area
        pages.append(PdfPage(number=page_number, width=float(page.rect.width), height=float(page.rect.height), text=page_text, blocks=blocks, text_length=text_length, density=density, has_text_layer=bool(page_text.strip())))
    needs_ocr = not any(page.has_text_layer for page in pages)
    return PdfDocument(
        document_id=document_id,
        source_file=path.name,
        path=str(path),
        sha256=sha256,
        file_size=path.stat().st_size,
        page_count=len(pages),
        extraction_method='pymupdf',
        pages=pages,
        warnings=[],
        needs_ocr=needs_ocr,
    )


def load_pdf_document(path: str | Path) -> PdfDocument:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f'PDF introuvable: {pdf_path}')
    try:
        return _extract_with_fitz(pdf_path)
    except Exception as exc:
        LOGGER.warning('Extraction PyMuPDF impossible pour %s: %s', pdf_path, exc)
        try:
            from referential_import.pdf_loader import load_pdf_document as fallback_loader  # type: ignore
            fallback = fallback_loader(pdf_path)
            raw = pdf_path.read_bytes()
            sha256 = hashlib.sha256(raw).hexdigest()
            document_id = hashlib.sha1(f'{pdf_path.name}|{sha256}'.encode('utf-8')).hexdigest()[:24]
            pages: list[PdfPage] = []
            for page in fallback.pages:
                blocks = [
                    PdfBlock(
                        block_id=hashlib.sha1(f'{document_id}:{page.number}:{block.order}:{block.line_number}:{block.text}'.encode('utf-8')).hexdigest()[:16],
                        page=page.number,
                        order=block.order,
                        text=block.text,
                        bbox=block.bbox,
                        font_size=None,
                        font_name='',
                        bold=False,
                        block_type='text',
                        line_count=1,
                        source_file=pdf_path.name,
                        document_id=document_id,
                    )
                    for block in page.blocks
                ]
                page_text = '\n'.join(block.text for block in blocks if block.text)
                text_length = len(page_text)
                area = max((page.width or 1.0) * (page.height or 1.0), 1.0)
                density = text_length / area
                pages.append(PdfPage(number=page.number, width=page.width, height=page.height, text=page_text, blocks=blocks, text_length=text_length, density=density, has_text_layer=bool(page_text.strip())))
            return PdfDocument(
                document_id=document_id,
                source_file=pdf_path.name,
                path=str(pdf_path),
                sha256=sha256,
                file_size=pdf_path.stat().st_size,
                page_count=len(pages),
                extraction_method=fallback.extraction_method,
                pages=pages,
                warnings=[f'Fallback texte: {exc}'],
                needs_ocr=not any(page.has_text_layer for page in pages),
            )
        except Exception:
            raise
