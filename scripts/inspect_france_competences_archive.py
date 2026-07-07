#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import zipfile
from io import BytesIO
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from charset_normalizer import from_bytes

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from data_sources.france_competences.schema_adapter import FranceCompetencesSchemaAdapter


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Inspecte une archive France Compétences RNCP/RS.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def _detect_encoding(payload: bytes) -> str:
    try:
        best = from_bytes(payload).best()
        if best and best.encoding:
            return best.encoding
    except Exception:
        pass
    return 'utf-8'


def _detect_separator(text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=[',', ';', '\t', '|'])
        return dialect.delimiter
    except Exception:
        counts = {sep: text.count(sep) for sep in [',', ';', '\t', '|']}
        return max(counts, key=counts.get) if any(counts.values()) else ','


def _preview_csv(payload: bytes) -> dict[str, object]:
    encoding = _detect_encoding(payload)
    text = payload.decode(encoding, errors='ignore')
    separator = _detect_separator(text[:20000])
    rows = list(csv.reader(text.splitlines(), delimiter=separator))
    header = rows[0] if rows else []
    preview = rows[1:4] if len(rows) > 1 else []
    return {
        'encoding': encoding,
        'separator': separator,
        'columns': header,
        'preview_rows': preview,
    }


def _preview_xml(payload: bytes) -> dict[str, object]:
    try:
        root = ET.fromstring(payload)
        children = [child.tag for child in list(root)[:20]]
        return {'root': root.tag, 'children': children}
    except Exception as exc:
        return {'error': str(exc)}


def inspect_archive(path: Path) -> dict[str, object]:
    report: dict[str, object] = {'archive': str(path), 'entries': []}
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        candidates = [item for item in sorted(path.rglob('*')) if item.is_file() and item.suffix.lower() in {'.zip', '.csv', '.xml', '.xsd'}]
        report['archive'] = str(path)
        for item in candidates:
            entry = {'name': str(item.relative_to(path)), 'format': item.suffix.lower().lstrip('.'), 'size': item.stat().st_size}
            if item.suffix.lower() == '.zip':
                with zipfile.ZipFile(item) as zf:
                    entry['entries'] = []
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        nested: dict[str, object] = {
                            'name': info.filename,
                            'format': Path(info.filename).suffix.lower().lstrip('.'),
                            'size': info.file_size,
                        }
                        with zf.open(info) as fh:
                            payload = fh.read(128 * 1024)
                        if info.filename.lower().endswith('.csv'):
                            nested.update(_preview_csv(payload))
                        elif info.filename.lower().endswith('.xml'):
                            nested.update(_preview_xml(payload))
                        elif info.filename.lower().endswith('.xsd'):
                            nested['preview'] = payload.decode('utf-8', errors='ignore')[:2000]
                        entry['entries'].append(nested)
            else:
                payload = item.read_bytes()[:128 * 1024]
                if item.suffix.lower() == '.csv':
                    entry.update(_preview_csv(payload))
                elif item.suffix.lower() == '.xml':
                    entry.update(_preview_xml(payload))
                elif item.suffix.lower() == '.xsd':
                    entry['preview'] = payload.decode('utf-8', errors='ignore')[:2000]
            report['entries'].append(entry)
        return report

    if path.suffix.lower() != '.zip':
        payload = path.read_bytes()[:128 * 1024]
        entry: dict[str, object] = {'name': path.name, 'format': path.suffix.lower().lstrip('.'), 'size': path.stat().st_size}
        if path.suffix.lower() == '.csv':
            entry.update(_preview_csv(payload))
        elif path.suffix.lower() == '.xml':
            entry.update(_preview_xml(payload))
        report['entries'] = [entry]
        return report
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            entry: dict[str, object] = {
                'name': info.filename,
                'format': Path(info.filename).suffix.lower().lstrip('.'),
                'size': info.file_size,
            }
            with zf.open(info) as fh:
                payload = fh.read(128 * 1024)
            if info.filename.lower().endswith('.csv'):
                entry.update(_preview_csv(payload))
            elif info.filename.lower().endswith('.xml'):
                entry.update(_preview_xml(payload))
            elif info.filename.lower().endswith('.xsd'):
                entry['preview'] = payload.decode('utf-8', errors='ignore')[:2000]
            report['entries'].append(entry)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    report = inspect_archive(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    LOGGER.info('Rapport écrit: %s', args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

