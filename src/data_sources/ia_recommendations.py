from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.text import clean_text, normalize_for_match, stable_hash


@dataclass
class IARecommendationQualityReport:
    total_lines: int = 0
    valid_lines: int = 0
    rejected_lines: int = 0
    empty_keyword: int = 0
    empty_recommendation: int = 0
    exact_duplicates: int = 0
    normalized_duplicates: int = 0
    ambiguous_lines: int = 0
    default_rules: int = 0
    rejected_samples: list[dict[str, Any]] = field(default_factory=list)
    duplicate_groups: list[list[str]] = field(default_factory=list)


def normalize_recommendation_keyword(text: str) -> str:
    if not text:
        return ''
    result = text.lower().strip()
    result = unicodedata.normalize('NFKD', result)
    result = ''.join(ch for ch in result if not unicodedata.combining(ch))
    result = result.replace("'", " ")
    result = re.sub(r'[^a-z0-9\s]', ' ', result)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def robust_parse_line(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s:
        return None
    s = re.sub(r';+$', '', s)
    if not s:
        return None
    if s.startswith('"') and s.endswith('"'):
        inner = s[1:-1]
        inner = inner.replace('""', '\x00')
        sep = '","'
        sep_idx = inner.find(sep)
        if sep_idx == -1:
            idx = inner.index(',')
            kw = inner[:idx].strip()
            rec = inner[idx + 1:].strip()
        else:
            kw = inner[:sep_idx].strip()
            rec = inner[sep_idx + len(sep):].strip()
        kw = kw.replace('\x00', '"')
        rec = rec.replace('\x00', '"')
        return kw, rec
    else:
        idx = s.index(',')
        kw = s[:idx].strip()
        rec = s[idx + 1:].strip()
        return kw, rec


def load_ia_recommendations_csv(path: Path) -> tuple[list[dict[str, Any]], IARecommendationQualityReport]:
    raw = path.read_bytes()
    text = raw.decode('utf-8-sig')
    lines = text.splitlines()
    report = IARecommendationQualityReport(total_lines=len(lines))
    records: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    source_file = str(path)

    for i, line in enumerate(lines):
        if i == 0:
            continue
        report.total_lines = i + 1
        try:
            result = robust_parse_line(line)
        except Exception:
            report.rejected_lines += 1
            report.ambiguous_lines += 1
            report.rejected_samples.append({'line': i + 1, 'reason': 'parse_error', 'preview': line[:100]})
            continue
        if result is None:
            report.rejected_lines += 1
            report.empty_keyword += 1
            report.rejected_samples.append({'line': i + 1, 'reason': 'empty_line', 'preview': line[:50]})
            continue
        kw, rec = result
        if not kw:
            report.rejected_lines += 1
            report.empty_keyword += 1
            report.rejected_samples.append({'line': i + 1, 'reason': 'empty_keyword', 'preview': line[:80]})
            continue
        if not rec:
            report.rejected_lines += 1
            report.empty_recommendation += 1
            report.rejected_samples.append({'line': i + 1, 'reason': 'empty_recommendation', 'preview': line[:80]})
            continue
        kw_normalized = normalize_recommendation_keyword(kw)
        if not kw_normalized:
            report.rejected_lines += 1
            report.empty_keyword += 1
            report.rejected_samples.append({'line': i + 1, 'reason': 'keyword_empty_after_normalization', 'preview': kw})
            continue
        exact_key = kw.lower().strip()
        if exact_key in seen_exact:
            report.exact_duplicates += 1
            report.duplicate_groups.append([kw])
            continue
        seen_exact.add(exact_key)
        if kw_normalized in seen_normalized:
            report.normalized_duplicates += 1
            report.duplicate_groups.append([kw])
            continue
        seen_normalized.add(kw_normalized)
        is_default = 'aucune' in kw.lower() or 'defaut' in kw.lower()
        if is_default:
            report.default_rules += 1
        record = {
            'recommendation_id': stable_hash('ia_rec', kw, rec, length=24),
            'keyword': kw,
            'keyword_normalized': kw_normalized,
            'recommendation': rec,
            'source_file': source_file,
            'is_default': is_default,
            'is_active': True,
            'category': None,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        records.append(record)
        report.valid_lines += 1

    return records, report


def validate_ia_recommendations(rows: list[dict[str, Any]]) -> IARecommendationQualityReport:
    report = IARecommendationQualityReport(total_lines=len(rows))
    seen_kw: set[str] = set()
    for row in rows:
        kw = row.get('keyword_normalized', '')
        if not kw:
            report.empty_keyword += 1
            continue
        if kw in seen_kw:
            report.normalized_duplicates += 1
            continue
        seen_kw.add(kw)
        if not row.get('recommendation'):
            report.empty_recommendation += 1
            continue
        if row.get('is_default'):
            report.default_rules += 1
        report.valid_lines += 1
    return report
