from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match, stable_hash

from .category_mapper import map_rule_categories
from .models import AIRecommendationImportReport, AIRecommendationRule, AIRecommendationRuleCategory
from .normalizer import normalize_ai_keyword

LOGGER = logging.getLogger(__name__)

SOURCE_FILENAME = 'dataset_recommandations_IA_complet.csv'
DEFAULT_RULE_TEXT = "Acculturation à l'IA et découverte de l'IA agentique"
DEFAULT_KEYWORD = '(aucune mention IA dans le référentiel)'

VERB_HINTS = (
    'utiliser', 'automatiser', 'générer', 'creer', 'créer', 'decouvrir', 'découvrir',
    'construire', 'faire', 'proposer', 'soutenir', 'analyser', 'adapter', 'mettre', 'concevoir',
)
TOOL_HINTS = (
    'chatgpt', 'claude', 'perplexity', 'feedly', 'copilot', 'gemini', 'canva', 'firefly',
    'gpt', 'llm', 'notebooklm', 'hubspot ai', 'zoho zia', 'genspark', 'midjourney',
)


def _strip_outer_quotes(text: str) -> str:
    value = text.strip()
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1].strip()
    return value


def _looks_short_recommendation(text: str) -> bool:
    value = clean_text(text)
    return len(value) < 24 or len(value.split()) < 4


def _keyword_anomalies(keyword: str, recommendation: str) -> list[str]:
    lowered = normalize_for_match(keyword)
    raw_lower = clean_text(keyword).lower()
    anomalies: list[str] = []
    if ',' in keyword:
        anomalies.append('keyword_contains_comma')
    if len(clean_text(keyword)) > 80:
        anomalies.append('keyword_too_long')
    if any(verb in lowered for verb in VERB_HINTS) or any(verb in raw_lower for verb in VERB_HINTS):
        anomalies.append('keyword_contains_verb')
    if '"' in keyword or '«' in keyword or '»' in keyword or '“' in keyword or '”' in keyword:
        anomalies.append('keyword_contains_quotes')
    if any(tool in lowered for tool in TOOL_HINTS):
        anomalies.append('keyword_contains_tool_name')
    if _looks_short_recommendation(recommendation):
        anomalies.append('recommendation_too_short')
    return anomalies


def _repair_misaligned_fields(keyword: str, recommendation: str) -> tuple[str, str, list[str], float, str]:
    anomalies = _keyword_anomalies(keyword, recommendation)
    kw = _strip_outer_quotes(clean_text(keyword))
    rec = _strip_outer_quotes(clean_text(recommendation))
    confidence = 0.95
    anomaly_type = 'none'

    if ',' in kw:
        head, tail = [part.strip() for part in kw.split(',', 1)]
        if head and tail:
            kw = head
            rec = ', '.join([tail, rec]).strip(', ').strip()
            anomalies.append('misaligned_keyword_fragment')
            anomaly_type = 'misaligned_keyword_fragment'
            confidence = 0.94

    if kw and any(verb in normalize_for_match(kw) for verb in VERB_HINTS):
        anomalies.append('keyword_contains_verb')
        anomaly_type = anomaly_type if anomaly_type != 'none' else 'keyword_contains_verb'
        confidence = min(confidence, 0.78)

    if len(clean_text(kw)) > 80:
        anomaly_type = anomaly_type if anomaly_type != 'none' else 'keyword_too_long'
        confidence = min(confidence, 0.6)

    if _looks_short_recommendation(rec):
        anomaly_type = anomaly_type if anomaly_type != 'none' else 'recommendation_too_short'
        confidence = min(confidence, 0.72)

    if '"' in kw or '"' in rec:
        anomaly_type = anomaly_type if anomaly_type != 'none' else 'keyword_contains_quotes'
        confidence = min(confidence, 0.7)

    if anomaly_type == 'none' and anomalies:
        anomaly_type = anomalies[0]
        confidence = 0.9 if anomaly_type == 'none' else 0.85

    return kw, rec, anomalies, round(confidence, 4), anomaly_type


def detect_ai_recommendation_anomalies(keyword: str, recommendation: str) -> list[str]:
    return _keyword_anomalies(keyword, recommendation)


def load_ai_recommendation_rules_csv(path: str | Path) -> tuple[list[dict[str, Any]], AIRecommendationImportReport, list[dict[str, Any]]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    report = AIRecommendationImportReport()
    seen_norm: set[str] = set()
    anomaly_counter: Counter[str] = Counter()
    if not source.exists():
        raise FileNotFoundError(source)

    with source.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError('En-têtes CSV invalides: 2 colonnes attendues.')
        for line_no, row in enumerate(reader, start=2):
            report.total_lines = line_no
            if not row or all(not clean_text(cell) for cell in row):
                continue
            original_keyword = clean_text(row[0])
            original_recommendation = clean_text(row[1]) if len(row) > 1 else ''
            if len(row) > 2:
                original_recommendation = ', '.join([original_recommendation, *[clean_text(item) for item in row[2:] if clean_text(item)]])
            keyword, recommendation, anomalies, confidence, anomaly_type = _repair_misaligned_fields(original_keyword, original_recommendation)
            normalized_keyword = normalize_ai_keyword(keyword)
            is_default = normalize_for_match(keyword) == normalize_for_match(DEFAULT_KEYWORD)
            for anomaly in set(anomalies):
                anomaly_counter[anomaly] += 1
            if is_default:
                recommendation = DEFAULT_RULE_TEXT
                anomalies = ['default_rule']
                anomaly_type = 'default_rule'
                confidence = 1.0

            if not keyword or not recommendation:
                report.review_lines += 1
                review_rows.append({
                    'source_line': line_no,
                    'original_keyword': original_keyword,
                    'original_recommendation': original_recommendation,
                    'proposed_keyword': keyword,
                    'proposed_recommendation': recommendation,
                    'anomaly_type': 'empty_field',
                    'confidence': 0.0,
                    'review_status': 'to_review',
                    'reviewer_comment': 'Champ vide détecté.',
                })
                continue

            if normalized_keyword in seen_norm and normalized_keyword:
                report.duplicate_lines += 1
                review_rows.append({
                    'source_line': line_no,
                    'original_keyword': original_keyword,
                    'original_recommendation': original_recommendation,
                    'proposed_keyword': keyword,
                    'proposed_recommendation': recommendation,
                    'anomaly_type': 'duplicate_normalized_keyword',
                    'confidence': 0.95,
                    'review_status': 'rejected',
                    'reviewer_comment': 'Doublon normalisé.',
                })
                continue
            if normalized_keyword:
                seen_norm.add(normalized_keyword)

            categories = map_rule_categories(keyword, recommendation)
            rule = AIRecommendationRule(
                id=stable_hash('airule', keyword, recommendation, source.name, str(line_no), length=12),
                keyword=keyword,
                normalized_keyword=normalized_keyword,
                categories=categories,
                recommendation=recommendation,
                match_type='hybrid',
                priority=50,
                enabled=True,
                is_default=is_default,
                source=SOURCE_FILENAME,
                source_line=line_no,
                review_status='auto_accepted' if confidence >= 0.9 and anomaly_type in {'none', 'default_rule'} else 'to_review',
                metadata={
                    'anomalies': anomalies,
                    'original_keyword': original_keyword,
                    'original_recommendation': original_recommendation,
                    'mapping_method': 'hybrid',
                },
            )
            rows.append({**asdict(rule), 'categories': [asdict(category) for category in categories]})
            if rule.review_status == 'auto_accepted':
                report.imported_lines += 1
            else:
                report.review_lines += 1
                review_rows.append({
                    'source_line': line_no,
                    'original_keyword': original_keyword,
                    'original_recommendation': original_recommendation,
                    'proposed_keyword': keyword,
                    'proposed_recommendation': recommendation,
                    'anomaly_type': anomaly_type,
                    'confidence': confidence,
                    'review_status': rule.review_status,
                    'reviewer_comment': '; '.join(anomalies) if anomalies else '',
                })

    report.anomalies = dict(anomaly_counter)
    report.warnings = [f'{count} lignes avec {name}' for name, count in anomaly_counter.items()]
    return rows, report, review_rows


def load_ai_recommendation_rules(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(payload, dict):
        rules = payload.get('rules') or payload.get('items') or []
        if isinstance(rules, list):
            return [dict(item) for item in rules if isinstance(item, dict)]
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    raise ValueError(f'Format de règles IA invalide: {path}')


def write_ai_recommendation_outputs(
    rules: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    *,
    output_csv: str | Path,
    output_json: str | Path,
    review_output: str | Path,
) -> tuple[Path, Path, Path]:
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    review_path = Path(review_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(rules)
    if not df.empty:
        df = df.copy()
        df['categories'] = df['categories'].apply(json.dumps)
        df['metadata'] = df['metadata'].apply(json.dumps)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    json_path.write_text(json.dumps({'generated_at': datetime.now(timezone.utc).isoformat(), 'source': SOURCE_FILENAME, 'rules': rules}, ensure_ascii=False, indent=2), encoding='utf-8')
    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(review_path, index=False, encoding='utf-8-sig')
    return csv_path, json_path, review_path


def import_ai_recommendation_dataset(
    input_path: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    review_output: str | Path,
) -> dict[str, Any]:
    rules, report, review_rows = load_ai_recommendation_rules_csv(input_path)
    csv_path, json_path, review_path = write_ai_recommendation_outputs(
        rules,
        review_rows,
        output_csv=output_csv,
        output_json=output_json,
        review_output=review_output,
    )
    return {
        'rules_count': len(rules),
        'review_count': len(review_rows),
        'report': report,
        'csv_path': csv_path,
        'json_path': json_path,
        'review_path': review_path,
    }
