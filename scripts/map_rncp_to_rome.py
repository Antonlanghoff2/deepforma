#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.text import clean_text
from referentials.france_competences import FranceCompetenceCertification, FranceCompetencesOpenDataImporter
from referentials.rncp_rome_mapper import RNCPRomeMapper
from referentials.rome_referential import RomeJob, RomeReferentialImporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Mappe les certifications RNCP/RS vers le ROME.')
    parser.add_argument('--rncp-path', type=Path, default=Path('data/raw/france_competences'))
    parser.add_argument('--rome-path', type=Path, default=Path('data/raw/rome'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/referentials/mappings'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--write', action='store_true')
    return parser


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + ('\n' if rows else ''), encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cert_from_dict(row: dict[str, Any]) -> FranceCompetenceCertification:
    return FranceCompetenceCertification(
        rncp_code=clean_text(row.get('rncp_code')),
        type=clean_text(row.get('type') or 'RNCP'),
        title=clean_text(row.get('title')),
        status=clean_text(row.get('status') or 'active'),
        level=row.get('level'),
        valid_until=clean_text(row.get('valid_until')) or None,
        activities=[clean_text(item) for item in row.get('activities', []) if clean_text(item)],
        target_jobs=[clean_text(item) for item in row.get('target_jobs', []) if clean_text(item)],
        sectors=[clean_text(item) for item in row.get('sectors', []) if clean_text(item)],
        block_ids=[clean_text(item) for item in row.get('block_ids', []) if clean_text(item)],
        source_url=clean_text(row.get('source_url')) or None,
        source_updated_at=clean_text(row.get('source_updated_at')) or None,
    )


def _job_from_dict(row: dict[str, Any]) -> RomeJob:
    return RomeJob(
        rome_code=clean_text(row.get('rome_code')),
        label=clean_text(row.get('label')),
        definition=clean_text(row.get('definition') or ''),
        alternative_titles=[clean_text(item) for item in row.get('alternative_titles', []) if clean_text(item)],
        activity_ids=[clean_text(item) for item in row.get('activity_ids', []) if clean_text(item)],
        skill_ids=[clean_text(item) for item in row.get('skill_ids', []) if clean_text(item)],
    )


def _labels_from_skill_ids(skill_ids: list[str], skills: dict[str, dict[str, Any]], key: str = 'official_label') -> list[str]:
    labels: list[str] = []
    for skill_id in skill_ids:
        skill = skills.get(skill_id)
        if skill and skill.get(key):
            labels.append(str(skill[key]))
    return labels


def main() -> None:
    args = build_parser().parse_args()
    rncp = FranceCompetencesOpenDataImporter(args.rncp_path).load(active_only=True)
    rome = RomeReferentialImporter(args.rome_path).load()
    mapper = RNCPRomeMapper()

    rncp_skills = {item['skill_id']: item for item in rncp.get('skills', []) if item.get('skill_id')}
    rome_skills = {item['rome_skill_id']: item for item in rome.get('skills', []) if item.get('rome_skill_id')}
    rome_jobs = [item for item in rome.get('jobs', []) if item.get('rome_code')]
    certifications = [item for item in rncp.get('certifications', []) if item.get('rncp_code')]

    matches: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for cert_row in certifications:
        cert = _cert_from_dict(cert_row)
        cert_skill_labels = _labels_from_skill_ids([item['skill_id'] for item in rncp.get('skills', []) if item.get('rncp_code') == cert.rncp_code], rncp_skills)
        for job_row in rome_jobs:
            job = _job_from_dict(job_row)
            rome_skill_labels = _labels_from_skill_ids(job.skill_ids, rome_skills)
            match = mapper.score(cert, job, cert_skill_labels=cert_skill_labels, rome_skill_labels=rome_skill_labels)
            if match.score >= mapper.review_threshold:
                row = match.to_dict()
                matches.append(row)
                if not row['validated']:
                    review_rows.append(row)

    print(json.dumps({'matches': len(matches), 'review': len(review_rows), 'sample': matches[:3]}, ensure_ascii=False, indent=2))
    if args.dry_run or not args.write:
        return
    out = args.output_dir
    _write_jsonl(out / 'rncp_rome_links.jsonl', matches)
    _write_csv(out / 'rncp_rome_review.csv', review_rows)


if __name__ == '__main__':
    main()
