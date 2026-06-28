from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from common.text import clean_text, normalize_for_match
from deepforma.training.cpf_dataset import load_jsonl


LOGGER = logging.getLogger(__name__)
DEFAULT_TEST = Path('data/training/cpf_test.jsonl')
DEFAULT_FORMATIONS = Path('data/processed/cpf/formations_with_skills.parquet')
DEFAULT_REPORT = Path('data/reports/cpf_training_metrics.json')
DEFAULT_ERRORS = Path('data/reports/cpf_training_errors.csv')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Évalue le recommender CPF sur plusieurs baselines')
    parser.add_argument('--test', type=Path, default=DEFAULT_TEST)
    parser.add_argument('--formations', type=Path, default=DEFAULT_FORMATIONS)
    parser.add_argument('--base-model', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--fine-tuned-model', type=Path, default=Path('models/cpf-recommender/final'))
    parser.add_argument('--output-metrics', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--output-errors', type=Path, default=DEFAULT_ERRORS)
    parser.add_argument('--limit', type=int, default=None)
    return parser


def _load_formations(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    rows = df.fillna('').to_dict(orient='records')
    return rows


def _candidate_text(row: dict[str, Any]) -> str:
    parts = [row.get('search_text') or row.get('title') or '', row.get('description') or '', row.get('objectives') or '', row.get('certification_label') or row.get('certification') or '']
    return '\n'.join(clean_text(part) for part in parts if clean_text(part))


def _skills_list(row: dict[str, Any]) -> list[str]:
    values = row.get('skills_normalized') or []
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = [part.strip() for part in values.split('|') if part.strip()]
    labels: list[str] = []
    for item in values:
        if isinstance(item, dict):
            label = item.get('canonical_label') or item.get('label')
        else:
            label = item
        label = clean_text(label)
        if label:
            labels.append(label)
    return list(dict.fromkeys(labels))


def _territory_match(query: dict[str, Any], candidate: dict[str, Any]) -> bool:
    q_dept = clean_text(query.get('department_code')) or None
    q_region = clean_text(query.get('region_code')) or None
    c_dept = clean_text(candidate.get('department_code')) or None
    c_region = clean_text(candidate.get('region_code')) or None
    if q_dept and c_dept and normalize_for_match(q_dept) == normalize_for_match(c_dept):
        return True
    if q_region and c_region and normalize_for_match(q_region) == normalize_for_match(c_region):
        return True
    return bool(candidate.get('remote') or candidate.get('distance_compatible'))


def _coverage(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    required = [_ for _ in (query.get('required_skills') or []) if _]
    if not required:
        return 0.0
    candidate_skills = {normalize_for_match(skill) for skill in _skills_list(candidate)}
    return len({normalize_for_match(skill) for skill in required} & candidate_skills) / len(required)


def _similarity_matrix(model: SentenceTransformer, queries: list[str], candidates: list[str]) -> np.ndarray:
    query_embeddings = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    candidate_embeddings = model.encode(candidates, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return cosine_similarity(query_embeddings, candidate_embeddings)


def _tfidf_matrix(queries: list[str], candidates: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(candidates + queries)
    candidate_matrix = matrix[: len(candidates)]
    query_matrix = matrix[len(candidates) :]
    return cosine_similarity(query_matrix, candidate_matrix)


def _metrics_from_ranks(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {'recall_at_1': 0.0, 'recall_at_5': 0.0, 'recall_at_10': 0.0, 'precision_at_5': 0.0, 'mrr': 0.0, 'ndcg_at_10': 0.0}
    total = len(ranks)
    return {
        'recall_at_1': round(sum(rank <= 1 for rank in ranks) / total, 4),
        'recall_at_5': round(sum(rank <= 5 for rank in ranks) / total, 4),
        'recall_at_10': round(sum(rank <= 10 for rank in ranks) / total, 4),
        'precision_at_5': round(sum(rank <= 5 for rank in ranks) / (total * 5), 4),
        'mrr': round(sum(1.0 / rank for rank in ranks) / total, 4),
        'ndcg_at_10': round(sum((1.0 / np.log2(rank + 1)) if rank <= 10 else 0.0 for rank in ranks) / total, 4),
    }


def evaluate_model(name: str, similarities: np.ndarray, queries: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranks: list[int] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    errors: list[dict[str, Any]] = []
    grouped: dict[str, list[int]] = {}
    candidate_ids = [str(row.get('formation_uid')) for row in candidates]
    for idx, query in enumerate(queries):
        row_scores = list(zip(candidate_ids, similarities[idx].tolist()))
        row_scores.sort(key=lambda item: item[1], reverse=True)
        positive_uid = str(query['positive_uid'])
        positive_rank = next((rank for rank, (uid, _) in enumerate(row_scores, start=1) if uid == positive_uid), len(row_scores) + 1)
        ranks.append(positive_rank)
        grouped.setdefault(clean_text(query.get('region_code')) or 'unknown', []).append(positive_rank)
        positive_index = candidate_ids.index(positive_uid)
        positive_scores.append(float(similarities[idx][positive_index]))
        negative_uid = str(query.get('negative_uid') or '')
        if negative_uid in candidate_ids:
            negative_scores.append(float(similarities[idx][candidate_ids.index(negative_uid)]))
        if positive_rank > 10:
            top_candidates = [uid for uid, _ in row_scores[:5]]
            covered = _coverage(query, candidates[positive_index])
            errors.append(
                {
                    'query': query.get('query'),
                    'expected_uid': positive_uid,
                    'expected_title': candidates[positive_index].get('title'),
                    'proposed_uids': ' | '.join(top_candidates),
                    'covered_skills': ' | '.join(query.get('required_skills') or []),
                    'missing_skills': ' | '.join(query.get('missing_skills') or []),
                    'territory': f"{query.get('region_code') or ''}/{query.get('department_code') or ''}",
                    'score': round(float(row_scores[0][1]) if row_scores else 0.0, 4),
                    'positive_rank': positive_rank,
                    'coverage': round(float(covered), 4),
                }
            )
    metrics = _metrics_from_ranks(ranks)
    metrics.update(
        {
            'validation_examples': len(queries),
            'mean_positive_similarity': round(float(np.mean(positive_scores)) if positive_scores else 0.0, 4),
            'mean_negative_similarity': round(float(np.mean(negative_scores)) if negative_scores else 0.0, 4),
            'by_region': {region: _metrics_from_ranks(region_ranks) for region, region_ranks in grouped.items()},
        }
    )
    return {'name': name, 'metrics': metrics}, errors


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    queries = load_jsonl(args.test)
    if args.limit is not None:
        queries = queries[: args.limit]
    candidates = _load_formations(args.formations)
    candidate_texts = [_candidate_text(row) for row in candidates]
    query_texts = [clean_text(row.get('query')) for row in queries]

    base_model = SentenceTransformer(args.base_model)
    tuned_model = SentenceTransformer(str(args.fine_tuned_model)) if args.fine_tuned_model.exists() else base_model

    base_sim = _similarity_matrix(base_model, query_texts, candidate_texts)
    tuned_sim = _similarity_matrix(tuned_model, query_texts, candidate_texts)
    tfidf_sim = _tfidf_matrix(query_texts, candidate_texts)

    base_result, base_errors = evaluate_model('base_model', base_sim, queries, candidates)
    tuned_result, tuned_errors = evaluate_model('fine_tuned_model', tuned_sim, queries, candidates)
    tfidf_result, tfidf_errors = evaluate_model('tfidf', tfidf_sim, queries, candidates)

    report = {
        'models': [base_result, tuned_result, tfidf_result],
        'summary': {
            'base_model': base_result['metrics'],
            'fine_tuned_model': tuned_result['metrics'],
            'tfidf': tfidf_result['metrics'],
        },
    }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    error_rows = base_errors + tuned_errors + tfidf_errors
    with args.output_errors.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['query', 'expected_uid', 'expected_title', 'proposed_uids', 'covered_skills', 'missing_skills', 'territory', 'score', 'positive_rank', 'coverage'])
        writer.writeheader()
        for row in error_rows:
            writer.writerow(row)
    LOGGER.info('Rapport métriques écrit dans %s', args.output_metrics)
    LOGGER.info('Rapport erreurs écrit dans %s', args.output_errors)


if __name__ == '__main__':
    main()
