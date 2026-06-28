from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from common.text import clean_text, normalize_for_match, stable_hash
from deepforma.training.cpf_dataset import (
    CPFTrainingExample,
    build_group_id,
    canonicalize_formation_row,
    ensure_list,
    load_jsonl,
    save_jsonl,
    split_by_group,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_FORMATIONS = Path('data/processed/cpf/formations_with_skills.parquet')
DEFAULT_TRAIN_DIR = Path('data/training')
DEFAULT_REVIEW = DEFAULT_TRAIN_DIR / 'cpf_pairs_review.csv'
DEFAULT_SPLITS = {
    'train': DEFAULT_TRAIN_DIR / 'cpf_train.jsonl',
    'validation': DEFAULT_TRAIN_DIR / 'cpf_validation.jsonl',
    'test': DEFAULT_TRAIN_DIR / 'cpf_test.jsonl',
}


class PairBuildError(RuntimeError):
    pass


def _available_processed_files() -> list[str]:
    processed_dir = Path('data/processed/cpf')
    if not processed_dir.exists():
        return []
    return [str(path.resolve()) for path in sorted(processed_dir.glob('*')) if path.is_file()]



def _missing_formations_error(path: Path, *, step: str) -> PairBuildError:
    available = _available_processed_files()
    return PairBuildError(
        f"Catalogue CPF enrichi introuvable: {path.resolve()}\n"
        f"Étape précédente à exécuter: {step}\n"
        f"Cible Make correspondante: cpf-enrich-skills\n"
        f"Fichiers présents dans data/processed/cpf: {available if available else ['(vide)']}"
    )


def _required_column_sets() -> list[set[str]]:
    return [
        {'formation_id', 'titre', 'texte_modele', 'competences_normalisees', 'source', 'source_version', 'record_type'},
        {'formation_uid', 'title', 'search_text', 'skills_normalized'},
    ]


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if clean_text(value):
                return value
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                return value
            continue
        size = getattr(value, 'size', None)
        if isinstance(size, int) and size == 0:
            continue
        try:
            if len(value) == 0:  # type: ignore[arg-type]
                continue
        except Exception:
            pass
        return value
    return None



def _adapt_formation_row(row: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(row)
    adapted.setdefault('formation_uid', adapted.get('formation_id'))
    adapted['title'] = clean_text(_first_present(adapted.get('title'), adapted.get('titre')))
    adapted['search_text'] = clean_text(_first_present(adapted.get('search_text'), adapted.get('texte_modele'), adapted.get('title')))
    adapted['description'] = clean_text(adapted.get('description'))
    adapted['objectives'] = clean_text(adapted.get('objectives'))
    adapted['certification_label'] = clean_text(_first_present(adapted.get('certification_label'), adapted.get('certification')))
    adapted['certification'] = clean_text(_first_present(adapted.get('certification'), adapted.get('certification_label')))
    adapted['certification_code'] = clean_text(_first_present(adapted.get('certification_code'), adapted.get('code_certification'), adapted.get('code_rncp'), adapted.get('code_rs')))
    adapted['level'] = clean_text(_first_present(adapted.get('level'), adapted.get('niveau')))
    adapted['nsf'] = clean_text(adapted.get('nsf'))
    adapted['organization'] = clean_text(_first_present(adapted.get('organization'), adapted.get('organisme')))
    adapted['siret'] = clean_text(adapted.get('siret'))
    adapted['region'] = clean_text(adapted.get('region'))
    adapted['region_code'] = clean_text(adapted.get('region_code'))
    adapted['department'] = clean_text(adapted.get('department'))
    adapted['department_code'] = clean_text(adapted.get('department_code'))
    adapted['referential_type'] = clean_text(adapted.get('referential_type'))
    adapted['distance_compatible'] = bool(adapted.get('distance_compatible')) if adapted.get('distance_compatible') not in (None, '') else False
    adapted['remote'] = bool(adapted.get('remote')) if adapted.get('remote') not in (None, '') else adapted['distance_compatible']
    adapted['skills_normalized'] = ensure_list(_first_present(adapted.get('skills_normalized'), adapted.get('competences_normalisees'), adapted.get('competences')))
    adapted['skills_explicit'] = ensure_list(_first_present(adapted.get('skills_explicit'), adapted.get('competences')))
    adapted['skills_inferred'] = ensure_list(adapted.get('skills_inferred'))
    adapted['skills_confidence'] = adapted.get('skills_confidence') or {}
    adapted['skills_evidence'] = adapted.get('skills_evidence') or {}
    adapted['group_id'] = _first_present(adapted.get('group_id'), adapted.get('formation_group_id')) or build_group_id(adapted)
    return adapted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construit les paires et triplets CPF pour l'entraînement")
    parser.add_argument('--formations', type=Path, default=DEFAULT_FORMATIONS)
    parser.add_argument('--offers-dir', type=Path, default=Path('data/france_travail/normalized'))
    parser.add_argument('--output', type=Path, default=DEFAULT_TRAIN_DIR / 'cpf_pairs.jsonl')
    parser.add_argument('--review-output', type=Path, default=DEFAULT_REVIEW)
    parser.add_argument('--train-output', type=Path, default=DEFAULT_SPLITS['train'])
    parser.add_argument('--validation-output', type=Path, default=DEFAULT_SPLITS['validation'])
    parser.add_argument('--test-output', type=Path, default=DEFAULT_SPLITS['test'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-queries', type=int, default=100)
    parser.add_argument('--min-skill-coverage', type=float, default=0.05)
    parser.add_argument('--min-semantic-similarity', type=float, default=0.2)
    parser.add_argument('--hard-negative-similarity', type=float, default=0.55)
    parser.add_argument('--territorial-margin', type=float, default=0.55)
    return parser


def _load_formations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise _missing_formations_error(path, step='make cpf-prepare puis make cpf-enrich-skills')
    if path.stat().st_size <= 0:
        raise PairBuildError(
            f"Le catalogue CPF enrichi est vide: {path.resolve()}\n"
            f"Étape précédente à exécuter: make cpf-enrich-skills CPF_SOURCE_FILE=data/raw/Dataset_Generaliste_CPF_V3.xlsx\n"
            f"Cible Make correspondante: cpf-enrich-skills\n"
            f"Fichiers présents dans data/processed/cpf: {_available_processed_files() or ['(vide)']}"
        )
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise PairBuildError(
            f"Impossible de lire le catalogue CPF enrichi: {path.resolve()}\n"
            f"Étape précédente à exécuter: make cpf-enrich-skills CPF_SOURCE_FILE=data/raw/Dataset_Generaliste_CPF_V3.xlsx\n"
            f"Cible Make correspondante: cpf-enrich-skills\n"
            f"Fichiers présents dans data/processed/cpf: {_available_processed_files() or ['(vide)']}"
        ) from exc
    if df.empty:
        raise PairBuildError(
            f"Le catalogue CPF enrichi est vide: {path.resolve()}\n"
            f"Étape précédente à exécuter: make cpf-enrich-skills CPF_SOURCE_FILE=data/raw/Dataset_Generaliste_CPF_V3.xlsx\n"
            f"Cible Make correspondante: cpf-enrich-skills\n"
            f"Fichiers présents dans data/processed/cpf: {_available_processed_files() or ['(vide)']}"
        )
    columns = set(str(col) for col in df.columns)
    if not any(required.issubset(columns) for required in _required_column_sets()):
        raise PairBuildError(
            f"Colonnes CPF enrichies insuffisantes dans {path.resolve()}: {sorted(columns)}\n"
            f"Étape précédente à exécuter: make cpf-enrich-skills CPF_SOURCE_FILE=data/raw/Dataset_Generaliste_CPF_V3.xlsx\n"
            f"Cible Make correspondante: cpf-enrich-skills\n"
            f"Fichiers présents dans data/processed/cpf: {_available_processed_files() or ['(vide)']}"
        )
    rows = [_adapt_formation_row(row) for row in df.fillna('').to_dict(orient='records')]
    rows = [canonicalize_formation_row(row) for row in rows]
    if not rows:
        raise PairBuildError(
            f"Aucune formation exploitable dans {path.resolve()}\n"
            f"Étape précédente à exécuter: make cpf-enrich-skills CPF_SOURCE_FILE=data/raw/Dataset_Generaliste_CPF_V3.xlsx\n"
            f"Cible Make correspondante: cpf-enrich-skills\n"
            f"Fichiers présents dans data/processed/cpf: {_available_processed_files() or ['(vide)']}"
        )
    return rows


def _load_offers(offers_dir: Path) -> list[dict[str, Any]]:
    offers: list[dict[str, Any]] = []
    if not offers_dir.exists():
        return offers
    for path in sorted(offers_dir.rglob('*.jsonl')):
        offers.extend(load_jsonl(path))
    return offers


def _extract_offer_profile(offer: dict[str, Any]) -> dict[str, Any]:
    skills = []
    for key in ('normalized_skills', 'skills_normalized', 'merged_skills', 'skills'):
        values = ensure_list(offer.get(key))
        if values:
            skills = values
            break
    normalized_skills = []
    for item in skills:
        if isinstance(item, dict):
            label = item.get('canonical_label') or item.get('label') or item.get('name')
        else:
            label = item
        label = clean_text(label)
        if label:
            normalized_skills.append(label)
    title = clean_text(offer.get('title') or offer.get('intitule'))
    return {
        'target_job': title or clean_text(offer.get('rome_label') or offer.get('job_title') or ''),
        'required_skills': list(dict.fromkeys(normalized_skills)),
        'missing_skills': list(dict.fromkeys(normalized_skills)),
        'region_code': clean_text(offer.get('region_code') or offer.get('codeRegion')) or None,
        'department_code': clean_text(offer.get('department_code') or offer.get('departement')) or None,
        'source': 'france_travail',
        'search_text': clean_text(offer.get('search_text') or offer.get('description') or offer.get('title') or ''),
        'territorial': offer,
    }


def _make_formation_profile(row: dict[str, Any]) -> dict[str, Any]:
    skills = [item.get('canonical_label') if isinstance(item, dict) else item for item in ensure_list(row.get('skills_normalized'))]
    skills = [clean_text(skill) for skill in skills if clean_text(skill)]
    title = clean_text(row.get('title'))
    return {
        'target_job': title,
        'required_skills': list(dict.fromkeys(skills)),
        'missing_skills': list(dict.fromkeys(skills)),
        'region_code': clean_text(row.get('region_code')) or None,
        'department_code': clean_text(row.get('department_code')) or None,
        'source': 'cpf_formation',
        'search_text': clean_text(row.get('search_text') or ''),
        'territorial': row,
    }


def _skill_overlap(required: list[str], candidate: list[str]) -> float:
    if not required:
        return 0.0
    req = {normalize_for_match(item) for item in required if item}
    cand = {normalize_for_match(item) for item in candidate if item}
    if not req:
        return 0.0
    return len(req & cand) / len(req)


def _territory_compatible(candidate: dict[str, Any], profile: dict[str, Any]) -> bool:
    if candidate.get('remote') or candidate.get('distance_compatible'):
        return True
    dept = clean_text(candidate.get('department_code')) or None
    region = clean_text(candidate.get('region_code')) or None
    profile_dept = profile.get('department_code')
    profile_region = profile.get('region_code')
    if profile_dept and dept and normalize_for_match(profile_dept) == normalize_for_match(dept):
        return True
    if profile_region and region and normalize_for_match(profile_region) == normalize_for_match(region):
        return True
    return False


def _formation_score(profile: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, float, float, bool]:
    candidate_skills = [item.get('canonical_label') if isinstance(item.get('canonical_label'), str) else item for item in ensure_list(candidate.get('skills_normalized'))]
    candidate_skills = [clean_text(skill) for skill in candidate_skills if clean_text(skill)]
    overlap = _skill_overlap(profile['required_skills'], candidate_skills)
    semantic = SequenceMatcher(None, normalize_for_match(profile['target_job']), normalize_for_match(candidate['title'])).ratio()
    if not semantic:
        semantic = SequenceMatcher(None, normalize_for_match(profile['search_text']), normalize_for_match(candidate['search_text'])).ratio()
    territory_ok = _territory_compatible(candidate, profile)
    return overlap, semantic, 1.0 if territory_ok else 0.0, territory_ok


def _choose_positive(profile: dict[str, Any], formations: list[dict[str, Any]], *, min_skill_coverage: float, min_semantic_similarity: float) -> dict[str, Any] | None:
    scored: list[tuple[float, dict[str, Any], float, float, bool]] = []
    for candidate in formations:
        overlap, semantic, territory_ratio, territory_ok = _formation_score(profile, candidate)
        text_length = len(clean_text(candidate.get('search_text')))
        if overlap < min_skill_coverage:
            continue
        if semantic < min_semantic_similarity:
            continue
        if text_length < 40:
            continue
        if not territory_ok and not candidate.get('online_mode') and not candidate.get('remote'):
            continue
        score = (overlap * 0.55) + (semantic * 0.3) + (territory_ratio * 0.15)
        scored.append((score, candidate, overlap, semantic, territory_ok))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _pick_easy_negative(profile: dict[str, Any], formations: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    candidates = []
    for candidate in formations:
        overlap, semantic, territory_ratio, territory_ok = _formation_score(profile, candidate)
        if overlap > 0.15 or semantic > 0.35:
            continue
        candidates.append((overlap + semantic, candidate))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return rng.choice(formations)


def _pick_hard_negative(profile: dict[str, Any], formations: list[dict[str, Any]], positive_uid: str) -> dict[str, Any]:
    candidates = []
    for candidate in formations:
        if candidate['formation_uid'] == positive_uid:
            continue
        overlap, semantic, territory_ratio, territory_ok = _formation_score(profile, candidate)
        title_sim = SequenceMatcher(None, normalize_for_match(profile['target_job']), normalize_for_match(candidate['title'])).ratio()
        cert_sim = SequenceMatcher(None, normalize_for_match(profile['territorial'].get('certification_label') or profile['territorial'].get('certification_code') or ''), normalize_for_match(candidate.get('certification_label') or candidate.get('certification_code') or '')).ratio()
        if max(title_sim, cert_sim) < 0.45:
            continue
        if overlap >= 0.35:
            continue
        candidates.append((max(title_sim, cert_sim), candidate))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return _pick_easy_negative(profile, formations, random.Random(0))


def _pick_territorial_negative(profile: dict[str, Any], formations: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for candidate in formations:
        overlap, semantic, territory_ratio, territory_ok = _formation_score(profile, candidate)
        if overlap < 0.35:
            continue
        if territory_ok:
            continue
        candidates.append((overlap + semantic, candidate))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return formations[0]


def _build_query_text(profile: dict[str, Any]) -> str:
    parts = [profile['target_job']]
    parts.extend(profile['required_skills'][:5])
    return ' | '.join(part for part in parts if part)


def _build_example(profile: dict[str, Any], positive: dict[str, Any], negative: dict[str, Any], negative_type: str, label_source: str, label_confidence: float) -> CPFTrainingExample:
    required = list(dict.fromkeys(profile['required_skills']))
    positive_skills = [item.get('canonical_label') if isinstance(item, dict) else item for item in ensure_list(positive.get('skills_normalized'))]
    positive_skills = [clean_text(skill) for skill in positive_skills if clean_text(skill)]
    covered = [skill for skill in required if normalize_for_match(skill) in {normalize_for_match(item) for item in positive_skills}]
    missing = [skill for skill in profile['missing_skills'] if normalize_for_match(skill) not in {normalize_for_match(item) for item in positive_skills}]
    query = _build_query_text(profile)
    query_id = stable_hash(query, positive['formation_uid'], negative['formation_uid'], negative_type)
    return CPFTrainingExample(
        query_id=query_id,
        query=query,
        target_job=profile['target_job'],
        required_skills=required,
        missing_skills=missing,
        region_code=clean_text(profile.get('region_code')) or None,
        department_code=clean_text(profile.get('department_code')) or None,
        positive_uid=positive['formation_uid'],
        positive_text=positive['search_text'],
        negative_uid=negative['formation_uid'],
        negative_text=negative['search_text'],
        negative_type=negative_type,
        label_source=label_source,
        label_confidence=round(label_confidence, 4),
        group_id=positive['group_id'],
        certification_code=positive.get('certification_code'),
        certification_label=positive.get('certification_label'),
        referential_type=positive.get('referential_type'),
        level=positive.get('level'),
        nsf=positive.get('nsf'),
        organization=positive.get('organization'),
        siret=positive.get('siret'),
        region=positive.get('region'),
        department=positive.get('department'),
    )


def _positive_confidence(profile: dict[str, Any], positive: dict[str, Any]) -> float:
    overlap = _skill_overlap(profile['required_skills'], [item.get('canonical_label') if isinstance(item, dict) else item for item in ensure_list(positive.get('skills_normalized'))])
    semantic = SequenceMatcher(None, normalize_for_match(profile['target_job']), normalize_for_match(positive['title'])).ratio()
    territory = 1.0 if _territory_compatible(positive, profile) else 0.0
    return min(0.99, round(overlap * 0.5 + semantic * 0.35 + territory * 0.15, 4))


def generate_pairs(formations_path: Path, offers_dir: Path, *, seed: int = 42, max_queries: int = 100, min_skill_coverage: float = 0.05, min_semantic_similarity: float = 0.2) -> list[dict[str, Any]]:
    formations = _load_formations(formations_path)
    offers = [_extract_offer_profile(offer) for offer in _load_offers(offers_dir)]
    rng = random.Random(seed)
    profiles = [_make_formation_profile(row) for row in formations] + offers
    rng.shuffle(profiles)
    rows: list[dict[str, Any]] = []
    for profile in profiles[:max_queries]:
        positive = _choose_positive(profile, formations, min_skill_coverage=min_skill_coverage, min_semantic_similarity=min_semantic_similarity)
        if not positive:
            continue
        neg_easy = _pick_easy_negative(profile, formations, rng)
        neg_hard = _pick_hard_negative(profile, formations, positive['formation_uid'])
        neg_territorial = _pick_territorial_negative(profile, formations)
        for negative, negative_type, confidence in [
            (neg_easy, 'easy', 0.72),
            (neg_hard, 'hard', 0.64),
            (neg_territorial, 'territorial', 0.68),
        ]:
            example = _build_example(profile, positive, negative, negative_type, 'heuristic', min(0.99, confidence * _positive_confidence(profile, positive)))
            rows.append(asdict(example))
    return rows


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ['reviewer_label', 'reviewer_comment', 'validated_at'] if rows else ['reviewer_label', 'reviewer_comment', 'validated_at']
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload.update({'reviewer_label': '', 'reviewer_comment': '', 'validated_at': ''})
            writer.writerow(payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    rows = generate_pairs(
        args.formations,
        args.offers_dir,
        seed=args.seed,
        max_queries=args.max_queries,
        min_skill_coverage=args.min_skill_coverage,
        min_semantic_similarity=args.min_semantic_similarity,
    )
    if not rows:
        raise PairBuildError("Aucune paire CPF n'a pu être générée.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output, rows)
    write_review_csv(args.review_output, rows)
    splits = split_by_group(rows, seed=args.seed)
    save_jsonl(args.train_output, splits['train'])
    save_jsonl(args.validation_output, splits['validation'])
    save_jsonl(args.test_output, splits['test'])
    LOGGER.info('Paires générées: %s', len(rows))
    LOGGER.info('Répartition: train=%s validation=%s test=%s', len(splits['train']), len(splits['validation']), len(splits['test']))


if __name__ == '__main__':
    main()
