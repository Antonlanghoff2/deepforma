
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common.text import clean_text, normalize_for_match, stable_hash
from deepforma.cpf.columns import anonymize_value, detect_columns, load_column_aliases
from deepforma.skills.normalizer import SkillTaxonomyNormalizer


LOGGER = logging.getLogger(__name__)
DEFAULT_V3_SHEET = 'Generaliste_CPF'
DEFAULT_SOURCE_CANDIDATES = [
    Path('data/raw/Dataset_Generaliste_CPF_V3.xlsx'),
    Path('Dataset_Generaliste_CPF_V3.xlsx'),
    Path('data/raw/Dataset_Generaliste_CPF_V2.xlsx'),
    Path('Dataset_Generaliste_CPF_V2.xlsx'),
    Path('data/raw/Dataset_Generaliste_CPF_V1.xlsx'),
    Path('Dataset_Generaliste_CPF_V1.xlsx'),
]
V3_SOURCE_NAME = 'Dataset_Generaliste_CPF'
V3_SOURCE_VERSION = 'CPF_V3'
V3_RECORD_TYPE = 'formation'
V3_REQUIRED_COLUMNS = [
    '#',
    'Secteur',
    'Organisme de formation',
    'Intitulé de la formation',
    'Type de certification',
    'Code certification',
    'Niveau',
    'Codes ROME',
    'Compétences majeures',
    'Modalité',
    'Durée',
    'Prix TTC (€)',
    'Tags',
    '✅ Relu / Validé (oui/non)',
    '🗒 Corrections / Remarques',
]
V3_COLUMN_MAPPING = {
    'source_row_id': '#',
    'secteur': 'Secteur',
    'organisme': 'Organisme de formation',
    'titre': 'Intitulé de la formation',
    'type_certification': 'Type de certification',
    'code_certification': 'Code certification',
    'niveau': 'Niveau',
    'codes_rome': 'Codes ROME',
    'competences': 'Compétences majeures',
    'modalite': 'Modalité',
    'duree': 'Durée',
    'prix': 'Prix TTC (€)',
    'tags': 'Tags',
    'relu_valide': '✅ Relu / Validé (oui/non)',
    'corrections_remarques': '🗒 Corrections / Remarques',
}
MULTI_SPLIT_RE = re.compile(r'\s*\|\s*|\s*;\s*|\n+|•|·')


@dataclass(frozen=True)
class CPFSourceInspection:
    path: str
    resolved_path: str
    sheet_names: list[str]
    selected_sheet: str
    row_count: int
    column_count: int
    columns: list[str]
    non_null_by_column: dict[str, int]
    examples: dict[str, list[str]]
    candidate_columns: dict[str, list[str]]
    missing_required_columns: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class CPFPreparedCatalog:
    frame: pd.DataFrame
    inspection: CPFSourceInspection
    column_mapping: dict[str, str]
    duplicates: pd.DataFrame
    report: dict[str, Any]


def _normalize_column_name(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ''
    normalized = unicodedata.normalize('NFKD', text)
    normalized = ''.join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r'[^a-z0-9#]+', ' ', normalized)
    return re.sub(r'\s+', ' ', normalized).strip()


def resolve_cpf_source(preferred: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    candidates.extend(DEFAULT_SOURCE_CANDIDATES)
    seen: set[str] = set()
    searched: list[str] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = str(resolved.resolve()) if resolved.exists() else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        searched.append(str(resolved))
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Aucun catalogue CPF trouvé. Fichiers recherchés: " + "; ".join(searched)
    )


def _load_workbook(path: Path) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(path)
    except Exception as exc:
        raise RuntimeError(f"Impossible d'ouvrir le classeur Excel CPF: {path}") from exc


def _select_sheet(xls: pd.ExcelFile, preferred: str | None = None) -> str:
    if preferred:
        for sheet in xls.sheet_names:
            if _normalize_column_name(sheet) == _normalize_column_name(preferred):
                return sheet
    for candidate in (DEFAULT_V3_SHEET, 'generaliste_cpf', 'Generaliste CPF'):
        for sheet in xls.sheet_names:
            if _normalize_column_name(sheet) == _normalize_column_name(candidate):
                return sheet
    for sheet in xls.sheet_names:
        df = xls.parse(sheet, nrows=5, dtype=object)
        if not df.empty and df.shape[1] > 0:
            return sheet
    if not xls.sheet_names:
        raise ValueError('Le classeur Excel ne contient aucune feuille exploitable.')
    return xls.sheet_names[0]


def _anon_examples(df: pd.DataFrame, limit: int = 3) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = {}
    for column in df.columns:
        values: list[str] = []
        for value in df[column].dropna().tolist():
            text = clean_text(value)
            if not text:
                continue
            values.append(anonymize_value(text))
            if len(values) >= limit:
                break
        examples[column] = values
    return examples


def inspect_cpf_source(path: str | Path, *, sheet_name: str | None = None, config_path: str | Path | None = None) -> CPFSourceInspection:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f'Fichier CPF introuvable: {source_path.resolve()}')
    xls = _load_workbook(source_path)
    selected_sheet = _select_sheet(xls, preferred=sheet_name)
    df = pd.read_excel(source_path, sheet_name=selected_sheet, dtype=object)
    df = df.dropna(how='all')
    columns = [str(col) for col in df.columns]
    alias_map = load_column_aliases(config_path)
    detection = detect_columns(columns, alias_map)
    missing_required = [column for column in V3_REQUIRED_COLUMNS if column not in columns]
    warnings: list[str] = []
    if not df.empty and df.shape[0] < 50:
        warnings.append('Le nombre de lignes est inférieur au volume attendu pour le catalogue V3.')
    if missing_required:
        warnings.append('Colonnes V3 manquantes: ' + ', '.join(missing_required))
    non_null = {column: int(df[column].notna().sum()) for column in df.columns}
    inspection = CPFSourceInspection(
        path=str(source_path),
        resolved_path=str(source_path.resolve()),
        sheet_names=list(xls.sheet_names),
        selected_sheet=selected_sheet,
        row_count=int(len(df)),
        column_count=int(len(columns)),
        columns=columns,
        non_null_by_column=non_null,
        examples=_anon_examples(df),
        candidate_columns=detection.candidates,
        missing_required_columns=missing_required,
        warnings=warnings,
    )
    LOGGER.info('Fichier CPF inspecté: %s', inspection.resolved_path)
    LOGGER.info('Feuilles disponibles: %s', ', '.join(inspection.sheet_names))
    LOGGER.info('Feuille sélectionnée: %s', inspection.selected_sheet)
    LOGGER.info('Lignes: %s | Colonnes: %s', inspection.row_count, inspection.column_count)
    LOGGER.info('Colonnes exactes: %s', ', '.join(inspection.columns))
    LOGGER.info('Colonnes candidates: %s', json.dumps(inspection.candidate_columns, ensure_ascii=False))
    for warning in inspection.warnings:
        LOGGER.warning('%s', warning)
    return inspection


def _flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(_flatten_values(item))
        return items
    if isinstance(value, tuple) or isinstance(value, set):
        items: list[str] = []
        for item in value:
            items.extend(_flatten_values(item))
        return items
    if hasattr(value, 'tolist') and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                return _flatten_values(converted)
        except Exception:
            pass
    text = clean_text(value)
    if not text:
        return []
    if text.startswith('[') and text.endswith(']'):
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return _flatten_values(loaded)
        except Exception:
            pass
    parts = [clean_text(part) for part in MULTI_SPLIT_RE.split(text) if clean_text(part)]
    if len(parts) > 1:
        return parts
    return [text]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = normalize_for_match(value)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _skill_payload(values: list[str], normalizer: SkillTaxonomyNormalizer) -> list[dict[str, Any]]:
    matches = normalizer.normalize_many(values, extraction_source='cpf_v3', confidence_floor=0.0)
    payload: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        match = next((item for item in matches if normalize_for_match(item.original_label) == normalize_for_match(value)), None)
        if match:
            key = match.canonical_id
            if key in seen:
                continue
            seen.add(key)
            payload.append(
                {
                    'canonical_id': match.canonical_id,
                    'canonical_label': match.canonical_label,
                    'aliases': match.aliases,
                    'original_label': value,
                    'confidence': round(float(match.confidence), 4),
                    'extraction_source': match.extraction_source,
                }
            )
            continue
        key = normalize_for_match(value)
        if key in seen:
            continue
        seen.add(key)
        payload.append(
            {
                'canonical_id': key,
                'canonical_label': value,
                'aliases': [],
                'original_label': value,
                'confidence': 0.0,
                'extraction_source': 'cpf_v3:raw',
            }
        )
    return payload


def _join_nonempty(parts: list[tuple[str, Any]]) -> str:
    lines: list[str] = []
    for label, value in parts:
        text = clean_text(value)
        if text:
            lines.append(f'{label}: {text}')
    return '\n'.join(lines)


def _build_text_modele(row: dict[str, Any]) -> str:
    competences = row.get('competences') or []
    if isinstance(competences, str):
        competences = [competences]
    tags = row.get('tags') or []
    if isinstance(tags, str):
        tags = [tags]
    codes_rome = row.get('codes_rome') or []
    if isinstance(codes_rome, str):
        codes_rome = [codes_rome]
    return _join_nonempty([
        ('Titre', row.get('titre')),
        ('Secteur', row.get('secteur')),
        ('Description', row.get('description')),
        ('Objectifs', row.get('objectifs')),
        ('Contenu', row.get('contenu')),
        ('Compétences', ' | '.join([clean_text(item) for item in competences if clean_text(item)])),
        ('Pré-requis', row.get('prerequis')),
        ('Métiers cibles', row.get('metiers_cibles')),
        ('Certification', row.get('certification')),
        ('Niveau', row.get('niveau')),
        ('Modalité', row.get('modalite')),
        ('Durée', row.get('duree')),
        ('Prix', row.get('prix')),
        ('Codes ROME', ' | '.join([clean_text(item) for item in codes_rome if clean_text(item)])),
        ('Tags', ' | '.join([clean_text(item) for item in tags if clean_text(item)])),
    ])


def _build_formation_id(source_version: str, source_row_id: Any, titre: str, organisme: str, certification: str, code_certification: str) -> str:
    return stable_hash(
        source_version,
        clean_text(source_row_id),
        normalize_for_match(titre),
        normalize_for_match(organisme),
        normalize_for_match(certification),
        normalize_for_match(code_certification),
        length=24,
    )


def _extract_v3_row(row: dict[str, Any], normalizer: SkillTaxonomyNormalizer) -> dict[str, Any]:
    source_row_id = clean_text(row.get('#'))
    titre = clean_text(row.get('Intitulé de la formation'))
    secteur = clean_text(row.get('Secteur'))
    organisme = clean_text(row.get('Organisme de formation'))
    type_certification = clean_text(row.get('Type de certification'))
    code_certification = clean_text(row.get('Code certification'))
    niveau = clean_text(row.get('Niveau'))
    codes_rome = _dedupe_preserve_order(_flatten_values(row.get('Codes ROME')))
    competences_originales = _dedupe_preserve_order(_flatten_values(row.get('Compétences majeures')))
    tags = _dedupe_preserve_order(_flatten_values(row.get('Tags')))
    competences_sources = _dedupe_preserve_order(competences_originales + [tag for tag in tags if tag not in competences_originales])
    competences_normalisees = _skill_payload(competences_sources, normalizer)
    certification = ' '.join(part for part in [type_certification, code_certification] if part).strip()
    code_rncp = code_certification if normalize_for_match(type_certification) == 'rncp' else ''
    code_rs = code_certification if normalize_for_match(type_certification) == 'rs' else ''
    modalite = clean_text(row.get('Modalité'))
    distance_compatible = bool(modalite) and any(token in normalize_for_match(modalite) for token in ['distance', 'distanciel', 'e learning', 'elearning', 'a distance', 'à distance', 'en ligne'])
    record = {
        'formation_id': _build_formation_id(V3_SOURCE_VERSION, source_row_id, titre, organisme, certification, code_certification),
        'titre': titre,
        'description': '',
        'objectifs': '',
        'competences': competences_sources,
        'competences_normalisees': competences_normalisees,
        'prerequis': '',
        'contenu': '',
        'metiers_cibles': '',
        'code_rome': ' | '.join(codes_rome),
        'codes_rome': codes_rome,
        'certification': certification,
        'code_rncp': code_rncp,
        'code_rs': code_rs,
        'organisme': organisme,
        'modalite': modalite,
        'duree': clean_text(row.get('Durée')),
        'prix': clean_text(row.get('Prix TTC (€)')),
        'commune': '',
        'departement': '',
        'region': '',
        'code_postal': '',
        'latitude': None,
        'longitude': None,
        'secteur': secteur,
        'tags': tags,
        'niveau': niveau,
        'source': V3_SOURCE_NAME,
        'source_version': V3_SOURCE_VERSION,
        'record_type': V3_RECORD_TYPE,
        'source_row_id': source_row_id,
        'texte_modele': '',
        'type_certification': type_certification,
        'code_certification': code_certification,
        'distance_compatible': distance_compatible,
        'remote': distance_compatible,
        'relu_valide': clean_text(row.get('✅ Relu / Validé (oui/non)')),
        'corrections_remarques': clean_text(row.get('🗒 Corrections / Remarques')),
    }
    record['texte_modele'] = _build_text_modele(record)
    return record


def _deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    seen_exact: set[str] = set()
    seen_cert: set[str] = set()
    seen_title_org: set[str] = set()
    seen_title_cert: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        formation_id = clean_text(row.get('formation_id'))
        code = clean_text(row.get('code_certification')) or clean_text(row.get('code_rncp')) or clean_text(row.get('code_rs'))
        title_key = normalize_for_match(row.get('titre'))
        org_key = normalize_for_match(row.get('organisme'))
        cert_key = normalize_for_match(row.get('certification'))
        exact_key = formation_id
        title_org_key = f'{title_key}||{org_key}' if title_key or org_key else ''
        title_cert_key = f'{title_key}||{cert_key}' if title_key or cert_key else ''
        if exact_key and exact_key in seen_exact:
            duplicates.append({**row, 'duplicate_reason': 'source_id'})
            continue
        if code and code in seen_cert:
            duplicates.append({**row, 'duplicate_reason': 'code_certification'})
            continue
        if title_org_key and title_org_key in seen_title_org:
            duplicates.append({**row, 'duplicate_reason': 'titre_organisme'})
            continue
        if title_cert_key and title_cert_key in seen_title_cert:
            duplicates.append({**row, 'duplicate_reason': 'titre_certification'})
            continue
        kept.append(row)
        if exact_key:
            seen_exact.add(exact_key)
        if code:
            seen_cert.add(code)
        if title_org_key:
            seen_title_org.add(title_org_key)
        if title_cert_key:
            seen_title_cert.add(title_cert_key)
    return kept, pd.DataFrame.from_records(duplicates)


def _serialize_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    serialised = frame.copy()
    for column in serialised.columns:
        serialised[column] = serialised[column].apply(
            lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        )
    return serialised


def _build_report(
    *,
    inspection: CPFSourceInspection,
    rows_initial: int,
    rows_kept: int,
    duplicates_count: int,
    frame: pd.DataFrame,
    duplicates: pd.DataFrame,
    column_mapping: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    title_non_empty = int(frame['titre'].fillna('').astype(str).str.strip().ne('').sum()) if not frame.empty and 'titre' in frame.columns else 0
    text_non_empty = int(frame['texte_modele'].fillna('').astype(str).str.strip().ne('').sum()) if not frame.empty and 'texte_modele' in frame.columns else 0
    skills_non_empty = int(frame['competences'].apply(lambda value: bool(_flatten_values(value))).sum()) if not frame.empty and 'competences' in frame.columns else 0
    with_location = int(frame[['commune', 'departement', 'region']].fillna('').astype(str).apply(lambda row: any(cell.strip() for cell in row), axis=1).sum()) if not frame.empty else 0
    with_certification = int(frame[['certification', 'code_rncp', 'code_rs']].fillna('').astype(str).apply(lambda row: any(cell.strip() for cell in row), axis=1).sum()) if not frame.empty else 0
    report = {
        'source_file': inspection.path,
        'source_file_abs': inspection.resolved_path,
        'source_version': V3_SOURCE_VERSION,
        'sheet_used': inspection.selected_sheet,
        'rows_initial': rows_initial,
        'rows_kept': rows_kept,
        'rows_rejected': rows_initial - rows_kept,
        'duplicates': duplicates_count,
        'formations_with_skills': skills_non_empty,
        'formations_without_skills': max(rows_kept - skills_non_empty, 0),
        'formations_with_location': with_location,
        'formations_with_certification': with_certification,
        'titles_non_empty': title_non_empty,
        'texts_modele_non_empty': text_non_empty,
        'columns_detected': inspection.columns,
        'column_mapping': column_mapping,
        'warnings': warnings,
        'errors': [],
    }
    return report


def prepare_cpf_v3_dataset(
    path: str | Path,
    output_dir: str | Path,
    *,
    sheet_name: str | None = None,
    config_path: str | Path | None = None,
) -> CPFPreparedCatalog:
    source_path = Path(path)
    inspection = inspect_cpf_source(source_path, sheet_name=sheet_name, config_path=config_path)
    df = pd.read_excel(source_path, sheet_name=inspection.selected_sheet, dtype=object).dropna(how='all')
    normalizer = SkillTaxonomyNormalizer()
    rows = [_extract_v3_row(row, normalizer) for row in df.fillna('').to_dict(orient='records')]
    rows, duplicates = _deduplicate_rows(rows)
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise RuntimeError("Aucune formation exploitable n'a été conservée après préparation du catalogue CPF V3.")
    frame['texte_modele'] = frame['texte_modele'].fillna('').astype(str).str.strip()
    frame = frame[frame['titre'].fillna('').astype(str).str.strip() != ''].copy()
    frame = frame[frame['texte_modele'].str.strip() != ''].copy()
    frame['source_row_id'] = frame['source_row_id'].astype(str)
    frame['formation_id'] = frame['formation_id'].astype(str)
    frame['record_type'] = V3_RECORD_TYPE
    frame['source_version'] = V3_SOURCE_VERSION
    frame['source'] = V3_SOURCE_NAME
    output_root = Path(output_dir)
    processed_dir = output_root
    cpf_dir = processed_dir / 'cpf'
    reports_dir = processed_dir / 'reports'
    for directory in [processed_dir, cpf_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    canonical_parquet = processed_dir / 'formations_cpf_v3.parquet'
    canonical_csv = processed_dir / 'formations_cpf_v3.csv'
    canonical_jsonl = processed_dir / 'formations_cpf_v3.jsonl'
    unified_parquet = processed_dir / 'dataset_unifie.parquet'
    mapping_report_path = reports_dir / 'cpf_v3_column_mapping.json'
    import_report_path = reports_dir / 'cpf_v3_import_report.json'
    duplicates_path = reports_dir / 'cpf_v3_duplicates.csv'
    cpf_normalized_parquet = cpf_dir / 'formations_normalized.parquet'
    cpf_normalized_csv = cpf_dir / 'formations_normalized.csv'
    cpf_normalized_jsonl = cpf_dir / 'formations_normalized.jsonl'
    cpf_compat_parquet = cpf_dir / 'formations.parquet'
    cpf_compat_csv = cpf_dir / 'formations.csv'
    cpf_compat_jsonl = cpf_dir / 'formations.jsonl'
    frame.to_parquet(canonical_parquet, index=False)
    _serialize_for_csv(frame).to_csv(canonical_csv, index=False, encoding='utf-8')
    canonical_jsonl.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in frame.replace({pd.NA: None}).to_dict(orient='records')) + '\n', encoding='utf-8')
    frame.to_parquet(unified_parquet, index=False)
    frame.to_parquet(cpf_normalized_parquet, index=False)
    _serialize_for_csv(frame).to_csv(cpf_normalized_csv, index=False, encoding='utf-8')
    cpf_normalized_jsonl.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in frame.replace({pd.NA: None}).to_dict(orient='records')) + '\n', encoding='utf-8')
    frame.to_parquet(cpf_compat_parquet, index=False)
    _serialize_for_csv(frame).to_csv(cpf_compat_csv, index=False, encoding='utf-8')
    cpf_compat_jsonl.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in frame.replace({pd.NA: None}).to_dict(orient='records')) + '\n', encoding='utf-8')
    duplicates.to_csv(duplicates_path, index=False, encoding='utf-8')
    column_mapping = {canonical: source for canonical, source in V3_COLUMN_MAPPING.items()}
    mapping_report_path.write_text(json.dumps(column_mapping, ensure_ascii=False, indent=2), encoding='utf-8')
    warnings = []
    if len(rows) < 100:
        warnings.append("Volume de formations plus faible qu'attendu pour le catalogue V3.")
    if frame['competences'].apply(lambda value: bool(_flatten_values(value))).mean() < 0.5:
        warnings.append("Fort taux de formations sans compétences explicites.")
    if frame['description'].fillna('').astype(str).str.strip().eq('').mean() > 0.5:
        warnings.append("Fort taux de descriptions vides.")
    report = _build_report(
        inspection=inspection,
        rows_initial=int(inspection.row_count),
        rows_kept=int(len(frame)),
        duplicates_count=int(len(duplicates)),
        frame=frame,
        duplicates=duplicates,
        column_mapping=column_mapping,
        warnings=warnings,
    )
    report['output_files'] = {
        'formations_cpf_v3_parquet': str(canonical_parquet),
        'formations_cpf_v3_csv': str(canonical_csv),
        'formations_cpf_v3_jsonl': str(canonical_jsonl),
        'dataset_unifie_parquet': str(unified_parquet),
        'formations_normalized_parquet': str(cpf_normalized_parquet),
        'formations_normalized_csv': str(cpf_normalized_csv),
        'formations_normalized_jsonl': str(cpf_normalized_jsonl),
        'duplicates_csv': str(duplicates_path),
        'mapping_report': str(mapping_report_path),
        'import_report': str(import_report_path),
    }
    import_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return CPFPreparedCatalog(
        frame=frame,
        inspection=inspection,
        column_mapping=column_mapping,
        duplicates=duplicates,
        report=report,
    )


def load_cpf_v3_frame(path: str | Path, *, sheet_name: str | None = None) -> pd.DataFrame:
    source_path = Path(path)
    inspection = inspect_cpf_source(source_path, sheet_name=sheet_name)
    df = pd.read_excel(source_path, sheet_name=inspection.selected_sheet, dtype=object).dropna(how='all')
    normalizer = SkillTaxonomyNormalizer()
    rows = [_extract_v3_row(row, normalizer) for row in df.fillna('').to_dict(orient='records')]
    rows, _ = _deduplicate_rows(rows)
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return frame
    frame['texte_modele'] = frame['texte_modele'].fillna('').astype(str).str.strip()
    frame = frame[frame['titre'].fillna('').astype(str).str.strip() != ''].copy()
    frame = frame[frame['texte_modele'].str.strip() != ''].copy()
    return frame
