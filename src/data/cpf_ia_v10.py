import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SHEET_NAME = 'Dataset_IA_V10'

REQUIRED_COLUMNS = [
    '#', 'Secteur', 'Organisme de formation', 'Intitulé de la formation',
    'Type de certification', 'Code certification', 'Compétences IA extraites',
    'Modalité', 'Tags TrendRadar',
]

OPTIONAL_COLUMNS = [
    'Niveau', 'Codes ROME', 'Durée', 'Prix TTC (€)',
    '✅ Relu / Validé (oui/non)', '🗒 Corrections / Remarques',
]

COLUMN_ALIASES: dict[str, str] = {}

ROME_PATTERN = re.compile(r'^[A-Z][0-9]{4}$')


@dataclass
class CPFIAFormation:
    source_row_id: int
    sector: str
    provider_name: str
    title: str
    certification_type: str
    certification_code: str
    level: str | None = None
    rome_codes: list[str] = field(default_factory=list)
    extracted_ai_skills_raw: str = ''
    modality: str = ''
    duration: str | None = None
    price_ttc: float | None = None
    trendradar_tags: list[str] = field(default_factory=list)
    reviewed: bool | None = None
    review_notes: str | None = None


@dataclass
class ParsedSkill:
    source_row_id: int
    certification_code: str
    formation_title: str
    sector: str
    provider_name: str
    skill_original: str
    skill_normalized: str
    detected_type: str = 'TO_REVIEW'
    quality_status: str = 'TO_REVIEW'
    reviewed: bool | None = None
    review_notes: str | None = None


@dataclass
class RomeMissingRow:
    source_row_id: int
    certification_code: str
    title: str
    sector: str
    provider_name: str
    extracted_ai_skills_raw: str
    suggested_rome_codes: str = ''
    review_status: str = 'pending'


@dataclass
class QualityReport:
    input_file: str = ''
    sheet: str = SHEET_NAME
    rows_total: int = 0
    rncp_count: int = 0
    rs_count: int = 0
    missing_rome_count: int = 0
    missing_level_count: int = 0
    missing_duration_count: int = 0
    duplicate_certification_codes: list[str] = field(default_factory=list)
    sectors: dict[str, int] = field(default_factory=dict)
    top_rome_codes: dict[str, int] = field(default_factory=dict)
    rows_to_review: int = 0
    skills_total: int = 0
    skills_to_review: int = 0
    generated_at: str = ''


def inspect_excel(path: str | Path) -> dict[str, Any]:
    import pandas as pd
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Fichier introuvable: {path}')
    xls = pd.ExcelFile(path)
    sheet_names = xls.sheet_names
    if SHEET_NAME not in sheet_names:
        raise ValueError(f'Feuille "{SHEET_NAME}" introuvable dans {path}. Feuilles: {sheet_names}')
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    columns = list(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    return {
        'path': str(path),
        'sheet_names': sheet_names,
        'selected_sheet': SHEET_NAME,
        'row_count': len(df),
        'column_count': len(columns),
        'columns': columns,
        'missing_required_columns': missing,
    }


def normalize_certification_code(raw: str) -> str:
    cleaned = raw.strip().upper().replace(' ', '')
    return cleaned


def parse_certification_type(code: str) -> str:
    code_upper = code.strip().upper()
    if code_upper.startswith('RNCP'):
        return 'RNCP'
    elif code_upper.startswith('RS'):
        return 'RS'
    return 'UNKNOWN'


def parse_rome_codes(value: str | None) -> list[str]:
    if not value or (isinstance(value, float) and str(value) == 'nan'):
        return []
    if isinstance(value, (int, float)):
        value = str(int(value))
    import re as _re
    parts = _re.split(r'[,;|]+', str(value))
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        code = part.strip().upper()
        if ROME_PATTERN.match(code):
            if code not in seen:
                seen.add(code)
                result.append(code)
        elif code:
            logger.warning('Code ROME invalide ignoré: %s', repr(code))
    return result


def split_extracted_ai_skills(raw_text: str) -> list[str]:
    if not raw_text or (isinstance(raw_text, float) and str(raw_text) == 'nan'):
        return []
    text = str(raw_text).strip()
    if not text:
        return []
    candidates: list[str] = []
    for sep in ('\n', '|', ';'):
        if sep in text:
            candidates = [s.strip() for s in text.split(sep) if s.strip()]
            break
    if not candidates:
        candidates = [text]
    return candidates


def normalize_skill_text(text: str) -> str:
    t = text.strip()
    t = re.sub(r'\s+', ' ', t)
    return t


def detect_skill_type(text: str) -> str:
    t = text.lower().strip()
    if len(t) < 5:
        return 'TO_REVIEW'
    if t.startswith('formation en ') or t.startswith('formation à '):
        return 'COURSE_CONTENT'
    if re.search(r'(machine learning|deep learning|intelligence artificielle|réseau de neurones|llm|nlp|computer vision|traitement automatique)', t):
        return 'SKILL'
    if re.search(r'(python|tensorflow|pytorch|spark|docker|kubernetes|git|jira|power bi|tableau|excel|word|powerpoint|chatgpt|gpt)', t):
        return 'TOOL'
    if re.search(r'(méthode|processus|démarche|approche|technique|protocole)', t):
        return 'METHOD'
    if len(t) > 200:
        return 'COURSE_CONTENT'
    if re.search(r'(apprenant|formation|stagiaire|module|semaine|jour|compétence)', t):
        return 'COURSE_CONTENT'
    return 'TO_REVIEW'


def determine_quality_status(text: str) -> str:
    t = text.strip()
    if len(t) < 5:
        return 'TOO_SHORT'
    if len(t) > 300:
        return 'TOO_LONG'
    if t.endswith('…') or t.endswith('...') or t.endswith('constitu'):
        return 'TRUNCATED'
    if len(t) < 15:
        return 'VAGUE'
    if t.lower().startswith('formation en ') or t.lower().startswith('formation à '):
        return 'NOT_A_SKILL'
    if 'accompagnent nos apprenants' in t.lower() or 'expert' in t.lower():
        return 'NOT_A_SKILL'
    if len(t) > 150:
        return 'TO_REVIEW'
    return 'OK'


def parse_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if str(value) == 'nan':
            return None
        return float(value)
    cleaned = str(value).strip().replace('€', '').replace(' ', '').replace(',', '.').replace('\u202f', '').replace('\xa0', '')
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def parse_tags(raw: str | None) -> list[str]:
    if not raw or (isinstance(raw, float) and str(raw) == 'nan'):
        return []
    return [t.strip() for t in str(raw).split('|') if t.strip()]


def parse_reviewed(raw: Any) -> bool | None:
    if raw is None or (isinstance(raw, float) and str(raw) == 'nan'):
        return None
    s = str(raw).strip().lower()
    if s in ('oui', 'yes', '1', 'true', 'x'):
        return True
    if s in ('non', 'no', '0', 'false', ''):
        return False
    return None


def parse_formation(row: dict, row_id: int) -> CPFIAFormation:
    cert_code_raw = str(row.get('Code certification', ''))
    cert_code = normalize_certification_code(cert_code_raw)
    cert_type = parse_certification_type(cert_code)
    rome_raw = row.get('Codes ROME')
    rome_codes = parse_rome_codes(rome_raw)
    skills_raw = str(row.get('Compétences IA extraites', ''))
    tags_raw = row.get('Tags TrendRadar')
    price_raw = row.get('Prix TTC (€)')
    reviewed_raw = row.get('✅ Relu / Validé (oui/non)')
    notes_raw = row.get('🗒 Corrections / Remarques')
    return CPFIAFormation(
        source_row_id=row_id,
        sector=str(row.get('Secteur', '')),
        provider_name=str(row.get('Organisme de formation', '')),
        title=str(row.get('Intitulé de la formation', '')),
        certification_type=cert_type,
        certification_code=cert_code,
        level=_safe_str(row.get('Niveau')),
        rome_codes=rome_codes,
        extracted_ai_skills_raw=skills_raw,
        modality=str(row.get('Modalité', '')),
        duration=_safe_str(row.get('Durée')),
        price_ttc=parse_price(price_raw),
        trendradar_tags=parse_tags(tags_raw),
        reviewed=parse_reviewed(reviewed_raw),
        review_notes=_safe_str(notes_raw),
    )


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and str(value) == 'nan':
        return None
    s = str(value).strip()
    return s if s else None


def build_quality_report(
    formations: list[CPFIAFormation],
    skills: list[ParsedSkill],
    input_path: str,
) -> QualityReport:
    sectors: dict[str, int] = {}
    top_rome: dict[str, int] = {}
    cert_codes: dict[str, int] = {}
    rncp = rs = missing_rome = missing_level = missing_duration = 0
    for f in formations:
        sectors[f.sector] = sectors.get(f.sector, 0) + 1
        if f.certification_type == 'RNCP':
            rncp += 1
        elif f.certification_type == 'RS':
            rs += 1
        cert_codes[f.certification_code] = cert_codes.get(f.certification_code, 0) + 1
        if not f.rome_codes:
            missing_rome += 1
        for code in f.rome_codes:
            top_rome[code] = top_rome.get(code, 0) + 1
        if not f.level:
            missing_level += 1
        if not f.duration:
            missing_duration += 1
    dupes = [code for code, count in cert_codes.items() if count > 1]
    rows_to_review = len([f for f in formations if not f.rome_codes or (f.certification_type == 'RNCP' and not f.level)])
    skills_to_review = len([s for s in skills if s.quality_status != 'OK'])
    top_rome_sorted = dict(sorted(top_rome.items(), key=lambda x: -x[1])[:20])
    return QualityReport(
        input_file=input_path,
        rows_total=len(formations),
        rncp_count=rncp,
        rs_count=rs,
        missing_rome_count=missing_rome,
        missing_level_count=missing_level,
        missing_duration_count=missing_duration,
        duplicate_certification_codes=dupes,
        sectors=sectors,
        top_rome_codes=top_rome_sorted,
        rows_to_review=rows_to_review,
        skills_total=len(skills),
        skills_to_review=skills_to_review,
        generated_at=datetime.now().isoformat(),
    )


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
) -> QualityReport:
    import pandas as pd
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Lecture de %s', input_path)
    inspection = inspect_excel(input_path)
    if inspection['missing_required_columns']:
        raise ValueError(f'Colonnes obligatoires manquantes: {inspection["missing_required_columns"]}')

    df = pd.read_excel(input_path, sheet_name=SHEET_NAME)
    formations: list[CPFIAFormation] = []
    all_skills: list[ParsedSkill] = []
    rome_missing: list[RomeMissingRow] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        row_id = idx + 1
        formation = parse_formation(row.to_dict(), row_id)
        formations.append(formation)
        skill_texts = split_extracted_ai_skills(formation.extracted_ai_skills_raw)
        for st in skill_texts:
            normalized = normalize_skill_text(st)
            ps = ParsedSkill(
                source_row_id=row_id,
                certification_code=formation.certification_code,
                formation_title=formation.title,
                sector=formation.sector,
                provider_name=formation.provider_name,
                skill_original=st,
                skill_normalized=normalized,
                detected_type=detect_skill_type(normalized),
                quality_status=determine_quality_status(normalized),
            )
            all_skills.append(ps)
        if not formation.rome_codes:
            rome_missing.append(RomeMissingRow(
                source_row_id=row_id,
                certification_code=formation.certification_code,
                title=formation.title,
                sector=formation.sector,
                provider_name=formation.provider_name,
                extracted_ai_skills_raw=formation.extracted_ai_skills_raw,
            ))

    report = build_quality_report(formations, all_skills, str(input_path))

    formations_path = output_dir / 'formations.parquet'
    formations_df = pd.DataFrame([asdict(f) for f in formations])
    formations_df['rome_codes'] = formations_df['rome_codes'].apply(lambda x: ','.join(x) if x else '')
    formations_df['trendradar_tags'] = formations_df['trendradar_tags'].apply(lambda x: '|'.join(x) if x else '')
    formations_df.to_parquet(formations_path, index=False)
    logger.info('Formations exportées: %s (%d lignes)', formations_path, len(formations_df))

    skills_path = output_dir / 'skills_to_review.csv'
    skills_df = pd.DataFrame([asdict(s) for s in all_skills])
    skills_df.to_csv(skills_path, index=False)
    logger.info('Compétences exportées: %s (%d lignes)', skills_path, len(skills_df))

    rome_path = output_dir / 'rome_missing.csv'
    rome_df = pd.DataFrame([asdict(r) for r in rome_missing])
    rome_df.to_csv(rome_path, index=False)
    logger.info('ROME manquants exportés: %s (%d lignes)', rome_path, len(rome_df))

    report_path = output_dir / 'quality_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    logger.info('Rapport qualité exporté: %s', report_path)

    return report


def export_training_candidates(
    skills: list[ParsedSkill],
    output_path: str | Path,
) -> int:
    accepted: list[dict[str, str]] = []
    for s in skills:
        if s.quality_status != 'OK':
            continue
        if s.detected_type == 'COURSE_CONTENT':
            continue
        accepted.append({
            'text': s.skill_normalized,
            'labels': json.dumps([]),
            'source': 'Dataset_IA_V10_CPF',
            'certification_code': s.certification_code,
            'formation_title': s.formation_title,
            'sector': s.sector,
            'detected_type': s.detected_type,
            'quality_status': s.quality_status,
        })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in accepted:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logger.info(
        'Training candidates exportés: %s (%d lignes)',
        output_path, len(accepted),
    )
    return len(accepted)


def match_referential_skills_to_cpf_ia_formations(
    referential_skills: list[dict[str, str]],
    formations: list[CPFIAFormation],
    *,
    min_similarity: float = 0.72,
) -> list[dict[str, Any]]:
    from rapidfuzz import fuzz

    recommendations: list[dict[str, Any]] = []
    formation_skills_cache: dict[str, set[str]] = {}

    def _get_formation_skills(f: CPFIAFormation) -> set[str]:
        if f.certification_code in formation_skills_cache:
            return formation_skills_cache[f.certification_code]
        raw = f.extracted_ai_skills_raw
        texts = split_extracted_ai_skills(raw)
        normalized = {normalize_skill_text(t).lower() for t in texts if t.strip()}
        formation_skills_cache[f.certification_code] = normalized
        return normalized

    for rskill in referential_skills:
        ref_label = (rskill.get('label') or rskill.get('official_label') or '').strip().lower()
        if not ref_label:
            continue
        best_score = 0.0
        best_formation: CPFIAFormation | None = None
        matched: list[str] = []
        for f in formations:
            f_skills = _get_formation_skills(f)
            for fs in f_skills:
                score = fuzz.ratio(ref_label, fs) / 100.0
                if score > best_score:
                    best_score = score
                    best_formation = f
                    matched = [fs]
                elif score == best_score and score > 0:
                    matched.append(fs)
        if best_formation and best_score >= min_similarity:
            rome_overlap = set(rskill.get('rome_codes', [])) & set(best_formation.rome_codes)
            tags_overlap = set(rskill.get('tags', [])) & set(best_formation.trendradar_tags)
            recommendations.append({
                'formation_title': best_formation.title,
                'provider_name': best_formation.provider_name,
                'certification_code': best_formation.certification_code,
                'certification_type': best_formation.certification_type,
                'sector': best_formation.sector,
                'matched_skills': list(matched),
                'missing_skills': [],
                'similarity_score': round(best_score, 4),
                'rome_overlap': list(rome_overlap),
                'tags_overlap': list(tags_overlap),
                'price_ttc': best_formation.price_ttc,
                'modality': best_formation.modality,
                'duration': best_formation.duration,
            })
    recommendations.sort(key=lambda r: -r['similarity_score'])
    return recommendations
