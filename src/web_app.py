from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import secrets
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request, Response, redirect, url_for, session
from markupsafe import Markup, escape
from werkzeug.exceptions import RequestEntityTooLarge

from common.text import clean_text, normalize_for_match, stable_hash
from config.thresholds import THRESHOLDS
from config.weights import SCORING_WEIGHTS
from data_sources.ia_recommendations import load_ia_recommendations_csv
from domain.models import MarketTarget, PdfAnalysis, RomeOccupation, Territory
from france_travail.client import FranceTravailAuthError, FranceTravailClient, FranceTravailError, FranceTravailRateLimitError, FranceTravailTimeoutError, SearchCriteria
from france_travail.normalizer import normalize_offer
from inference.deepforma_predictor import DeepformaPredictor, get_predictor
from models.analysis_result import (
    AnalysisResult, CheckpointAuditInfo, ClassificationInfo, IAClassificationInfo,
    MarketComparisonItem, MarketSkillInfo, ModelMetadata, OpenExtractedSkill,
    QualityInfo, Recommendation, SkillExtractionInfo, SkillInfo,
    TerritorialMarketInfo,
)
from skills.merge_offer_skills import extract_skills_from_text, merge_offer_skills
from continual_learning.auth import require_admin_auth
from continual_learning.store import ContinualLearningStore
from referential_import import ReferentialImportService
from referential_import.editing_service import ReferentialEditingService
from referential_import.models import DerivedSkill, OfficialCompetency
from referential_learning.store import AnnotationStore
from referential_import.import_service import analysis_from_export, build_export_payload
from referential_import.pdf_loader import load_pdf_document
from referentials.rome_referential import RomeService, validate_rome_code, validate_rome_codes
from skills.open_extractor import extract_skills as open_extract_skills
from services.analysis_result_builder import build_analysis_result
from services.certification_market_comparison import CertificationMarketComparator, collect_market_offers, write_comparison_outputs
from services.market_context import build_market_context, fetch_offers_by_rome, fetch_offers_by_rome_codes, serialize_record
from services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / 'templates'
STATIC_DIR = PROJECT_ROOT / 'static'
DEFAULT_CACHE_TTL_SECONDS = int(os.getenv('DEEPFORMA_CACHE_TTL_SECONDS', '600'))
DEFAULT_MAX_OFFERS = int(os.getenv('DEEPFORMA_MAX_OFFERS', '25'))
DEFAULT_PAGE_SIZE = int(os.getenv('DEEPFORMA_PAGE_SIZE', '10'))
DEFAULT_MAX_PAGES = int(os.getenv('DEEPFORMA_MAX_PAGES', '3'))
MAX_ROME_CODES_PER_SEARCH = int(os.getenv('MAX_ROME_CODES_PER_SEARCH', '10'))
DEFAULT_THRESHOLD = float(os.getenv('DEEPFORMA_DEFAULT_THRESHOLD', str(THRESHOLDS.medium_confidence)))
DEFAULT_MAX_CONTENT_LENGTH = int(os.getenv('DEEPFORMA_MAX_CONTENT_LENGTH', '10485760'))
DEFAULT_MAX_FORM_MEMORY_SIZE = int(os.getenv('DEEPFORMA_MAX_FORM_MEMORY_SIZE', '2097152'))
DEFAULT_MAX_FORM_PARTS = int(os.getenv('DEEPFORMA_MAX_FORM_PARTS', '2000'))
MODEL_SCORE_STD_MIN = float(os.getenv('DEEPFORMA_MODEL_SCORE_STD_MIN', '0.05'))
MODEL_SCORE_MAX_MIN = float(os.getenv('DEEPFORMA_MODEL_SCORE_MAX_MIN', '0.70'))
MODEL_SCORE_GAP_MIN = float(os.getenv('DEEPFORMA_MODEL_SCORE_GAP_MIN', '0.05'))

DEPARTMENT_CODES = [
    f'{code:02d}' for code in range(1, 96)
] + ['2A', '2B', '971', '972', '973', '974', '976']

EXPERIMENTAL_WARNING = (
    'Resultat experimental. Le modele doit encore etre valide '
    'avant utilisation operationnelle.'
)


class TTLCache:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        now = datetime.now(timezone.utc).timestamp()
        item = self._data.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at < now:
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        expires_at = datetime.now(timezone.utc).timestamp() + self.ttl_seconds
        self._data[key] = (expires_at, value)


class DiagnosticLogger:
    _logged: set[str] = set()

    @classmethod
    def log_once(cls, key: str, message: str) -> None:
        if key not in cls._logged:
            cls._logged.add(key)
            logger.info('[DIAGNOSTIC] %s', message)


def _make_cache_key(departement: str, keywords: str | None) -> str:
    normalized_keywords = (keywords or '').strip().lower()
    return f"{departement.strip()}::{normalized_keywords}"

def parse_rome_codes_from_request(req: Any) -> list[str]:
    values: list[str] = []
    if hasattr(req, 'form'):
        try:
            values.extend(req.form.getlist('rome_codes'))
            values.extend(req.form.getlist('selected_rome_codes'))
        except Exception:
            pass
        single = clean_text(req.form.get('rome_code') or req.form.get('selected_rome_code') or '')
        if single:
            values.append(single)
    if getattr(req, 'is_json', False):
        payload = req.get_json(silent=True) or {}
        if isinstance(payload, dict):
            json_values = payload.get('rome_codes') or payload.get('selected_rome_codes') or []
            if isinstance(json_values, list):
                values.extend(str(item) for item in json_values if clean_text(item))
            single = clean_text(payload.get('rome_code') or '')
            if single:
                values.append(single)
    return [clean_text(value).replace(' ', '').upper() for value in values if clean_text(value)]


def _rome_occupation_payload(code: str, label: str | None = None) -> dict[str, Any]:
    return {'code': clean_text(code).replace(' ', '').upper(), 'label': clean_text(label or '') or clean_text(code).replace(' ', '').upper()}


def _format_request_payload_log(req: Any) -> str:
    content_length = req.content_length if req.content_length is not None else req.headers.get('Content-Length', 'unknown')
    if req.method in {'POST', 'PUT', 'PATCH'}:
        try:
            field_names = sorted(req.form.keys())
        except Exception:
            field_names = []
        names = ', '.join(field_names[:20])
        return f'{req.method} {req.path} | Content-Length={content_length} | Fields={len(field_names)} | Names={names}'
    return f'{req.method} {req.path} | Content-Length={content_length}'


def _ensure_csrf_token() -> str:
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token


def _validate_csrf_token() -> None:
    submitted = clean_text(request.form.get('csrf_token') or request.headers.get('X-CSRF-Token') or '')
    expected = session.get('_csrf_token') or ''
    if not submitted or not expected or submitted != expected:
        raise ValueError('Jeton CSRF invalide.')


def _admin_filters_from_request() -> dict[str, str]:
    keys = ['status', 'territory', 'job_family', 'min_confidence', 'max_confidence', 'source', 'model_version', 'disagreement']
    return {key: clean_text(request.args.get(key) or request.form.get(key) or '') for key in keys}


def _admin_redirect_url(*, offer_row_id: int | None = None, filters: dict[str, str] | None = None) -> str:
    filters = filters or {}
    query: dict[str, Any] = {key: value for key, value in filters.items() if value}
    if offer_row_id not in (None, '', 0):
        query['offer_row_id'] = int(offer_row_id)
    return url_for('admin_continual_learning', **query)


def _admin_validation_log(message: str) -> None:
    logger.info('[ADMIN_CONTINUAL_LEARNING] %s', message)



def _build_france_travail_client() -> FranceTravailClient:
    return FranceTravailClient(timeout=int(os.getenv('FRANCE_TRAVAIL_TIMEOUT', '20')))



def _load_predictor() -> tuple[DeepformaPredictor | None, str | None]:
    try:
        return get_predictor(), None
    except Exception as exc:
        return None, str(exc)


def _available_france_travail_config() -> bool:
    return bool(
        os.getenv('FRANCE_TRAVAIL_CLIENT_ID')
        and os.getenv('FRANCE_TRAVAIL_CLIENT_SECRET')
    )


def _serialize_report(report: Any) -> dict[str, Any]:
    return {
        'formation_skills': report.formation_skills,
        'market_skills': [asdict(item) for item in report.market_skills],
        'covered_skills': report.covered_skills,
        'missing_priority_skills': [asdict(item) for item in report.missing_priority_skills],
        'coverage_score': report.coverage_score,
        'offer_count': report.offer_count,
        'matched_market_offers': report.matched_market_offers,
    }


def _skill_confidence(score: float) -> str:
    return THRESHOLDS.get_confidence_level(score)


def _check_ia_classifier_quality(skills_result: dict[str, Any]) -> IAClassificationInfo:
    score_std = skills_result.get('score_std', 0.0)
    score_max = skills_result.get('score_max', 0.0)
    score_mean = skills_result.get('score_mean', 0.0)
    score_min = skills_result.get('score_min', 0.0)
    discriminating = score_std > 0.05 or score_max > 0.70
    warnings: list[str] = []
    if not discriminating:
        warnings.append(
            'Le modele specialise dans les 18 categories IA ne produit pas de scores '
            'suffisamment discriminants (ecart-type={:.4f}, max={:.4f}). '
            'Ses resultats sont desactives. '
            'Cette anomalie n empeche pas l extraction directe des competences depuis le texte.'.format(
                score_std, score_max
            )
        )
    if score_max < 0.50:
        warnings.append('Aucune categorie IA ne depasse 50%% de probabilite.')
    if score_min > 0.40 and score_max < 0.60:
        warnings.append(
            'Tous les scores sont compris entre {:.2f} et {:.2f}.'.format(score_min, score_max)
        )
    predictions = skills_result.get('predictions', [])
    categories = [
        {'label': p['label'], 'probability': p['probability']}
        for p in predictions if p['probability'] >= 0.35
    ]
    families = skills_result.get('family_groups', [])
    status = 'success' if discriminating and categories else (
        'unreliable' if not discriminating else 'unavailable'
    )
    return IAClassificationInfo(
        status=status,
        categories=categories,
        families=families,
        scores=skills_result.get('all_scores', []),
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
        score_std=score_std,
        discriminating=discriminating,
        warnings=warnings,
        threshold_applied=skills_result.get("threshold_applied", 0.35),
    )


def _build_skill_extraction(text: str) -> SkillExtractionInfo:
    extracted = open_extract_skills(text)
    if not extracted:
        return SkillExtractionInfo(
            status='failed',
            warnings=['Aucune competence extraite du texte avec les regles linguistiques.'],
        )
    skills = []
    tools = []
    knowledge = []
    seen_labels: set[str] = set()
    for e in extracted:
        key = e.normalized_label.lower()
        if key in seen_labels:
            continue
        seen_labels.add(key)
        item = OpenExtractedSkill(
            source_label=e.source_label,
            normalized_label=e.normalized_label,
            type=e.type,
            source_text=e.source_text,
            start=e.start,
            end=e.end,
            confidence=e.confidence,
            method=e.method,
            referential_id=e.referential_id,
            referential_source=e.referential_source,
        )
        if e.type in ('tool', 'tool_with_context'):
            tools.append(item)
        elif e.type == 'knowledge':
            knowledge.append(item)
        else:
            skills.append(item)
    total = len(skills) + len(tools) + len(knowledge)
    status = 'success' if total >= 3 else 'partial'
    return SkillExtractionInfo(status=status, skills=skills, tools=tools,
                                knowledge_items=knowledge)


def _build_analysis_result(
    analysis: dict[str, Any],
    normalized_offers: list[dict[str, Any]],
    recommendation: Any,
    territorial_stats: Any,
    departement: str,
    threshold: float,
    skill_extraction: SkillExtractionInfo | None = None,
    ia_recommendation_records: list[dict[str, Any]] | None = None,
) -> AnalysisResult:
    return build_analysis_result(
        analysis=analysis,
        normalized_offers=normalized_offers,
        recommendation=recommendation,
        territorial_stats=territorial_stats,
        departement=departement,
        threshold=threshold,
        skill_extraction=skill_extraction,
        ia_recommendation_records=ia_recommendation_records,
    )


def normalize_skill_label(label: str) -> str:
    from common.text import normalize_for_match
    return normalize_for_match(label)


def create_app(
    predictor: DeepformaPredictor | None = None,
    cache_ttl_seconds: int | None = None,
    france_travail_client_factory: Any | None = None,
) -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
    app.secret_key = os.getenv('DEEPFORMA_SECRET_KEY', 'deepforma-dev-secret')

    @app.template_filter('sigmoid_pct')
    def _fmt_sigmoid_pct(value):
        """Format a sigmoid probability as percentage (e.g. 0.5234 -> '52.3%')."""
        return f"{float(value) * 100:.1f}%"

    app.config.update(
        CACHE_TTL_SECONDS=cache_ttl_seconds or DEFAULT_CACHE_TTL_SECONDS,
        MAX_OFFERS=DEFAULT_MAX_OFFERS,
        PAGE_SIZE=DEFAULT_PAGE_SIZE,
        MAX_PAGES=DEFAULT_MAX_PAGES,
        DEFAULT_THRESHOLD=DEFAULT_THRESHOLD,
        MAX_CONTENT_LENGTH=DEFAULT_MAX_CONTENT_LENGTH,
        MAX_FORM_MEMORY_SIZE=DEFAULT_MAX_FORM_MEMORY_SIZE,
        MAX_FORM_PARTS=DEFAULT_MAX_FORM_PARTS,
    )

    predictor_error = None
    if predictor is None:
        predictor, predictor_error = _load_predictor()
    app.extensions['deepforma_predictor'] = predictor
    app.extensions['deepforma_predictor_error'] = predictor_error

    ia_recommendation_records: list[dict[str, Any]] = []
    ia_csv_path = PROJECT_ROOT / 'data' / 'raw' / 'recommandations_IA_consolide.csv'
    if ia_csv_path.exists():
        try:
            ia_recommendation_records, _ = load_ia_recommendations_csv(ia_csv_path)
        except Exception:
            logger.warning('Impossible de charger les recommandations IA depuis %s', ia_csv_path)
    app.extensions['ia_recommendation_records'] = ia_recommendation_records
    app.extensions['recommendation_service'] = RecommendationService()
    app.extensions['certification_market_comparator'] = CertificationMarketComparator()
    app.extensions['market_cache'] = TTLCache(app.config['CACHE_TTL_SECONDS'])
    app.extensions['referential_analysis_cache'] = TTLCache(app.config['CACHE_TTL_SECONDS'])
    app.extensions['rome_service'] = RomeService()
    app.extensions['france_travail_client_factory'] = france_travail_client_factory or _build_france_travail_client

    @app.context_processor
    def _inject_csrf_token() -> dict[str, Any]:
        return {'csrf_token': _ensure_csrf_token}

    @app.errorhandler(RequestEntityTooLarge)
    def _handle_request_too_large(exc: RequestEntityTooLarge):
        content_length = request.headers.get('Content-Length', request.content_length or 'unknown')
        logger.warning('[ADMIN_CONTINUAL_LEARNING] %s %s | Content-Length=%s', request.method, request.path, content_length)
        message = 'La requête est trop volumineuse. Réduisez la taille du formulaire et réessayez.'
        if request.is_json:
            return jsonify({'error': message, 'status': 413}), 413
        return Response(f'<h1>413 Request Entity Too Large</h1><p>{message}</p>', 413, {'Content-Type': 'text/html; charset=utf-8'})

    def _require_valid_csrf() -> None:
        _validate_csrf_token()

    def _log_admin_request() -> None:
        _admin_validation_log(_format_request_payload_log(request))

    def get_predictor_instance() -> DeepformaPredictor | None:
        return app.extensions.get('deepforma_predictor')

    def get_market_client() -> FranceTravailClient:
        factory = app.extensions['france_travail_client_factory']
        return factory()

    def get_rome_service() -> RomeService:
        return app.extensions['rome_service']

    def _referential_state_cache() -> TTLCache:
        return app.extensions['referential_analysis_cache']

    def _build_territory_from_form(form: Any, *, fallback_code: str | None = None, fallback_label: str | None = None) -> Territory:
        raw_code = clean_text(form.get('territory_code') or form.get('territory') or form.get('departement') or fallback_code or '')
        raw_label = clean_text(form.get('territory_label') or fallback_label or raw_code or '')
        radius_raw = clean_text(form.get('radius_km') or form.get('distance_km') or '')
        radius: int | None = None
        if radius_raw:
            try:
                radius = int(radius_raw)
            except ValueError:
                radius = None
        territory_type: str | None = None
        department_code: str | None = None
        if raw_code.isdigit() and len(raw_code) == 5:
            territory_type = 'commune'
        elif (raw_code.isdigit() and len(raw_code) in {2, 3}) or raw_code in {'2A', '2B'}:
            territory_type = 'departement'
            department_code = raw_code
        elif raw_code:
            territory_type = 'custom'
        if territory_type == 'departement' and not department_code:
            department_code = raw_code or None
        return Territory(code=raw_code or None, label=raw_label or None, type=territory_type, radius_km=radius, department_code=department_code, region_code=None, remote_allowed=True)

    def _build_referential_state(analysis: dict[str, Any], *, source_path: str, departement: str) -> dict[str, Any]:
        export = build_export_payload(analysis)
        document = analysis['document']
        analysis_id = stable_hash(getattr(document, 'sha256', '') or source_path, getattr(document, 'file_name', '') or '', getattr(document, 'title', '') or '', length=24)
        page_texts: list[str] = []
        try:
            document_loader = load_pdf_document(Path(source_path))
            page_texts = [page.text for page in document_loader.pages if getattr(page, 'text', '')]
        except Exception:
            page_texts = []
        overview = _build_referential_extraction_overview(analysis, _build_referential_keywords_candidates(analysis, page_texts), page_texts=page_texts) if page_texts else {'job_title': getattr(document, 'title', '') or '', 'job_title_source': 'document_metadata', 'keywords': [], 'keywords_source': 'manual', 'competency_labels': [], 'derived_skill_labels': [], 'tool_labels': [], 'criterion_labels': []}
        market_target = {
            'rome_code': '',
            'rome_label': '',
            'selected_rome_occupations': [],
            'territory_code': departement or '',
            'territory_label': departement or '',
            'radius_km': None,
            'contract_type': '',
            'territory_type': 'departement' if departement else None,
        }
        state = {
            'analysis_id': analysis_id,
            'analysis_export': export,
            'source_path': source_path,
            'departement': departement,
            'overview': overview,
            'analysis': analysis,
            'market_target': market_target,
            'rome_candidates': [],
            'rome_query': overview.get('job_title') or getattr(document, 'title', '') or '',
            'market_target_confirmed': False,
            'analysis_status': 'PDF_ANALYZED',
            'market_search_status': 'WAITING_FOR_ROME',
        }
        _referential_state_cache().set(analysis_id, state)
        return state

    def _load_referential_state(analysis_id: str | None, analysis_json: str | None = None) -> dict[str, Any]:
        if analysis_id:
            cached = _referential_state_cache().get(analysis_id)
            if cached is not None:
                return cached
        if analysis_json:
            export = json.loads(analysis_json)
            analysis = analysis_from_export(export)
            source_path = getattr(analysis['document'], 'source_path', '') or ''
            departement = clean_text(getattr(analysis['document'], 'department', '') or '')
            state = _build_referential_state(analysis, source_path=source_path, departement=departement)
            if analysis_id:
                state['analysis_id'] = analysis_id
                _referential_state_cache().set(analysis_id, state)
            return state
        raise ValueError('Analyse du référentiel manquante.')

    def _offer_matches_focus_terms(offer: dict[str, Any], focus_terms: list[str]) -> bool:
        if not focus_terms:
            return True
        title = normalize_for_match(clean_text(offer.get('title') or ''))
        description = normalize_for_match(clean_text(offer.get('description') or ''))
        normalized_skills = {normalize_for_match(clean_text(label)) for label in (offer.get('normalized_skills') or []) if clean_text(label)}
        structured_skills = set()
        for item in offer.get('structured_skills') or []:
            if isinstance(item, dict):
                candidate = item.get('canonical_label') or item.get('canonical_name') or item.get('label')
            else:
                candidate = item
            candidate = clean_text(candidate)
            if candidate:
                structured_skills.add(normalize_for_match(candidate))

        haystacks = [title, description, *normalized_skills, *structured_skills]
        haystack_tokens = set()
        for item in haystacks:
            if item:
                haystack_tokens.update(item.split())

        for term in focus_terms:
            normalized_term = normalize_for_match(term)
            if not normalized_term:
                continue
            if any(normalized_term in haystack for haystack in haystacks if haystack):
                return True
            term_tokens = set(normalized_term.split())
            if term_tokens and term_tokens.issubset(haystack_tokens):
                return True
        return False

    def analyze_market(departement: str, keywords: str | None, focus_terms: list[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cache = app.extensions['market_cache']
        focus_signature = '::'.join(sorted({normalize_for_match(term) for term in (focus_terms or []) if normalize_for_match(term)}))
        cache_key = _make_cache_key(departement, keywords) + (f'::{focus_signature}' if focus_signature else '')
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        client = get_market_client()
        criteria = SearchCriteria(
            departement=departement,
            keywords=keywords or None,
            size=app.config['PAGE_SIZE'],
        )
        normalized_offers: list[dict[str, Any]] = []
        raw_offers: list[dict[str, Any]] = []

        for offer in client.iter_offers(
            criteria,
            max_pages=app.config['MAX_PAGES'],
            max_offers=app.config['MAX_OFFERS'],
            page_size=app.config['PAGE_SIZE'],
        ):
            raw_offers.append(offer)
            normalized = normalize_offer(offer)
            explicit_skills = extract_skills_from_text(normalized.offer_text)
            merged_skills = merge_offer_skills(
                structured_skills=normalized.structured_skills,
                explicit_skills=explicit_skills,
                model_skills=normalized.model_skills,
                rome_skills=[],
            )
            normalized_dict = normalized.to_dict()
            normalized_dict['merged_skills'] = merged_skills
            normalized_dict['normalized_skills'] = [item['canonical_label'] for item in merged_skills]
            if _offer_matches_focus_terms(normalized_dict, focus_terms or ([] if keywords is None else [keywords])):
                normalized_offers.append(normalized_dict)

        result = {
            'raw_offers': raw_offers,
            'normalized_offers': normalized_offers,
        }
        cache.set(cache_key, (normalized_offers, result))
        return normalized_offers, result

    def _parse_request_payload() -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if request.is_json:
            incoming = request.get_json(silent=True) or {}
            if isinstance(incoming, dict):
                payload.update(incoming)
        payload.update(request.form.to_dict(flat=True))
        return payload

    def _extract_inputs(payload: dict[str, Any]) -> tuple[str, str, str | None, float, bool]:
        text = clean_text(payload.get('program') or payload.get('programme') or payload.get('text') or '')
        departement = clean_text(payload.get('departement') or payload.get('department') or '')
        keywords = clean_text(payload.get('keywords') or '') or None
        threshold_raw = payload.get('threshold')
        model_only = str(payload.get('model_only') or payload.get('skip_market') or '').lower() in {'1', 'true', 'yes', 'on'}
        threshold = app.config['DEFAULT_THRESHOLD']
        if threshold_raw not in (None, ''):
            try:
                threshold = float(threshold_raw)
            except (TypeError, ValueError):
                raise ValueError('Le seuil doit etre un nombre compris entre 0 et 1.')
        if not 0.0 <= threshold <= 1.0:
            raise ValueError('Le seuil doit etre compris entre 0 et 1.')
        if not text:
            raise ValueError('Le programme de formation est obligatoire.')
        if not departement:
            raise ValueError('Le departement est obligatoire.')
        return text, departement, keywords, threshold, model_only


    def _extract_referential_inputs(payload: dict[str, Any], files: Any) -> tuple[Path, str]:
        departement = clean_text(payload.get('departement') or payload.get('department') or '')
        uploaded = files.get('pdf') if files is not None else None
        if not uploaded or not getattr(uploaded, 'filename', ''):
            raise ValueError('Le PDF du référentiel est obligatoire.')
        if not departement:
            raise ValueError('Le departement est obligatoire.')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = Path(temp_file.name)
        temp_file.close()
        uploaded.save(temp_path)
        return temp_path, departement

    def _normalize_referential_title_candidate(text: str) -> str:
        candidate = clean_text(text)
        if not candidate:
            return ''
        marker_re = re.compile(
            r"(?:référentiel|referentiel|modalités d['’]?évaluation|modalites d evaluation|critères d['’]?évaluation|criteres d evaluation)",
            flags=re.IGNORECASE,
        )
        match = marker_re.search(candidate)
        if match:
            candidate = candidate[: match.start()].strip()
        candidate = re.sub(
            r"^(?:référentiel|referentiel)(?:\s+(?:d['’]?activit(?:é|e)s?|de\s+compétences?|de\s+competences?|d['’]?évaluation|d['’]?evaluation))?(?:\s*[-:–—]\s*)?",
            '',
            candidate,
            flags=re.IGNORECASE,
        ).strip()
        candidate = re.sub(r"^(?:référentiel|referentiel)\s+", '', candidate, flags=re.IGNORECASE).strip()
        candidate = candidate.strip(' -:–—')
        if not candidate:
            return ''
        if candidate.isupper() or sum(1 for char in candidate if char.isupper()) >= max(2, len(candidate) // 2):
            candidate = candidate.lower()
            candidate = candidate[:1].upper() + candidate[1:] if candidate else candidate
        return candidate

    def _infer_referential_title(page_texts: list[str], analysis: dict[str, Any]) -> str:
        document = analysis.get('document')
        document_title = _normalize_referential_title_candidate(getattr(document, 'title', '') or '') if document is not None else ''
        if document_title:
            return document_title

        title_re = re.compile(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’&(),./ -]{8,}$")
        header_markers = (
            "référentiel d'activités",
            'référentiel de compétences',
            'modalités d’évaluation',
            'modalites d evaluation',
            'critères d’évaluation',
            'criteres d evaluation',
        )
        code_markers = (
            'bloc ',
            'activité ',
            'activite ',
            'a1.',
            'c1.',
            'ce1.',
        )
        for page_text in page_texts[:2]:
            first_fallback = ''
            for raw_line in page_text.splitlines():
                line = clean_text(raw_line)
                if not line:
                    continue
                if not first_fallback:
                    first_fallback = line
                normalized = line.lower()
                if any(marker in normalized for marker in header_markers):
                    parts = re.split(r"(?:référentiel d'activités|référentiel de compétences|modalités d’évaluation|modalites d evaluation|critères d’évaluation|criteres d evaluation)", line, maxsplit=1, flags=re.IGNORECASE)
                    prefix = _normalize_referential_title_candidate(parts[0])
                    suffix = _normalize_referential_title_candidate(parts[1]) if len(parts) > 1 else ''
                    for candidate in (prefix, suffix):
                        if candidate and len(candidate.split()) >= 2 and title_re.match(candidate):
                            return candidate
                    continue
                if any(normalized.startswith(marker) for marker in code_markers):
                    continue
                normalized_line = _normalize_referential_title_candidate(line)
                if line.isupper() and len(line) >= 10 and normalized_line:
                    return normalized_line
                if title_re.match(normalized_line) and len(normalized_line.split()) >= 2:
                    return normalized_line
            if first_fallback:
                normalized_fallback = _normalize_referential_title_candidate(first_fallback)
                if normalized_fallback and len(normalized_fallback.split()) >= 2:
                    return normalized_fallback
        return ''

    def _build_referential_keywords_candidates(analysis: dict[str, Any], page_texts: list[str]) -> list[str]:
        title = _infer_referential_title(page_texts, analysis)
        if not title:
            document = analysis.get('document')
            title = clean_text(getattr(document, 'title', '') or '') if document is not None else ''

        candidates: list[str] = []
        seen: set[str] = set()

        def add(label: str) -> None:
            normalized = clean_text(label)
            if not normalized:
                return
            normalized = re.sub(r'^référentiel\s+', '', normalized, flags=re.IGNORECASE).strip()
            normalized = re.sub(r'^referentiel\s+', '', normalized, flags=re.IGNORECASE).strip()
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            candidates.append(normalized)

        add(title)
        if title:
            stripped = re.sub(r'\(.*?\)', '', title).strip()
            if stripped:
                add(stripped)
        return candidates[:3]

    def _build_referential_extraction_overview(analysis: dict[str, Any], keyword_candidates: list[str], *, page_texts: list[str]) -> dict[str, Any]:
        competencies = referential_editing_service.get_active_competencies(analysis)
        criteria = analysis.get('criteria', [])
        derived_skills = analysis.get('derived_skills', [])
        tools_methods = analysis.get('tools_methods', [])
        inferred_title = _infer_referential_title(page_texts, analysis)

        return {
            'job_title': inferred_title,
            'job_title_source': 'first_page_text' if inferred_title else 'document_metadata',
            'keywords': [candidate for candidate in keyword_candidates if candidate],
            'keywords_source': 'job_title_for_france_travail',
            'competency_labels': [clean_text(getattr(item, 'official_label', '') or '') for item in competencies if clean_text(getattr(item, 'official_label', '') or '')],
            'derived_skill_labels': [clean_text(getattr(item, 'canonical_label', '') or getattr(item, 'label', '') or '') for item in derived_skills if clean_text(getattr(item, 'canonical_label', '') or getattr(item, 'label', '') or '')],
            'tool_labels': [clean_text(getattr(item, 'canonical_label', '') or getattr(item, 'label', '') or '') for item in tools_methods if clean_text(getattr(item, 'canonical_label', '') or getattr(item, 'label', '') or '')],
            'criterion_labels': [clean_text(getattr(item, 'criterion_label', '') or '') for item in criteria if clean_text(getattr(item, 'criterion_label', '') or '')],
        }

    def _market_offer_score(offer: dict[str, Any], focus_labels: list[str]) -> tuple[int, int, int, int, str]:
        focus = {normalize_for_match(label) for label in focus_labels if normalize_for_match(label)}
        focus_tokens = {token for label in focus for token in label.split() if token}
        normalized_skills = offer.get('normalized_skills') or []
        structured_skills = offer.get('structured_skills') or []
        skill_labels: list[str] = []
        for label in normalized_skills:
            skill_labels.append(clean_text(label))
        for item in structured_skills:
            if isinstance(item, dict):
                candidate = item.get('canonical_label') or item.get('canonical_name') or item.get('label')
            else:
                candidate = item
            candidate = clean_text(candidate)
            if candidate:
                skill_labels.append(candidate)
        title = clean_text(offer.get('title') or '')
        title_key = normalize_for_match(title)
        title_tokens = set(title_key.split())
        skill_tokens = set()
        for label in skill_labels:
            skill_tokens.update(normalize_for_match(label).split())
        token_overlap = len(skill_tokens & focus_tokens)
        exact_count = sum(1 for label in normalized_skills if normalize_for_match(label) in focus)
        structured_count = sum(1 for item in structured_skills if normalize_for_match((item.get('canonical_label') or item.get('canonical_name') or item.get('label')) if isinstance(item, dict) else item) in focus)
        title_match = len(title_tokens & focus_tokens)
        return (token_overlap, exact_count, structured_count, title_match, title_key)

    def _build_context(text: str, departement: str, keywords: str | None, threshold: float, model_only: bool, allow_market_failure: bool = False, market_keyword_candidates: list[str | None] | None = None) -> dict[str, Any]:
        predictor_instance = get_predictor_instance()
        if predictor_instance is None:
            raise RuntimeError(app.extensions.get('deepforma_predictor_error') or 'Les modeles ne sont pas disponibles.')

        # 1. Run the model (binary + 18-label IA classifier)
        analysis = predictor_instance.analyze(text, threshold=threshold)

        # 2. Run the open extractor (primary skill extraction)
        skill_extraction = _build_skill_extraction(text)

        normalized_offers: list[dict[str, Any]] = []
        recommendation = None
        territorial_stats = None
        market_error = None
        market_status = 'skipped' if model_only else 'not_requested'
        market_offers_used: list[dict[str, Any]] = []
        market_offers_preview: list[dict[str, Any]] = []
        market_offers_more_count = 0
        market_analysis = None

        if not model_only:
            keyword_candidates = market_keyword_candidates or ([keywords] if keywords else [None])
            for candidate in keyword_candidates:
                try:
                    normalized_offers, _ = analyze_market(
                        departement,
                        candidate,
                        focus_terms=(
                            [candidate]
                            if candidate
                            else [
                                *[item.normalized_label for item in skill_extraction.skills if getattr(item, 'normalized_label', '')],
                                *[item.normalized_label for item in skill_extraction.tools if getattr(item, 'normalized_label', '')],
                            ]
                        ),
                    )
                except ValueError as exc:
                    if not allow_market_failure:
                        raise RuntimeError('Configuration France Travail absente ou invalide.') from exc
                    market_error = 'Configuration France Travail absente ou invalide.'
                    market_status = 'error'
                    break
                except FranceTravailRateLimitError as exc:
                    if not allow_market_failure:
                        raise RuntimeError('France Travail a repondu avec une limite de debit (429).') from exc
                    market_error = 'France Travail a repondu avec une limite de debit (429).'
                    market_status = 'error'
                    break
                except FranceTravailTimeoutError as exc:
                    if not allow_market_failure:
                        raise RuntimeError("Le delai d'attente France Travail a expire.") from exc
                    market_error = "Le delai d'attente France Travail a expire."
                    market_status = 'error'
                    break
                except FranceTravailAuthError as exc:
                    if not allow_market_failure:
                        raise RuntimeError('Authentification France Travail invalide ou expiree.') from exc
                    market_error = 'Authentification France Travail invalide ou expiree.'
                    market_status = 'error'
                    break
                except FranceTravailError as exc:
                    message = str(exc)
                    if '429' in message:
                        if not allow_market_failure:
                            raise RuntimeError('France Travail a repondu avec une limite de debit (429).') from exc
                        market_error = 'France Travail a repondu avec une limite de debit (429).'
                    else:
                        if not allow_market_failure:
                            raise RuntimeError('Erreur France Travail lors de la recuperation des offres.') from exc
                        market_error = 'Erreur France Travail lors de la recuperation des offres.'
                    market_status = 'error'
                    break
                except requests.Timeout as exc:
                    if not allow_market_failure:
                        raise RuntimeError("Le delai d'attente France Travail a expire.") from exc
                    market_error = "Le delai d'attente France Travail a expire."
                    market_status = 'error'
                    break
                except requests.RequestException as exc:
                    if not allow_market_failure:
                        raise RuntimeError("Erreur reseau lors de l'appel a France Travail.") from exc
                    market_error = "Erreur reseau lors de l'appel a France Travail."
                    market_status = 'error'
                    break

                if not normalized_offers:
                    continue

                service: RecommendationService = app.extensions['recommendation_service']
                # Comparison uses open-extracted skills (normalized), NOT the 18 sigmoid outputs
                market_context = build_market_context(
                    skill_extraction=skill_extraction,
                    normalized_offers=normalized_offers,
                    departement=departement,
                    recommendation_service=service,
                )
                recommendation = market_context['recommendation']
                market_analysis = market_context['market_analysis']
                territorial_stats = market_context['territorial_stats']
                market_status = 'ok'
                market_offers_used = normalized_offers
                market_offers_preview = market_context['market_offers_preview']
                market_offers_more_count = market_context['market_offers_more_count']
                break

            if not normalized_offers and market_error is None:
                market_status = 'empty'

        analysis_result = _build_analysis_result(
            analysis, normalized_offers, recommendation, territorial_stats,
            departement, threshold, skill_extraction=skill_extraction,
            ia_recommendation_records=app.extensions.get('ia_recommendation_records'),
        )

        return {
            'analysis': analysis,
            'context': {
                'normalized_offers': normalized_offers,
                'market_offers_used': market_offers_used,
                'market_offers_preview': market_offers_preview,
                'market_offers_more_count': market_offers_more_count,
                'territorial_stats': territorial_stats,
                'recommendation': recommendation,
                'market_analysis': market_analysis,
                'market_status': market_status,
                'market_error': market_error,
            },
            'analysis_result': analysis_result,
            'department': departement,
            'keywords': keywords,
            'threshold': threshold,
            'model_only': model_only,
            'warning': EXPERIMENTAL_WARNING,
        }

    def _render_error(message: str, status_code: int = 400):
        if request.path.startswith('/api/'):
            return jsonify({'ok': False, 'error': message}), status_code
        return render_template('index.html', error=message, department_options=DEPARTMENT_CODES, default_threshold=app.config['DEFAULT_THRESHOLD']), status_code

    def _render_home_page(*, error: str | None = None, referential_analysis: dict[str, Any] | None = None, referential_validation: dict[str, Any] | None = None, referential_error: str | None = None, referential_success: str | None = None):
        return render_template(
            'index.html',
            error=error,
            department_options=DEPARTMENT_CODES,
            default_threshold=app.config['DEFAULT_THRESHOLD'],
            referential_analysis=referential_analysis,
            referential_validation=referential_validation,
            referential_error=referential_error,
            referential_success=referential_success,
        )

    def _build_referential_preview_payload(analysis: dict[str, Any], *, source_path: str) -> dict[str, Any]:
        return {
            'document': analysis['document'],
            'report': analysis['report'],
            'blocks': analysis['blocks'],
            'activities': analysis['activities'],
            'competencies': analysis['competencies'],
            'criteria': analysis['criteria'],
            'derived_skills': analysis['derived_skills'],
            'tools_methods': analysis['tools_methods'],
            'analysis_json': json.dumps(build_export_payload(analysis), ensure_ascii=False, indent=2),
            'source_path': source_path,
        }

    referential_editing_service = ReferentialEditingService()

    def _apply_referential_import_edits(analysis: dict[str, Any], form: Any) -> dict[str, Any]:
        return referential_editing_service.apply_edits(analysis, form)

    def _render_referential_validation(analysis: dict[str, Any], *, source_path: str, departement: str, analysis_id: str | None = None, referential_success: str | None = None, referential_error: str | None = None, rome_query: str | None = None, rome_candidates: list[dict[str, Any]] | None = None, market_target: dict[str, Any] | None = None, market_target_confirmed: bool = False, selected_rome_occupations: list[dict[str, Any]] | None = None):
        preview = _build_referential_preview_payload(analysis, source_path=source_path)
        preview['analysis_id'] = analysis_id or stable_hash(getattr(analysis['document'], 'sha256', '') or source_path, getattr(analysis['document'], 'file_name', '') or '', getattr(analysis['document'], 'title', '') or '', length=24)
        return _render_home_page(
            referential_analysis=preview,
            referential_validation={
                'analysis_id': preview['analysis_id'],
                'analysis_json': preview['analysis_json'],
                'source_path': source_path,
                'departement': departement,
                'validated_title': getattr(analysis['document'], 'title', '') or '',
                'competencies': [
                    {
                        'code': competency.code,
                        'official_label': competency.official_label,
                        'review_status': competency.review_status,
                        'block_code': competency.block_code,
                        'activity_code': competency.activity_code,
                    }
                    for competency in analysis.get('competencies', [])
                ],
                'derived_skills': [
                    {
                        'label': getattr(s, 'label', '') or '',
                        'canonical_label': getattr(s, 'canonical_label', '') or '',
                        'category': getattr(s, 'category', '') or '',
                        'source_code': getattr(s, 'source_code', '') or '',
                        'source_type': getattr(s, 'source_type', '') or '',
                        'confidence': getattr(s, 'confidence', 0.0) or 0.0,
                        'review_status': getattr(s, 'review_status', 'pending') or 'pending',
                    }
                    for s in analysis.get('derived_skills', [])
                ],
                'criteria': [
                    {
                        'code': c.code,
                        'competency_code': c.competency_code,
                        'criterion_label': c.criterion_label,
                        'review_status': c.review_status,
                        'provenance': c.provenance,
                    }
                    for c in analysis.get('criteria', [])
                ],
                'rome_query': rome_query or getattr(analysis['document'], 'title', '') or '',
                'rome_candidates': rome_candidates or [],
                'market_target': market_target or {},
                'selected_rome_occupations': selected_rome_occupations or (market_target or {}).get('selected_rome_occupations', []),
                'market_target_confirmed': market_target_confirmed,
            },
            referential_success=referential_success,
            referential_error=referential_error,
        )

    @app.get('/')
    def index():
        return _render_home_page()

    @app.post('/referential/import')
    def referential_import_preview():
        uploaded = request.files.get('pdf')
        if not uploaded or not uploaded.filename:
            return _render_home_page(referential_error='Fichier PDF manquant.'), 400
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = Path(temp_file.name)
        temp_file.close()
        uploaded.save(temp_path)
        departement = clean_text(request.form.get('departement') or '')
        analysis = referential_import_service.analyze(temp_path)
        state = _build_referential_state(analysis, source_path=str(temp_path), departement=departement)
        return redirect(url_for('referential_import_editor', analysis_id=state['analysis_id']))

    @app.get('/referential/import/<analysis_id>/edit')
    def referential_import_editor(analysis_id: str):
        cached = _referential_state_cache().get(analysis_id)
        if cached is None:
            return _render_home_page(referential_error='Session expiree. Veuillez re-importer le PDF.'), 404
        return _render_referential_validation(
            cached['analysis'],
            source_path=cached['source_path'],
            departement=cached['departement'],
            analysis_id=analysis_id,
            rome_query=cached.get('rome_query'),
            rome_candidates=cached.get('rome_candidates') or [],
            market_target=cached.get('market_target') or {},
            market_target_confirmed=cached.get('market_target_confirmed', False),
            selected_rome_occupations=cached.get('selected_rome_occupations') or cached.get('market_target', {}).get('selected_rome_occupations', []),
        )

    @app.post('/referential/import/<analysis_id>/skills/<skill_code>/reject')
    def referential_import_reject_skill(analysis_id: str, skill_code: str):
        cached = _referential_state_cache().get(analysis_id)
        if cached is None:
            return jsonify({'ok': False, 'error': 'Session expiree.'}), 404
        analysis = cached['analysis']
        analysis = referential_editing_service.reject_skill(analysis, skill_code)
        analysis['export'] = build_export_payload(analysis)
        cached['analysis'] = analysis
        _referential_state_cache().set(analysis_id, cached)
        return redirect(url_for('referential_import_editor', analysis_id=analysis_id))

    @app.post('/referential/import/<analysis_id>/skills/<skill_code>/restore')
    def referential_import_restore_skill(analysis_id: str, skill_code: str):
        cached = _referential_state_cache().get(analysis_id)
        if cached is None:
            return jsonify({'ok': False, 'error': 'Session expiree.'}), 404
        analysis = cached['analysis']
        analysis = referential_editing_service.restore_skill(analysis, skill_code)
        analysis['export'] = build_export_payload(analysis)
        cached['analysis'] = analysis
        _referential_state_cache().set(analysis_id, cached)
        return redirect(url_for('referential_import_editor', analysis_id=analysis_id))

    @app.post('/referential/import/<analysis_id>/skills/add')
    def referential_import_add_skill(analysis_id: str):
        cached = _referential_state_cache().get(analysis_id)
        if cached is None:
            return jsonify({'ok': False, 'error': 'Session expiree.'}), 404
        label = clean_text(request.form.get('label') or request.form.get('new_competency_labels') or '')
        if not label:
            return redirect(url_for('referential_import_editor', analysis_id=analysis_id))
        analysis = cached['analysis']
        analysis = referential_editing_service.add_skill(analysis, label)
        analysis['export'] = build_export_payload(analysis)
        cached['analysis'] = analysis
        _referential_state_cache().set(analysis_id, cached)
        return redirect(url_for('referential_import_editor', analysis_id=analysis_id))

    @app.post('/referential/import/<analysis_id>/update')
    def referential_import_update(analysis_id: str):
        cached = _referential_state_cache().get(analysis_id)
        if cached is None:
            return jsonify({'ok': False, 'error': 'Session expiree.'}), 404
        analysis = cached['analysis']
        analysis = referential_editing_service.apply_edits(analysis, request.form)
        analysis['export'] = build_export_payload(analysis)
        cached['analysis'] = analysis
        _referential_state_cache().set(analysis_id, cached)
        return redirect(url_for('referential_import_editor', analysis_id=analysis_id))

    @app.post('/referential/import/<analysis_id>/derived/<int:skill_index>/reject')
    def referential_import_reject_derived_skill(analysis_id: str, skill_index: int):
        cached = _referential_state_cache().get(analysis_id)
        if cached is None:
            return jsonify({'ok': False, 'error': 'Session expiree.'}), 404
        analysis = cached['analysis']
        analysis = referential_editing_service.reject_derived_skill(analysis, skill_index)
        analysis['export'] = build_export_payload(analysis)
        cached['analysis'] = analysis
        _referential_state_cache().set(analysis_id, cached)
        return redirect(url_for('referential_import_editor', analysis_id=analysis_id))

    @app.post('/referential/import/<analysis_id>/derived/<int:skill_index>/restore')
    def referential_import_restore_derived_skill(analysis_id: str, skill_index: int):
        cached = _referential_state_cache().get(analysis_id)
        if cached is None:
            return jsonify({'ok': False, 'error': 'Session expiree.'}), 404
        analysis = cached['analysis']
        analysis = referential_editing_service.restore_derived_skill(analysis, skill_index)
        analysis['export'] = build_export_payload(analysis)
        cached['analysis'] = analysis
        _referential_state_cache().set(analysis_id, cached)
        return redirect(url_for('referential_import_editor', analysis_id=analysis_id))

    @app.post('/referential/import/<analysis_id>/derived/add')
    def referential_import_add_derived_skill(analysis_id: str):
        cached = _referential_state_cache().get(analysis_id)
        if cached is None:
            return jsonify({'ok': False, 'error': 'Session expiree.'}), 404
        label = clean_text(request.form.get('label') or '')
        category = clean_text(request.form.get('category') or 'skill')
        canonical = clean_text(request.form.get('canonical') or '')
        if not label:
            return redirect(url_for('referential_import_editor', analysis_id=analysis_id))
        analysis = cached['analysis']
        analysis = referential_editing_service.add_derived_skill(analysis, label, category, canonical)
        analysis['export'] = build_export_payload(analysis)
        cached['analysis'] = analysis
        _referential_state_cache().set(analysis_id, cached)
        return redirect(url_for('referential_import_editor', analysis_id=analysis_id))

    def _apply_referential_validation_edits(analysis: dict[str, Any], form: Any) -> dict[str, Any]:
        return _apply_referential_import_edits(analysis, form)

    def _build_referential_market_context(
        referential_analysis: dict[str, Any],
        *,
        source_path: str,
        departement: str,
        selected_rome_code: str,
        selected_rome_label: str | None = None,
        selected_rome_codes: list[str] | None = None,
        territory: Territory,
        contract_type: str | None = None,
        radius_km: int | None = None,
    ) -> dict[str, Any]:
        document = referential_analysis['document']
        page_texts: list[str] = []
        try:
            document_loader = load_pdf_document(Path(source_path))
            page_texts = [page.text for page in document_loader.pages if getattr(page, 'text', '')]
        except Exception:
            page_texts = []
        overview = _build_referential_extraction_overview(referential_analysis, _build_referential_keywords_candidates(referential_analysis, page_texts), page_texts=page_texts) if page_texts else {'job_title': getattr(document, 'title', '') or '', 'job_title_source': 'document_metadata', 'keywords': [], 'keywords_source': 'manual', 'competency_labels': [], 'derived_skill_labels': [], 'tool_labels': [], 'criterion_labels': []}
        market_territory = territory or Territory(code=departement or None, label=departement or None, department_code=departement or None, region_code=None, remote_allowed=True)
        requested_codes = selected_rome_codes or ([selected_rome_code] if selected_rome_code else [])
        validated_codes = validate_rome_codes(requested_codes, max_codes=MAX_ROME_CODES_PER_SEARCH, service=get_rome_service())
        selected_occupations = []
        for code in validated_codes:
            rome_job = get_rome_service().get(code)
            selected_occupations.append({
                'code': code,
                'label': rome_job.label if rome_job and rome_job.label else code,
                'alternative_labels': list(getattr(rome_job, 'alternative_titles', []) or []),
                'domain': getattr(rome_job, 'domain', None),
            })
        multi_result = fetch_offers_by_rome_codes(
            validated_codes,
            market_territory,
            radius_km=radius_km,
            contract_types=[contract_type] if contract_type else None,
            max_results_per_code=app.config['MAX_OFFERS'],
            client=get_market_client(),
        )
        normalized_offers = list(multi_result.offers)
        service: RecommendationService = app.extensions['recommendation_service']
        market_context = build_market_context(
            skill_extraction=_build_skill_extraction(' '.join(page_texts) if page_texts else getattr(document, 'title', '') or ''),
            normalized_offers=normalized_offers,
            departement=departement,
            recommendation_service=service,
        )
        recommendation = market_context['recommendation']
        market_analysis = market_context['market_analysis']
        territorial_stats = market_context['territorial_stats']
        market_status = 'ok' if normalized_offers else ('empty' if not multi_result.warnings else 'error')
        market_offers_used = normalized_offers
        market_offers_preview = market_context['market_offers_preview']
        market_offers_more_count = market_context['market_offers_more_count']
        raw_count = sum(stat.raw_count for stat in multi_result.stats_by_rome)
        accepted_count = len(normalized_offers)
        rejected_count = len(multi_result.rejected_offers)
        market_error = None
        if raw_count > 0 and accepted_count == 0:
            market_error = 'Le code ROME confirme est valide, mais aucune offre ne correspond exactement au code demandé.'
        elif raw_count == 0 and validated_codes:
            market_error = 'Aucune offre France Travail ne correspond aux codes ROME confirmés.'
        rejected_reasons = dict(Counter(item.reason for item in multi_result.rejected_offers))
        rome_distribution = {
            code: sum(
                1
                for offer in normalized_offers
                if code in set(offer.get('matched_requested_rome_codes') or [])
            )
            for code in validated_codes
        }
        market_offer_audit = {
            'raw_count': raw_count,
            'accepted_count': accepted_count,
            'rejected_count': rejected_count,
            'rejected_reasons': rejected_reasons,
            'rejections': [serialize_record(item) for item in multi_result.rejected_offers],
            'stats_by_rome': [serialize_record(item) for item in multi_result.stats_by_rome],
            'rome_distribution': rome_distribution,
        }
        context = {
            'analysis': referential_analysis,
            'context': {
                'normalized_offers': normalized_offers,
                'market_offers_used': market_offers_used,
                'market_offers_preview': market_offers_preview,
                'market_offers_more_count': market_offers_more_count,
                'territorial_stats': territorial_stats,
                'recommendation': recommendation,
                'market_analysis': market_analysis,
                'market_status': market_status,
                'market_error': market_error,
                'market_offer_audit': market_offer_audit,
                'market_raw_count': raw_count,
                'market_accepted_count': accepted_count,
                'market_rejected_count': rejected_count,
                'market_rejection_reasons': rejected_reasons,
                'market_rejections': [serialize_record(item) for item in multi_result.rejected_offers],
                'market_stats_by_rome': [serialize_record(item) for item in multi_result.stats_by_rome],
            },
            'analysis_result': _build_analysis_result(
                _build_context(' '.join(page_texts) if page_texts else getattr(document, 'title', '') or '', departement, None, app.config['DEFAULT_THRESHOLD'], True, allow_market_failure=True)['analysis'],
                normalized_offers,
                recommendation,
                territorial_stats,
                departement,
                app.config['DEFAULT_THRESHOLD'],
                skill_extraction=_build_skill_extraction(' '.join(page_texts) if page_texts else getattr(document, 'title', '') or ''),
                ia_recommendation_records=app.extensions.get('ia_recommendation_records'),
            ),
            'department': departement,
            'keywords': selected_rome_label or (selected_occupations[0]['label'] if selected_occupations else selected_rome_code),
            'threshold': app.config['DEFAULT_THRESHOLD'],
            'model_only': False,
            'warning': EXPERIMENTAL_WARNING,
            'referential_analysis': referential_analysis,
            'referential_overview': overview,
            'market_target': {
                'rome_code': selected_rome_code if len(validated_codes) == 1 else (validated_codes[0] if validated_codes else ''),
                'rome_label': selected_rome_label or (selected_occupations[0]['label'] if selected_occupations else ''),
                'selected_rome_codes': validated_codes,
                'selected_rome_occupations': selected_occupations,
                'territory_code': market_territory.code,
                'territory_label': market_territory.label,
                'radius_km': market_territory.radius_km,
                'contract_type': contract_type,
            },
            'selected_rome_occupations': selected_occupations,
        }
        if market_error:
            context['warning'] = context['warning'] + ' Analyse territoriale partielle: ' + market_error
        return context

    @app.post('/analyze')
    def analyze():
        try:
            payload = _parse_request_payload()
            action = clean_text(payload.get('action'))
            if action == 'validate_referential':
                analysis_id = clean_text(payload.get('analysis_id') or '')
                cached = _referential_state_cache().get(analysis_id) if analysis_id else None
                if cached is not None:
                    referential_analysis = cached['analysis']
                    source_path = cached.get('source_path') or clean_text(payload.get('source_path') or referential_analysis['document'].source_path)
                    departement = cached.get('departement') or clean_text(payload.get('departement') or '')
                else:
                    export_raw = payload.get('analysis_json')
                    if not export_raw:
                        raise ValueError('Analyse du référentiel manquante.')
                    try:
                        export = json.loads(export_raw)
                    except Exception:
                        raise ValueError('JSON du référentiel invalide.')
                    referential_analysis = analysis_from_export(export)
                    source_path = clean_text(payload.get('source_path') or referential_analysis['document'].source_path)
                    departement = clean_text(payload.get('departement') or '')
                referential_analysis = _apply_referential_validation_edits(referential_analysis, payload)
                referential_analysis['export'] = build_export_payload(referential_analysis)
                validated_by = clean_text(payload.get('validated_by') or 'human_review') or 'human_review'
                referential_import_service.approve(referential_analysis, validated_by=validated_by)
                if not departement:
                    raise ValueError('Le departement est obligatoire.')
                state = _build_referential_state(referential_analysis, source_path=source_path, departement=departement)
                context = {'analysis': referential_analysis, 'state': state, 'analysis_result': None}
                return render_template(
                    'index.html',
                    department_options=DEPARTMENT_CODES,
                    default_threshold=app.config['DEFAULT_THRESHOLD'],
                    referential_analysis=_build_referential_preview_payload(referential_analysis, source_path=source_path) | {'analysis_id': state['analysis_id']},
                    referential_validation={
                        'analysis_id': state['analysis_id'],
                        'analysis_json': json.dumps(state['analysis_export'], ensure_ascii=False, indent=2),
                        'source_path': source_path,
                        'departement': departement,
                        'validated_title': getattr(referential_analysis['document'], 'title', '') or '',
                        'competencies': [
                            {
                                'code': competency.code,
                                'official_label': competency.official_label,
                                'review_status': competency.review_status,
                                'block_code': competency.block_code,
                                'activity_code': competency.activity_code,
                            }
                            for competency in referential_analysis.get('competencies', [])
                        ],
                        'derived_skills': [
                            {
                                'label': getattr(s, 'label', '') or '',
                                'canonical_label': getattr(s, 'canonical_label', '') or '',
                                'category': getattr(s, 'category', '') or '',
                                'source_code': getattr(s, 'source_code', '') or '',
                                'source_type': getattr(s, 'source_type', '') or '',
                                'confidence': getattr(s, 'confidence', 0.0) or 0.0,
                                'review_status': getattr(s, 'review_status', 'pending') or 'pending',
                            }
                            for s in referential_analysis.get('derived_skills', [])
                        ],
                        'criteria': [
                            {
                                'code': c.code,
                                'competency_code': c.competency_code,
                                'criterion_label': c.criterion_label,
                                'review_status': c.review_status,
                                'provenance': c.provenance,
                            }
                            for c in referential_analysis.get('criteria', [])
                        ],
                        'rome_query': state['rome_query'],
                        'rome_candidates': [],
                        'market_target': {},
                        'market_target_confirmed': False,
                    },
                    referential_success='Le référentiel a été validé. Vous pouvez rechercher un métier cible.',
                )
            if action == 'search_rome_candidates':
                state = _load_referential_state(clean_text(payload.get('analysis_id') or ''), payload.get('analysis_json'))
                referential_analysis = state['analysis']
                query = clean_text(payload.get('rome_query') or state.get('rome_query') or '')
                candidates = [job.to_dict() for job in get_rome_service().search(query, limit=10)]
                state['rome_query'] = query
                state['rome_candidates'] = candidates
                _referential_state_cache().set(state['analysis_id'], state)
                return _render_referential_validation(
                    referential_analysis,
                    source_path=state['source_path'],
                    departement=state['departement'],
                    analysis_id=state['analysis_id'],
                    rome_query=query,
                    rome_candidates=candidates,
                    market_target=state.get('market_target') or {},
                    referential_success='Résultats de recherche métier affichés.',
                )
            if action in {'select_market_target', 'add_market_target'}:
                state = _load_referential_state(clean_text(payload.get('analysis_id') or ''), payload.get('analysis_json'))
                referential_analysis = state['analysis']
                rome_code = validate_rome_code(clean_text(payload.get('rome_code') or payload.get('selected_rome_code') or ''), service=get_rome_service())
                rome_job = get_rome_service().get(rome_code)
                selected = list((state.get('market_target') or {}).get('selected_rome_occupations') or state.get('selected_rome_occupations') or [])
                if any(clean_text(item.get('code') if isinstance(item, dict) else getattr(item, 'code', '')).replace(' ', '').upper() == rome_code for item in selected):
                    raise ValueError(f'Le code ROME {rome_code} est déjà sélectionné')
                if len(selected) >= MAX_ROME_CODES_PER_SEARCH:
                    raise ValueError(f'Vous pouvez sélectionner au maximum {MAX_ROME_CODES_PER_SEARCH} codes ROME')
                selected.append(_rome_occupation_payload(rome_code, rome_job.label if rome_job and rome_job.label else clean_text(payload.get('rome_label') or '')))
                territory = _build_territory_from_form(payload, fallback_code=state.get('departement') or '', fallback_label=state.get('departement') or '')
                market_target = dict(state.get('market_target') or {})
                market_target.update({
                    'rome_code': selected[0]['code'],
                    'rome_label': selected[0]['label'],
                    'territory_code': territory.code or market_target.get('territory_code') or state.get('departement') or '',
                    'territory_label': territory.label or market_target.get('territory_label') or state.get('departement') or '',
                    'radius_km': territory.radius_km,
                    'contract_type': clean_text(payload.get('contract_type') or market_target.get('contract_type') or ''),
                    'selected_rome_codes': [item['code'] for item in selected],
                    'selected_rome_occupations': selected,
                })
                state['market_target'] = market_target
                state['selected_rome_occupations'] = selected
                state['market_target_confirmed'] = True
                state['analysis_status'] = 'ROME_CONFIRMED'
                state['market_search_status'] = 'WAITING_FOR_SEARCH'
                _referential_state_cache().set(state['analysis_id'], state)
                return _render_referential_validation(
                    referential_analysis,
                    source_path=state['source_path'],
                    departement=state['departement'],
                    analysis_id=state['analysis_id'],
                    rome_query=state.get('rome_query'),
                    rome_candidates=state.get('rome_candidates') or [],
                    market_target=market_target,
                    market_target_confirmed=True,
                    selected_rome_occupations=selected,
                    referential_success=f'Code ROME ajouté: {rome_code}.',
                )
            if action == 'remove_market_target':
                state = _load_referential_state(clean_text(payload.get('analysis_id') or ''), payload.get('analysis_json'))
                referential_analysis = state['analysis']
                rome_code = validate_rome_code(clean_text(payload.get('rome_code') or payload.get('selected_rome_code') or ''), service=get_rome_service())
                selected = list((state.get('market_target') or {}).get('selected_rome_occupations') or state.get('selected_rome_occupations') or [])
                selected = [item for item in selected if clean_text(item.get('code') if isinstance(item, dict) else getattr(item, 'code', '')).replace(' ', '').upper() != rome_code]
                market_target = dict(state.get('market_target') or {})
                market_target['selected_rome_occupations'] = selected
                market_target['selected_rome_codes'] = [item['code'] for item in selected]
                if selected:
                    market_target['rome_code'] = selected[0]['code']
                    market_target['rome_label'] = selected[0]['label']
                    state['market_target_confirmed'] = True
                    state['analysis_status'] = 'ROME_CONFIRMED'
                    state['market_search_status'] = 'WAITING_FOR_SEARCH'
                else:
                    market_target['rome_code'] = ''
                    market_target['rome_label'] = ''
                    state['market_target_confirmed'] = False
                    state['analysis_status'] = 'WAITING_FOR_ROME'
                    state['market_search_status'] = 'WAITING_FOR_ROME'
                state['market_target'] = market_target
                state['selected_rome_occupations'] = selected
                _referential_state_cache().set(state['analysis_id'], state)
                return _render_referential_validation(
                    referential_analysis,
                    source_path=state['source_path'],
                    departement=state['departement'],
                    analysis_id=state['analysis_id'],
                    rome_query=state.get('rome_query'),
                    rome_candidates=state.get('rome_candidates') or [],
                    market_target=market_target,
                    market_target_confirmed=bool(selected),
                    selected_rome_occupations=selected,
                    referential_success='Le code ROME a été retiré.' if selected else 'Aucun code ROME sélectionné.',
                )
            if action == 'run_market_search':
                state = _load_referential_state(clean_text(payload.get('analysis_id') or ''), payload.get('analysis_json'))
                referential_analysis = state['analysis']
                market_target = state.get('market_target') or {}
                selected_occupations = list(market_target.get('selected_rome_occupations') or state.get('selected_rome_occupations') or [])
                selected_codes = [clean_text(item.get('code') if isinstance(item, dict) else getattr(item, 'code', '')).replace(' ', '').upper() for item in selected_occupations if clean_text(item.get('code') if isinstance(item, dict) else getattr(item, 'code', ''))]
                if not selected_codes and clean_text(market_target.get('rome_code') or payload.get('rome_code') or ''):
                    selected_codes = [validate_rome_code(clean_text(market_target.get('rome_code') or payload.get('rome_code') or ''), service=get_rome_service())]
                if not selected_codes:
                    raise ValueError('Sélectionnez au moins un code ROME')
                territory = _build_territory_from_form(payload, fallback_code=market_target.get('territory_code') or state.get('departement') or '', fallback_label=market_target.get('territory_label') or state.get('departement') or '')
                context = _build_referential_market_context(
                    referential_analysis,
                    source_path=state['source_path'],
                    departement=state['departement'],
                    selected_rome_code=selected_codes[0],
                    selected_rome_label=clean_text((selected_occupations[0] if selected_occupations else {}).get('label') or market_target.get('rome_label') or ''),
                    selected_rome_codes=selected_codes,
                    territory=territory,
                    contract_type=clean_text(market_target.get('contract_type') or payload.get('contract_type') or '') or None,
                    radius_km=territory.radius_km,
                )
                state['analysis_status'] = 'MARKET_ANALYZED'
                state['market_search_status'] = 'MARKET_ANALYZED'
                _referential_state_cache().set(state['analysis_id'], state)
                result = context['analysis_result']
                return render_template(
                    'result.html',
                    **context,
                    department_options=DEPARTMENT_CODES,
                    result_json=result.to_json(),
                    result_dict=result.to_dict(),
                )
            elif request.files.get('pdf') and request.files.get('pdf').filename:
                pdf_path, departement = _extract_referential_inputs(payload, request.files)
                referential_analysis = referential_import_service.analyze(pdf_path)
                state = _build_referential_state(referential_analysis, source_path=str(pdf_path), departement=departement)
                return _render_referential_validation(referential_analysis, source_path=str(pdf_path), departement=departement, analysis_id=state['analysis_id'], rome_query=state['rome_query'])
            else:
                text, departement, keywords, threshold, model_only = _extract_inputs(payload)
                context = _build_context(text, departement, keywords, threshold, model_only, allow_market_failure=False)
        except ValueError as exc:
            return _render_error(str(exc), 400)
        except RuntimeError as exc:
            return _render_error(str(exc), 503)

        result = context['analysis_result']
        return render_template(
            'result.html',
            **context,
            department_options=DEPARTMENT_CODES,
            result_json=result.to_json(),
            result_dict=result.to_dict(),
        )

    @app.post('/api/analyze')
    def api_analyze():
        try:
            text, departement, keywords, threshold, model_only = _extract_inputs(_parse_request_payload())
            context = _build_context(text, departement, keywords, threshold, model_only)
        except ValueError as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 400
        except RuntimeError as exc:
            message = str(exc)
            status = 503
            if '429' in message:
                status = 429
            elif 'authentification' in message.lower():
                status = 401
            elif 'delai' in message.lower():
                status = 504
            return jsonify({'ok': False, 'error': message}), status

        result = context['analysis_result']
        return jsonify({
            'ok': True,
            'warning': context['warning'],
            'department': context['department'],
            'keywords': context['keywords'],
            'threshold': context['threshold'],
            'model_only': context['model_only'],
            'analysis': context['analysis'],
            'result': result.to_dict(),
        })

    @app.get('/health')
    def health():
        predictor_instance = get_predictor_instance()
        return jsonify(
            {
                'status': 'ok' if predictor_instance else 'degraded',
                'models_available': bool(predictor_instance),
                'device': str(predictor_instance.device) if predictor_instance else None,
                'france_travail_configured': _available_france_travail_config(),
                'predictor_error': app.extensions.get('deepforma_predictor_error'),
            }
        )

    @app.post('/api/analyze/export/json')
    def export_json():
        try:
            text, departement, keywords, threshold, model_only = _extract_inputs(_parse_request_payload())
            context = _build_context(text, departement, keywords, threshold, model_only)
        except (ValueError, RuntimeError) as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 400

        result = context['analysis_result']
        return Response(
            result.to_json(),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment; filename=deepforma_analysis.json'},
        )

    @app.post('/api/analyze/export/csv')
    def export_csv():
        try:
            text, departement, keywords, threshold, model_only = _extract_inputs(_parse_request_payload())
            context = _build_context(text, departement, keywords, threshold, model_only)
        except (ValueError, RuntimeError) as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 400

        result = context['analysis_result']
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Competence', 'Presente_formation', 'Confiance_detection',
                         'Frequence_offres', 'Nb_offres', 'Niveau_couverture', 'Priorite'])
        for item in result.formation_market_comparison:
            writer.writerow([
                item.skill,
                'Oui' if item.in_formation else 'Non',
                f"{item.detection_confidence:.2f}",
                f"{item.frequency_in_offers:.1f}%",
                item.offer_count,
                item.coverage_level,
                item.priority,
            ])
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=deepforma_comparison.csv'},
        )

    referential_import_service = ReferentialImportService()
    app.extensions['referential_import_service'] = referential_import_service

    @app.get('/admin/referential-import')
    @require_admin_auth
    def admin_referential_import():
        return render_template(
            'admin_referential_import.html',
            analysis=None,
            report=None,
            document=None,
            competencies=[],
            criteria=[],
            derived_skills=[],
            blocks=[],
            activities=[],
            analysis_json='',
            source_path='',
        )

    def _render_referential_import_context(analysis: dict[str, Any], *, source_path: str, success: str | None = None, error: str | None = None):
        analysis_json = json.dumps(build_export_payload(analysis), ensure_ascii=False, indent=2)
        return render_template(
            'admin_referential_import.html',
            analysis=analysis,
            report=analysis['report'],
            document=analysis['document'],
            competencies=analysis['competencies'],
            criteria=analysis['criteria'],
            derived_skills=analysis['derived_skills'],
            blocks=analysis['blocks'],
            activities=analysis['activities'],
            analysis_json=analysis_json,
            source_path=source_path,
            success=success,
            error=error,
        )

    def _apply_manual_corrections(analysis: dict[str, Any]) -> dict[str, Any]:
        analysis = _apply_referential_import_edits(analysis, request.form)
        for criterion in analysis.get('criteria', []):
            label_key = f'criterion_label__{criterion.code}'
            status_key = f'criterion_status__{criterion.code}'
            if label_key in request.form:
                criterion.criterion_label = clean_text(request.form.get(label_key)) or criterion.criterion_label
                criterion.normalized_label = clean_text(request.form.get(label_key)) or criterion.normalized_label
            if status_key in request.form:
                criterion.review_status = clean_text(request.form.get(status_key)) or criterion.review_status
        return analysis

    @app.post('/admin/referential-import')
    @require_admin_auth
    def admin_referential_import_post():
        action = clean_text(request.form.get('action') or 'analyze')
        if action == 'approve':
            payload = request.form.get('analysis_json')
            if not payload:
                return render_template('admin_referential_import.html', analysis=None, report=None, document=None, competencies=[], criteria=[], derived_skills=[], blocks=[], activities=[], analysis_json='', source_path='', error='Analyse manquante.'), 400
            try:
                export = json.loads(payload)
            except Exception:
                return render_template('admin_referential_import.html', analysis=None, report=None, document=None, competencies=[], criteria=[], derived_skills=[], blocks=[], activities=[], analysis_json='', source_path='', error='JSON d analyse invalide.'), 400
            analysis = analysis_from_export(export)
            analysis = _apply_manual_corrections(analysis)
            analysis['export'] = build_export_payload(analysis)
            validated_by = clean_text(request.form.get('validated_by') or 'human_review') or 'human_review'
            output_path = referential_import_service.approve(analysis, validated_by=validated_by)
            return _render_referential_import_context(analysis, source_path=clean_text(request.form.get('source_path')), success=f'Import approuve: {output_path}')

        uploaded = request.files.get('pdf')
        if not uploaded or not uploaded.filename:
            return render_template('admin_referential_import.html', analysis=None, report=None, document=None, competencies=[], criteria=[], derived_skills=[], blocks=[], activities=[], analysis_json='', source_path='', error='Fichier PDF manquant.'), 400
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_path = Path(temp_file.name)
        temp_file.close()
        uploaded.save(temp_path)
        analysis = referential_import_service.analyze(temp_path)
        return _render_referential_import_context(analysis, source_path=str(temp_path))

    @app.route('/admin/ai-certification-market-comparison', methods=['GET', 'POST'])
    @require_admin_auth
    def admin_ai_certification_market_comparison():
        def _split_form_values(raw: str | None) -> list[str]:
            if not raw:
                return []
            parts = [part.strip() for part in re.split(r'[\n,;|]+', raw) if part and part.strip()]
            return [part for part in parts if part]

        def _parse_int(value: str | None) -> int | None:
            value = clean_text(value)
            if not value:
                return None
            try:
                return int(value)
            except ValueError:
                return None

        job_titles_default = 'ingénieur intelligence artificielle,AI Engineer,Machine Learning Engineer,Data Scientist,MLOps Engineer,ingénieur Machine Learning,ingénieur NLP,ingénieur Deep Learning,ingénieur IA générative,Data Engineer IA,chef de projet IA'
        rome_codes_default = 'M1805'
        territory = clean_text(request.values.get('territory') or '75056')
        commune = clean_text(request.values.get('commune') or '') or None
        departement = clean_text(request.values.get('departement') or '') or None
        if not commune and not departement:
            if len(territory) == 5 and territory.isdigit():
                commune = territory
            elif len(territory) == 2 and territory.isdigit():
                departement = territory
        radius_km = _parse_int(request.values.get('radius_km'))
        date_min = clean_text(request.values.get('date_min') or '') or None
        date_max = clean_text(request.values.get('date_max') or '') or None
        job_titles = _split_form_values(request.values.get('job_titles') or job_titles_default)
        rome_codes = _split_form_values(request.values.get('rome_codes') or rome_codes_default)
        max_pages = _parse_int(request.values.get('max_pages')) or DEFAULT_MAX_PAGES
        max_offers = _parse_int(request.values.get('max_offers')) or DEFAULT_MAX_OFFERS
        page_size = _parse_int(request.values.get('page_size')) or DEFAULT_PAGE_SIZE
        report = None
        output_paths: dict[str, Path] | None = None
        source_queries: list[str] = []
        error = None
        if request.method == 'POST':
            try:
                client = get_market_client()
                offers, source_queries = collect_market_offers(
                    client,
                    commune=commune,
                    departement=departement,
                    distance_km=radius_km,
                    date_min=date_min,
                    date_max=date_max,
                    job_titles=job_titles,
                    rome_codes=rome_codes,
                    max_pages=max_pages,
                    max_offers=max_offers,
                    page_size=page_size,
                )
                comparator: CertificationMarketComparator = app.extensions['certification_market_comparator']
                territory_label = ' / '.join([part for part in [commune, departement, territory] if part])
                report = comparator.compare(
                    offers,
                    territory=territory_label,
                    radius_km=radius_km,
                    date_min=date_min,
                    date_max=date_max,
                    job_titles=job_titles,
                    rome_codes=rome_codes,
                    source_queries=source_queries,
                )
                output_paths = write_comparison_outputs(report, PROJECT_ROOT / 'data' / 'reports' / 'ai_certification_market')
            except ValueError as exc:
                error = str(exc)
            except RuntimeError as exc:
                error = str(exc)
        return render_template(
            'admin_ai_certification_market_comparison.html',
            report=report.to_dict() if report else None,
            report_obj=report,
            output_paths=output_paths,
            error=error,
            territory=territory,
            commune=commune or '',
            departement=departement or '',
            radius_km=radius_km or '',
            date_min=date_min or '',
            date_max=date_max or '',
            job_titles='\n'.join(job_titles),
            rome_codes='\n'.join(rome_codes),
            max_pages=max_pages,
            max_offers=max_offers,
            page_size=page_size,
            dry_run=str(request.values.get('dry_run') or '').lower() in {'1', 'true', 'yes', 'on'},
            source_queries=source_queries,
        )

    ner_annotation_store = AnnotationStore(PROJECT_ROOT / 'data' / 'annotation' / 'referential_ner_candidates.jsonl')
    multilabel_annotation_store = AnnotationStore(PROJECT_ROOT / 'data' / 'annotation' / 'referential_multilabel_candidates.jsonl')
    app.extensions['referential_ner_annotation_store'] = ner_annotation_store
    app.extensions['referential_multilabel_annotation_store'] = multilabel_annotation_store

    REFERENTIAL_ENTITY_LABELS = ['SKILL', 'METHOD', 'TOOL', 'DOMAIN', 'SOFT_SKILL', 'OTHER']
    REFERENTIAL_FAMILY_LABELS = ['Machine Learning', 'Deep Learning', 'NLP', 'MLOps', 'Other']

    def _referential_annotation_records() -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for kind, store in (('ner', ner_annotation_store), ('multilabel', multilabel_annotation_store)):
            for record in store.load():
                item = dict(record)
                item['kind'] = kind
                item['record_id'] = item.get('record_id') or stable_hash(
                    kind,
                    item.get('document_id', ''),
                    item.get('page', 0),
                    item.get('block_id', ''),
                    item.get('text', ''),
                    length=24,
                )
                item.setdefault('status', 'pending')
                records.append(item)
        records.sort(key=lambda item: (
            clean_text(item.get('source_file') or ''),
            int(item.get('page') or 0),
            clean_text(item.get('section') or ''),
            clean_text(item.get('record_id') or ''),
        ))
        return records

    def _persist_referential_records(records: list[dict[str, Any]]) -> None:
        ner_rows = [dict(record) for record in records if clean_text(record.get('kind')) == 'ner']
        multilabel_rows = [dict(record) for record in records if clean_text(record.get('kind')) == 'multilabel']
        ner_annotation_store.save(ner_rows)
        multilabel_annotation_store.save(multilabel_rows)

    def _highlight_referential_text(text: str, entities: list[dict[str, Any]] | None) -> Markup:
        safe_text = clean_text(text)
        if not safe_text:
            return Markup('')
        spans = []
        for entity in entities or []:
            try:
                start = int(entity.get('start', 0))
                end = int(entity.get('end', 0))
            except Exception:
                continue
            if start < 0 or end <= start or end > len(safe_text):
                continue
            spans.append((start, end, entity))
        spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        parts: list[str] = []
        cursor = 0
        for start, end, entity in spans:
            if start < cursor:
                continue
            parts.append(str(escape(safe_text[cursor:start])))
            label = clean_text(entity.get('approved_label') or entity.get('predicted_label') or '') or 'OTHER'
            canonical = clean_text(entity.get('canonical_name') or '')
            annotation = f"{label}"
            if canonical and canonical != label:
                annotation = f"{label} · {canonical}"
            parts.append(
                f"<mark class='entity-highlight entity-{normalize_for_match(label)}'>"
                f"{escape(safe_text[start:end])}"
                f"<span>{escape(annotation)}</span>"
                f"</mark>"
            )
            cursor = end
        parts.append(str(escape(safe_text[cursor:])))
        return Markup(''.join(parts))

    def _render_referential_annotation_index(*, selected_record_id: str | None = None, success: str | None = None, error: str | None = None):
        records = _referential_annotation_records()
        selected = None
        if selected_record_id:
            selected = next((record for record in records if clean_text(record.get('record_id') or '') == clean_text(selected_record_id)), None)
        if selected is None and records:
            selected = records[0]
        if selected is not None:
            selected = dict(selected)
            selected['highlighted_text'] = _highlight_referential_text(selected.get('text', ''), selected.get('entities', []))
        return render_template(
            'admin_referential_annotation.html',
            records=records,
            selected=selected,
            success=success,
            error=error,
            section_labels=['TITLE', 'PROVIDER', 'REFERENCE', 'DURATION', 'LEVEL', 'FORMAT', 'PRICE', 'CPF', 'CERTIFICATION', 'PUBLIC', 'PREREQUISITES', 'OBJECTIVES', 'PROGRAM', 'MODULE', 'SKILLS', 'TOOLS', 'DOMAINS', 'FOOTER', 'OTHER'],
            entity_labels=REFERENTIAL_ENTITY_LABELS,
            family_labels=REFERENTIAL_FAMILY_LABELS,
        )

    @app.get('/admin/referential-annotation')
    @require_admin_auth
    def admin_referential_annotation():
        return _render_referential_annotation_index(selected_record_id=request.args.get('record_id'))

    def _update_annotation_status(record: dict[str, Any], *, action: str, form: Any) -> None:
        kind = clean_text(record.get('kind') or 'ner')
        if kind == 'ner':
            if action in {'approve_entity', 'approve_all_entities'}:
                entity_id = clean_text(form.get('entity_id') or '')
                approved_label = clean_text(form.get('approved_label') or '')
                canonical_name = clean_text(form.get('canonical_name') or '')
                referential_id = clean_text(form.get('referential_id') or '')
                for entity in record.get('entities', []):
                    current_id = clean_text(entity.get('entity_id') or '')
                    if action == 'approve_all_entities' or (current_id and current_id == entity_id):
                        entity['approved_label'] = approved_label or entity.get('predicted_label')
                        entity['canonical_name'] = canonical_name or entity.get('canonical_name')
                        entity['referential_id'] = referential_id or entity.get('referential_id')
                        entity['status'] = 'approved'
            elif action in {'reject_entity', 'reject_all_entities'}:
                entity_id = clean_text(form.get('entity_id') or '')
                for entity in record.get('entities', []):
                    if action == 'reject_all_entities' or clean_text(entity.get('entity_id') or '') == entity_id:
                        entity['status'] = 'rejected'
            elif action == 'add_entity':
                text_value = clean_text(form.get('text') or '')
                if text_value:
                    record.setdefault('entities', []).append({
                        'entity_id': f"manual-entity-{len(record.get('entities', [])) + 1}",
                        'start': int(form.get('start') or 0),
                        'end': int(form.get('end') or len(text_value)),
                        'text': text_value,
                        'predicted_label': clean_text(form.get('entity_label') or 'SKILL') or 'SKILL',
                        'approved_label': clean_text(form.get('approved_label') or form.get('entity_label') or 'SKILL') or 'SKILL',
                        'canonical_name': clean_text(form.get('canonical_name') or text_value) or text_value,
                        'confidence': 1.0,
                        'page': int(form.get('page') or record.get('page') or 0),
                        'block_id': clean_text(form.get('block_id') or record.get('block_id') or ''),
                        'source_file': record.get('source_file', ''),
                        'document_id': record.get('document_id', ''),
                        'status': 'approved',
                        'referential_id': clean_text(form.get('referential_id') or ''),
                        'evidence': clean_text(form.get('evidence') or ''),
                    })
            elif action == 'validate_document':
                pass
        else:
            if action in {'approve_multilabel', 'save_multilabel', 'approve_all_labels'}:
                if action == 'approve_all_labels':
                    approved_labels = [clean_text(label) for label in record.get('predicted_labels') or [] if clean_text(label)]
                else:
                    approved_labels = [clean_text(label) for label in form.getlist('approved_labels') if clean_text(label)]
                    if not approved_labels:
                        approved_labels = [clean_text(form.get('approved_labels') or '')] if clean_text(form.get('approved_labels') or '') else []
                if approved_labels:
                    record['approved_labels'] = approved_labels
                    record['status'] = 'approved'
            elif action in {'reject_multilabel', 'reject_all_labels'}:
                record['approved_labels'] = []
                record['status'] = 'rejected'
            elif action == 'validate_document':
                pass

    @app.post('/admin/referential-annotation/action')
    @require_admin_auth
    def admin_referential_annotation_action():
        _require_valid_csrf()
        record_id = clean_text(request.form.get('record_id') or '')
        action = clean_text(request.form.get('action') or '')
        if not record_id:
            raise ValueError('record_id requis.')
        records = _referential_annotation_records()
        record = next((item for item in records if clean_text(item.get('record_id') or '') == record_id), None)
        if record is None:
            raise ValueError('Enregistrement introuvable.')
        _update_annotation_status(record, action=action, form=request.form)
        for idx, item in enumerate(records):
            if clean_text(item.get('record_id') or '') == record_id:
                records[idx] = record
                break
        if action == 'validate_document':
            document_id = clean_text(record.get('document_id') or '')
            for idx, item in enumerate(records):
                if clean_text(item.get('document_id') or '') == document_id:
                    records[idx] = item
                    records[idx]['status'] = 'validated'
        _persist_referential_records(records)
        return redirect(url_for('admin_referential_annotation', record_id=record_id))

    continual_store = ContinualLearningStore(PROJECT_ROOT / 'data' / 'continual_learning' / 'continual_learning.sqlite3')
    app.extensions['continual_learning_store'] = continual_store

    def _parse_float(value: str | None) -> float | None:
        if value in (None, ''):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _json_list(value: str | None) -> list[dict[str, Any]]:
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    def _offer_confidence(offer: dict[str, Any]) -> float:
        confidence = _json_list(offer.get('confidence_json'))
        if isinstance(offer.get('confidence_json'), str):
            try:
                payload = json.loads(offer['confidence_json'])
                if isinstance(payload, dict):
                    values = [float(v) for v in payload.values() if isinstance(v, (int, float))]
                    return max(values) if values else 0.0
            except Exception:
                return 0.0
        return 0.0

    def _offer_structured_labels(offer: dict[str, Any]) -> set[str]:
        labels = set()
        for item in _json_list(offer.get('structured_skills_json')):
            label = normalize_skill_label(item.get('canonical_name') or item.get('label') or '')
            if label:
                labels.add(label)
        return labels

    def _offer_predicted_labels(offer: dict[str, Any]) -> set[str]:
        labels = set()
        for item in _json_list(offer.get('predicted_skills_json')):
            label = normalize_skill_label(item.get('canonical_name') or item.get('label') or '')
            if label:
                labels.add(label)
        return labels

    def _offer_disagreement(offer: dict[str, Any]) -> bool:
        structured = _offer_structured_labels(offer)
        predicted = _offer_predicted_labels(offer)
        if not structured or not predicted:
            return False
        return structured != predicted

    def _continual_learning_model_diagnostics(offer: dict[str, Any]) -> dict[str, Any]:
        predictions = [
            item for item in _json_list(offer.get('predicted_skills_json'))
            if isinstance(item, dict) and (clean_text(item.get('source')) == 'camembert_multilabel' or clean_text(item.get('provenance')) == 'model_prediction')
        ]
        scores = [float(item.get('confidence', 0.0)) for item in predictions if isinstance(item.get('confidence', 0.0), (int, float)) or str(item.get('confidence', '')).strip()]
        if not scores:
            return {
                'reliable': False,
                'reason': 'Aucune catégorie du classifieur multilabel.',
                'score_min': 0.0,
                'score_max': 0.0,
                'score_mean': 0.0,
                'score_std': 0.0,
                'score_gap': 0.0,
                'scores': [],
                'top_predictions': predictions,
                'ignored_predictions_count': 0,
            }
        scores_sorted = sorted(scores, reverse=True)
        score_min = min(scores)
        score_max = max(scores)
        score_mean = sum(scores) / len(scores)
        variance = sum((score - score_mean) ** 2 for score in scores) / len(scores)
        score_std = variance ** 0.5
        score_gap = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else scores_sorted[0]
        all_close_to_mid = all(0.40 <= score <= 0.60 for score in scores)
        reliable = not (
            score_std < MODEL_SCORE_STD_MIN
            or score_max < MODEL_SCORE_MAX_MIN
            or all_close_to_mid
            or score_gap < MODEL_SCORE_GAP_MIN
        )
        if reliable:
            reason = 'Scores suffisamment discriminants.'
        elif score_std < MODEL_SCORE_STD_MIN:
            reason = 'Écart-type trop faible.'
        elif score_max < MODEL_SCORE_MAX_MIN:
            reason = 'Aucun score ne dépasse le seuil de confiance.'
        elif all_close_to_mid:
            reason = 'Tous les scores sont proches de 0.5.'
        else:
            reason = 'Les deux meilleurs scores sont trop proches.'
        return {
            'reliable': reliable,
            'reason': reason,
            'score_min': round(score_min, 4),
            'score_max': round(score_max, 4),
            'score_mean': round(score_mean, 4),
            'score_std': round(score_std, 4),
            'score_gap': round(score_gap, 4),
            'scores': scores,
            'top_predictions': predictions,
            'ignored_predictions_count': len(predictions) if not reliable else 0,
        }

    def _build_continual_learning_offer_view(offer: dict[str, Any]) -> dict[str, Any]:
        annotations = list(offer.get('annotations') or [])
        diagnostics = _continual_learning_model_diagnostics(offer)
        france_travail_skills = [
            item for item in annotations
            if clean_text(item.get('provenance')) == 'france_travail_api'
        ]
        text_skills = [
            item for item in annotations
            if clean_text(item.get('provenance')) in {'exact_reference_match', 'semantic_match'}
            or clean_text(item.get('source')) in {'text_explicit', 'text_fallback'}
        ]
        model_categories = [
            item for item in annotations
            if clean_text(item.get('provenance')) == 'model_prediction'
        ]
        if diagnostics['top_predictions']:
            existing_signatures = {
                (
                    clean_text(item.get('canonical_name') or item.get('label')),
                    round(float(item.get('confidence') or 0.0), 4),
                )
                for item in model_categories
            }
            for index, item in enumerate(diagnostics['top_predictions'], start=1):
                signature = (
                    clean_text(item.get('canonical_name') or item.get('label')),
                    round(float(item.get('confidence') or 0.0), 4),
                )
                if signature in existing_signatures:
                    continue
                model_categories.append({
                    'id': None,
                    'offer_row_id': offer.get('id'),
                    'canonical_name': clean_text(item.get('canonical_name') or item.get('label')),
                    'canonical_label': clean_text(item.get('canonical_label') or item.get('label')),
                    'surface_form': clean_text(item.get('surface_form')) or None,
                    'confidence': float(item.get('confidence', 0.0)),
                    'source': clean_text(item.get('source') or 'camembert_multilabel'),
                    'provenance': clean_text(item.get('provenance') or 'model_prediction'),
                    'validation_status': 'pending',
                    'text_sentence': clean_text(item.get('text_sentence')) or None,
                    'referential_code': clean_text(item.get('referential_code')) or None,
                    'referential_label': clean_text(item.get('referential_label')) or None,
                    'start': item.get('start'),
                    'end': item.get('end'),
                    'is_explicit': bool(item.get('is_explicit', False)),
                    'model_rank': index,
                })
                existing_signatures.add(signature)
        human_additions = [
            item for item in annotations
            if clean_text(item.get('provenance')) == 'human_review'
        ]
        rejected_annotations = [item for item in annotations if clean_text(item.get('validation_status')) == 'rejected']
        corrected_annotations = [item for item in annotations if clean_text(item.get('validation_status')) == 'corrected']
        approved_annotations = [item for item in annotations if clean_text(item.get('validation_status')) == 'approved']
        pending_model_predictions = [item for item in model_categories if clean_text(item.get('validation_status')) == 'pending']
        reviewable_pending = [
            item for item in annotations
            if clean_text(item.get('validation_status')) == 'pending'
            and (clean_text(item.get('provenance')) != 'model_prediction' or diagnostics['reliable'])
        ]
        ignored_predictions = [item for item in model_categories if not diagnostics['reliable']]
        return {
            'annotations': annotations,
            'france_travail_skills': france_travail_skills,
            'text_skills': text_skills,
            'model_categories': model_categories,
            'human_additions': human_additions,
            'rejected_annotations': rejected_annotations,
            'corrected_annotations': corrected_annotations,
            'approved_annotations': approved_annotations,
            'pending_model_predictions': pending_model_predictions,
            'reviewable_pending': reviewable_pending,
            'ignored_predictions': ignored_predictions,
            'model_diagnostics': diagnostics,
            'pending_review_count': len(reviewable_pending),
            'validation_summary': {
                'accepted': len(approved_annotations),
                'corrected': len(corrected_annotations),
                'rejected': len(rejected_annotations),
                'added': len(human_additions),
                'ignored': len(ignored_predictions),
            },
        }

    def _filter_admin_offers(all_offers: list[dict[str, Any]], filters: dict[str, str]) -> list[dict[str, Any]]:
        min_conf = _parse_float(filters.get('min_confidence'))
        max_conf = _parse_float(filters.get('max_confidence'))
        status = clean_text(filters.get('status'))
        territory = clean_text(filters.get('territory'))
        job_family = clean_text(filters.get('job_family'))
        source = clean_text(filters.get('source'))
        model_version = clean_text(filters.get('model_version'))
        disagreement = filters.get('disagreement') in {'1', 'true', 'yes', 'on'}
        result = []
        for offer in all_offers:
            offer_status = clean_text(offer.get('validation_status'))
            offer_territory = clean_text(offer.get('territory'))
            offer_job_family = clean_text(offer.get('job_family'))
            offer_model_version = clean_text(offer.get('model_version'))
            confidence = _offer_confidence(offer)
            annotations = continual_store.list_annotations('offer_row_id = ?', (offer['id'],))
            sources = {clean_text(item.get('source')) for item in annotations}
            if status and offer_status != status:
                continue
            if territory and territory not in offer_territory:
                continue
            if job_family and job_family not in offer_job_family:
                continue
            if model_version and model_version not in offer_model_version:
                continue
            if source and source not in sources:
                continue
            if min_conf is not None and confidence < min_conf:
                continue
            if max_conf is not None and confidence > max_conf:
                continue
            if disagreement and not _offer_disagreement(offer):
                continue
            result.append({**offer, 'annotations': annotations, 'confidence_value': confidence, 'disagreement': _offer_disagreement(offer)})
        return result

    @app.get('/admin/continual-learning')
    @require_admin_auth
    def admin_continual_learning():
        filters = {
            'status': request.args.get('status', ''),
            'territory': request.args.get('territory', ''),
            'job_family': request.args.get('job_family', ''),
            'min_confidence': request.args.get('min_confidence', ''),
            'max_confidence': request.args.get('max_confidence', ''),
            'source': request.args.get('source', ''),
            'model_version': request.args.get('model_version', ''),
            'disagreement': request.args.get('disagreement', ''),
        }
        all_offers = continual_store.list_offers(limit=500)
        filtered_offers = _filter_admin_offers(all_offers, filters)
        selected_id = request.args.get('offer_row_id')
        selected_offer = None
        if selected_id:
            try:
                selected_offer = continual_store.get_offer_with_annotations(int(selected_id))
            except ValueError:
                selected_offer = None
        if selected_offer is None and filtered_offers:
            selected_offer = continual_store.get_offer_with_annotations(int(filtered_offers[0]['id']))
        selected_offer_view = _build_continual_learning_offer_view(selected_offer) if selected_offer else None
        next_offer_id = None
        if selected_offer:
            ids = [int(offer['id']) for offer in filtered_offers]
            if selected_offer['id'] in ids:
                idx = ids.index(selected_offer['id'])
                if idx + 1 < len(ids):
                    next_offer_id = ids[idx + 1]
        validation_counts = Counter(item.get('validation_status', 'pending') for item in all_offers)
        filters_query = urlencode({key: value for key, value in filters.items() if value})
        return render_template(
            'admin_continual_learning.html',
            offers=filtered_offers,
            selected_offer=selected_offer,
            selected_offer_view=selected_offer_view,
            next_offer_id=next_offer_id,
            filters=filters,
            filters_query=filters_query,
            validation_counts=dict(validation_counts),
            admin_enabled=True,
        )

    @app.post('/admin/continual-learning/action')
    @require_admin_auth
    def admin_continual_learning_action():
        _log_admin_request()
        _require_valid_csrf()
        form = request.form
        action = clean_text(form.get('action'))
        offer_row_id = int(form.get('offer_row_id') or 0)
        annotation_id = form.get('annotation_id')
        validated_by = clean_text(form.get('validated_by') or os.getenv('DEEPFORMA_ADMIN_USER') or 'admin')
        note = clean_text(form.get('note')) or None
        redirect_offer = form.get('redirect_offer_row_id') or offer_row_id
        selected_filters = _admin_filters_from_request()
        if action == 'mark_offer_approved':
            offer = continual_store.get_offer_with_annotations(offer_row_id)
            if not offer:
                raise ValueError('Offre introuvable.')
            offer_view = _build_continual_learning_offer_view(offer)
            if offer_view['pending_review_count'] > 0 and clean_text(form.get('confirm_pending')) not in {'1', 'true', 'yes', 'on'}:
                raise ValueError('Des propositions sont encore en attente. Confirmez la validation explicite avant de valider l’offre.')
            continual_store.mark_offer_status(offer_row_id, 'approved', validation_actor=validated_by, validation_note=note)
            continual_store.add_validation_event(offer_row_id, validated_by, 'offer_approved', {'note': note, 'pending_review_count': offer_view['pending_review_count']})
        elif action in {'approve_annotation', 'reject_annotation', 'correct_annotation'}:
            if not annotation_id:
                raise ValueError('annotation_id requis.')
            annotation = continual_store.get_annotation(int(annotation_id))
            if not annotation:
                raise ValueError('Annotation introuvable.')
            if action == 'approve_annotation':
                continual_store.update_annotation_fields(
                    int(annotation_id),
                    validation_status='approved',
                    validated_at=datetime.now(timezone.utc).isoformat(),
                    validated_by=validated_by,
                )
            elif action == 'reject_annotation':
                continual_store.update_annotation_fields(
                    int(annotation_id),
                    validation_status='rejected',
                    rejected_reason=note,
                    validated_at=datetime.now(timezone.utc).isoformat(),
                    validated_by=validated_by,
                )
            else:
                corrected_name = clean_text(form.get('corrected_name') or annotation['canonical_name'])
                corrected_surface = clean_text(form.get('corrected_surface') or annotation['surface_form'])
                normalized = normalize_skill_label(corrected_name)
                continual_store.update_annotation_fields(
                    int(annotation_id),
                    canonical_name=corrected_name,
                    surface_form=corrected_surface,
                    normalized_name=normalized,
                    validation_status='corrected',
                    correction_json={'corrected_name': corrected_name, 'corrected_surface': corrected_surface, 'note': note},
                    validated_at=datetime.now(timezone.utc).isoformat(),
                    validated_by=validated_by,
                )
        elif action == 'add_annotation':
            canonical_name = clean_text(form.get('canonical_name') or '')
            if not canonical_name:
                raise ValueError('canonical_name requis.')
            surface_form = clean_text(form.get('surface_form') or canonical_name)
            start = form.get('start')
            end = form.get('end')
            start_i = int(start) if start not in (None, '') else None
            end_i = int(end) if end not in (None, '') else None
            offer = continual_store.get_offer(offer_row_id)
            if not offer:
                raise ValueError('Offre introuvable.')
            continual_store.upsert_annotation(
                offer_row_id=offer_row_id,
                offer_id=offer['offer_id'],
                content_version=offer['content_version'],
                canonical_name=canonical_name,
                surface_form=surface_form,
                normalized_name=normalize_skill_label(canonical_name),
                label='SKILL',
                start=start_i,
                end=end_i,
                confidence=1.0,
                source='human_review',
                provenance='human_review',
                is_explicit=start_i is not None and end_i is not None,
                text_sentence=clean_text(form.get('text_sentence')) or offer['description_original'],
                validation_status='approved',
                validated_at=datetime.now(timezone.utc).isoformat(),
                validated_by=validated_by,
            )
        elif action == 'exclude_offer':
            continual_store.mark_offer_status(offer_row_id, 'excluded', validation_actor=validated_by, validation_note=note)
        else:
            raise ValueError(f"Action inconnue: {action}")

        if redirect_offer in (None, '', '0'):
            return redirect(_admin_redirect_url(filters=selected_filters))
        return redirect(_admin_redirect_url(offer_row_id=int(redirect_offer), filters=selected_filters))
    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    create_app().run(host='127.0.0.1', port=5000, debug=False)


if __name__ == '__main__':
    main()
