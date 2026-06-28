from __future__ import annotations

import csv
import io
import json
import logging
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request, Response

from analytics.territorial_skills import compute_territorial_stats
from common.text import clean_text
from config.thresholds import THRESHOLDS
from config.weights import SCORING_WEIGHTS
from france_travail.client import FranceTravailAuthError, FranceTravailClient, FranceTravailError, FranceTravailRateLimitError, FranceTravailTimeoutError, SearchCriteria
from france_travail.normalizer import normalize_offer
from inference.deepforma_predictor import DeepformaPredictor, get_predictor
from models.analysis_result import (
    AnalysisResult, CheckpointAuditInfo, ClassificationInfo, MarketComparisonItem,
    MarketSkillInfo, ModelMetadata, QualityInfo, Recommendation,
    SkillInfo, TerritorialMarketInfo,
)
from skills.merge_offer_skills import extract_skills_from_text, merge_offer_skills
from services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / 'templates'
STATIC_DIR = PROJECT_ROOT / 'static'
DEFAULT_CACHE_TTL_SECONDS = int(os.getenv('DEEPFORMA_CACHE_TTL_SECONDS', '600'))
DEFAULT_MAX_OFFERS = int(os.getenv('DEEPFORMA_MAX_OFFERS', '25'))
DEFAULT_PAGE_SIZE = int(os.getenv('DEEPFORMA_PAGE_SIZE', '10'))
DEFAULT_MAX_PAGES = int(os.getenv('DEEPFORMA_MAX_PAGES', '3'))
DEFAULT_THRESHOLD = float(os.getenv('DEEPFORMA_DEFAULT_THRESHOLD', '0.35'))

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


def _check_quality(skills_result: dict[str, Any]) -> QualityInfo:
    score_std = skills_result.get('score_std', 0.0)
    score_max = skills_result.get('score_max', 0.0)
    score_mean = skills_result.get('score_mean', 0.0)
    score_min = skills_result.get('score_min', 0.0)
    discriminating = score_std > 0.05 or score_max > 0.70
    warnings: list[str] = []
    if not discriminating:
        warnings.append(
            'Les scores de competences sont tous regroupes autour de {:.3f} (ecart-type={:.4f}). '
            'Le modele ne discrimine pas correctement.'.format(score_mean, score_std)
        )
    if score_max < 0.50:
        warnings.append('Aucune competence ne depasse 50%% de probabilite.')
    if score_min > 0.40 and score_max < 0.60:
        warnings.append(
            'Tous les scores sont compris entre {:.2f} et {:.2f}. '
            'Les resultats ne sont pas fiables.'.format(score_min, score_max)
        )
    return QualityInfo(
        model_loaded=True,
        skills_discriminating=discriminating,
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
        score_std=score_std,
        offers_sufficient=False,
        warnings=warnings,
    )


def _build_analysis_result(
    analysis: dict[str, Any],
    normalized_offers: list[dict[str, Any]],
    recommendation: Any,
    territorial_stats: Any,
    departement: str,
    threshold: float,
) -> AnalysisResult:
    result = AnalysisResult()

    binary = analysis['binary']
    skills_result = analysis['skills']
    predictions = skills_result.get('predictions', [])
    all_scores = skills_result.get('all_scores', [])

    class_state = THRESHOLDS.get_classification_state(
        binary['probability_ia'], binary['probability_non_ia']
    )
    result.classification = ClassificationInfo(
        is_ia=binary['is_ia'],
        predicted_class=binary['predicted_class'],
        probability_ia=binary['probability_ia'],
        probability_non_ia=binary['probability_non_ia'],
        state=class_state['state'],
        state_description=class_state['description'],
        gap=class_state['gap'],
    )

    result.quality = _check_quality(skills_result)
    result.quality.offers_sufficient = len(normalized_offers) >= THRESHOLDS.min_offers_for_conclusion

    discriminating = result.quality.skills_discriminating
    score_std = result.quality.score_std
    score_max = result.quality.score_max

    # ---- Build skills ----
    detected_skills: list[SkillInfo] = []
    low_confidence_skills: list[SkillInfo] = []
    rejected_skills: list[SkillInfo] = []
    indeterminate_skills: list[SkillInfo] = []

    for p in predictions:
        prob = p['probability']
        label = p['label']
        confidence = _skill_confidence(prob)
        skill = SkillInfo(
            label=label,
            score_brut=round(prob, 4),
            niveau_confiance=confidence,
            seuil_applique=threshold,
            methode_detection='camembert_multilabel',
        )

        if not discriminating:
            skill.presence = 'indeterminate'
            skill.statut = 'indetermine'
            indeterminate_skills.append(skill)
        elif prob >= threshold and confidence in ('forte', 'moyenne'):
            skill.presence = 'present'
            skill.statut = 'central' if prob >= 0.70 else 'secondaire'
            detected_skills.append(skill)
        elif prob >= threshold * 0.5:
            skill.presence = 'indeterminate'
            skill.statut = 'a_verifier'
            low_confidence_skills.append(skill)
        else:
            skill.presence = 'absent'
            skill.statut = 'rejete'
            rejected_skills.append(skill)

    detected_skills.sort(key=lambda s: s.score_brut, reverse=True)
    low_confidence_skills.sort(key=lambda s: s.score_brut, reverse=True)
    indeterminate_skills.sort(key=lambda s: s.score_brut, reverse=True)

    result.detected_skills = detected_skills
    result.low_confidence_skills = low_confidence_skills
    result.rejected_skills = rejected_skills
    result.indeterminate_skills = indeterminate_skills

    # ---- Formation analysis status ----
    if not discriminating:
        result.formation_analysis_status = 'unreliable'
        result.skills_presence = 'indeterminate'
        result.comparison_available = False
        result.recommendations_available = False
        result.blocking_reasons = ['skill_scores_not_discriminant']
    elif len(detected_skills) == 0:
        result.formation_analysis_status = 'no_skills_detected'
        result.skills_presence = 'indeterminate'
        result.comparison_available = False
        result.recommendations_available = False
        result.blocking_reasons = ['no_skills_detected']
    else:
        result.formation_analysis_status = 'reliable'
        result.skills_presence = 'determinate'
        result.comparison_available = True
        result.recommendations_available = True
        result.blocking_reasons = []

    # ---- Model metadata ----
    binary_model_checkpoint = str(getattr(analysis, 'binary_model_dir', 'models/binary_ia_v2/final'))
    multilabel_model_checkpoint = str(getattr(analysis, 'multilabel_model_dir', 'models/multilabel_competences_v2/final'))
    checkpoint_audit_raw = analysis.get('checkpoint_audit', {})

    result.checkpoint_audit = CheckpointAuditInfo(
        config_present=checkpoint_audit_raw.get('config_present', False),
        weights_present=checkpoint_audit_raw.get('weights_present', False),
        weights_size_bytes=checkpoint_audit_raw.get('weights_size_bytes', 0),
        architecture_declared=checkpoint_audit_raw.get('architecture_declared', ''),
        num_labels_declared=checkpoint_audit_raw.get('num_labels_declared', 0),
        problem_type=checkpoint_audit_raw.get('problem_type', ''),
        id2label_count=checkpoint_audit_raw.get('id2label_count', 0),
        label2id_count=checkpoint_audit_raw.get('label2id_count', 0),
        strict_load_success=checkpoint_audit_raw.get('strict_load_success', False),
        missing_keys=checkpoint_audit_raw.get('missing_keys', []),
        unexpected_keys=checkpoint_audit_raw.get('unexpected_keys', []),
        ignored_keys=checkpoint_audit_raw.get('ignored_keys', []),
        classifier_weight_shape=checkpoint_audit_raw.get('classifier_weight_shape', ''),
        classifier_weight_mean=checkpoint_audit_raw.get('classifier_weight_mean', 0.0),
        classifier_weight_std=checkpoint_audit_raw.get('classifier_weight_std', 0.0),
        classifier_weight_min=checkpoint_audit_raw.get('classifier_weight_min', 0.0),
        classifier_weight_max=checkpoint_audit_raw.get('classifier_weight_max', 0.0),
        classifier_bias_mean=checkpoint_audit_raw.get('classifier_bias_mean'),
        appears_random_init=checkpoint_audit_raw.get('appears_random_init', True),
    )

    result.model_metadata = ModelMetadata(
        binary_model='CamemBERT (CamembertForSequenceClassification)',
        multilabel_model='CamemBERT (CamembertForSequenceClassification)',
        binary_checkpoint=binary_model_checkpoint,
        multilabel_checkpoint=multilabel_model_checkpoint,
        device=analysis.get('device', 'cpu'),
        max_length=512,
        num_labels=len(skills_result.get('predictions', [])),
        labels=[p['label'] for p in predictions] if predictions else [],
        thresholds={'multilabel': threshold, 'binary': None},
        inference_time_ms=analysis.get('inference_time_ms', 0.0),
        classifier_weight_stats={
            'mean': result.checkpoint_audit.classifier_weight_mean,
            'std': result.checkpoint_audit.classifier_weight_std,
            'min': result.checkpoint_audit.classifier_weight_min,
            'max': result.checkpoint_audit.classifier_weight_max,
            'appears_random_init': result.checkpoint_audit.appears_random_init,
        },
    )

    # ---- Territorial market (independent) ----
    if territorial_stats:
        market_skills_sorted = sorted(
            territorial_stats.skill_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_skills = [
            MarketSkillInfo(
                label=label,
                offer_count=count,
                share_percent=round(
                    count / territorial_stats.offer_count * 100, 2
                ) if territorial_stats.offer_count else 0.0,
            )
            for label, count in market_skills_sorted[:20]
        ]
        robust = 'forte' if territorial_stats.offer_count >= THRESHOLDS.statistical_robustness_min else (
            'moyenne' if territorial_stats.offer_count >= THRESHOLDS.min_offers_for_conclusion else 'faible'
        )
        alert = ''
        if territorial_stats.offer_count < THRESHOLDS.min_offers_for_conclusion:
            alert = (
                f"Nombre d'offres trop faible ({territorial_stats.offer_count}) "
                "pour une analyse territoriale fiable."
            )
        elif territorial_stats.offer_count < THRESHOLDS.statistical_robustness_min:
            alert = (
                f"Volume d'offres modere ({territorial_stats.offer_count}). "
                "Les tendances restent indicatives."
            )
        result.territorial_market = TerritorialMarketInfo(
            territory=departement,
            period='Derniers mois (source: France Travail)',
            offer_count=territorial_stats.offer_count,
            exploitable_offers=len(normalized_offers),
            top_skills=top_skills,
            contract_types=getattr(territorial_stats, 'contract_types', {}),
            statistical_robustness=robust,
            alert=alert,
        )

    # ---- Comparison and recommendations (only if reliable) ----
    if recommendation and result.comparison_available:
        formation_labels = set(normalize_skill_label(s.label) for s in detected_skills)
        market_lookup = {}
        for ms in recommendation.market_skills:
            market_lookup[normalize_skill_label(ms.label)] = ms

        comparison_items: list[MarketComparisonItem] = []
        all_compared_labels = set()

        for skill_key, ms in market_lookup.items():
            in_formation = skill_key in formation_labels
            detection_conf = 0.0
            for ds in detected_skills:
                if normalize_skill_label(ds.label) == skill_key:
                    detection_conf = ds.score_brut
                    break
            coverage = 'complete' if in_formation else 'absente'
            priority = 'haute' if ms.offer_count >= 5 else 'moyenne'
            comparison_items.append(MarketComparisonItem(
                skill=ms.label,
                in_formation=in_formation,
                detection_confidence=detection_conf,
                frequency_in_offers=ms.share_percent,
                offer_count=ms.offer_count,
                coverage_level=coverage,
                priority=priority,
            ))
            all_compared_labels.add(skill_key)

        covered = [c for c in comparison_items if c.in_formation]
        overrepresented = [
            c for c in comparison_items
            if c.in_formation and c.frequency_in_offers < 5.0
        ]
        missing = [c for c in comparison_items if not c.in_formation]

        result.formation_market_comparison = comparison_items
        result.comparison_categories = {
            'covered': covered,
            'overrepresented': overrepresented,
            'missing': missing,
        }
        result.missing_skills = [
            MarketSkillInfo(label=c.skill, offer_count=c.offer_count, share_percent=c.frequency_in_offers)
            for c in missing
        ]

        sub_score_values = {}
        if len(detected_skills) > 0 and len(market_lookup) > 0:
            coverage_pct = len(covered) / max(len(market_lookup), 1)
            sub_score_values['couverture_competences'] = coverage_pct * 100
            sub_score_values['pertinence_metier'] = min(100.0, coverage_pct * 120)
            sub_score_values['adequation_territoriale'] = min(100.0, coverage_pct * 100)
            sub_score_values['niveau_experience'] = 50.0
            sub_score_values['employabilite'] = min(100.0, coverage_pct * 150)
            sub_score_values['actualite_programme'] = 50.0
            result.global_score = SCORING_WEIGHTS.compute_global(sub_score_values)

        recommendations: list[Recommendation] = []
        seen_recs: set[str] = set()

        for c in missing[:5]:
            if c.skill not in seen_recs:
                seen_recs.add(c.skill)
                recommendations.append(Recommendation(
                    type='competence_a_ajouter',
                    skill=c.skill,
                    justification=(
                        f"Competence demandee dans {c.offer_count} offres locales "
                        f"({c.frequency_in_offers:.1f}%) mais absente de la formation."
                    ),
                    impact_estime='eleve' if c.offer_count >= 5 else 'moyen',
                    offer_count=c.offer_count,
                    offer_percent=round(c.frequency_in_offers, 1),
                    priorite='haute' if c.offer_count >= 5 else 'moyenne',
                    niveau_confiance='forte' if c.offer_count >= 10 else 'moyenne',
                ))

        for ds in detected_skills:
            skill_key = normalize_skill_label(ds.label)
            if skill_key not in market_lookup and ds.score_brut >= 0.70:
                if ds.label not in seen_recs:
                    seen_recs.add(ds.label)
                    recommendations.append(Recommendation(
                        type='competence_peu_utile_localement',
                        skill=ds.label,
                        justification=(
                            f"Competence '{ds.label}' bien detectee dans la formation "
                            "mais peu presente dans les offres locales."
                        ),
                        impact_estime='faible',
                        offer_count=0,
                        offer_percent=0.0,
                        priorite='basse',
                        niveau_confiance='moyenne',
                    ))

        if len(overrepresented) > 0:
            for c in overrepresented[:3]:
                if c.skill not in seen_recs:
                    seen_recs.add(c.skill)
                    recommendations.append(Recommendation(
                        type='contenu_surrepresente',
                        skill=c.skill,
                        justification=(
                            f"Competence '{c.skill}' presente dans la formation "
                            f"mais faiblement demandee localement ({c.frequency_in_offers:.1f}% des offres)."
                        ),
                        impact_estime='moyen',
                        offer_count=c.offer_count,
                        offer_percent=round(c.frequency_in_offers, 1),
                        priorite='moyenne',
                        niveau_confiance='moyenne',
                    ))

        priorities = {'haute': 0, 'moyenne': 1, 'basse': 2}
        recommendations.sort(key=lambda r: (priorities.get(r.priorite, 99), -r.offer_count))
        result.recommendations = recommendations

    # ---- Summary ----
    result.summary = {
        'formation_analysis_status': result.formation_analysis_status,
        'total_skills_detected': len(detected_skills),
        'total_skills_low_confidence': len(low_confidence_skills),
        'total_skills_indeterminate': len(indeterminate_skills),
        'total_skills_rejected': len(rejected_skills),
        'total_offers_analyzed': len(normalized_offers),
        'classification_state': class_state['state'],
        'global_score': result.global_score.get('global_score') if result.global_score else None,
        'inference_time_ms': analysis.get('inference_time_ms', 0.0),
        'analyzed_at': datetime.now(timezone.utc).isoformat(),
    }

    return result


def normalize_skill_label(label: str) -> str:
    from common.text import normalize_for_match
    return normalize_for_match(label)


def create_app(
    predictor: DeepformaPredictor | None = None,
    cache_ttl_seconds: int | None = None,
    france_travail_client_factory: Any | None = None,
) -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
    app.config.update(
        CACHE_TTL_SECONDS=cache_ttl_seconds or DEFAULT_CACHE_TTL_SECONDS,
        MAX_OFFERS=DEFAULT_MAX_OFFERS,
        PAGE_SIZE=DEFAULT_PAGE_SIZE,
        MAX_PAGES=DEFAULT_MAX_PAGES,
        DEFAULT_THRESHOLD=DEFAULT_THRESHOLD,
    )

    predictor_error = None
    if predictor is None:
        predictor, predictor_error = _load_predictor()
    app.extensions['deepforma_predictor'] = predictor
    app.extensions['deepforma_predictor_error'] = predictor_error
    app.extensions['recommendation_service'] = RecommendationService()
    app.extensions['market_cache'] = TTLCache(app.config['CACHE_TTL_SECONDS'])
    app.extensions['france_travail_client_factory'] = france_travail_client_factory or _build_france_travail_client

    def get_predictor_instance() -> DeepformaPredictor | None:
        return app.extensions.get('deepforma_predictor')

    def get_market_client() -> FranceTravailClient:
        factory = app.extensions['france_travail_client_factory']
        return factory()

    def analyze_market(departement: str, keywords: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cache = app.extensions['market_cache']
        cache_key = _make_cache_key(departement, keywords)
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

    def _build_context(text: str, departement: str, keywords: str | None, threshold: float, model_only: bool) -> dict[str, Any]:
        predictor_instance = get_predictor_instance()
        if predictor_instance is None:
            raise RuntimeError(app.extensions.get('deepforma_predictor_error') or 'Les modeles ne sont pas disponibles.')

        analysis = predictor_instance.analyze(text, threshold=threshold)
        normalized_offers: list[dict[str, Any]] = []
        recommendation = None
        territorial_stats = None
        market_status = 'skipped' if model_only else 'not_requested'

        if not model_only:
            try:
                normalized_offers, _ = analyze_market(departement, keywords)
            except ValueError as exc:
                raise RuntimeError('Configuration France Travail absente ou invalide.') from exc
            except FranceTravailRateLimitError as exc:
                raise RuntimeError('France Travail a repondu avec une limite de debit (429).') from exc
            except FranceTravailTimeoutError as exc:
                raise RuntimeError("Le delai d'attente France Travail a expire.") from exc
            except FranceTravailAuthError as exc:
                raise RuntimeError('Authentification France Travail invalide ou expiree.') from exc
            except FranceTravailError as exc:
                message = str(exc)
                if '429' in message:
                    raise RuntimeError('France Travail a repondu avec une limite de debit (429).') from exc
                raise RuntimeError('Erreur France Travail lors de la recuperation des offres.') from exc
            except requests.Timeout as exc:
                raise RuntimeError("Le delai d'attente France Travail a expire.") from exc
            except requests.RequestException as exc:
                raise RuntimeError("Erreur reseau lors de l'appel a France Travail.") from exc

            service: RecommendationService = app.extensions['recommendation_service']
            recommendation = service.compare(analysis['skills']['predictions'], normalized_offers)
            territorial_stats = compute_territorial_stats(normalized_offers, territory_key=departement)
            market_status = 'ok'

        analysis_result = _build_analysis_result(
            analysis, normalized_offers, recommendation, territorial_stats,
            departement, threshold,
        )

        return {
            'analysis': analysis,
            'context': {
                'normalized_offers': normalized_offers,
                'territorial_stats': territorial_stats,
                'recommendation': recommendation,
                'market_status': market_status,
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

    @app.get('/')
    def index():
        return render_template(
            'index.html',
            error=None,
            department_options=DEPARTMENT_CODES,
            default_threshold=app.config['DEFAULT_THRESHOLD'],
        )

    @app.post('/analyze')
    def analyze():
        try:
            text, departement, keywords, threshold, model_only = _extract_inputs(_parse_request_payload())
            context = _build_context(text, departement, keywords, threshold, model_only)
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

    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    create_app().run(host='127.0.0.1', port=5000, debug=False)


if __name__ == '__main__':
    main()
