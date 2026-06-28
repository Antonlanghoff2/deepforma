from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, render_template, request

from analytics.territorial_skills import compute_territorial_stats
from common.text import clean_text
from france_travail.client import FranceTravailAuthError, FranceTravailClient, FranceTravailError, FranceTravailRateLimitError, FranceTravailTimeoutError, SearchCriteria
from france_travail.normalizer import normalize_offer
from inference.deepforma_predictor import DeepformaPredictor, get_predictor
from skills.merge_offer_skills import extract_skills_from_text, merge_offer_skills
from services.recommendation_service import RecommendationService


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

EXPERIMENTAL_WARNING = 'Résultat expérimental. Le modèle doit encore être validé avant utilisation opérationnelle.'


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
                raise ValueError('Le seuil doit être un nombre compris entre 0 et 1.')
        if not 0.0 <= threshold <= 1.0:
            raise ValueError('Le seuil doit être compris entre 0 et 1.')
        if not text:
            raise ValueError('Le programme de formation est obligatoire.')
        if not departement:
            raise ValueError('Le département est obligatoire.')
        return text, departement, keywords, threshold, model_only

    def _build_context(text: str, departement: str, keywords: str | None, threshold: float, model_only: bool) -> dict[str, Any]:
        predictor_instance = get_predictor_instance()
        if predictor_instance is None:
            raise RuntimeError(app.extensions.get('deepforma_predictor_error') or 'Les modèles ne sont pas disponibles.')

        analysis = predictor_instance.analyze(text, threshold=threshold)
        market_context: dict[str, Any] = {
            'normalized_offers': [],
            'territorial_stats': None,
            'recommendation': None,
            'market_status': 'skipped' if model_only else 'not_requested',
        }

        if not model_only:
            try:
                normalized_offers, _ = analyze_market(departement, keywords)
            except ValueError as exc:
                raise RuntimeError('Configuration France Travail absente ou invalide.') from exc
            except FranceTravailRateLimitError as exc:
                raise RuntimeError('France Travail a répondu avec une limite de débit (429).') from exc
            except FranceTravailTimeoutError as exc:
                raise RuntimeError("Le délai d'attente France Travail a expiré.") from exc
            except FranceTravailAuthError as exc:
                raise RuntimeError('Authentification France Travail invalide ou expirée.') from exc
            except FranceTravailError as exc:
                message = str(exc)
                if '429' in message:
                    raise RuntimeError('France Travail a répondu avec une limite de débit (429).') from exc
                raise RuntimeError('Erreur France Travail lors de la récupération des offres.') from exc
            except requests.Timeout as exc:
                raise RuntimeError("Le délai d'attente France Travail a expiré.") from exc
            except requests.RequestException as exc:
                raise RuntimeError("Erreur réseau lors de l'appel à France Travail.") from exc

            service: RecommendationService = app.extensions['recommendation_service']
            recommendation = service.compare(analysis['skills'], normalized_offers)
            territorial_stats = compute_territorial_stats(normalized_offers, territory_key=departement)
            market_context.update(
                {
                    'normalized_offers': normalized_offers,
                    'territorial_stats': territorial_stats,
                    'recommendation': recommendation,
                    'market_status': 'ok',
                }
            )

        return {
            'analysis': analysis,
            'context': market_context,
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

        return render_template(
            'result.html',
            **context,
            department_options=DEPARTMENT_CODES,
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
            elif 'délai' in message.lower():
                status = 504
            return jsonify({'ok': False, 'error': message}), status

        recommendation = context['context']['recommendation']
        territorial_stats = context['context']['territorial_stats']
        payload = {
            'ok': True,
            'warning': context['warning'],
            'department': context['department'],
            'keywords': context['keywords'],
            'threshold': context['threshold'],
            'model_only': context['model_only'],
            'analysis': context['analysis'],
            'market': {
                'status': context['context']['market_status'],
                'offer_count': context['context']['territorial_stats'].offer_count if territorial_stats else 0,
                'territorial_stats': asdict(territorial_stats) if territorial_stats else None,
                'recommendation': _serialize_report(recommendation) if recommendation else None,
                'normalized_offer_count': len(context['context']['normalized_offers']),
            },
        }
        return jsonify(payload)

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

    return app


def main() -> None:
    create_app().run(host='127.0.0.1', port=5000, debug=False)


if __name__ == '__main__':
    main()
