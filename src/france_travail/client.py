
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import urljoin

import requests

from os import getenv

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
FRANCE_TRAVAIL_QUERY_ROME_KEY = 'codeROME'
FRANCE_TRAVAIL_RESULT_ROME_KEY = 'romeCode'
ROME_CODE_RE = re.compile(r'^[A-Z][0-9]{4}$')


def normalize_rome_code(value: str | None) -> str:
    return (value or '').strip().upper()


def remove_empty_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in params.items()
        if value not in (None, '', [], ())
    }


def build_search_params(
    *,
    rome_code: str,
    territory_code: str,
    territory_type: str,
    range_value: str,
    keywords: str | None = None,
) -> dict[str, str]:
    params: dict[str, str] = {}
    normalized_rome_code = normalize_rome_code(rome_code)
    if normalized_rome_code:
        params[FRANCE_TRAVAIL_QUERY_ROME_KEY] = normalized_rome_code
    territory_value = (territory_code or '').strip()
    territory_key = (territory_type or '').strip().lower()
    if territory_value:
        if territory_key == 'commune':
            params['commune'] = territory_value
        elif territory_key == 'departement':
            params['departement'] = territory_value
    normalized_range = (range_value or '').strip()
    if normalized_range:
        params['distance'] = normalized_range
    normalized_keywords = (keywords or '').strip()
    if normalized_keywords:
        params['motsCles'] = normalized_keywords
    return remove_empty_params(params)


def _extract_result_rome_code(offer: dict[str, Any]) -> str:
    candidate = offer.get(FRANCE_TRAVAIL_RESULT_ROME_KEY) or offer.get('rome_code') or offer.get('code_rome')
    normalized = normalize_rome_code(candidate if isinstance(candidate, str) else str(candidate or ''))
    if normalized:
        return normalized
    rome = offer.get('rome')
    if isinstance(rome, dict):
        candidate = rome.get(FRANCE_TRAVAIL_RESULT_ROME_KEY) or rome.get('code')
        return normalize_rome_code(candidate if isinstance(candidate, str) else str(candidate or ''))
    return ''


def count_returned_rome_codes(offers: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for offer in offers:
        if not isinstance(offer, dict):
            counts['MALFORMED_OFFER'] += 1
            continue
        code = _extract_result_rome_code(offer)
        counts[code or 'UNKNOWN'] += 1
    return dict(counts)


@dataclass(frozen=True)
class SearchCriteria:
    keywords: str | None = None
    rome_code: str | None = None
    commune: str | None = None
    departement: str | None = None
    distance_km: int | None = None
    contract_type: str | None = None
    date_min: str | None = None
    date_max: str | None = None
    offset: int = 0
    size: int = 20
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_params(self) -> dict[str, Any]:
        territory_code = self.commune or self.departement or ''
        territory_type = 'commune' if self.commune else 'departement' if self.departement else ''
        params: dict[str, Any] = build_search_params(
            rome_code=self.rome_code or '',
            territory_code=territory_code,
            territory_type=territory_type,
            range_value=str(self.distance_km) if self.distance_km is not None else '',
            keywords=self.keywords,
        )
        if self.contract_type not in (None, ''):
            params['typeContrat'] = self.contract_type
        if self.date_min not in (None, ''):
            params['dateCreationMin'] = self.date_min
        if self.date_max not in (None, ''):
            params['dateCreationMax'] = self.date_max
        params.update({k: v for k, v in self.extra_params.items() if v is not None})
        return params

    @property
    def range_header(self) -> str:
        end = self.offset + self.size - 1
        return f'items={self.offset}-{end}'


@dataclass
class SearchResult:
    offers: list[dict[str, Any]]
    content_range: str | None
    status_code: int
    raw: dict[str, Any] | list[Any] | None = None


class FranceTravailError(RuntimeError):
    pass


class FranceTravailAuthError(FranceTravailError):
    pass


class FranceTravailRateLimitError(FranceTravailError):
    pass


class FranceTravailTimeoutError(FranceTravailError):
    pass


class FranceTravailClient:
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        scope: str | None = None,
        token_url: str | None = None,
        api_base_url: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
        load_env: bool = True,
    ) -> None:
        if load_env:
            load_dotenv()

        self.client_id = client_id or getenv('FRANCE_TRAVAIL_CLIENT_ID')
        self.client_secret = client_secret or getenv('FRANCE_TRAVAIL_CLIENT_SECRET')
        self.scope = scope or getenv('FRANCE_TRAVAIL_SCOPE', 'api_offresdemploiv2 o2dsoffre')
        self.token_url = token_url or getenv(
            'FRANCE_TRAVAIL_TOKEN_URL',
            'https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire',
        )
        self.api_base_url = api_base_url or getenv(
            'FRANCE_TRAVAIL_API_BASE_URL',
            'https://api.francetravail.io/partenaire/offresdemploi/v2',
        )
        self.timeout = timeout
        self.session = session or requests.Session()
        self._token: str | None = None
        self._token_expires_at: datetime | None = None

        if not self.client_id or not self.client_secret:
            raise ValueError(
                'FRANCE_TRAVAIL_CLIENT_ID et FRANCE_TRAVAIL_CLIENT_SECRET doivent être définis.'
            )

    def _is_token_valid(self) -> bool:
        if not self._token or not self._token_expires_at:
            return False
        return datetime.now(timezone.utc) < self._token_expires_at

    def authenticate(self, force_refresh: bool = False) -> str:
        if not force_refresh and self._is_token_valid():
            return self._token or ''

        payload = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': self.scope,
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = self.session.post(self.token_url, data=payload, headers=headers, timeout=self.timeout)
        if response.status_code != 200:
            raise FranceTravailAuthError(
                f'Authentification France Travail impossible (status={response.status_code}).'
            )
        data = response.json()
        token = data.get('access_token')
        expires_in = int(data.get('expires_in', 3600))
        if not token:
            raise FranceTravailAuthError('Réponse OAuth2 invalide: access_token manquant.')
        self._token = token
        self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(60, expires_in - 60))
        LOGGER.info('Token OAuth2 France Travail obtenu et mis en cache.')
        return token

    def _auth_headers(self) -> dict[str, str]:
        token = self.authenticate()
        return {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> requests.Response:
        url = urljoin(self.api_base_url.rstrip('/') + '/', path.lstrip('/'))
        current_headers = self._auth_headers()
        if headers:
            current_headers.update(headers)

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    headers=current_headers,
                    timeout=self.timeout,
                )
            except requests.Timeout as exc:
                raise FranceTravailTimeoutError(f'Timeout France Travail sur {path}.') from exc
            if response.status_code == 401 and attempt < MAX_RETRIES:
                self._token = None
                current_headers = self._auth_headers()
                continue
            if response.status_code in {403, 404}:
                raise FranceTravailError(
                    f'Erreur France Travail {response.status_code} sur {path}.'
                )
            if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                wait = 2 ** (attempt - 1)
                LOGGER.warning('Retry France Travail status=%s attempt=%s path=%s', response.status_code, attempt, path)
                time.sleep(wait)
                continue
            if response.status_code == 429:
                raise FranceTravailRateLimitError(f'Limite de débit France Travail atteinte sur {path}.')
            if response.status_code >= 400:
                raise FranceTravailError(
                    f'Erreur France Travail {response.status_code} sur {path}.'
                )
            return response
        if last_exc:
            raise last_exc
        raise FranceTravailError(f'Échec de la requête {method} {path}.')

    def search_offers(self, criteria: SearchCriteria | None = None, **kwargs: Any) -> SearchResult:
        if criteria is None:
            criteria = SearchCriteria(**kwargs)
        elif kwargs:
            raise TypeError('Utiliser soit criteria, soit les arguments nommés, pas les deux.')

        params = criteria.to_params()
        headers = {'Range': criteria.range_header}
        LOGGER.info(
            'France Travail search endpoint=%s rome_code=%s params=%s range=%s',
            '/offres/search',
            normalize_rome_code(criteria.rome_code),
            {key: value for key, value in params.items() if key not in {'Authorization', 'access_token'}},
            headers.get('Range'),
        )
        response = self._request('GET', '/offres/search', params=params, headers=headers)
        content_range = response.headers.get('Content-Range')
        payload = response.json()
        offers = self._extract_offers(payload)
        LOGGER.info(
            'France Travail response url=%s status=%s count=%s rome_distribution=%s',
            response.url,
            response.status_code,
            len(offers),
            count_returned_rome_codes(offers),
        )
        return SearchResult(
            offers=offers,
            content_range=content_range,
            status_code=response.status_code,
            raw=payload,
        )

    def search_offers_by_rome(
        self,
        rome_code: str,
        *,
        commune: str | None = None,
        departement: str | None = None,
        distance_km: int | None = None,
        contract_type: str | None = None,
        date_min: str | None = None,
        date_max: str | None = None,
        offset: int = 0,
        size: int = 20,
        extra_params: dict[str, Any] | None = None,
    ) -> SearchResult:
        criteria = SearchCriteria(
            rome_code=rome_code,
            commune=commune,
            departement=departement,
            distance_km=distance_km,
            contract_type=contract_type,
            date_min=date_min,
            date_max=date_max,
            offset=offset,
            size=size,
            extra_params=extra_params or {},
        )
        return self.search_offers(criteria)

    def get_offer(self, offer_id: str) -> dict[str, Any]:
        response = self._request('GET', f'/offres/{offer_id}')
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        raise FranceTravailError('Réponse offer invalide: dictionnaire attendu.')

    def iter_offers(
        self,
        criteria: SearchCriteria | None = None,
        *,
        max_pages: int | None = None,
        max_offers: int | None = None,
        page_size: int = 20,
        pause_seconds: float = 0.0,
    ) -> Iterable[dict[str, Any]]:
        criteria = criteria or SearchCriteria()
        seen: set[str] = set()
        total_emitted = 0
        page = 0
        while max_pages is None or page < max_pages:
            current = SearchCriteria(
                keywords=criteria.keywords,
                rome_code=criteria.rome_code,
                commune=criteria.commune,
                departement=criteria.departement,
                distance_km=criteria.distance_km,
                contract_type=criteria.contract_type,
                date_min=criteria.date_min,
                date_max=criteria.date_max,
                offset=criteria.offset + page * page_size,
                size=page_size,
                extra_params=criteria.extra_params,
            )
            result = self.search_offers(current)
            offers = result.offers
            if not offers:
                break
            yield_count = 0
            for offer in offers:
                offer_id = str(offer.get('id') or offer.get('idOffre') or offer.get('reference') or '')
                if not offer_id or offer_id in seen:
                    continue
                seen.add(offer_id)
                yield offer
                yield_count += 1
                total_emitted += 1
                if max_offers is not None and total_emitted >= max_offers:
                    return
            if len(offers) < page_size or yield_count == 0:
                break
            page += 1
            if pause_seconds > 0:
                time.sleep(pause_seconds)

    @staticmethod
    def _extract_offers(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            for key in ('resultats', 'results', 'items', 'offres'):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []
