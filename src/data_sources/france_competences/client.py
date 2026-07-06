from __future__ import annotations

import json
import logging
import os
import os.path
import tempfile
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from deepforma.cpf.io import ensure_parent, sha256_file


LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET_SLUG = os.getenv(
    'FRANCE_COMPETENCES_DATASET_SLUG',
    'repertoire-national-des-certifications-professionnelles-et-repertoire-specifique',
)
DEFAULT_INCLUDE_RNCP = os.getenv('FRANCE_COMPETENCES_INCLUDE_RNCP', 'true').lower() in {'1', 'true', 'yes', 'on'}
DEFAULT_INCLUDE_RS = os.getenv('FRANCE_COMPETENCES_INCLUDE_RS', 'true').lower() in {'1', 'true', 'yes', 'on'}
DEFAULT_ACTIVE_ONLY = os.getenv('FRANCE_COMPETENCES_ACTIVE_ONLY', 'true').lower() in {'1', 'true', 'yes', 'on'}
DEFAULT_KEEP_EVALUATION = os.getenv('FRANCE_COMPETENCES_KEEP_EVALUATION', 'false').lower() in {'1', 'true', 'yes', 'on'}
DEFAULT_FORCE_DOWNLOAD = os.getenv('FRANCE_COMPETENCES_FORCE_DOWNLOAD', 'false').lower() in {'1', 'true', 'yes', 'on'}
DEFAULT_TIMEOUT = int(os.getenv('FRANCE_COMPETENCES_TIMEOUT', '60'))
DATASET_API_URL = 'https://www.data.gouv.fr/api/1/datasets/{slug}/'
SUPPORTED_FORMATS = {'.zip', '.csv', '.xml'}


class FranceCompetencesArchiveError(RuntimeError):
    pass


class FranceCompetencesDownloadError(FranceCompetencesArchiveError):
    pass


class FranceCompetencesVerificationError(FranceCompetencesArchiveError):
    pass


class FranceCompetencesResourceSelectionError(FranceCompetencesArchiveError):
    pass


@dataclass(slots=True)
class DownloadVerification:
    ok: bool
    path: str
    sha256: str
    sha1: str
    size: int
    expected_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'ok': self.ok,
            'path': self.path,
            'sha256': self.sha256,
            'sha1': self.sha1,
            'size': self.size,
            'expected_size': self.expected_size,
        }


def _normalize_resource_name(resource: dict[str, Any]) -> str:
    return (str(resource.get('title') or resource.get('url') or '')).lower()


def _resource_kind(resource: dict[str, Any]) -> str | None:
    text = _normalize_resource_name(resource)
    if 'rncp' in text:
        return 'rncp'
    if 'rs' in text:
        return 'rs'
    return None


def _resource_datetime(resource: dict[str, Any]) -> datetime:
    for key in ('last_modified', 'created_at'):
        value = resource.get(key)
        if isinstance(value, str) and value:
            try:
                text = value.replace('Z', '+00:00')
                return datetime.fromisoformat(text)
            except Exception:
                continue
    return datetime.fromtimestamp(0, tz=timezone.utc)


def _resource_basename(resource: dict[str, Any]) -> str:
    title = str(resource.get('title') or '').strip()
    if title:
        return title
    url = str(resource.get('url') or '').strip()
    if url:
        return Path(url).name or 'france_competences_resource'
    return 'france_competences_resource'


class FranceCompetencesClient:
    def __init__(
        self,
        *,
        dataset_slug: str | None = None,
        timeout: int | None = None,
        include_rncp: bool | None = None,
        include_rs: bool | None = None,
        active_only: bool | None = None,
        keep_evaluation: bool | None = None,
        force_download: bool | None = None,
        session: requests.Session | None = None,
        retries: int = 4,
        backoff_factor: float = 0.5,
    ) -> None:
        self.dataset_slug = dataset_slug or DEFAULT_DATASET_SLUG
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.include_rncp = DEFAULT_INCLUDE_RNCP if include_rncp is None else include_rncp
        self.include_rs = DEFAULT_INCLUDE_RS if include_rs is None else include_rs
        self.active_only = DEFAULT_ACTIVE_ONLY if active_only is None else active_only
        self.keep_evaluation = DEFAULT_KEEP_EVALUATION if keep_evaluation is None else keep_evaluation
        self.force_download = DEFAULT_FORCE_DOWNLOAD if force_download is None else force_download
        self._metadata_cache: dict[str, Any] | None = None
        self.session = session or requests.Session()
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            allowed_methods={'GET', 'HEAD'},
            status_forcelist={429, 500, 502, 503, 504},
            backoff_factor=backoff_factor,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

    def fetch_dataset_metadata(self) -> dict[str, Any]:
        if self._metadata_cache is not None:
            return self._metadata_cache
        url = DATASET_API_URL.format(slug=self.dataset_slug)
        LOGGER.info('Récupération des métadonnées data.gouv.fr: %s', url)
        response = self.session.get(url, timeout=self.timeout, headers={'Accept': 'application/json'})
        if response.status_code >= 400:
            raise FranceCompetencesDownloadError(f'Impossible de récupérer le dataset data.gouv.fr: HTTP {response.status_code}')
        payload = response.json()
        if not isinstance(payload, dict):
            raise FranceCompetencesDownloadError('Métadonnées data.gouv.fr invalides: objet JSON attendu.')
        self._metadata_cache = payload
        return payload

    def list_resources(self) -> list[dict[str, Any]]:
        payload = self.fetch_dataset_metadata()
        resources = payload.get('resources', [])
        if not isinstance(resources, list):
            return []
        return [resource for resource in resources if isinstance(resource, dict)]

    def _filter_resources(self, resources: Iterable[dict[str, Any]], *, include_rncp: bool, include_rs: bool) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for resource in resources:
            kind = _resource_kind(resource)
            if kind == 'rncp' and not include_rncp:
                continue
            if kind == 'rs' and not include_rs:
                continue
            if kind is None:
                continue
            fmt = str(resource.get('format') or '').lower()
            if fmt not in {'zip', 'csv', 'xml'}:
                continue
            filtered.append(resource)
        return filtered

    def select_latest_resource(
        self,
        resources: Iterable[dict[str, Any]] | None = None,
        *,
        include_rncp: bool | None = None,
        include_rs: bool | None = None,
    ) -> dict[str, Any]:
        include_rncp = self.include_rncp if include_rncp is None else include_rncp
        include_rs = self.include_rs if include_rs is None else include_rs
        candidates = list(resources or self.list_resources())
        filtered = self._filter_resources(candidates, include_rncp=include_rncp, include_rs=include_rs)
        if not filtered:
            raise FranceCompetencesResourceSelectionError('Aucune ressource RNCP/RS compatible trouvée.')
        filtered.sort(key=lambda item: (_resource_datetime(item), int(item.get('filesize') or 0)), reverse=True)
        return filtered[0]

    def _stream_download(self, url: str, destination: Path) -> None:
        LOGGER.info('Téléchargement: %s', url)
        with self.session.get(url, stream=True, timeout=self.timeout, headers={'Accept': '*/*'}) as response:
            if response.status_code >= 400:
                raise FranceCompetencesDownloadError(f'Téléchargement impossible: HTTP {response.status_code} pour {url}')
            response.raise_for_status()
            with destination.open('wb') as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

    @staticmethod
    def _digest_file(path: Path) -> tuple[str, str]:
        sha256_digest = sha256()
        sha1_digest = sha1()
        with path.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                sha256_digest.update(chunk)
                sha1_digest.update(chunk)
        return sha256_digest.hexdigest(), sha1_digest.hexdigest()

    def verify_download(self, path: Path, resource: dict[str, Any]) -> DownloadVerification:
        if not path.exists():
            raise FranceCompetencesVerificationError(f'Fichier manquant: {path}')
        size = path.stat().st_size
        expected_size = int(resource.get('filesize') or 0) or None
        checksum_sha256, checksum_sha1 = self._digest_file(path)
        ok = True
        if expected_size is not None and size != expected_size:
            ok = False
        checksum_meta = resource.get('checksum') or {}
        if isinstance(checksum_meta, dict):
            checksum_type = str(checksum_meta.get('type') or '').lower()
            checksum_value = str(checksum_meta.get('value') or '').strip()
            if checksum_type == 'sha1' and checksum_value and checksum_value != checksum_sha1:
                ok = False
        return DownloadVerification(ok=ok, path=str(path), sha256=checksum_sha256, sha1=checksum_sha1, size=size, expected_size=expected_size)

    def download_resource(
        self,
        resource: dict[str, Any],
        output_dir: str | Path,
        *,
        force: bool | None = None,
    ) -> tuple[Path, DownloadVerification]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        force = self.force_download if force is None else force
        target = output_dir / _resource_basename(resource)
        if target.exists() and not force:
            verification = self.verify_download(target, resource)
            if verification.ok:
                LOGGER.info('Ressource déjà présente et vérifiée: %s', target)
                return target, verification
            LOGGER.info('Ressource existante invalide, re-téléchargement: %s', target)

        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f'{target.stem}.', suffix='.download', dir=str(output_dir))
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            self._stream_download(str(resource.get('url') or ''), tmp_path)
            if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
                raise FranceCompetencesDownloadError(f'Téléchargement vide pour {resource.get("title")}')
            os.replace(tmp_path, target)
            verification = self.verify_download(target, resource)
            if not verification.ok:
                raise FranceCompetencesVerificationError(f'Checksum ou taille invalide pour {target}')
            LOGGER.info('Ressource téléchargée: %s', target)
            return target, verification
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def write_manifest(
        self,
        path: str | Path,
        *,
        dataset_metadata: dict[str, Any],
        resources: list[dict[str, Any]],
        downloaded_at: str | None = None,
        parser_version: str = '1.0',
    ) -> dict[str, Any]:
        path = Path(path)
        ensure_parent(path)
        payload_resources: list[dict[str, Any]] = []
        for item in resources:
            verification = item.get('verification') or {}
            payload_resources.append(
                {
                    'resource_id': item.get('resource_id'),
                    'resource_title': item.get('resource_title'),
                    'resource_url': item.get('resource_url'),
                    'format': item.get('format'),
                    'checksum': item.get('checksum'),
                    'size': item.get('size'),
                    'source_last_modified': item.get('source_last_modified'),
                    'local_path': item.get('local_path'),
                    'verification': verification,
                }
            )
        latest = payload_resources[0] if payload_resources else {}
        manifest = {
            'dataset_id': dataset_metadata.get('id') or dataset_metadata.get('dataset_id') or dataset_metadata.get('slug') or self.dataset_slug,
            'dataset_slug': dataset_metadata.get('slug') or self.dataset_slug,
            'resource_id': latest.get('resource_id'),
            'resource_title': latest.get('resource_title'),
            'resource_url': latest.get('resource_url'),
            'format': latest.get('format'),
            'checksum': latest.get('checksum'),
            'size': latest.get('size'),
            'source_last_modified': latest.get('source_last_modified'),
            'downloaded_at': downloaded_at or datetime.now(timezone.utc).isoformat(),
            'parser_version': parser_version,
            'resources': payload_resources,
        }
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
        LOGGER.info('Manifeste écrit: %s', path)
        return manifest



def safe_extract_zip(archive: str | Path, destination: str | Path) -> list[Path]:
    destination_path = Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    archive_path = Path(archive)
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            member_path = destination_path / member.filename
            resolved = member_path.resolve()
            if os.path.commonpath([str(destination_path), str(resolved)]) != str(destination_path):
                raise FranceCompetencesVerificationError(f'Archive ZIP invalide (path traversal): {member.filename}')
            if member.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, member_path.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(member_path)
    return extracted
