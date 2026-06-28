from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from deepforma.cpf.io import ensure_parent, json_dump, sha256_file


LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET_ID = "moncompteformation_catalogueformation"


def _guess_dataset_id(url: str) -> str | None:
    match = re.search(r"/dataset/([^/]+)/?", url)
    if match:
        return match.group(1)
    match = re.search(r"/catalog/datasets/([^/]+)/", url)
    if match:
        return match.group(1)
    return None


def _build_opendatasoft_csv_url(source_url: str, limit: int | None, api_page_size: int) -> str:
    parsed = urlparse(source_url)
    dataset_id = _guess_dataset_id(source_url) or DEFAULT_DATASET_ID
    base = f"{parsed.scheme or 'https'}://{parsed.netloc or 'opendata.caissedesdepots.fr'}"
    query = [f"format=csv"]
    if limit is not None:
        query.append(f"limit={int(limit)}")
    elif api_page_size:
        query.append(f"limit={int(api_page_size)}")
    return f"{base}/api/explore/v2.1/catalog/datasets/{dataset_id}/exports/csv?{'&'.join(query)}"


def _download_stream(url: str, destination: Path, timeout: int = 120) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def _copy_local(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Télécharge le catalogue CPF dans data/raw/cpf")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, default=None)
    parser.add_argument("--source-url", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api-page-size", type=int, default=50_000)
    return parser


def download_catalog(
    *,
    output_dir: Path,
    source_file: Path | None = None,
    source_url: str | None = None,
    force: bool = False,
    limit: int | None = None,
    api_page_size: int = 50_000,
) -> dict[str, str]:
    """Télécharge le catalogue CPF et produit un manifeste de téléchargement."""

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "cpf_catalog.csv"
    manifest_path = output_dir / "cpf_download_manifest.json"

    if target.exists() and not force:
        LOGGER.info("Le fichier existe déjà: %s. Utiliser --force pour le remplacer.", target)
        checksum = sha256_file(target)
        return {
            "path": str(target),
            "checksum_sha256": checksum,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "status": "existing",
        }

    tmp_fd, tmp_name = tempfile.mkstemp(prefix="cpf_catalog_", suffix=".tmp", dir=str(output_dir))
    tmp_path = Path(tmp_name)
    os.close(tmp_fd)
    try:
        if source_file:
            LOGGER.info("Copie du fichier source local: %s", source_file)
            _copy_local(source_file, tmp_path)
            source_kind = "local"
        elif source_url:
            parsed = urlparse(source_url)
            if "opendata.caissedesdepots.fr" in parsed.netloc and "/explore/dataset/" in parsed.path:
                source_url = _build_opendatasoft_csv_url(source_url, limit=limit, api_page_size=api_page_size)
                LOGGER.info("URL OpenDataSoft transformée en export CSV API: %s", source_url)
            elif parsed.netloc and "api.explore" in parsed.netloc:
                LOGGER.info("Utilisation de l'URL API fournie: %s", source_url)
            else:
                LOGGER.info("Téléchargement direct depuis: %s", source_url)
            _download_stream(source_url, tmp_path)
            source_kind = "remote"
        else:
            raise ValueError("Indiquer --source-file ou --source-url.")

        if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
            raise RuntimeError("Le fichier téléchargé est vide.")

        shutil.move(str(tmp_path), target)
        checksum = sha256_file(target)
        manifest = {
            "dataset_id": DEFAULT_DATASET_ID,
            "path": str(target),
            "checksum_sha256": checksum,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "source_kind": source_kind,
            "source_file": str(source_file) if source_file else None,
            "source_url": source_url,
            "limit": limit,
            "api_page_size": api_page_size,
        }
        json_dump(manifest_path, manifest)
        LOGGER.info("Téléchargement terminé: %s", target)
        LOGGER.info("Checksum SHA-256: %s", checksum)
        return manifest
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    download_catalog(
        output_dir=args.output_dir,
        source_file=args.source_file,
        source_url=args.source_url,
        force=args.force,
        limit=args.limit,
        api_page_size=args.api_page_size,
    )


if __name__ == "__main__":
    main()

