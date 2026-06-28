from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from charset_normalizer import from_path


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectedFormat:
    """Format détecté pour un fichier texte tabulaire."""

    encoding: str
    separator: str
    has_header: bool = True


def detect_encoding(path: Path, fallback: str = "utf-8") -> str:
    """Détecte l'encodage probable d'un fichier."""

    try:
        result = from_path(path).best()
        if result and result.encoding:
            return str(result.encoding)
    except Exception:
        LOGGER.debug("Détection d'encodage impossible pour %s", path, exc_info=True)
    return fallback


def detect_separator(path: Path, encoding: str, sample_size: int = 32_768) -> str:
    """Détecte le séparateur d'un CSV de manière tolérante."""

    sample = path.read_bytes()[:sample_size]
    try:
        text = sample.decode(encoding, errors="ignore")
    except LookupError:
        text = sample.decode("utf-8", errors="ignore")

    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(text, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        candidates = {sep: text.count(sep) for sep in [",", ";", "\t", "|"]}
        return max(candidates, key=candidates.get) if any(candidates.values()) else ","


def detect_text_format(path: Path) -> DetectedFormat:
    """Retourne l'encodage et le séparateur les plus probables."""

    encoding = detect_encoding(path)
    separator = detect_separator(path, encoding)
    return DetectedFormat(encoding=encoding, separator=separator)


def ensure_parent(path: Path) -> None:
    """Crée le dossier parent si nécessaire."""

    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcule le checksum SHA-256 d'un fichier."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Écrit un JSON UTF-8 lisible."""

    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=indent), encoding="utf-8")


def chunked(iterable: Iterable[Any], size: int) -> list[list[Any]]:
    """Découpe un itérable en lots de taille fixe."""

    batch: list[Any] = []
    batches: list[list[Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches
