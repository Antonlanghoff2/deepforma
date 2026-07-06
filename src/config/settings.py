from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from domain.errors import ConfigurationError


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"Valeur booléenne invalide: {value!r}")


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"Valeur entière invalide: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class DeepformaSettings:
    project_root: Path
    data_dir: Path
    models_dir: Path
    reports_dir: Path
    default_territory: str | None
    cache_ttl_seconds: int
    max_offers: int
    page_size: int
    debug: bool

    @classmethod
    def load(cls) -> "DeepformaSettings":
        project_root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.getenv("DEEPFORMA_DATA_DIR", project_root / "data"))
        models_dir = Path(os.getenv("DEEPFORMA_MODELS_DIR", project_root / "models"))
        reports_dir = Path(os.getenv("DEEPFORMA_REPORTS_DIR", project_root / "reports"))
        default_territory = os.getenv("DEEPFORMA_DEFAULT_TERRITORY") or None
        cache_ttl_seconds = _parse_int(os.getenv("DEEPFORMA_CACHE_TTL_SECONDS"), 3600)
        max_offers = _parse_int(os.getenv("DEEPFORMA_MAX_OFFERS"), 200)
        page_size = _parse_int(os.getenv("DEEPFORMA_PAGE_SIZE"), 20)
        debug = _parse_bool(os.getenv("DEEPFORMA_DEBUG"), False)
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            models_dir=models_dir,
            reports_dir=reports_dir,
            default_territory=default_territory,
            cache_ttl_seconds=cache_ttl_seconds,
            max_offers=max_offers,
            page_size=page_size,
            debug=debug,
        )


SETTINGS = DeepformaSettings.load()
