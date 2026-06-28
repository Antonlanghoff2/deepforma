from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common.text import clean_text, normalize_for_match
from deepforma.cpf.cleaning import CPFDeduper, normalize_row, row_hash
from deepforma.cpf.columns import detect_columns, load_column_aliases
from deepforma.cpf.embeddings import compute_corpus_hash
from deepforma.cpf.io import detect_text_format, json_dump
from deepforma.cpf.skill_extractor import CPFSkillExtractor


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparationStats:
    """Statistiques de nettoyage du catalogue CPF."""

    rows_read: int
    rows_kept: int
    rows_rejected: int
    exact_duplicates: int
    near_duplicates: int
    enriched_rows: int
    unique_formations: int


def _write_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    """Écrit les enregistrements au format parquet si pyarrow est disponible."""

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - dépendance optionnelle
        raise ImportError(
            "pyarrow est nécessaire pour écrire le parquet CPF. Installer pyarrow ou exécuter ce script dans l'environnement cible."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(pd.DataFrame.from_records(records), preserve_index=False)
    pq.write_table(table, output_path)


def _safe_value(value: Any) -> Any:
    if isinstance(value, list):
        return [
            _safe_value(item)
            for item in value
        ]
    if isinstance(value, dict):
        return {key: _safe_value(item) for key, item in value.items()}
    if value in ("", None):
        return None
    return value


def prepare_catalog(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    chunksize: int = 25_000,
    sample_limit: int = 1_000,
    similarity_threshold: float = 0.96,
) -> dict[str, Any]:
    """Nettoie et normalise le catalogue CPF en streaming."""

    source_path = Path(csv_path)
    if source_path.suffix.lower() in {'.xlsx', '.xls'}:
        from data.cpf_loader import prepare_cpf_v3_dataset

        prepared = prepare_cpf_v3_dataset(source_path, output_dir, config_path=config_path)
        frame = prepared.frame
        stats = PreparationStats(
            rows_read=int(prepared.report['rows_initial']),
            rows_kept=int(prepared.report['rows_kept']),
            rows_rejected=int(prepared.report['rows_rejected']),
            exact_duplicates=int(prepared.report['duplicates']),
            near_duplicates=0,
            enriched_rows=int(prepared.report['formations_with_skills']),
            unique_formations=int(frame['formation_id'].nunique()) if not frame.empty else 0,
        )
        return {
            'stats': stats,
            'report': prepared.report,
            'parquet_path': Path(prepared.report['output_files']['formations_normalized_parquet']),
            'sample_path': Path(prepared.report['output_files']['formations_normalized_csv']),
            'report_path': Path(prepared.report['output_files']['import_report']),
            'kept_rows': frame.to_dict(orient='records'),
            'corpus_hash': prepared.report.get('corpus_hash'),
        }

    output_root = Path(output_dir)
    processed_dir = output_root / "processed" / "cpf"
    reports_dir = output_root / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    fmt = detect_text_format(source_path)
    alias_map = load_column_aliases(config_path)
    deduper = CPFDeduper(similarity_threshold=similarity_threshold)
    extractor = CPFSkillExtractor()

    kept_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    rows_read = 0

    first_chunk = True
    columns: list[str] = []
    column_detection = None

    for chunk in pd.read_csv(
        source_path,
        sep=fmt.separator,
        encoding=fmt.encoding,
        chunksize=chunksize,
        low_memory=False,
    ):
        rows_read += len(chunk)
        if first_chunk:
            columns = list(chunk.columns)
            column_detection = detect_columns(columns, alias_map)
            first_chunk = False

        for raw_row in chunk.to_dict(orient="records"):
            normalized = normalize_row(raw_row, column_detection)  # type: ignore[arg-type]
            normalized["row_hash"] = row_hash(normalized)
            if deduper.is_duplicate(normalized):
                continue
            skill_result = extractor.extract(normalized)
            normalized.update(
                {
                    "skills_explicit": skill_result.skills_explicit,
                    "skills_inferred": skill_result.skills_inferred,
                    "skills_normalized": skill_result.skills_normalized,
                    "skills_confidence": skill_result.skills_confidence,
                    "skills_evidence": skill_result.skills_evidence,
                }
            )
            deduper.register(normalized)
            kept_rows.append(normalized)
            if len(sample_rows) < sample_limit:
                sample_rows.append({key: _safe_value(value) for key, value in normalized.items()})

    if not kept_rows:
        LOGGER.warning("Aucune formation conservée après nettoyage.")

    parquet_path = processed_dir / "formations.parquet"
    sample_path = processed_dir / "formations_sample.csv"
    report_path = reports_dir / "cpf_cleaning_report.json"

    _write_parquet(kept_rows, parquet_path)
    pd.DataFrame.from_records(sample_rows).to_csv(sample_path, index=False, encoding="utf-8")

    corpus_hash = compute_corpus_hash(kept_rows)
    report = {
        "source_path": str(source_path),
        "encoding": fmt.encoding,
        "separator": fmt.separator,
        "rows_read": rows_read,
        "rows_kept": len(kept_rows),
        "rows_rejected": rows_read - len(kept_rows),
        "exact_duplicates": deduper.stats.exact_duplicates,
        "near_duplicates": deduper.stats.near_duplicates,
        "enriched_rows": sum(1 for row in kept_rows if row.get("skills_normalized")),
        "unique_formations": len({row.get("formation_uid") for row in kept_rows}),
        "corpus_hash": corpus_hash,
        "resolved_columns": column_detection.resolved if column_detection else {},
        "candidate_columns": column_detection.candidates if column_detection else {},
        "alias_map": alias_map,
        "sample_rows": sample_rows[:sample_limit],
        "output_files": {
            "parquet": str(parquet_path),
            "sample_csv": str(sample_path),
        },
    }
    json_dump(report_path, report)
    return {
        "stats": PreparationStats(
            rows_read=rows_read,
            rows_kept=len(kept_rows),
            rows_rejected=rows_read - len(kept_rows),
            exact_duplicates=deduper.stats.exact_duplicates,
            near_duplicates=deduper.stats.near_duplicates,
            enriched_rows=sum(1 for row in kept_rows if row.get("skills_normalized")),
            unique_formations=len({row.get("formation_uid") for row in kept_rows}),
        ),
        "report": report,
        "parquet_path": parquet_path,
        "sample_path": sample_path,
        "report_path": report_path,
        "kept_rows": kept_rows,
        "corpus_hash": corpus_hash,
    }

