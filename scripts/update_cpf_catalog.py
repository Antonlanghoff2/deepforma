from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.download_cpf_catalog import download_catalog
from deepforma.cpf.prepare import prepare_catalog
from deepforma.cpf.embeddings import compute_corpus_hash
from deepforma.cpf.io import json_dump
from deepforma.cpf.update import build_update_report


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Met à jour le catalogue CPF de manière incrémentale")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-file", type=Path, default=None)
    parser.add_argument("--source-url", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api-page-size", type=int, default=50_000)
    return parser


def _diff_metadata(old_df: pd.DataFrame, new_df: pd.DataFrame) -> dict[str, list[str]]:
    old = old_df.set_index("formation_uid")
    new = new_df.set_index("formation_uid")
    added = sorted(set(new.index) - set(old.index))
    removed = sorted(set(old.index) - set(new.index))
    modified = sorted(
        uid for uid in set(old.index) & set(new.index)
        if str(old.loc[uid, "row_hash"]) != str(new.loc[uid, "row_hash"])
    )
    return {"added": added, "modified": modified, "removed": removed}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    raw_manifest = download_catalog(
        output_dir=args.output_dir / "raw" / "cpf",
        source_file=args.source_file,
        source_url=args.source_url,
        force=args.force,
        limit=args.limit,
        api_page_size=args.api_page_size,
    )
    result = prepare_catalog(
        args.output_dir / "raw" / "cpf" / "cpf_catalog.csv",
        args.output_dir,
        config_path=Path("config/cpf_columns.yaml"),
    )
    metadata_path = args.output_dir / "processed" / "cpf" / "formations.parquet"
    report_path = args.output_dir / "reports" / "cpf_update_report.json"
    new_df = pd.DataFrame.from_records(result["kept_rows"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(metadata_path, index=False)
    manifest_path = args.output_dir / "indexes" / "cpf" / "index_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_hash = compute_corpus_hash(result["kept_rows"])
    json_dump(manifest_path, {"corpus_hash": corpus_hash, "download": raw_manifest})
    report = build_update_report(
        new_metadata_path=metadata_path,
        previous_metadata_path=metadata_path,
        new_manifest_path=manifest_path,
        previous_manifest_path=manifest_path,
    )
    report.update({
        "download": raw_manifest,
        "rows_kept": result["stats"].rows_kept,
        "rows_read": result["stats"].rows_read,
    })
    json_dump(report_path, report)
    LOGGER.info("Rapport de mise à jour écrit dans %s", report_path)


if __name__ == "__main__":
    main()

