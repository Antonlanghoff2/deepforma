from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from deepforma.cpf.embeddings import (
    build_embeddings,
    build_encoder,
    choose_backend,
    compute_corpus_hash,
    make_manifest,
    manifest_to_dict,
)
from deepforma.cpf.io import ensure_parent, json_dump


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construit les embeddings CPF et l'index vectoriel")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("data/indexes/cpf/metadata.parquet"))
    parser.add_argument("--index", type=Path, default=Path("data/indexes/cpf/faiss.index"))
    parser.add_argument("--manifest", type=Path, default=Path("data/indexes/cpf/index_manifest.json"))
    parser.add_argument("--model-name", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def _read_metadata(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Impossible de lire les métadonnées parquet: {path}") from exc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args()
    metadata = _read_metadata(args.input)
    records = metadata.to_dict(orient="records")
    corpus_hash = compute_corpus_hash(records)
    encoder = build_encoder(args.model_name)
    backend = choose_backend(prefer_faiss=True)
    vectors, backend = build_embeddings(records, encoder, batch_size=args.batch_size, backend=backend)
    backend.save(args.index)
    ensure_parent(args.metadata)
    metadata.to_parquet(args.metadata, index=False)
    manifest = make_manifest(
        model_name=args.model_name,
        corpus_hash=corpus_hash,
        record_count=len(records),
        backend_name=backend.__class__.__name__,
        vectors=vectors,
        records=records,
    )
    json_dump(args.manifest, manifest_to_dict(manifest))
    LOGGER.info("Index vectoriel écrit dans %s", args.index)
    LOGGER.info("Métadonnées écrites dans %s", args.metadata)
    LOGGER.info("Manifeste écrit dans %s", args.manifest)


if __name__ == "__main__":
    main()

