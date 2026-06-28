from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from deepforma.cpf.embeddings import build_embeddings, choose_backend, compute_corpus_hash, make_manifest, manifest_to_dict
from deepforma.training.cpf_trainer import resolve_device
from deepforma.cpf.io import ensure_parent, json_dump


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construit les embeddings CPF et l'index vectoriel")
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, default=Path('data/indexes/cpf/metadata.parquet'))
    parser.add_argument('--index', type=Path, default=Path('data/indexes/cpf/faiss.index'))
    parser.add_argument('--manifest', type=Path, default=Path('data/indexes/cpf/index_manifest.json'))
    parser.add_argument('--model', type=str, default=None, help='Chemin local ou identifiant Sentence-Transformer finetuné')
    parser.add_argument('--model-name', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--batch-size', type=int, default=64)
    return parser


def _read_metadata(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f'Impossible de lire les métadonnées parquet: {path}') from exc


def _text_row(row: dict[str, object]) -> str:
    parts = [row.get('search_text') or row.get('title') or '', row.get('description') or '', row.get('objectives') or '']
    return '\n'.join(str(part) for part in parts if str(part).strip())


def _model_identifier(model: str | None, default_name: str) -> str:
    return str(model or default_name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    args = build_parser().parse_args()
    metadata = _read_metadata(args.input)
    records = metadata.to_dict(orient='records')
    corpus_hash = compute_corpus_hash(records)
    model_name = _model_identifier(args.model, args.model_name)
    device = resolve_device()
    LOGGER.info("Device retenu pour l'indexation: %s", device)
    encoder = SentenceTransformer(model_name, device=device)
    backend = choose_backend(prefer_faiss=True)
    vectors, backend = build_embeddings(records, encoder, batch_size=args.batch_size, backend=backend)
    backend.save(args.index)
    ensure_parent(args.metadata)
    metadata.to_parquet(args.metadata, index=False)
    manifest = make_manifest(
        model_name=model_name,
        corpus_hash=corpus_hash,
        record_count=len(records),
        backend_name=backend.__class__.__name__,
        vectors=vectors,
        records=records,
    )
    manifest_payload = manifest_to_dict(manifest) | {
        'model_identifier': model_name,
        'vector_metric': 'cosine',
        'normalized_embeddings': True,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'dataset_hash': corpus_hash,
        'record_count': len(records),
        'embedding_dim': int(vectors.shape[1]) if vectors.size else 0,
        'faiss_index_type': backend.__class__.__name__,
    }
    json_dump(args.manifest, manifest_payload)
    LOGGER.info('Index vectoriel écrit dans %s', args.index)
    LOGGER.info('Métadonnées écrites dans %s', args.metadata)
    LOGGER.info('Manifeste écrit dans %s', args.manifest)


if __name__ == '__main__':
    main()
