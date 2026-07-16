from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from common.text import clean_text, normalize_for_match
from deepforma.cpf.embeddings import build_encoder

from .loader import load_ai_recommendation_rules
from .normalizer import normalize_ai_keyword

LOGGER = logging.getLogger(__name__)

INDEX_ROOT = Path(__file__).resolve().parents[2] / 'data' / 'indexes' / 'ai_recommendations'
DEFAULT_EMBEDDING_MODEL = Path(__file__).resolve().parents[2] / 'models' / 'cpf-recommender' / 'final'


@dataclass(frozen=True, slots=True)
class AIRecommendationSemanticIndex:
    rules_path: str
    model_name: str
    rules_hash: str
    model_hash: str
    generated_at: str
    rule_ids: list[str]
    keywords: list[str]
    vectors: list[list[float]]
    metadata: dict[str, Any]


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ''
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _model_hash(model_path: Path | None) -> str:
    if model_path is None or not model_path.exists():
        return ''
    if model_path.is_file():
        return _file_hash(model_path)
    digest = hashlib.sha256()
    for path in sorted(model_path.rglob('*')):
        if path.is_file():
            digest.update(path.relative_to(model_path).as_posix().encode('utf-8'))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_encoder(embedding_model: str | Path | None = None):
    model_path = Path(embedding_model or DEFAULT_EMBEDDING_MODEL)
    if not model_path.exists():
        return None, model_path
    try:
        return build_encoder(str(model_path)), model_path
    except Exception:
        LOGGER.warning('Encodeur d embeddings indisponible pour %s', model_path)
        return None, model_path


def _index_text(rule: dict[str, Any]) -> str:
    parts = [rule.get('keyword', ''), rule.get('recommendation', '')]
    categories = rule.get('categories') or []
    if isinstance(categories, list):
        parts.extend(str(item.get('label', '')) for item in categories if isinstance(item, dict))
    return clean_text(' '.join(clean_text(part) for part in parts if clean_text(part)))


def _manifest_path(index_dir: Path) -> Path:
    return index_dir / 'manifest.json'


def _vectors_path(index_dir: Path) -> Path:
    return index_dir / 'vectors.npz'


def _rules_path(index_dir: Path) -> Path:
    return index_dir / 'rules.json'


def _hash_rules(rules: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for rule in sorted(rules, key=lambda item: str(item.get('id', ''))):
        digest.update(json.dumps(rule, sort_keys=True, ensure_ascii=False).encode('utf-8'))
    return digest.hexdigest()


def _load_index_payload(index_dir: Path) -> AIRecommendationSemanticIndex | None:
    manifest_file = _manifest_path(index_dir)
    vectors_file = _vectors_path(index_dir)
    rules_file = _rules_path(index_dir)
    if not manifest_file.exists() or not vectors_file.exists() or not rules_file.exists():
        return None
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    vectors_payload = np.load(vectors_file, allow_pickle=True)
    vectors = vectors_payload['vectors'].tolist()
    return AIRecommendationSemanticIndex(
        rules_path=str(rules_file),
        model_name=str(manifest.get('model_name', '')),
        rules_hash=str(manifest.get('rules_hash', '')),
        model_hash=str(manifest.get('model_hash', '')),
        generated_at=str(manifest.get('generated_at', '')),
        rule_ids=list(manifest.get('rule_ids', [])),
        keywords=list(manifest.get('keywords', [])),
        vectors=vectors,
        metadata=dict(manifest.get('metadata', {}) or {}),
    )


def rebuild_index(
    rules_path: str | Path,
    index_dir: str | Path = INDEX_ROOT,
    *,
    embedding_model: str | Path | None = None,
) -> AIRecommendationSemanticIndex:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    rules = load_ai_recommendation_rules(rules_path)
    encoder, model_path = _load_encoder(embedding_model)
    texts = [_index_text(rule) for rule in rules]
    vectors = np.zeros((len(texts), 1), dtype=np.float32)
    if encoder is not None and texts:
        encoded = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        vectors = np.asarray(encoded, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
    rule_ids = [str(rule.get('id', '')) for rule in rules]
    keywords = [str(rule.get('normalized_keyword') or normalize_ai_keyword(rule.get('keyword', ''))) for rule in rules]
    manifest = {
        'model_name': str(model_path or ''),
        'rules_hash': _hash_rules(rules),
        'model_hash': _model_hash(model_path),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'rule_ids': rule_ids,
        'keywords': keywords,
        'metadata': {
            'record_count': len(rules),
            'embedding_dim': int(vectors.shape[1]) if vectors.size else 0,
            'rules_path': str(rules_path),
        },
    }
    _manifest_path(index_dir).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    _rules_path(index_dir).write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(_vectors_path(index_dir), vectors=vectors)
    return AIRecommendationSemanticIndex(
        rules_path=str(_rules_path(index_dir)),
        model_name=manifest['model_name'],
        rules_hash=manifest['rules_hash'],
        model_hash=manifest['model_hash'],
        generated_at=manifest['generated_at'],
        rule_ids=rule_ids,
        keywords=keywords,
        vectors=vectors.tolist(),
        metadata=manifest['metadata'],
    )


def load_index(index_dir: str | Path = INDEX_ROOT) -> AIRecommendationSemanticIndex | None:
    return _load_index_payload(Path(index_dir))


def build_or_load_index(
    rules_path: str | Path,
    index_dir: str | Path = INDEX_ROOT,
    *,
    embedding_model: str | Path | None = None,
) -> AIRecommendationSemanticIndex:
    index_dir = Path(index_dir)
    rules = load_ai_recommendation_rules(rules_path)
    rules_hash = _hash_rules(rules)
    encoder, model_path = _load_encoder(embedding_model)
    model_hash = _model_hash(model_path)
    existing = _load_index_payload(index_dir)
    if existing and existing.rules_hash == rules_hash and existing.model_hash == model_hash:
        return existing
    return rebuild_index(rules_path, index_dir, embedding_model=embedding_model)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Reconstruit l index sémantique des recommandations IA.')
    parser.add_argument('--rules-path', type=Path, default=Path('data/referentials/ai_recommendation_rules.json'))
    parser.add_argument('--index-dir', type=Path, default=INDEX_ROOT)
    parser.add_argument('--embedding-model', type=Path, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument('--rebuild', action='store_true')
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    if args.rebuild:
        index = rebuild_index(args.rules_path, args.index_dir, embedding_model=args.embedding_model)
    else:
        index = build_or_load_index(args.rules_path, args.index_dir, embedding_model=args.embedding_model)
    print(json.dumps({
        'rules_path': index.rules_path,
        'model_name': index.model_name,
        'rules_hash': index.rules_hash,
        'model_hash': index.model_hash,
        'generated_at': index.generated_at,
        'record_count': len(index.rule_ids),
        'embedding_dim': int(index.metadata.get('embedding_dim', 0)),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
