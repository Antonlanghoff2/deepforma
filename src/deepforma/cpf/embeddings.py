from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np


LOGGER = logging.getLogger(__name__)


class EmbeddingEncoder(Protocol):
    """Protocole minimal pour un encodeur d'embeddings."""

    def encode(self, texts: list[str], batch_size: int = 32, show_progress_bar: bool = False, normalize_embeddings: bool = False) -> Any:  # noqa: E501
        ...


@dataclass(frozen=True)
class EmbeddingManifest:
    """Métadonnées du corpus indexé."""

    model_name: str
    embedding_dim: int
    generated_at: str
    corpus_hash: str
    record_count: int
    backend: str
    records_hashes: dict[str, str]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalise des vecteurs à la norme unitaire."""

    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def compute_text_hash(text: str) -> str:
    """Hash SHA-256 d'un texte normalisé."""

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def compute_corpus_hash(records: list[dict[str, Any]]) -> str:
    """Hash déterministe du corpus."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item.get("formation_uid") or "")):
        digest.update(str(record.get("formation_uid") or "").encode("utf-8"))
        digest.update(str(record.get("search_text") or "").encode("utf-8"))
        digest.update(str(record.get("row_hash") or "").encode("utf-8"))
    return digest.hexdigest()


def _ensure_2d(array: Any) -> np.ndarray:
    vectors = np.asarray(array, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return vectors


class VectorIndexBackend(Protocol):
    """Backend d'index vectoriel interchangeable."""

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        ...

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        ...

    def save(self, path: Path) -> None:
        ...


class NumpyVectorIndex:
    """Index vectoriel simple basé sur NumPy."""

    def __init__(self, vectors: np.ndarray | None = None, ids: list[str] | None = None) -> None:
        self.vectors = normalize_vectors(_ensure_2d(vectors)) if vectors is not None else np.zeros((0, 0), dtype=np.float32)
        self.ids = list(ids or [])

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        normalized = normalize_vectors(_ensure_2d(vectors))
        if self.vectors.size == 0:
            self.vectors = normalized
        else:
            self.vectors = np.vstack([self.vectors, normalized])
        self.ids.extend(ids)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if self.vectors.size == 0 or not self.ids:
            return []
        query = normalize_vectors(_ensure_2d(vector))[0]
        scores = self.vectors @ query
        order = np.argsort(-scores)[:top_k]
        return [(self.ids[int(idx)], float(scores[int(idx)])) for idx in order]

    def save(self, path: Path) -> None:
        payload = {"vectors": self.vectors, "ids": self.ids}
        with path.open("wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Path) -> "NumpyVectorIndex":
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return cls(vectors=payload["vectors"], ids=payload["ids"])


class FaissVectorIndex:
    """Wrapper optionnel autour de FAISS."""

    def __init__(self, dimension: int | None = None) -> None:
        import faiss  # type: ignore

        self.faiss = faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) if dimension else None
        self.ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        normalized = normalize_vectors(_ensure_2d(vectors))
        if self.index is None:
            self.dimension = int(normalized.shape[1])
            self.index = self.faiss.IndexFlatIP(self.dimension)
        self.index.add(normalized.astype(np.float32))
        self.ids.extend(ids)

    def search(self, vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if self.index is None or not self.ids:
            return []
        query = normalize_vectors(_ensure_2d(vector)).astype(np.float32)
        scores, indices = self.index.search(query, top_k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            results.append((self.ids[int(idx)], float(score)))
        return results

    def save(self, path: Path) -> None:
        self.faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "FaissVectorIndex":
        import faiss  # type: ignore

        index = faiss.read_index(str(path))
        instance = cls(dimension=index.d)
        instance.index = index
        return instance


def build_encoder(model_name: str):
    """Construit un encodeur sentence-transformers si disponible."""

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - dépendance optionnelle
        raise ImportError(
            "sentence-transformers n'est pas installé. Installer la dépendance pour générer les embeddings."
        ) from exc
    return SentenceTransformer(model_name)


def choose_backend(prefer_faiss: bool = True, dimension: int | None = None) -> VectorIndexBackend:
    """Sélectionne un backend d'index vectoriel."""

    if prefer_faiss:
        try:
            return FaissVectorIndex(dimension=dimension)
        except Exception:
            LOGGER.info("FAISS indisponible, repli sur NumPy.")
    return NumpyVectorIndex()


def build_embeddings(
    records: list[dict[str, Any]],
    encoder: EmbeddingEncoder,
    *,
    batch_size: int = 64,
    backend: VectorIndexBackend | None = None,
) -> tuple[np.ndarray, VectorIndexBackend]:
    """Encode des textes par lots et alimente un index vectoriel."""

    texts = [str(record.get("search_text") or "") for record in records]
    ids = [str(record.get("formation_uid") or "") for record in records]
    embeddings: list[np.ndarray] = []
    backend = backend or NumpyVectorIndex()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_ids = ids[start : start + batch_size]
        encoded = encoder.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=False)
        batch_vectors = _ensure_2d(encoded).astype(np.float32)
        batch_vectors = normalize_vectors(batch_vectors)
        embeddings.append(batch_vectors)
        backend.add(batch_vectors, batch_ids)
    stacked = np.vstack(embeddings) if embeddings else np.zeros((0, 0), dtype=np.float32)
    return stacked, backend


def make_manifest(
    *,
    model_name: str,
    corpus_hash: str,
    record_count: int,
    backend_name: str,
    vectors: np.ndarray,
    records: list[dict[str, Any]],
) -> EmbeddingManifest:
    """Construit le manifeste d'indexation."""

    hashes = {str(record.get("formation_uid") or ""): str(record.get("row_hash") or "") for record in records}
    embedding_dim = int(vectors.shape[1]) if vectors.size else 0
    return EmbeddingManifest(
        model_name=model_name,
        embedding_dim=embedding_dim,
        generated_at=datetime.now(timezone.utc).isoformat(),
        corpus_hash=corpus_hash,
        record_count=record_count,
        backend=backend_name,
        records_hashes=hashes,
    )


def manifest_to_dict(manifest: EmbeddingManifest) -> dict[str, Any]:
    """Convertit un manifeste en dictionnaire JSON."""

    return {
        "model_name": manifest.model_name,
        "embedding_dim": manifest.embedding_dim,
        "generated_at": manifest.generated_at,
        "corpus_hash": manifest.corpus_hash,
        "record_count": manifest.record_count,
        "backend": manifest.backend,
        "records_hashes": manifest.records_hashes,
    }

