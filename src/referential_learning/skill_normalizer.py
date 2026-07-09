from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match
from services.skill_normalization import normalize_skill_label

from .ml_dl_taxonomy import alias_map as ml_dl_alias_map, canonicalize_term

LOGGER = logging.getLogger(__name__)

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parents[2] / 'data' / 'referentials' / 'skills.json'

_REFERENTIAL_KEYS = ('skills', 'competencies', 'competences', 'entries', 'items', 'data')


def normalize_skill_entry(entry: Any) -> dict | None:
    if isinstance(entry, str):
        text = clean_text(entry)
        if not text:
            LOGGER.warning('Entrée chaîne vide ignorée dans le référentiel')
            return None
        return {'label': text, 'aliases': [], 'skill_id': text}
    if isinstance(entry, dict):
        label = (clean_text(entry.get('label'))
                 or clean_text(entry.get('name'))
                 or clean_text(entry.get('skill'))
                 or clean_text(entry.get('competence'))
                 or clean_text(entry.get('title')))
        if not label:
            LOGGER.warning('Entrée dictionnaire sans label ignorée : %s', entry)
            return None
        result = dict(entry)
        result['label'] = label
        result.setdefault('aliases', [])
        if 'skill_id' not in result or not result['skill_id']:
            result['skill_id'] = label
        return result
    LOGGER.warning('Entrée de type %s ignorée dans le référentiel', type(entry).__name__)
    return None


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    surface_form: str
    canonical_name: str | None
    confidence: float
    referential_id: str | None
    provenance: str = 'exact_match'
    ambiguity: float = 0.0

    @property
    def accepted(self) -> bool:
        return bool(self.canonical_name) and self.confidence >= 0.55 and self.ambiguity < 0.18


@lru_cache(maxsize=8)
def load_referential(skills_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(skills_path or DEFAULT_SKILLS_PATH)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(raw, dict):
        for key in _REFERENTIAL_KEYS:
            entries = raw.get(key)
            if isinstance(entries, list):
                return entries
        LOGGER.warning('Aucune liste de compétences trouvée dans %s (clés: %s)', path, list(raw.keys()))
        return []
    if isinstance(raw, list):
        return raw
    LOGGER.warning('Format inattendu dans %s (type: %s)', path, type(raw).__name__)
    return []


class SkillNormalizer:
    def __init__(self, skills_path: str | Path | None = None, *, embedding_model: str | None = None, threshold: float = 0.62) -> None:
        self.skills_path = Path(skills_path or DEFAULT_SKILLS_PATH)
        raw_reference = load_referential(self.skills_path)
        self.reference: list[dict[str, Any]] = []
        self.embedding_model = embedding_model
        self.threshold = threshold
        self._index: list[tuple[dict[str, Any], str]] = []
        for entry in raw_reference:
            skill = normalize_skill_entry(entry)
            if skill is None:
                continue
            self.reference.append(skill)
            label = clean_text(skill.get('label', ''))
            if not label:
                continue
            self._index.append((skill, normalize_for_match(label)))
            for alias in skill.get('aliases', []) or []:
                self._index.append((skill, normalize_for_match(alias)))
        for alias_norm, canonical in ml_dl_alias_map().items():
            self._index.append(({'label': canonical, 'skill_id': f'ml-dl::{normalize_for_match(canonical)}'}, alias_norm))

        self._model = None
        self._model_ready = False

    def _ensure_model(self) -> Any | None:
        if self._model_ready:
            return self._model
        self._model_ready = True
        if not self.embedding_model:
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.embedding_model)
        except Exception:
            self._model = None
        return self._model

    def _embed(self, texts: list[str]) -> Any | None:
        model = self._ensure_model()
        if model is None:
            return None
        try:
            return model.encode(texts, normalize_embeddings=True)
        except Exception:
            return None

    def normalize(self, candidate: str) -> NormalizationResult:
        text = clean_text(candidate)
        if not text:
            return NormalizationResult(surface_form='', canonical_name=None, confidence=0.0, referential_id=None, provenance='empty')
        norm = normalize_for_match(text)
        if not norm:
            return NormalizationResult(surface_form=text, canonical_name=None, confidence=0.0, referential_id=None, provenance='empty')

        shared_label = normalize_skill_label(text)
        if shared_label and normalize_for_match(shared_label) == norm:
            canonical_term = shared_label
        else:
            canonical_term, _, _ = canonicalize_term(text)
        if canonical_term and normalize_for_match(canonical_term) == norm:
            return NormalizationResult(surface_form=text, canonical_name=canonical_term, confidence=0.95, referential_id=f'ml-dl::{normalize_for_match(canonical_term)}', provenance='taxonomy_alias')

        exact_hits: list[dict[str, Any]] = []
        partial_hits: list[tuple[float, dict[str, Any]]] = []
        for skill, alias_norm in self._index:
            label_norm = normalize_for_match(skill.get('label', ''))
            if norm == label_norm or norm == alias_norm:
                return NormalizationResult(surface_form=text, canonical_name=skill.get('label'), confidence=1.0, referential_id=skill.get('skill_id'), provenance='exact_match')
            if norm and (norm in label_norm or label_norm in norm):
                exact_hits.append(skill)
            elif alias_norm and norm in alias_norm:
                partial_hits.append((0.68, skill))

        if exact_hits:
            best = exact_hits[0]
            return NormalizationResult(surface_form=text, canonical_name=best.get('label'), confidence=0.76, referential_id=best.get('skill_id'), provenance='substring_match')

        if partial_hits:
            best = max(partial_hits, key=lambda item: item[0])[1]
            return NormalizationResult(surface_form=text, canonical_name=best.get('label'), confidence=0.68, referential_id=best.get('skill_id'), provenance='alias_match')

        if self.embedding_model:
            embeddings = self._embed([text, *[skill.get('label', '') for skill in self.reference]])
            if embeddings is not None and len(embeddings) > 1:
                import numpy as np  # type: ignore
                candidate_vector = embeddings[0]
                refs = embeddings[1:]
                sims = refs @ candidate_vector
                if len(sims) > 0:
                    order = np.argsort(-sims)
                    best_idx = int(order[0])
                    best_score = float(sims[best_idx])
                    second_score = float(sims[int(order[1])]) if len(order) > 1 else -1.0
                    ambiguity = max(0.0, best_score - second_score)
                    if best_score >= self.threshold and ambiguity >= 0.08:
                        ref = self.reference[best_idx]
                        return NormalizationResult(surface_form=text, canonical_name=ref.get('label'), confidence=round(best_score, 4), referential_id=ref.get('skill_id'), provenance='embedding_match', ambiguity=round(ambiguity, 4))
        return NormalizationResult(surface_form=text, canonical_name=None, confidence=0.0, referential_id=None, provenance='rejected')

    def normalize_many(self, candidates: Iterable[str]) -> tuple[list[dict[str, Any]], list[str]]:
        normalized: list[dict[str, Any]] = []
        rejected: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            result = self.normalize(candidate)
            if result.accepted and result.canonical_name:
                key = normalize_for_match(result.canonical_name)
                if key not in seen:
                    seen.add(key)
                    normalized.append({
                        'surface_form': result.surface_form,
                        'canonical_name': result.canonical_name,
                        'confidence': result.confidence,
                        'referential_id': result.referential_id,
                        'provenance': result.provenance,
                    })
            else:
                cleaned = clean_text(candidate)
                if cleaned:
                    rejected.append(cleaned)
        return normalized, rejected
