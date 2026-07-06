from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common.text import clean_text, normalize_for_match


DEFAULT_REFERENTIAL_PATH = Path(
    os.getenv(
        'AI_CERTIFICATION_REFERENTIAL_PATH',
        'data/referentials/ai_engineer_certification_2025.json',
    )
)
DEFAULT_EMBEDDING_MODEL = os.getenv('AI_CERT_SKILL_EMBEDDING_MODEL', '').strip() or None


@dataclass(frozen=True, slots=True)
class ReferentialSearchResult:
    referential_id: str
    code: str
    label: str
    official_description: str
    normalized_label: str
    aliases: list[str]
    score: float
    match_type: str


def _normalize_aliases(values: Iterable[str]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in values:
        alias = clean_text(value)
        if not alias:
            continue
        key = normalize_for_match(alias)
        if not key or key in seen:
            continue
        seen.add(key)
        aliases.append(alias)
    return aliases


@lru_cache(maxsize=8)
def _load_json(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f'Reférentiel IA introuvable: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Reférentiel IA invalide: {path}')
    return payload


class AICertificationReferential:
    """Référentiel de certification IA utilisé pour la normalisation des offres."""

    def __init__(self, referential_path: str | Path | None = None, *, embedding_model: str | None = None) -> None:
        self.referential_path = Path(referential_path or DEFAULT_REFERENTIAL_PATH)
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL or None
        self._payload: dict[str, Any] | None = None
        self._skills: list[dict[str, Any]] = []
        self._skill_by_id: dict[str, dict[str, Any]] = {}
        self._exact_index: dict[str, dict[str, Any]] = {}
        self._alias_index: dict[str, dict[str, Any]] = {}
        self._encoder: Any | None = None
        self._embedding_ready = False

    @property
    def payload(self) -> dict[str, Any]:
        if self._payload is None:
            self.load()
        assert self._payload is not None
        return self._payload

    def load(self) -> dict[str, Any]:
        if self._payload is not None:
            return self._payload
        payload = _load_json(str(self.referential_path))
        skills = payload.get('skills', [])
        if not isinstance(skills, list):
            raise ValueError("Le champ 'skills' du référentiel doit être une liste.")

        normalized_skills: list[dict[str, Any]] = []
        self._skill_by_id = {}
        self._exact_index = {}
        self._alias_index = {}
        for raw_skill in skills:
            if not isinstance(raw_skill, dict):
                continue
            skill = dict(raw_skill)
            skill['id'] = clean_text(skill.get('id') or '')
            skill['block'] = clean_text(skill.get('block') or '')
            skill['activity'] = clean_text(skill.get('activity') or '')
            skill['code'] = clean_text(skill.get('code') or '')
            skill['label'] = clean_text(skill.get('label') or '')
            skill['official_description'] = clean_text(skill.get('official_description') or '')
            skill['normalized_label'] = clean_text(skill.get('normalized_label') or normalize_for_match(skill['label']))
            skill['aliases'] = _normalize_aliases(skill.get('aliases') or [])
            skill['source_page'] = int(skill.get('source_page') or 0)
            skill['active'] = bool(skill.get('active', True))
            skill['category'] = clean_text(skill.get('category') or '')
            skill['subcategory'] = clean_text(skill.get('subcategory') or '')
            technical_keywords = skill.get('technical_keywords') or skill.get('keywords') or []
            if isinstance(technical_keywords, str):
                technical_keywords = [technical_keywords]
            skill['technical_keywords'] = _normalize_aliases(technical_keywords)
            skill['origin_document'] = clean_text(skill.get('origin_document') or payload.get('metadata', {}).get('source_pdf', '') or '')
            if not skill['id']:
                continue
            normalized_skills.append(skill)
            self._skill_by_id[skill['id']] = skill
            self._exact_index[self.normalize_label(skill['normalized_label'])] = skill
            self._exact_index[self.normalize_label(skill['label'])] = skill
            for alias in skill['aliases']:
                self._alias_index[self.normalize_label(alias)] = skill

        self._payload = {**payload, 'skills': normalized_skills}
        self._skills = normalized_skills
        self._embedding_ready = False
        self._encoder = None
        return self._payload

    def get_all_skills(self) -> list[dict[str, Any]]:
        self.load()
        return list(self._skills)

    def get_skill_by_id(self, referential_id: str) -> dict[str, Any] | None:
        self.load()
        return self._skill_by_id.get(clean_text(referential_id))

    @staticmethod
    def normalize_label(label: str | None) -> str:
        return normalize_for_match(clean_text(label))

    def search_exact(self, label: str) -> dict[str, Any] | None:
        self.load()
        return self._exact_index.get(self.normalize_label(label))

    def search_alias(self, alias: str) -> dict[str, Any] | None:
        self.load()
        return self._alias_index.get(self.normalize_label(alias))

    def _load_sentence_encoder(self) -> Any | None:
        if self._embedding_ready:
            return self._encoder
        self._embedding_ready = True
        self._encoder = None
        if not self.embedding_model:
            return None
        model_path = Path(self.embedding_model)
        if not model_path.exists() or not model_path.is_dir():
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            return None
        try:
            self._encoder = SentenceTransformer(str(model_path))
        except Exception:
            self._encoder = None
        return self._encoder

    def _skill_texts(self) -> list[str]:
        texts: list[str] = []
        for skill in self.get_all_skills():
            parts = [skill.get('label', ''), skill.get('official_description', ''), *skill.get('aliases', [])]
            text = clean_text(' '.join(clean_text(part) for part in parts if clean_text(part)))
            if text:
                texts.append(text)
        return texts

    def search_semantic(self, text: str, *, top_k: int = 5) -> list[ReferentialSearchResult]:
        self.load()
        query = clean_text(text)
        if not query:
            return []

        encoder = self._load_sentence_encoder()
        results: list[ReferentialSearchResult] = []
        if encoder is not None and self._skills:
            try:
                query_vec = encoder.encode([query], normalize_embeddings=True)
                skill_texts = self._skill_texts()
                skill_vecs = encoder.encode(skill_texts, normalize_embeddings=True)
                scores = np.asarray(skill_vecs, dtype=np.float32) @ np.asarray(query_vec, dtype=np.float32)[0]
                order = list(np.argsort(-scores)[:top_k])
                for index in order:
                    skill = self._skills[int(index)]
                    score = float(scores[int(index)])
                    results.append(
                        ReferentialSearchResult(
                            referential_id=skill['id'],
                            code=skill['code'],
                            label=skill['label'],
                            official_description=skill['official_description'],
                            normalized_label=skill['normalized_label'],
                            aliases=list(skill['aliases']),
                            score=round(score, 4),
                            match_type='semantic',
                        )
                    )
                return results
            except Exception:
                pass

        norm_query = normalize_for_match(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for skill in self._skills:
            candidates = [skill['label'], skill['official_description'], *skill['aliases']]
            best = 0.0
            for candidate in candidates:
                cand_norm = normalize_for_match(candidate)
                if not cand_norm:
                    continue
                if cand_norm in norm_query or norm_query in cand_norm:
                    best = max(best, 0.92)
                else:
                    import difflib

                    best = max(best, difflib.SequenceMatcher(None, norm_query, cand_norm).ratio())
            if best > 0:
                scored.append((best, skill))
        scored.sort(key=lambda item: item[0], reverse=True)
        for score, skill in scored[:top_k]:
            results.append(
                ReferentialSearchResult(
                    referential_id=skill['id'],
                    code=skill['code'],
                    label=skill['label'],
                    official_description=skill['official_description'],
                    normalized_label=skill['normalized_label'],
                    aliases=list(skill['aliases']),
                    score=round(float(score), 4),
                    match_type='semantic',
                )
            )
        return results
