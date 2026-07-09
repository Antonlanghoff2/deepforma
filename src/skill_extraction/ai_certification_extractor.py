from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from common.text import clean_text, normalize_for_match
from referentials.ai_certification_referential import AICertificationReferential, ReferentialSearchResult


DEFAULT_REFERENTIAL_PATH = Path(
    os.getenv(
        'AI_CERTIFICATION_REFERENTIAL_PATH',
        'data/referentials/ai_engineer_certification_2025.json',
    )
)
EXACT_THRESHOLD = float(os.getenv('AI_CERT_SKILL_EXACT_THRESHOLD', '1.0'))
ALIAS_THRESHOLD = float(os.getenv('AI_CERT_SKILL_ALIAS_THRESHOLD', '0.92'))
SEMANTIC_THRESHOLD = float(os.getenv('AI_CERT_SKILL_SEMANTIC_THRESHOLD', '0.72'))
IMPLICIT_THRESHOLD = float(os.getenv('AI_CERT_SKILL_IMPLICIT_THRESHOLD', '0.80'))
DEFAULT_EMBEDDING_MODEL = os.getenv('AI_CERT_SKILL_EMBEDDING_MODEL', '').strip() or None

TITLE_PREFIX_RE = re.compile(
    r'^(?:offre de|poste\s*:|intitulé\s*:|intitule\s*:|profil recherché\s*:|profil\s*:|job\s*:|recrutement\s*:)',
    flags=re.IGNORECASE,
)
TITLE_SUFFIX_RE = re.compile(
    r'(?:\s*[-–—]\s*(?:h/f|hf|f/h|alternance|stage|junior|senior|lead|n[ée]gociable|remote|t[ée]l[ée]travail.*))$',
    flags=re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n+|•|·|\u2022|;')
STOP_RE = re.compile(
    r"(?:\[\s*comp|mise en situation professionnelle|conditions pratiques de réalisation|modalit[ée]s d[’']évaluation|criteres d[’']évaluation|critéres d[’']évaluation|jeux? de rôle|études? de cas|cas d’usage)",
    flags=re.IGNORECASE,
)
NEGATIVE_HINTS = (
    'bac +',
    'bac+',
    'diplôme',
    'diplome',
    'expérience',
    'experience',
    'télétravail',
    'teletravail',
    'salaire',
    'rémunération',
    'remuneration',
    'avantage',
    'localisation',
    'entreprise',
    'profil recherché',
    'qualités',
    "qualites",
    "modalité d'évaluation",
    "modalités d'évaluation",
    "modalite d'evaluation",
    "modalites d'evaluation",
    "critère d'évaluation",
    "critères d'évaluation",
    "critere d'evaluation",
    "criteres d'evaluation",
    "mise en situation professionnelle",
    "jeu de rôle",
    "jeux de rôle",
    "etude de cas",
    "étude de cas",
    "conditions pratiques de réalisation",
)
ACTION_HINTS = (
    'préparer',
    'preparer',
    'nettoyer',
    'normaliser',
    'entraîner',
    'entrainer',
    'déployer',
    'deployer',
    'construire',
    'concevoir',
    'piloter',
    'mettre en place',
    'mettre en œuvre',
    'mettre en oeuvre',
    'évaluer',
    'evaluer',
    'monitorer',
    'automatiser',
    'optimiser',
    'analyser',
    'classifier',
    'segmenter',
    'vectoriser',
    'tokeniser',
)


@dataclass(frozen=True, slots=True)
class AIExtractionMatch:
    referential_id: str
    code: str
    libelle: str
    libelle_officiel: str
    evidence: str
    confidence: float
    match_type: str
    _priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'referential_id': self.referential_id,
            'code': self.code,
            'libelle': self.libelle,
            'libelle_officiel': self.libelle_officiel,
            'evidence': self.evidence,
            'confidence': round(float(self.confidence), 4),
            'match_type': self.match_type,
        }


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for part in SENTENCE_SPLIT_RE.split(text or ''):
        cleaned = clean_text(part)
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _normalize_title(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ''
    text = TITLE_PREFIX_RE.sub('', text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = TITLE_SUFFIX_RE.sub('', text).strip(' -–—:;,.')
    return text


def _looks_like_title(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if not (1 < len(words) <= 10):
        return False
    norm = normalize_for_match(text)
    if any(hint in norm for hint in ('nous recherchons', 'vos missions', 'missions', 'profil recherché', 'description')):
        return False
    return True


def _title_from_description(description: str) -> str | None:
    for line in _sentences(description)[:5]:
        candidate = _normalize_title(line)
        if _looks_like_title(candidate):
            return candidate
    return None


def _clean_candidate(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ''
    text = STOP_RE.split(text, maxsplit=1)[0]
    text = re.sub(r'\s+', ' ', text)
    return text.strip(' -–—:;,.')


def _short_label(official_description: str) -> str:
    text = _clean_candidate(official_description)
    if not text:
        return ''
    split_markers = [
        r'\bafin de\b',
        r"\bafin d['’]\b",
        r'\bpour\b',
        r'\ben\b',
        r'\bdans le but de\b',
        r'\btout en\b',
        r'\bafin que\b',
        r'\bde manière à\b',
        r'\bde maniere a\b',
    ]
    for marker in split_markers:
        parts = re.split(marker, text, maxsplit=1, flags=re.IGNORECASE)
        if parts:
            candidate = clean_text(parts[0])
            if candidate:
                text = candidate
                break
    words = text.split()
    if len(words) > 12:
        text = ' '.join(words[:12])
    if text.isupper() or sum(1 for char in text if char.isupper()) > max(4, len(text) // 2):
        text = text[:1].upper() + text[1:].lower() if text else text
    return text


def _evidence_snippet(sentence: str, phrase: str) -> str:
    cleaned_sentence = clean_text(sentence)
    cleaned_phrase = clean_text(phrase)
    if not cleaned_sentence:
        return cleaned_phrase
    if not cleaned_phrase:
        return cleaned_sentence
    return cleaned_sentence


def _confidence_priority(match_type: str) -> int:
    return {'exact': 4, 'alias': 3, 'semantic': 2, 'implicit': 1}.get(match_type, 0)


@lru_cache(maxsize=4)
def _build_referential(path_str: str, embedding_model: str | None) -> AICertificationReferential:
    return AICertificationReferential(path_str, embedding_model=embedding_model)


class AICertificationSkillExtractor:
    """Extrait les compétences IA d'une offre à partir du référentiel certification."""

    def __init__(
        self,
        referential: AICertificationReferential | None = None,
        *,
        referential_path: str | Path | None = None,
        embedding_model: str | None = None,
        exact_threshold: float = EXACT_THRESHOLD,
        alias_threshold: float = ALIAS_THRESHOLD,
        semantic_threshold: float = SEMANTIC_THRESHOLD,
        implicit_threshold: float = IMPLICIT_THRESHOLD,
    ) -> None:
        if referential is None:
            referential = _build_referential(str(Path(referential_path or DEFAULT_REFERENTIAL_PATH)), embedding_model or DEFAULT_EMBEDDING_MODEL)
        self.referential = referential
        self.exact_threshold = exact_threshold
        self.alias_threshold = alias_threshold
        self.semantic_threshold = semantic_threshold
        self.implicit_threshold = implicit_threshold

    def normalize_label(self, label: str | None) -> str:
        return self.referential.normalize_label(label)

    def _split_title(self, title: str | None, description: str) -> str | None:
        explicit = _normalize_title(title or '')
        if explicit:
            return explicit
        return _title_from_description(description)

    def _sentence_is_plausible(self, sentence: str) -> bool:
        sentence_norm = normalize_for_match(sentence)
        if not sentence_norm:
            return False
        if any(hint in sentence_norm for hint in NEGATIVE_HINTS):
            return False
        return any(hint in sentence_norm for hint in ACTION_HINTS) or len(sentence_norm.split()) >= 5

    def _score_sentence(self, sentence: str, skill: dict[str, Any], *, _semantic_lookup: dict[str, ReferentialSearchResult] | None = None) -> tuple[str | None, float, str]:
        sentence_norm = normalize_for_match(sentence)
        candidates = [skill['label'], skill['official_description'], *skill.get('aliases', [])]
        for candidate in candidates:
            candidate_norm = normalize_for_match(candidate)
            if not candidate_norm:
                continue
            if candidate_norm == sentence_norm:
                return candidate, 1.0, 'exact'
            if candidate_norm in sentence_norm:
                return candidate, max(self.exact_threshold, 0.98), 'exact'

        for candidate in candidates[1:]:
            candidate_norm = normalize_for_match(candidate)
            if not candidate_norm:
                continue
            if candidate_norm in sentence_norm or sentence_norm in candidate_norm:
                return candidate, max(self.alias_threshold, 0.92), 'alias'

        if _semantic_lookup is not None:
            hit = _semantic_lookup.get(skill['id'])
            if hit is not None and hit.score >= self.semantic_threshold:
                match_type = 'semantic' if hit.score < self.implicit_threshold else 'implicit'
                return skill['official_description'], float(hit.score), match_type
        else:
            semantic_hits = self.referential.search_semantic(sentence, top_k=8)
            for hit in semantic_hits:
                if hit.referential_id != skill['id']:
                    continue
                if hit.score >= self.semantic_threshold and self._sentence_is_plausible(sentence):
                    match_type = 'semantic' if hit.score < self.implicit_threshold else 'implicit'
                    return skill['official_description'], float(hit.score), match_type
        return None, 0.0, ''

    def extract(self, title: str | None, description: str) -> dict[str, Any]:
        raw_description = description or ''
        clean_description = clean_text(description)
        intitule_poste = self._split_title(title, raw_description)
        sentences = _sentences(clean_description)
        matches: dict[str, AIExtractionMatch] = {}
        for sentence in sentences:
            if not self._sentence_is_plausible(sentence):
                continue
            semantic_lookup: dict[str, ReferentialSearchResult] | None = None
            semantic_hits = self.referential.search_semantic(sentence, top_k=8)
            if semantic_hits:
                semantic_lookup = {hit.referential_id: hit for hit in semantic_hits if hit.score >= self.semantic_threshold}
            for skill in self.referential.get_all_skills():
                if not skill.get('active', True):
                    continue
                candidate_text, score, match_type = self._score_sentence(sentence, skill, _semantic_lookup=semantic_lookup)
                if not match_type:
                    continue
                reference_id = skill['id']
                current = matches.get(reference_id)
                if current is None or _confidence_priority(match_type) > _confidence_priority(current.match_type) or (
                    match_type == current.match_type and score > current.confidence
                ):
                    evidence = _evidence_snippet(sentence, candidate_text or skill['official_description'] or skill['label'])
                    matches[reference_id] = AIExtractionMatch(
                        referential_id=reference_id,
                        code=skill['code'],
                        libelle=skill['label'],
                        libelle_officiel=skill['official_description'],
                        evidence=evidence,
                        confidence=score,
                        match_type=match_type,
                        _priority=_confidence_priority(match_type),
                    )

        competences = [item.to_dict() for item in sorted(matches.values(), key=lambda item: (-item._priority, -item.confidence, item.libelle))]
        return {'intitule_poste': intitule_poste, 'competences': competences}
