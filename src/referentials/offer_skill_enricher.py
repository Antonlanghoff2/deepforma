from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match
from .unified_skill_referential import UnifiedSkillReferential

DEFAULT_UNIFIED_REFERENTIAL_PATH = Path(os.getenv('RNCP_ROME_UNIFIED_REFERENTIAL_PATH', 'data/referentials/unified/skills.jsonl'))
DEFAULT_MAPPINGS_PATH = Path(os.getenv('RNCP_ROME_MAPPINGS_PATH', 'data/referentials/mappings/rncp_rome_links.jsonl'))

_PRIORITY = {'exact': 3, 'alias': 2, 'semantic': 1, 'implicit': 0}
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+|\n+|[•·;]+')


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _safe_list(values: Iterable[Any] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = clean_text(value)
        if text:
            result.append(text)
    return result


def _token_set(text: str) -> set[str]:
    return {token for token in normalize_for_match(text).split() if token}


def _best_sentence(text: str, phrases: list[str]) -> str:
    sentences = [clean_text(part) for part in _SENTENCE_SPLIT.split(text or '') if clean_text(part)]
    if not sentences:
        return clean_text(text)
    best_sentence = ''
    best_score = 0.0
    for sentence in sentences:
        norm_sentence = normalize_for_match(sentence)
        for phrase in phrases:
            norm_phrase = normalize_for_match(phrase)
            if not norm_phrase:
                continue
            if norm_phrase in norm_sentence:
                return sentence
            sentence_tokens = _token_set(sentence)
            phrase_tokens = _token_set(phrase)
            if not sentence_tokens or not phrase_tokens:
                continue
            score = len(sentence_tokens & phrase_tokens) / len(sentence_tokens | phrase_tokens)
            if score > best_score:
                best_score = score
                best_sentence = sentence
    return best_sentence or clean_text(text)


@lru_cache(maxsize=4)
def _load_mappings(path_text: str) -> list[dict[str, Any]]:
    return _read_jsonl(Path(path_text))


class RNCPROMEOfferEnricher:
    def __init__(self, unified_referential_path: str | Path | None = None, mappings_path: str | Path | None = None) -> None:
        self.unified_referential_path = Path(unified_referential_path or DEFAULT_UNIFIED_REFERENTIAL_PATH)
        self.mappings_path = Path(mappings_path or DEFAULT_MAPPINGS_PATH)
        self._referential = UnifiedSkillReferential(self.unified_referential_path)
        self._skills = self._referential.get_all_skills()
        self._mappings = _load_mappings(str(self.mappings_path))
        self._skills_by_id = {item['canonical_skill_id']: item for item in self._skills if item.get('canonical_skill_id')}

    def _rncp_candidates(self, rome_code: str | None) -> list[dict[str, Any]]:
        if not rome_code:
            return []
        candidates: dict[str, dict[str, Any]] = {}
        for row in self._mappings:
            if clean_text(row.get('rome_code')) != clean_text(rome_code):
                continue
            rncp_code = clean_text(row.get('rncp_code'))
            if not rncp_code:
                continue
            score = float(row.get('score', 0.0) or 0.0)
            current = candidates.get(rncp_code)
            if current is None or score > float(current.get('mapping_score', 0.0)):
                candidates[rncp_code] = {'rncp_code': rncp_code, 'mapping_score': score}
        return sorted(candidates.values(), key=lambda item: (-float(item.get('mapping_score', 0.0)), item['rncp_code']))

    def _candidate_skills(self, rome_code: str | None, candidate_labels: list[str]) -> list[dict[str, Any]]:
        if not rome_code:
            return self._skills
        rncp_codes = {item['rncp_code'] for item in self._rncp_candidates(rome_code)}
        if not rncp_codes:
            return self._skills
        filtered: list[dict[str, Any]] = []
        for skill in self._skills:
            sources = skill.get('sources', []) or []
            if any(clean_text(source.get('source_id')).startswith(code) for code in rncp_codes for source in sources if isinstance(source, dict) and clean_text(source.get('source')) == 'france_competences'):
                filtered.append(skill)
                continue
            if any(clean_text(source.get('source')) == 'rome' and clean_text(source.get('source_id')) == clean_text(rome_code) for source in sources if isinstance(source, dict)):
                filtered.append(skill)
        if filtered:
            return filtered
        return self._skills

    def _match_skill(self, text: str, skill: dict[str, Any], *, candidate_labels: list[str]) -> dict[str, Any] | None:
        canonical_label = clean_text(skill.get('canonical_label') or '')
        aliases = [clean_text(alias) for alias in skill.get('aliases', []) if clean_text(alias)]
        phrases = [canonical_label, *aliases, *candidate_labels]
        text_norm = normalize_for_match(text)
        if not text_norm:
            return None
        for phrase in [canonical_label, *aliases, *candidate_labels]:
            phrase_norm = normalize_for_match(phrase)
            if not phrase_norm:
                continue
            if phrase_norm in text_norm:
                evidence = _best_sentence(text, [phrase])
                if normalize_for_match(evidence):
                    match_type = 'exact' if phrase_norm == normalize_for_match(canonical_label) else 'alias'
                    confidence = 1.0 if match_type == 'exact' else 0.92
                    return {
                        'canonical_skill_id': skill['canonical_skill_id'],
                        'canonical_label': canonical_label,
                        'evidence': evidence,
                        'confidence': confidence,
                        'match_type': match_type,
                        'source_links': skill.get('sources', []),
                    }
        sentence = _best_sentence(text, phrases)
        if not sentence:
            return None
        sentence_norm = normalize_for_match(sentence)
        if not sentence_norm:
            return None
        sentence_tokens = _token_set(sentence)
        best_score = 0.0
        for phrase in phrases:
            phrase_tokens = _token_set(phrase)
            if not phrase_tokens:
                continue
            coverage = len(sentence_tokens & phrase_tokens) / len(phrase_tokens)
            overlap = len(sentence_tokens & phrase_tokens) / len(sentence_tokens | phrase_tokens)
            score = max(coverage, overlap)
            if score > best_score:
                best_score = score
        if best_score >= 0.72:
            return {
                'canonical_skill_id': skill['canonical_skill_id'],
                'canonical_label': canonical_label,
                'evidence': sentence,
                'confidence': round(max(0.72, best_score), 4),
                'match_type': 'semantic',
                'source_links': skill.get('sources', []),
            }
        if best_score >= 0.58 and canonical_label:
            return {
                'canonical_skill_id': skill['canonical_skill_id'],
                'canonical_label': canonical_label,
                'evidence': sentence,
                'confidence': round(max(0.65, best_score), 4),
                'match_type': 'implicit',
                'source_links': skill.get('sources', []),
            }
        return None

    def extract(
        self,
        *,
        title: str | None,
        description: str,
        rome_code: str | None = None,
        rome_label: str | None = None,
        structured_skills: Iterable[dict[str, Any]] | None = None,
        model_skills: Iterable[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        title = clean_text(title)
        description = clean_text(description)
        rome_label = clean_text(rome_label)
        text = '\n'.join(part for part in [title, rome_label, description] if part)
        candidate_labels = []
        for item in list(structured_skills or []) + list(model_skills or []):
            if isinstance(item, dict):
                label = clean_text(item.get('label') or item.get('canonical_label') or item.get('libelle') or '')
                if label:
                    candidate_labels.append(label)
        candidate_labels = list(dict.fromkeys(candidate_labels))
        rncp_candidates = self._rncp_candidates(rome_code)
        candidate_skills = self._candidate_skills(rome_code, candidate_labels)
        matches: dict[str, dict[str, Any]] = {}
        for skill in candidate_skills:
            match = self._match_skill(text, skill, candidate_labels=candidate_labels)
            if not match:
                continue
            existing = matches.get(match['canonical_skill_id'])
            if existing is None or _PRIORITY.get(match['match_type'], -1) > _PRIORITY.get(existing['match_type'], -1) or (match['confidence'], len(match['evidence'])) > (existing['confidence'], len(existing['evidence'])):
                matches[match['canonical_skill_id']] = match
        competences = sorted(matches.values(), key=lambda item: (-_PRIORITY.get(item['match_type'], -1), -float(item['confidence']), item['canonical_label']))
        return {
            'rncp_candidates': rncp_candidates,
            'competences': competences,
        }
