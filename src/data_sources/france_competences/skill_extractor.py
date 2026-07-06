from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from common.text import clean_text, normalize_for_match
from referential_learning.ai_certification_taxonomy import infer_skill_taxonomy, normalize_market_alias


EVALUATION_MARKERS = (
    'evaluation :',
    'modalites d evaluation',
    "modalités d'évaluation",
    'criteres d evaluation',
    "critères d'évaluation",
    'mises en situation professionnelle',
    'mise en situation professionnelle',
    'conditions pratiques',
    'jury',
    'rapport écrit',
    'soutenance',
    'presentation orale',
)

NEGATIVE_HINTS = (
    'jury',
    'modalité',
    'modalite',
    'critère',
    'critere',
    'évaluation',
    'evaluation',
    'condition de réalisation',
    'conditions de réalisation',
)

BULLET_LINE_RE = re.compile(r'^(?:[-*•·]|\d+\s*[-.)])\s*(.+)$')
COMPOSITE_SPLIT_RE = re.compile(r'\s*(?:,| et | puis |/)\s*', re.IGNORECASE)


def is_excluded_text(text: str) -> bool:
    norm = normalize_for_match(text)
    if not norm:
        return True
    return any(marker in norm for marker in NEGATIVE_HINTS)


def split_evaluation_sections(text: str) -> list[str]:
    raw = str(text or '')
    if not raw.strip():
        return []
    raw_lower = raw.lower()
    sections: list[str] = []
    marker = 'evaluation :'
    start_index = 0
    while True:
        match_index = raw_lower.find(marker, start_index)
        if match_index < 0:
            break
        chunk = raw[start_index:match_index].strip()
        if chunk:
            sections.append(chunk)
        start_index = match_index + len(marker)
    return sections or [raw.strip()]


def extract_bullets(text: str) -> list[str]:
    raw = str(text or '')
    if not raw.strip():
        return []
    lines = [clean_text(line) for line in raw.splitlines() if clean_text(line)]
    bullets: list[str] = []
    for line in lines:
        match = BULLET_LINE_RE.match(line)
        if match:
            bullets.append(clean_text(match.group(1)))
    if bullets:
        return bullets
    cleaned = clean_text(raw)
    candidates = [clean_text(part) for part in re.split(r'(?<=[.;:])\s+', cleaned) if clean_text(part)]
    return candidates or [cleaned]


def decompose_composite_skill(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = [clean_text(part) for part in COMPOSITE_SPLIT_RE.split(cleaned) if clean_text(part)]
    if len(parts) <= 1:
        return [cleaned]
    base_object = ''
    match = re.search(r'(?:les|des|la|du|de la|de l[ae]|d[eu])\s+(.+)$', cleaned, flags=re.IGNORECASE)
    if match:
        base_object = clean_text(match.group(1))
    decomposed: list[str] = []
    for part in parts:
        if base_object and not re.search(r'(?:les|des|la|du|de la|de l[ae]|d[eu])', part, flags=re.IGNORECASE):
            decomposed.append(clean_text(f'{part} {base_object}'))
        else:
            decomposed.append(part)
    return [item for item in dict.fromkeys(decomposed) if item]


@dataclass(slots=True)
class ExtractedSkill:
    referential_id: str
    code: str
    libelle: str
    libelle_officiel: str
    evidence: str
    confidence: float
    match_type: str
    canonical_name: str
    category: str
    subcategory: str
    technical_keywords: list[str]
    origin_document: str
    block_code: str
    block_name: str
    activity_code: str
    activity_name: str
    source_page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'referential_id': self.referential_id,
            'code': self.code,
            'libelle': self.libelle,
            'libelle_officiel': self.libelle_officiel,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'match_type': self.match_type,
            'canonical_name': self.canonical_name,
            'category': self.category,
            'subcategory': self.subcategory,
            'technical_keywords': self.technical_keywords,
            'origin_document': self.origin_document,
            'block_code': self.block_code,
            'block_name': self.block_name,
            'activity_code': self.activity_code,
            'activity_name': self.activity_name,
            'source_page': self.source_page,
        }


class FranceCompetencesSkillExtractor:
    def __init__(self, *, taxonomy_version: str | None = None) -> None:
        self.taxonomy_version = taxonomy_version or '2026-07-05'

    def normalize_label(self, label: str) -> str:
        return normalize_market_alias(label)

    def classify(self, label: str) -> tuple[str, str, list[str]]:
        taxonomy = infer_skill_taxonomy(label)
        return taxonomy.category, taxonomy.subcategory, list(taxonomy.technical_keywords)

    def _build_skill(self, *, block_code: str, block_name: str, activity_code: str, activity_name: str, skill_code: str, skill_text: str, source_page: int | None, origin_document: str) -> ExtractedSkill:
        canonical = self.normalize_label(skill_text)
        category, subcategory, keywords = self.classify(skill_text)
        return ExtractedSkill(
            referential_id=f'{block_code}-{activity_code}-{skill_code}',
            code=skill_code,
            libelle=clean_text(skill_text),
            libelle_officiel=clean_text(skill_text),
            evidence=clean_text(skill_text),
            confidence=0.92,
            match_type='exact',
            canonical_name=canonical,
            category=category,
            subcategory=subcategory,
            technical_keywords=keywords,
            origin_document=origin_document,
            block_code=block_code,
            block_name=block_name,
            activity_code=activity_code,
            activity_name=activity_name,
            source_page=source_page,
        )

    def extract_block_skills(
        self,
        *,
        block_code: str,
        block_name: str,
        text: str,
        source_page: int | None,
        origin_document: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        activities: list[dict[str, Any]] = []
        skills: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        sections = [section for section in split_evaluation_sections(text) if clean_text(section)]
        if not sections:
            sections = [clean_text(text)]
        for activity_index, section in enumerate(sections, start=1):
            activity_code = f'A{activity_index}'
            activity_name = clean_text(section[:200])
            activities.append(
                {
                    'block_code': block_code,
                    'block_name': block_name,
                    'activity_code': activity_code,
                    'activity_name': activity_name,
                    'source_page': source_page,
                    'origin_document': origin_document,
                }
            )
            bullets = extract_bullets(section)
            for skill_index, bullet in enumerate(bullets, start=1):
                if is_excluded_text(bullet):
                    negatives.append(
                        {
                            'text': bullet,
                            'label': 'NOT_SKILL',
                            'block_code': block_code,
                            'activity_code': activity_code,
                            'source_page': source_page,
                            'origin_document': origin_document,
                        }
                    )
                    continue
                for variant in decompose_composite_skill(bullet):
                    skill = self._build_skill(
                        block_code=block_code,
                        block_name=block_name,
                        activity_code=activity_code,
                        activity_name=activity_name,
                        skill_code=f'C{skill_index:03d}',
                        skill_text=variant,
                        source_page=source_page,
                        origin_document=origin_document,
                    )
                    skills.append(skill.to_dict())
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in skills:
            key = (str(item.get('referential_id') or ''), normalize_for_match(item.get('evidence') or ''))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return activities, deduped, negatives
