from __future__ import annotations

from dataclasses import dataclass

from common.text import clean_text, normalize_for_match

SECTION_LABELS = [
    "TITLE", "PROVIDER", "REFERENCE", "DURATION", "LEVEL", "FORMAT", "PRICE", "CPF",
    "CERTIFICATION", "PUBLIC", "PREREQUISITES", "OBJECTIVES", "PROGRAM", "MODULE",
    "SKILLS", "TOOLS", "DOMAINS", "FOOTER", "OTHER",
]

SECTION_VARIANTS: dict[str, tuple[str, ...]] = {
    "TITLE": ("titre", "intitule", "nom de la formation", "formation"),
    "PROVIDER": ("organisme", "provider", "éditeur", "editeur", "centre de formation"),
    "REFERENCE": ("reference", "référence", "code rncp", "rncp", "certification"),
    "DURATION": ("duree", "durée", "temps de formation"),
    "LEVEL": ("niveau", "level", "bac+5", "bac +5", "niveau 7"),
    "FORMAT": ("format", "modalite", "modalité", "presentiel", "distanciel", "online"),
    "PRICE": ("prix", "tarif", "cout", "coût", "sur devis"),
    "CPF": ("cpf", "eligible cpf", "éligible cpf"),
    "CERTIFICATION": ("certification", "titre rncp", "rncp"),
    "PUBLIC": ("public", "pour qui", "participants", "profil"),
    "PREREQUISITES": ("prerequis", "pré-requis", "prérequis", "conditions d acces", "conditions d'accès"),
    "OBJECTIVES": ("objectifs", "objectifs pedagogiques", "objectifs d apprentissage", "enjeux business"),
    "PROGRAM": ("programme", "contenu", "programme detaille", "déroulé", "deroule"),
    "MODULE": ("module", "modules", "etape", "étape", "partie"),
    "SKILLS": ("competences", "compétences", "competences acquises", "compétences acquises", "competences visees", "compétences visées"),
    "TOOLS": ("outils", "logiciels", "technologies"),
    "DOMAINS": ("domaines", "secteurs", "familles"),
    "FOOTER": ("copyright", "compilation", "document de recherche", "page ", "ref."),
}

NER_LABELS = [
    "SKILL", "SOFT_SKILL", "TOOL", "METHOD", "KNOWLEDGE", "DOMAIN", "DEGREE",
    "CERTIFICATION", "DURATION", "PRICE", "REFERENCE", "PROVIDER", "OTHER",
]
NER_BIO_LABELS = ["O"] + [f"{prefix}-{label}" for label in NER_LABELS for prefix in ("B", "I")]
SECTION_BIO_LABELS = ["O"] + [f"{prefix}-{label}" for label in SECTION_LABELS for prefix in ("B", "I")]


@dataclass(frozen=True, slots=True)
class SectionMatch:
    label: str
    confidence: float
    evidence: str


def normalize_heading(text: str) -> str:
    return normalize_for_match(clean_text(text))


def classify_section_label(text: str) -> SectionMatch:
    normalized = normalize_heading(text)
    if not normalized:
        return SectionMatch("OTHER", 0.0, "")
    for label, variants in SECTION_VARIANTS.items():
        for variant in variants:
            normalized_variant = normalize_heading(variant)
            if normalized == normalized_variant or normalized.startswith(normalized_variant):
                confidence = 0.98 if normalized == normalized_variant else 0.84
                return SectionMatch(label, confidence, variant)
    if normalized.startswith("bloc"):
        return SectionMatch("PROGRAM", 0.72, "bloc")
    if normalized.startswith("a ") or normalized.startswith("c ") or normalized.startswith("ce "):
        return SectionMatch("OTHER", 0.4, "code")
    return SectionMatch("OTHER", 0.2, "")
