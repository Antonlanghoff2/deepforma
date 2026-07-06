# AI Certification Skill Extraction

This document covers the referential-driven extraction pipeline for the PDF
`Referentiel de certification - Ingenieur en intelligence artificelle janvier 2025.pdf`.

## Goal

Extract only:

- the certification title;
- skills contained in the `REFERENTIEL DE COMPETENCES` column.

Do not extract activities, evaluation criteria, practical cases, or conditions
of realization as skills.

## Main Outputs

- `data/referentials/ai_engineer_certification_2025.json`
- `data/referentials/ai_engineer_certification_2025.csv`
- `data/referentials/ai_engineer_certification_2025.metadata.json`

## Canonical Fields

Each skill keeps:

- block code and block label;
- competence code such as `A4-C5`;
- original wording;
- short normalized wording;
- category and subcategory;
- technical keywords;
- source document.

## Extraction Rules

- keep the source wording intact;
- strip page headers and decorative text;
- reject evaluation criteria and non-skill items;
- keep provenance and page number;
- generate aliases conservatively;
- never mark annotations as approved automatically.

## Matching Thresholds

- `AI_CERT_SKILL_EXACT_THRESHOLD=1.0`
- `AI_CERT_SKILL_ALIAS_THRESHOLD=0.92`
- `AI_CERT_SKILL_SEMANTIC_THRESHOLD=0.72`
- `AI_CERT_SKILL_IMPLICIT_THRESHOLD=0.80`

## Commands

Build the referential:

```bash
python scripts/build_ai_certification_skill_referential.py
```

Compare the extracted skills with offers:

```bash
python scripts/compare_ai_certification_to_market.py
```

Extract skill matches into offers:

```bash
python scripts/extract_ai_certification_skills.py --dry-run
python scripts/extract_ai_certification_skills.py --write
```

## Limits

The referential is centered on the AI engineer certification. It is useful as a
controlled source for the extractor, but it is not a general-purpose skill
ontology.
