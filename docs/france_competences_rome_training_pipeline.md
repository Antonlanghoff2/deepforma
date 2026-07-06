# France Compétences / ROME Training Pipeline

This pipeline aligns France Compétences RNCP/RS with France Travail ROME and
uses the mapping to enrich skill extraction and training datasets.

## Goal

- keep the job offer text as the primary signal;
- use ROME as a context signal, not as a replacement for text analysis;
- link certifications to ROME codes when the evidence is explicit or strong
  enough for review.

## Main Outputs

- `data/referentials/france_competences/certifications.jsonl`
- `data/referentials/france_competences/blocks.jsonl`
- `data/referentials/france_competences/skills.jsonl`
- `data/referentials/rome/jobs.jsonl`
- `data/referentials/rome/job_titles.jsonl`
- `data/referentials/rome/skills.jsonl`
- `data/referentials/mappings/rncp_rome_links.jsonl`
- `data/referentials/unified/skills.jsonl`
- `data/training/skill_extraction/train.jsonl`
- `data/training/skill_extraction/validation.jsonl`
- `data/training/skill_extraction/test.jsonl`

## Thresholds

- `RNCP_ROME_AUTO_VALIDATE_THRESHOLD=0.82`
- `RNCP_ROME_REVIEW_THRESHOLD=0.65`

## Commands

```bash
python scripts/import_france_competences.py --active-only --dry-run
python scripts/import_france_competences.py --active-only --write
python scripts/import_rome_referential.py --dry-run
python scripts/import_rome_referential.py --write
python scripts/map_rncp_to_rome.py --dry-run
python scripts/map_rncp_to_rome.py --write
python scripts/build_unified_skill_referential.py --dry-run
python scripts/build_unified_skill_referential.py --write
python scripts/enrich_offers_with_rome_rncp.py --input <source> --dry-run
python scripts/build_rome_rncp_training_dataset.py
python scripts/train_skill_extractor.py --train data/training/skill_extraction/train.jsonl --validation data/training/skill_extraction/validation.jsonl --test data/training/skill_extraction/test.jsonl
```

## Rules

- do not invent ROME or RNCP codes;
- keep mappings explicit and explainable;
- keep uncertain mappings in review;
- keep source links in exports;
- avoid injecting unmapped skills as positives.
