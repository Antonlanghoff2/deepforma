# France Compétences Pipeline

This pipeline imports the official RNCP / RS data from data.gouv.fr and turns it
into normalized certification, block, and skill tables.

## Sources

- `data.gouv.fr` dataset: `repertoire-national-des-certifications-professionnelles-et-repertoire-specifique`
- official open data exports only as the reproducible fallback
- API-based discovery only when an explicit documentated source is configured

## Environment Variables

- `FRANCE_COMPETENCES_DATASET_SLUG`
- `FRANCE_COMPETENCES_INCLUDE_RNCP`
- `FRANCE_COMPETENCES_INCLUDE_RS`
- `FRANCE_COMPETENCES_ACTIVE_ONLY`
- `FRANCE_COMPETENCES_KEEP_EVALUATION`
- `FRANCE_COMPETENCES_FORCE_DOWNLOAD`
- `FRANCE_COMPETENCES_TIMEOUT`

## Main Commands

```bash
python scripts/download_france_competences.py
python scripts/inspect_france_competences_archive.py
python scripts/normalize_france_competences.py
python scripts/build_france_competences_training_dataset.py
```

## Make Targets

- `make france-competences-download`
- `make france-competences-inspect`
- `make france-competences-normalize`
- `make france-competences-build-training`
- `make france-competences-all`

## Produced Files

Raw data:

- `data/raw/france_competences/manifest.json`

Processed data:

- `data/processed/france_competences/certifications.parquet`
- `data/processed/france_competences/blocks.parquet`
- `data/processed/france_competences/skills.parquet`
- `data/processed/france_competences/certification_rome_links.parquet`
- `data/processed/france_competences/quality_report.json`
- `data/processed/france_competences/review_queue.csv`

Training data:

- `data/training/france_competences/ner_*.jsonl`
- `data/training/france_competences/skill_classification_*.jsonl`
- `data/training/france_competences/skill_normalization_*.jsonl`
- `data/training/france_competences/semantic_pairs_*.jsonl`

## Rules

- keep only active certifications in the main corpus by default;
- exclude evaluation criteria, juries, and administrative text from skills;
- keep source text for audit;
- split by certification to avoid leakage;
- use human review for ambiguous items.
