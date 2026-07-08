# DeepForma

DeepForma is a skill intelligence application. It collects job offers by territory, extracts and normalizes skills, compares market needs with profiles or training catalogs, and recommends relevant training paths with explanations.

The project currently focuses on four business flows:

- job offer collection and normalization;
- skill extraction and matching;
- France Compétences RNCP/RS and ROME referentials;
- CPF catalog analysis and training recommendation.

## Quick Start

### Install

```bash
python -m pip install -r requirements.txt
```

### Run the web app

```bash
python -m src.web_app
```

Open `http://127.0.0.1:5000`.

### Run tests

```bash
make test
# or
.venv/bin/python -m pytest -q
```

## Main Pipelines

### 1. IA skill extraction

Build datasets, train the multilabel classifier, then evaluate it:

```bash
make ia-prepare
make ia-train
make ia-evaluate
```

Full flow:

```bash
make ia-all
```

### 2. CPF recommendation pipeline

Prepare the CPF corpus, build pairs, train the recommender, and evaluate it:

```bash
make cpf-general-prepare
make cpf-pairs
make cpf-train
make cpf-all
```

### 3. France Compétences RNCP/RS

Download, inspect, normalize, and build training data:

```bash
make france-competences-download
make france-competences-inspect
make france-competences-normalize
make france-competences-build-training
make france-competences-all
```

### 4. RNCP / ROME / unified referential

Import official referentials and build mappings:

```bash
make import-france-competences
make import-rome-referential
make map-rncp-to-rome
make build-unified-skill-referential
make build-rome-rncp-training-dataset
make train-skill-extractor
```

### 5. Referential PDF extraction

Build annotation candidates, export reviewed data, train the NER and multilabel models, and evaluate them:

```bash
make build-referential-ner-candidates
make build-referential-multilabel-candidates
make export-referential-training-data
make train-referential-ner
make train-referential-multilabel
make evaluate-referential-models
make test-referential-ml-dl
```

### 6. IA recommendations (pedagogical knowledge base)

A CSV of 243 IA‑related pedagogical recommendations is loaded at app startup and matched
against the skills extracted from the formation text.  Three confidence levels appear on the
result page:

| Level | Score | Badge |
|-------|-------|-------|
| `HIGH` | ≥ 1.0 (exact phrase match) | Vert – *Conseillé* |
| `MEDIUM` | ≥ 0.82 (inclusion / alias) | Jaune – *À envisager* |
| `LOW` | < 0.82 | Orange – *À vérifier* |

Matching pipeline: **EXACT** (phrase) → **ALIAS** (phrase via alias) → **INCLUSION** (significant word overlap) → **EMBEDDING** (semantic, optional) → **DEFAULT** (fallback rule).

```bash
make ia-recommendations-validate   # dry-run import + quality report
make ia-recommendations-import     # write clean parquet + csv
make ia-recommendations-demo       # demo against RNCP41966 skills
```

Key files:
- `data/raw/recommandations_IA_consolide.csv` – source dataset (BOM UTF‑8, comma‑separated, mixed quoting)
- `src/data_sources/ia_recommendations.py` – robust CSV loader, normalizer, quality report
- `src/domain/ia_recommendation_matching.py` – matching service (phrase, alias, inclusion, embedding)
- `src/domain/models.py` – `IARecommendation`, `IARecommendationMatch` dataclasses
- `src/services/analysis_result_builder.py` – wired into `build_analysis_result` via `ia_recommendation_records`
- `tests/test_ia_recommendations.py` – 49 unit tests (loader, normalizer, matching, demo scenario)

## Key Documentation

- [Architecture](docs/architecture.md)
- [Data Flow](docs/data_flow.md)
- [Development](docs/development.md)
- [Deployment](docs/deployment.md)
- [AI certification skill extraction](docs/ai_certification_skill_extraction.md)
- [France Compétences pipeline](docs/france_competences_pipeline.md)
- [France Compétences / ROME training pipeline](docs/france_competences_rome_training_pipeline.md)

## Repository Layout

- `src/` business logic, services, data sources, ML helpers, and web app
- `scripts/` data preparation, import, training, and evaluation CLIs
- `data/` raw, processed, annotated, and training data
- `models/` local model artifacts, ignored by Git
- `reports/` metrics, audits, and exports
- `tests/` unit and integration tests

## Notes

- The web app is still the main entry point, but the business logic is being progressively moved into services.
- The repository keeps legacy pipelines for compatibility, but the current direction is to reuse one shared domain model, one normalization path, and one scoring path.
- Do not commit model weights or generated datasets.
