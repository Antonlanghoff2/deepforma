# Development

## Setup

```bash
python -m pip install -r requirements.txt
```

If you use a virtual environment, activate it first.

## Common Commands

Run the web app:

```bash
python -m src.web_app
```

Run the full test suite:

```bash
make test
```

Run a focused test file:

```bash
.venv/bin/python -m pytest -q tests/test_web_app.py
```

## Main Make Targets

- `make ia-prepare`
- `make ia-train`
- `make ia-evaluate`
- `make ia-all`
- `make cpf-general-prepare`
- `make cpf-pairs`
- `make cpf-train`
- `make cpf-all`
- `make france-competences-download`
- `make france-competences-inspect`
- `make france-competences-normalize`
- `make france-competences-build-training`
- `make france-competences-all`
- `make build-referential-ner-candidates`
- `make build-referential-multilabel-candidates`
- `make export-referential-training-data`
- `make train-referential-ner`
- `make train-referential-multilabel`
- `make evaluate-referential-models`
- `make test-referential-ml-dl`

## Code Conventions

- Prefer the shared `src/domain/` models.
- Prefer the shared skill normalization service.
- Keep new logic in `src/services/` or `src/data_sources/` instead of adding route logic.
- Use `pathlib` for filesystem operations.

## Testing

- Add unit tests for new parsing, normalization, and scoring logic.
- Add regression tests when moving code out of `src/web_app.py`.
- Keep the behavior of existing routes and exports stable while refactoring.
