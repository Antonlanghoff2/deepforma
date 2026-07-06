# Deployment

## Local Run

```bash
python -m src.web_app
```

The app listens on Flask's default local server unless you provide a different entry point or WSGI server.

## Production Notes

- Do not commit model weights or generated datasets.
- Keep `models/`, `checkpoints/`, and cache directories out of Git.
- Use the existing deployment scripts when they are already wired in the environment.
- Keep configuration in environment variables instead of hard-coding paths or secrets.

## Useful Checks

```bash
make test
.venv/bin/python -m pytest -q tests/test_deployment.py
```

## Operational Rules

- Prefer safe, reversible changes.
- Verify model artifacts before promoting them.
- Keep referential imports and training data generation separate from deployment.
