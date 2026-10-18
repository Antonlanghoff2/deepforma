"""WSGI entry point used by Gunicorn in production."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference.deepforma_predictor import DeepformaPredictor  # noqa: E402
from web_app import create_app  # noqa: E402


MODELS_ROOT = Path(os.getenv("DEEPFORMA_MODELS_DIR", str(PROJECT_ROOT / "models"))).expanduser()
predictor = DeepformaPredictor(
    binary_model_dir=MODELS_ROOT / "binary_ia_v2" / "final",
    multilabel_model_dir=MODELS_ROOT / "multilabel_competences_v2" / "final",
)
application = create_app(predictor=predictor)
