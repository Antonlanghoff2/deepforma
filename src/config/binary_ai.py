from __future__ import annotations

import os
from dataclasses import dataclass

from domain.errors import ConfigurationError


ALLOWED_BINARY_AI_BACKENDS = {"existing", "ml_from_scratch", "textcnn_from_scratch"}


@dataclass(frozen=True, slots=True)
class BinaryAISettings:
    backend: str = "existing"

    @classmethod
    def load(cls) -> "BinaryAISettings":
        backend = os.getenv("DEEPFORMA_BINARY_AI_BACKEND", "existing").strip().lower()
        if backend not in ALLOWED_BINARY_AI_BACKENDS:
            raise ConfigurationError(
                f"Backend binary AI invalide: {backend!r}. Valeurs attendues: {sorted(ALLOWED_BINARY_AI_BACKENDS)}"
            )
        return cls(backend=backend)


BINARY_AI_SETTINGS = BinaryAISettings.load()

