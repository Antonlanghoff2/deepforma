from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .models import TrainingDocument


def to_export_dict(document: TrainingDocument) -> dict[str, Any]:
    return document.to_dict()


def write_json(document: TrainingDocument, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_export_dict(document), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_jsonl(documents: list[TrainingDocument], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for document in documents:
            fh.write(json.dumps(to_export_dict(document), ensure_ascii=False) + "\n")
    return path


def write_csv(documents: list[TrainingDocument], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["titre", "organisme", "référence", "durée", "niveau", "format", "CPF", "compétences", "outils", "domaines", "confiance", "besoin_de_revue"])
        for document in documents:
            program = document.program
            writer.writerow([
                program.title,
                document.provider.canonical_name or document.provider.name,
                program.reference,
                program.duration_text,
                program.level,
                program.format,
                program.cpf,
                " | ".join(skill.canonical_name for skill in program.skills),
                " | ".join(tool.canonical_name for tool in program.tools),
                " | ".join(domain.canonical_name for domain in program.domains),
                f"{document.confidence:.2f}",
                "oui" if document.review_required else "non",
            ])
    return path


def write_parquet(documents: list[TrainingDocument], output: str | Path) -> Path:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("Export Parquet indisponible: pandas/pyarrow non installés.") from exc
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [to_export_dict(document) for document in documents]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
