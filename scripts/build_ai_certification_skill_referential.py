#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import fitz  # type: ignore

from common.text import clean_text, normalize_for_match


DEFAULT_INPUT = ROOT / "data" / "raw" / "Referentiel de certification - Ingenieur en intelligence artificelle janvier 2025.pdf"
DEFAULT_JSON = ROOT / "data" / "referentials" / "ai_engineer_certification_2025.json"
DEFAULT_CSV = ROOT / "data" / "referentials" / "ai_engineer_certification_2025.csv"
DEFAULT_METADATA = ROOT / "data" / "referentials" / "ai_engineer_certification_2025.metadata.json"

COMPETENCE_RE = re.compile(r"\b(A\d+[AB]?-C\d+)\s*[)\.:\-]?", flags=re.IGNORECASE)
BLOCK_RE = re.compile(r"\bBLOC\s*(1|2|3|4|5A|5B)\b", flags=re.IGNORECASE)
ACTIVITY_HINT_RE = re.compile(r"\bA(\d+[AB]?)\.", flags=re.IGNORECASE)
STOP_RE = re.compile(
    r"(?:\[ ?comp|mise en situation professionnelle|conditions pratiques de réalisation|modalit[ée]s d[’']évaluation|crit[eè]res d[’']évaluation|jeux? de rôle|études? de cas|cas d’usage)",
    flags=re.IGNORECASE,
)
ALIAS_SPLIT_RE = re.compile(r"\b(?:afin de|afin d['’]|pour|en|dans le but de|tout en|de manière à|de maniere a)\b", flags=re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construit le référentiel de certification IA à partir du PDF RNCP.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    return parser


def _pdf_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = ROOT / "data" / "raw" / "Referentiel de certification - Ingenieur en intelligence artificelle janvier 2025.pdf"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"PDF introuvable: {path}")


def _load_pages(pdf_path: Path) -> tuple[fitz.Document, str]:
    doc = fitz.open(pdf_path)
    sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return doc, sha256


def _clean(text: str) -> str:
    text = clean_text(text)
    text = STOP_RE.split(text, maxsplit=1)[0]
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -–—:;,.")


def _detect_block(page_text: str, current: str | None) -> str | None:
    match = BLOCK_RE.search(page_text)
    if match:
        value = clean_text(match.group(1)).upper()
        if value in {"1", "2", "3", "4"}:
            return f"B{value}"
        if value == "5A":
            return "B5A"
        if value == "5B":
            return "B5B"
    return current


def _detect_title(page_text: str) -> str:
    lines = [clean_text(line) for line in page_text.splitlines() if clean_text(line)]
    for line in lines[:5]:
        if "INTELLIGENCE ARTIFICIELLE" in normalize_for_match(line):
            cleaned = clean_text(line)
            cleaned = re.sub(r"^REFERENTIEL DE CERTIFICATION\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\(MS\)", "", cleaned, flags=re.IGNORECASE).strip()
            return cleaned.title() if cleaned.isupper() else cleaned
    return "Ingénieur en intelligence artificielle"


def _page_blocks(page: fitz.Page) -> list[tuple[float, float, float, float, str]]:
    blocks = []
    for item in page.get_text("blocks"):
        x0, y0, x1, y1, text = item[:5]
        text = clean_text(text)
        if text:
            blocks.append((float(x0), float(y0), float(x1), float(y1), text))
    return blocks


def _competence_block_texts(page: fitz.Page) -> list[str]:
    selected: list[tuple[float, str]] = []
    for x0, y0, x1, y1, text in _page_blocks(page):
        if 160 <= x0 <= 470 and x1 <= 560:
            selected.append((y0, text))
    selected.sort(key=lambda item: item[0])
    return [text for _, text in selected]


def _normalize_code(code: str) -> str:
    return clean_text(code).upper().replace(" ", "")


def _short_label(official_description: str) -> str:
    text = _clean(official_description)
    if not text:
        return ""
    parts = ALIAS_SPLIT_RE.split(text, maxsplit=1)
    candidate = clean_text(parts[0] if parts else text)
    words = candidate.split()
    if len(words) > 12:
        candidate = " ".join(words[:12])
    if candidate.isupper() or sum(1 for char in candidate if char.isupper()) > max(4, len(candidate) // 2):
        candidate = candidate[:1].upper() + candidate[1:].lower() if candidate else candidate
    return candidate


def _aliases_for_skill(skill_id: str, label: str, official_description: str, block: str, activity: str) -> list[str]:
    aliases: list[str] = []
    candidates = [label]
    short = _short_label(official_description)
    if short and short not in candidates:
        candidates.append(short)
    text = normalize_for_match(f"{label} {official_description}")
    manual_aliases: list[str] = []
    if any(token in text for token in ["langage", "texte", "text", "tokenis", "embedding", "transformer"]):
        manual_aliases.extend(["NLP", "traitement du langage naturel", "tokenisation", "embeddings", "word2vec", "bert", "gpt", "transformers"])
    if any(token in text for token in ["image", "vision", "visuel", "ordinateur"]):
        manual_aliases.extend(["computer vision", "vision par ordinateur", "cnn", "deep learning"])
    if any(token in text for token in ["sonor", "audio", "dialogue", "parole"]):
        manual_aliases.extend(["audio", "speech", "signal sonore"])
    if any(token in text for token in ["déploi", "deploi", "production", "infrastructure", "pipeline", "monitor", "maintenance", "portabilité", "portabilite"]):
        manual_aliases.extend(["MLOps", "docker", "mlflow", "fastapi", "ci/cd", "aws sagemaker", "gcp vertex ai"])
    if any(token in text for token in ["modèle", "modele", "apprentissage automatique", "machine learning", "prédiction", "prediction", "validation croisée", "cross"]):
        manual_aliases.extend(["Machine Learning", "scikit-learn", "random forest", "xgboost", "svm", "k-means", "dbscan", "pca"])
    if any(token in text for token in ["apprentissage profond", "réseaux de neurones", "reseaux de neurones", "deep learning"]):
        manual_aliases.extend(["Deep Learning", "tensorflow", "keras", "pytorch", "cnn", "rnn", "lstm", "transformers"])
    if any(token in text for token in ["data", "donnée", "donnee", "préparer", "preparer", "nettoyer", "normaliser"]):
        manual_aliases.extend(["préparation des données", "nettoyage des données", "normalisation", "prétraitement"])
    if block == "B5B":
        manual_aliases.extend(["MLOps", "docker", "mlflow", "fastapi", "ci/cd"])
    if block == "B5A":
        manual_aliases.extend(["Deep Learning", "NLP", "Computer Vision", "MLOps"])

    for value in [*candidates, *manual_aliases]:
        alias = clean_text(value)
        if alias and normalize_for_match(alias) not in {normalize_for_match(item) for item in aliases}:
            aliases.append(alias)
    return aliases


def _parse_competency_chunk(chunk: str, *, block: str, source_page: int) -> dict[str, Any] | None:
    text = _clean(chunk)
    match = COMPETENCE_RE.search(text)
    if not match:
        return None
    code = _normalize_code(match.group(1))
    description = clean_text(text[match.end() :])
    description = _clean(description)
    if not description:
        return None
    activity_match = ACTIVITY_HINT_RE.search(code.replace("-", "."))
    activity = f"A{activity_match.group(1)}" if activity_match else code.split("-", 1)[0]
    label = _short_label(description) or description[:120]
    normalized_label = normalize_for_match(label)
    skill_id = f"{block}-{code}" if block else code
    aliases = _aliases_for_skill(skill_id, label, description, block, activity)
    return {
        "id": skill_id,
        "block": block,
        "activity": activity,
        "code": code,
        "label": label,
        "official_description": description,
        "normalized_label": normalized_label,
        "aliases": aliases,
        "source_page": source_page,
        "active": True,
    }


def build_referential(pdf_path: Path) -> dict[str, Any]:
    doc, sha256 = _load_pages(pdf_path)
    title = ""
    block = None
    skills_by_id: dict[str, dict[str, Any]] = {}
    page_ranges: dict[str, list[int]] = defaultdict(lambda: [10**9, 0])

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        page_text = clean_text(page.get_text("text"))
        if not title:
            title = _detect_title(page_text)
        block = _detect_block(page_text, block)
        if not block:
            continue
        candidate_texts = _competence_block_texts(page)
        if not candidate_texts:
            continue
        for candidate in candidate_texts:
            chunks = re.split(r"(?=\bA\d+[AB]?[-\.]C\d+\s*[)\.:]?)", candidate)
            for chunk in chunks:
                skill = _parse_competency_chunk(chunk, block=block, source_page=page_index + 1)
                if not skill:
                    continue
                existing = skills_by_id.get(skill["id"])
                if existing is None:
                    skills_by_id[skill["id"]] = skill
                    page_ranges[skill["id"]][0] = min(page_ranges[skill["id"]][0], page_index + 1)
                    page_ranges[skill["id"]][1] = max(page_ranges[skill["id"]][1], page_index + 1)
                    continue
                if len(skill["official_description"]) > len(existing["official_description"]):
                    existing["official_description"] = skill["official_description"]
                    existing["label"] = skill["label"]
                    existing["normalized_label"] = skill["normalized_label"]
                    existing["aliases"] = skill["aliases"]
                existing["source_page"] = min(int(existing["source_page"]), page_index + 1)
                existing["active"] = True

    skills = sorted(
        skills_by_id.values(),
        key=lambda item: (item["block"], item["activity"], item["code"], item["source_page"], item["label"]),
    )
    metadata = {
        "source_pdf": str(pdf_path),
        "sha256": sha256,
        "page_count": len(doc),
        "skill_count": len(skills),
        "blocks": sorted({skill["block"] for skill in skills}),
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    return {
        "referential_id": "ingenieur_ia_2025",
        "title": "Ingénieur en intelligence artificielle",
        "version": "2025-01",
        "skills": skills,
        "metadata": metadata,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, skills: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "block", "activity", "code", "label", "official_description", "normalized_label", "aliases", "source_page", "active"],
        )
        writer.writeheader()
        for skill in skills:
            row = dict(skill)
            row["aliases"] = json.dumps(row.get("aliases", []), ensure_ascii=False)
            writer.writerow(row)


def main() -> None:
    args = build_parser().parse_args()
    pdf_path = _pdf_path(args.input)
    referential = build_referential(pdf_path)
    _write_json(args.json_output, referential)
    _write_csv(args.csv_output, referential["skills"])
    _write_json(args.metadata_output, referential["metadata"])
    print(json.dumps({"skills": len(referential["skills"]), "json": str(args.json_output), "csv": str(args.csv_output), "metadata": str(args.metadata_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
