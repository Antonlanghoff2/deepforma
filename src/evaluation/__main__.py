from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._common import timestamp_slug
from . import (
    RecommendationCase,
    evaluate_binary_classification,
    evaluate_multilabel_classification,
    evaluate_recommendation,
    evaluate_skill_extraction,
    optimize_binary_threshold,
    optimize_thresholds,
    save_binary_threshold_json,
    save_thresholds_json,
    write_evaluation_artifacts,
)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _parse_binary_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_score: list[float] = []
    for row in rows:
        truth = row.get("y_true", row.get("label", row.get("target", 0)))
        score = row.get("y_score", row.get("score", row.get("probability", 0.0)))
        y_true.append(int(float(truth)))
        y_score.append(float(score))
    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)


def _parse_multilabel_rows(rows: list[dict[str, Any]], labels: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = labels or []
    if not labels:
        first = rows[0] if rows else {}
        multi_hot = first.get("multi_hot")
        if isinstance(multi_hot, str):
            try:
                multi_hot = ast.literal_eval(multi_hot)
            except Exception:
                multi_hot = []
        scores_value = first.get("scores") or first.get("probabilities")
        if isinstance(scores_value, str):
            try:
                scores_value = ast.literal_eval(scores_value)
            except Exception:
                scores_value = []
        if isinstance(first.get("labels"), list):
            labels = [str(item) for item in first["labels"]]
        elif isinstance(multi_hot, list):
            labels = [f"label_{index}" for index in range(len(multi_hot))]
        elif isinstance(scores_value, list):
            labels = [f"label_{index}" for index in range(len(scores_value))]
        else:
            labels = [str(item) for item in first.get("label_names", [])]
    y_true: list[list[int]] = []
    y_score: list[list[float]] = []
    for row in rows:
        multi_hot = row.get("multi_hot")
        if isinstance(multi_hot, str):
            try:
                multi_hot = ast.literal_eval(multi_hot)
            except Exception:
                multi_hot = []
        if isinstance(multi_hot, list):
            truth = [int(value) for value in multi_hot]
        else:
            raw_labels = row.get("labels") or row.get("gold_labels") or row.get("true_labels") or []
            if isinstance(raw_labels, str):
                try:
                    parsed = ast.literal_eval(raw_labels)
                    if isinstance(parsed, list):
                        raw_labels = parsed
                    else:
                        raw_labels = [part.strip() for part in raw_labels.split("|") if part.strip()]
                except Exception:
                    raw_labels = [part.strip() for part in raw_labels.split("|") if part.strip()]
            truth = [1 if label in raw_labels else 0 for label in labels]
        scores_value = row.get("scores") or row.get("probabilities")
        if isinstance(scores_value, str):
            try:
                scores_value = ast.literal_eval(scores_value)
            except Exception:
                scores_value = []
        if isinstance(scores_value, list):
            score_row = [float(value) for value in scores_value]
        else:
            score_row = [float(row.get(label, 0.0)) for label in labels]
        y_true.append(truth)
        y_score.append(score_row)
    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float), labels


def _load_hf_sequence_classifier(model_path: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def _predict_binary_scores(texts: list[str], model_path: Path) -> np.ndarray:
    import torch

    tokenizer, model = _load_hf_sequence_classifier(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    scores: list[float] = []
    for text in texts:
        encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
            if logits.shape[-1] == 2:
                prob = torch.softmax(logits, dim=-1)[0, 1].item()
            else:
                prob = torch.sigmoid(logits)[0, 0].item()
        scores.append(float(prob))
    return np.asarray(scores, dtype=float)




def _load_binary_label_names(model_path: Path | None) -> tuple[str, str]:
    if not model_path:
        return ("non_ia", "ia")
    config_path = model_path / 'config.json'
    if not config_path.exists():
        return ("non_ia", "ia")
    try:
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
    except Exception:
        return ("non_ia", "ia")
    id2label = cfg.get('id2label') or {}
    if len(id2label) == 2:
        ordered = [id2label[key] for key in sorted(id2label, key=lambda item: int(item) if str(item).isdigit() else str(item))]
        return (str(ordered[0]), str(ordered[1]))
    return ("non_ia", "ia")

def _load_model_labels(model_path: Path) -> list[str]:
    config_path = model_path / 'config.json'
    if not config_path.exists():
        return []
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    id2label = cfg.get('id2label') or {}
    ordered = [id2label[key] for key in sorted(id2label, key=lambda item: int(item) if str(item).isdigit() else str(item))] if id2label else []
    return [str(label) for label in ordered]


def _predict_multilabel_scores(texts: list[str], model_path: Path) -> tuple[np.ndarray, list[str]]:
    import torch

    tokenizer, model = _load_hf_sequence_classifier(model_path)
    labels = _load_model_labels(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    rows: list[list[float]] = []
    for text in texts:
        encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
            probas = torch.sigmoid(logits)[0].detach().cpu().numpy().tolist()
        rows.append([float(value) for value in probas])
    return np.asarray(rows, dtype=float), labels


def _dict_like(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, '__dict__'):
        return dict(value.__dict__)
    return {'value': value}


def _binary_text(row: dict[str, Any]) -> str:
    return str(row.get('texte_modele') or row.get('text') or row.get('description') or '')


def _binary_truth(row: dict[str, Any]) -> int:
    label = str(row.get('statut_annotation') or row.get('label') or row.get('y_true') or '').strip().lower()
    if label in {'ia_confirmee', '1', 'true', 'yes', 'ia'}:
        return 1
    return 0


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [part.strip() for part in value.split('|') if part.strip()]
    return []


def _load_sentence_transformer(model_path: Path):
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(str(model_path))
    return model


def _encode_scores(model, query: str, candidates: list[str]) -> list[float]:
    import numpy as np

    query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
    candidate_embeddings = model.encode(candidates, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    scores = np.asarray(candidate_embeddings, dtype=float) @ np.asarray(query_embedding, dtype=float)
    return scores.tolist()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Évalue les modèles Deepforma")
    parser.add_argument("--task", required=True, choices=["binary", "multilabel", "extraction", "recommendation"])
    parser.add_argument("--model-path", type=Path, required=False)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluations"))
    parser.add_argument("--model-name", type=str, default="deepforma-model")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--threshold-mode", type=str, default="maximize_f1", choices=["maximize_f1", "maximize_youden_j", "min_precision", "min_recall"])
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--semantic-threshold", type=float, default=0.75)
    parser.add_argument("--embedding-model", type=Path, default=None)
    return parser


def _run_binary(args: argparse.Namespace) -> Path:
    rows = _load_records(args.test_file)
    run_version = args.version or timestamp_slug()
    label_names = _load_binary_label_names(args.model_path if args.model_path and args.model_path.exists() else None)
    y_true = np.asarray([_binary_truth(row) for row in rows], dtype=int)
    if args.model_path and args.model_path.exists():
        y_score = _predict_binary_scores([_binary_text(row) for row in rows], args.model_path)
    else:
        _y_true, y_score = _parse_binary_rows(rows)
        y_true = _y_true
    threshold = 0.5
    threshold_optimization: dict[str, Any] = {}
    if args.validation_file and args.validation_file.exists() and args.model_path and args.model_path.exists():
        validation_rows = _load_records(args.validation_file)
        validation_scores = _predict_binary_scores([_binary_text(row) for row in validation_rows], args.model_path)
        validation_true = np.asarray([_binary_truth(row) for row in validation_rows], dtype=int)
        calibration = optimize_binary_threshold(
            validation_true,
            validation_scores,
            mode=args.threshold_mode,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        threshold = calibration.threshold
        threshold_optimization = calibration.to_dict()
        save_binary_threshold_json(
            args.output_dir / args.model_name / run_version / 'threshold.json',
            calibration,
            model_name=args.model_name,
            version=run_version,
            metric=args.threshold_mode,
        )
    report = evaluate_binary_classification(y_true, y_score, threshold=threshold, label_names=label_names)
    report.threshold_optimization = threshold_optimization
    if not threshold_optimization:
        report.warnings.append('Aucun jeu de validation binaire fourni; seuil de décision conservé à 0.5.')
    return write_evaluation_artifacts(report, args.output_dir, model_name=args.model_name, task="binary", version=run_version)


def _run_multilabel(args: argparse.Namespace) -> Path:
    rows = _load_records(args.test_file)
    run_version = args.version or timestamp_slug()
    if args.model_path and args.model_path.exists():
        texts = [str(row.get('text') or row.get('texte_modele') or '') for row in rows]
        y_score, labels = _predict_multilabel_scores(texts, args.model_path)
        y_true, _, _ = _parse_multilabel_rows(rows, labels=labels)
    else:
        y_true, y_score, labels = _parse_multilabel_rows(rows)
    thresholds = {label: 0.5 for label in labels}
    if args.validation_file and args.validation_file.exists():
        validation_rows = _load_records(args.validation_file)
        if args.model_path and args.model_path.exists():
            validation_texts = [str(row.get('text') or row.get('texte_modele') or '') for row in validation_rows]
            y_val_score, labels = _predict_multilabel_scores(validation_texts, args.model_path)
            y_val_true, _, _ = _parse_multilabel_rows(validation_rows, labels=labels)
        else:
            y_val_true, y_val_score, labels = _parse_multilabel_rows(validation_rows, labels=labels)
        calibration = optimize_thresholds(
            y_val_true,
            y_val_score,
            labels,
            mode=args.threshold_mode,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        thresholds = calibration.thresholds
        save_thresholds_json(
            args.output_dir / args.model_name / run_version / "thresholds.json",
            calibration,
            model_name=args.model_name,
            version=run_version,
            metric=args.threshold_mode,
        )
    report = evaluate_multilabel_classification(y_true, y_score, labels, thresholds=thresholds)
    return write_evaluation_artifacts(report, args.output_dir, model_name=args.model_name, task="multilabel", version=run_version)


def _run_extraction(args: argparse.Namespace) -> Path:
    rows = _load_records(args.test_file)
    report = evaluate_skill_extraction(rows, embedding_model=args.embedding_model)
    return write_evaluation_artifacts(report, args.output_dir, model_name=args.model_name, task="extraction", version=args.version or None)


def _run_recommendation(args: argparse.Namespace) -> Path:
    rows = _load_records(args.test_file)
    cases: list[RecommendationCase] = []
    rankings: dict[str, list[dict[str, Any]]] = {}
    model = _load_sentence_transformer(args.model_path) if args.model_path and args.model_path.exists() else None
    for index, row in enumerate(rows):
        case_id = str(row.get("case_id") or row.get("query_id") or row.get("id") or index)
        query = str(row.get("query") or row.get("target_job") or row.get("profile") or "")
        positive_uid = str(row.get("positive_uid") or row.get("relevant_formation_id") or row.get("relevant_id") or "")
        negative_uid = str(row.get("negative_uid") or row.get("irrelevant_formation_id") or row.get("negative_id") or "")
        positive_text = str(row.get("positive_text") or row.get("relevant_text") or "")
        negative_text = str(row.get("negative_text") or row.get("irrelevant_text") or "")
        relevant = row.get("relevant_formations") or row.get("labels") or {}
        if not relevant and positive_uid:
            relevant = {positive_uid: 3}
        if isinstance(relevant, list):
            relevant = {str(item): 1 for item in relevant}
        if isinstance(relevant, str):
            relevant = {item.strip(): 1 for item in relevant.split("|") if item.strip()}
        cases.append(
            RecommendationCase(
                case_id=case_id,
                profile=str(row.get("profile") or query),
                owned_skills=_as_list(row.get("owned_skills")),
                missing_skills=_as_list(row.get("missing_skills")),
                target_job=str(row.get("target_job") or query),
                rome_codes=_as_list(row.get("rome_codes") or row.get("codes_rome")),
                territory=str(row.get("territory") or row.get("department_code") or row.get("region_code") or ""),
                relevant_formations={str(key): int(value) for key, value in dict(relevant).items()},
            )
        )
        ranking_items: list[dict[str, Any]] = []
        if positive_uid and positive_text:
            ranking_items.append({"formation_id": positive_uid, "text": positive_text})
        if negative_uid and negative_text:
            ranking_items.append({"formation_id": negative_uid, "text": negative_text})
        if model is not None and query and ranking_items:
            scores = _encode_scores(model, query, [item["text"] for item in ranking_items])
            for item, score in zip(ranking_items, scores):
                item["score"] = float(score)
        elif ranking_items:
            for item in ranking_items:
                item["score"] = float(len(item.get("text", "")))
        ranking_items.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        rankings[case_id] = ranking_items
    report = evaluate_recommendation(cases, rankings)
    return write_evaluation_artifacts(report, args.output_dir, model_name=args.model_name, task="recommendation", version=args.version or None)


def main() -> None:
    args = build_parser().parse_args()
    if args.task == "binary":
        root = _run_binary(args)
    elif args.task == "multilabel":
        root = _run_multilabel(args)
    elif args.task == "extraction":
        root = _run_extraction(args)
    else:
        root = _run_recommendation(args)
    print(root)


if __name__ == "__main__":
    main()
