from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.binary_ai import BinaryAISettings
from deepforma.evaluation.binary_classification_metrics import evaluate_binary_classification, optimize_binary_threshold
from deepforma.training.binary_ai_dataset import build_binary_ai_dataset, group_stratified_split, load_tabular_file
from deepforma.training.binary_ai_ml import BinaryAITrainingConfig, fit_binary_ai_ml, load_binary_ai_ml, predict_binary_ai_ml
from deepforma.training.binary_ai_textcnn import BinaryAITextCNNConfig, fit_binary_ai_textcnn, predict_binary_ai_textcnn
from inference import binary_ai_predictor


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "formation_id": "1",
                "intitule": "Formation IA",
                "description": "Apprendre l IA",
                "est_lie_ia": 1,
                "formation_group_id": "g1",
            },
            {
                "formation_id": "2",
                "intitule": "Marketing digital",
                "description": "SEO social media",
                "est_lie_ia": 0,
                "formation_group_id": "g2",
            },
            {
                "formation_id": "3",
                "intitule": "IA generative",
                "description": "ChatGPT",
                "est_lie_ia": 1,
                "formation_group_id": "g3",
            },
            {
                "formation_id": "4",
                "intitule": "Comptabilite",
                "description": "Bilan",
                "est_lie_ia": 0,
                "formation_group_id": "g4",
            },
            {
                "formation_id": "5",
                "intitule": "Automatisation IA",
                "description": "No code",
                "est_lie_ia": 1,
                "formation_group_id": "g5",
            },
            {
                "formation_id": "6",
                "intitule": "Gestion",
                "description": "Gestion de projet",
                "est_lie_ia": 0,
                "formation_group_id": "g6",
            },
        ]
    )


def _write_sample_sources(tmp_path: Path) -> dict[str, Path]:
    frame = _sample_rows()
    csv_path = tmp_path / "sample.csv"
    xlsx_path = tmp_path / "sample.xlsx"
    jsonl_path = tmp_path / "sample.jsonl"
    parquet_path = tmp_path / "sample.parquet"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    frame.to_excel(xlsx_path, index=False)
    jsonl_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in frame.to_dict(orient="records")), encoding="utf-8")
    frame.to_parquet(parquet_path, index=False)
    return {"csv": csv_path, "xlsx": xlsx_path, "jsonl": jsonl_path, "parquet": parquet_path}


def _prepare_splits(tmp_path: Path):
    frame = _sample_rows()
    source_path = tmp_path / "source.csv"
    frame.to_csv(source_path, index=False, encoding="utf-8")
    dataset, audit, duplicates, conflicts = build_binary_ai_dataset([source_path])
    assert len(dataset) == 6
    assert audit.positives == 3
    assert audit.negatives == 3
    assert duplicates.empty
    assert conflicts.empty
    splits, manifest = group_stratified_split(dataset, seed=42)
    return dataset, audit, splits, manifest


def test_load_tabular_file_supports_csv_xlsx_jsonl_parquet(tmp_path):
    sources = _write_sample_sources(tmp_path)
    assert len(load_tabular_file(sources["csv"])) == 6
    assert len(load_tabular_file(sources["xlsx"])) == 6
    assert len(load_tabular_file(sources["jsonl"])) == 6
    assert len(load_tabular_file(sources["parquet"])) == 6


def test_dataset_building_and_group_splits_are_stable(tmp_path):
    dataset, audit, splits, manifest = _prepare_splits(tmp_path)
    assert dataset["text"].fillna("").astype(str).str.strip().ne("").all()
    assert manifest.seed == 42
    assert set(splits["train"]["group_id"]).isdisjoint(set(splits["validation"]["group_id"]))
    assert set(splits["train"]["group_id"]).isdisjoint(set(splits["test"]["group_id"]))
    assert set(splits["validation"]["group_id"]).isdisjoint(set(splits["test"]["group_id"]))
    splits_again, manifest_again = group_stratified_split(dataset, seed=42)
    assert manifest.group_ids_hash == manifest_again.group_ids_hash
    assert splits["train"]["group_id"].tolist() == splits_again["train"]["group_id"].tolist()


def test_binary_metrics_and_threshold_optimization():
    y_true = [0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.8, 0.2]
    threshold_result = optimize_binary_threshold(y_true, y_score, mode="maximize_f1")
    assert 0.0 <= threshold_result.threshold <= 1.0
    report = evaluate_binary_classification(y_true, y_score, threshold=threshold_result.threshold)
    assert report.metrics["accuracy"] == 1.0
    assert report.confusion_counts == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert report.per_class["ia"]["f1"] == 1.0
    assert report.per_class["non_ia"]["f1"] == 1.0


def test_ml_training_save_load_and_inference_format(tmp_path):
    _, _, splits, _ = _prepare_splits(tmp_path)
    model_dir = tmp_path / "ml"
    artifacts = fit_binary_ai_ml(
        splits["train"],
        splits["validation"],
        splits["test"],
        output_dir=model_dir,
        config=BinaryAITrainingConfig(word_min_df=1, char_min_df=1, word_max_features=1000, char_max_features=1000, cv_splits=2),
    )
    assert (model_dir / "pipeline.joblib").exists()
    assert (model_dir / "metadata.json").exists()
    pipeline, metadata = load_binary_ai_ml(model_dir)
    assert metadata["pretrained_model"] is False
    prediction = predict_binary_ai_ml("Formation sur l intelligence artificielle", model_dir=model_dir)
    assert set(prediction) == {"label", "probability_ai", "threshold", "model_name", "model_version", "pretrained", "latency_ms"}
    assert prediction["pretrained"] is False
    empty_prediction = predict_binary_ai_ml("", model_dir=model_dir)
    assert 0.0 <= empty_prediction["probability_ai"] <= 1.0
    long_prediction = predict_binary_ai_ml("IA " * 1000, model_dir=model_dir)
    assert 0.0 <= long_prediction["probability_ai"] <= 1.0
    assert artifacts.threshold == metadata["threshold"]


def test_textcnn_training_save_load_and_deterministic_inference(tmp_path):
    _, _, splits, _ = _prepare_splits(tmp_path)
    model_dir = tmp_path / "textcnn"
    artifacts = fit_binary_ai_textcnn(
        splits["train"],
        splits["validation"],
        splits["test"],
        output_dir=model_dir,
        config=BinaryAITextCNNConfig(epochs=1, batch_size=2, vocab_size=1000, max_length=32, device="cpu"),
    )
    assert (model_dir / "model.pt").exists()
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["pretrained_model"] is False
    assert metadata["pretrained_embeddings"] is False
    first = predict_binary_ai_textcnn("Formation sur l intelligence artificielle", model_dir=model_dir)
    second = predict_binary_ai_textcnn("Formation sur l intelligence artificielle", model_dir=model_dir)
    assert first["label"] == second["label"]
    assert first["probability_ai"] == second["probability_ai"]
    assert first["threshold"] == second["threshold"]
    assert first["model_name"] == second["model_name"]
    assert first["model_version"] == second["model_version"]
    assert first["pretrained"] == second["pretrained"]
    empty_prediction = predict_binary_ai_textcnn("", model_dir=model_dir)
    assert 0.0 <= empty_prediction["probability_ai"] <= 1.0
    long_prediction = predict_binary_ai_textcnn("IA " * 1000, model_dir=model_dir)
    assert 0.0 <= long_prediction["probability_ai"] <= 1.0
    assert artifacts.threshold == metadata["threshold"]


def test_common_facade_returns_expected_schema(tmp_path, monkeypatch):
    _, _, splits, _ = _prepare_splits(tmp_path)
    ml_dir = tmp_path / "ml"
    textcnn_dir = tmp_path / "textcnn"
    fit_binary_ai_ml(
        splits["train"],
        splits["validation"],
        splits["test"],
        output_dir=ml_dir,
        config=BinaryAITrainingConfig(word_min_df=1, char_min_df=1, word_max_features=1000, char_max_features=1000, cv_splits=2),
    )
    fit_binary_ai_textcnn(
        splits["train"],
        splits["validation"],
        splits["test"],
        output_dir=textcnn_dir,
        config=BinaryAITextCNNConfig(epochs=1, batch_size=2, vocab_size=1000, max_length=32, device="cpu"),
    )
    monkeypatch.setattr(binary_ai_predictor, "BINARY_AI_SETTINGS", BinaryAISettings(backend="ml_from_scratch"))
    binary_ai_predictor.DEFAULT_ML_MODEL_DIR = ml_dir
    result_ml = binary_ai_predictor.predict_binary_ai("Formation sur l intelligence artificielle", model_name="ml")
    assert set(result_ml) == {"label", "probability_ai", "threshold", "model_name", "model_version", "pretrained", "latency_ms"}
    monkeypatch.setattr(binary_ai_predictor, "BINARY_AI_SETTINGS", BinaryAISettings(backend="textcnn_from_scratch"))
    binary_ai_predictor.DEFAULT_TEXTCNN_MODEL_DIR = textcnn_dir
    result_textcnn = binary_ai_predictor.predict_binary_ai("Formation sur l intelligence artificielle", model_name="textcnn")
    assert set(result_textcnn) == {"label", "probability_ai", "threshold", "model_name", "model_version", "pretrained", "latency_ms"}


def test_textcnn_checkpoint_marks_random_initialization(tmp_path):
    _, _, splits, _ = _prepare_splits(tmp_path)
    model_dir = tmp_path / "textcnn"
    fit_binary_ai_textcnn(
        splits["train"],
        splits["validation"],
        splits["test"],
        output_dir=model_dir,
        config=BinaryAITextCNNConfig(epochs=1, batch_size=2, vocab_size=1000, max_length=32, device="cpu"),
    )
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["pretrained_model"] is False
    assert metadata["pretrained_embeddings"] is False
    assert metadata["random_initialization"] is True

