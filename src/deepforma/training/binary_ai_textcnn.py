from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common.text import clean_text
from deepforma.evaluation.binary_classification_metrics import (
    BinaryClassificationReport,
    evaluate_binary_classification,
    optimize_binary_threshold,
    save_thresholds_json,
)


LOGGER = logging.getLogger(__name__)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


@dataclass(frozen=True, slots=True)
class BinaryAITextCNNConfig:
    seed: int = 42
    vocab_size: int = 30_000
    max_length: int = 256
    embedding_dim: int = 128
    num_filters: int = 128
    kernel_sizes: tuple[int, ...] = (3, 4, 5)
    dense_dim: int = 128
    dropout: float = 0.4
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    patience: int = 3
    grad_clip: float = 1.0
    threshold_mode: str = "maximize_f1"
    min_recall: float | None = None
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class BinaryAITextCNNArtifacts:
    model_dir: str
    model_name: str
    model_version: str
    threshold: float
    train_report: dict[str, Any]
    validation_report: dict[str, Any]
    test_report: dict[str, Any]
    training_history: list[dict[str, Any]]


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9']+")


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def tokenize(text: str) -> list[str]:
    text = clean_text(text).lower()
    return [token for token in TOKEN_RE.findall(text) if token]


def build_vocabulary(texts: Iterable[str], *, max_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for index, (token, _) in enumerate(ordered[: max(0, max_size - 2)], start=2):
        vocab[token] = index
    return vocab


def encode_text(text: str, vocab: dict[str, int], *, max_length: int) -> list[int]:
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)]
    token_ids = token_ids[:max_length]
    if len(token_ids) < max_length:
        token_ids.extend([vocab[PAD_TOKEN]] * (max_length - len(token_ids)))
    return token_ids


class BinaryAIDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: dict[str, int], *, max_length: int) -> None:
        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.labels = frame["is_ai"].astype(int).tolist()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = encode_text(self.texts[index], self.vocab, max_length=self.max_length)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(float(self.labels[index]), dtype=torch.float32)


class TextCNN(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: Iterable[int],
        dense_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        kernel_sizes = list(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size) for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(num_filters * len(kernel_sizes), dense_dim)
        self.output = nn.Linear(dense_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # (batch, seq, embed)
        x = x.transpose(1, 2)  # (batch, embed, seq)
        pooled = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pooled.append(torch.amax(conv_out, dim=2))
        x = torch.cat(pooled, dim=1)
        x = self.dropout(torch.relu(self.projection(x)))
        return self.output(x).squeeze(-1)


def _build_loader(dataset: BinaryAIDataset, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _evaluate_model(
    model: TextCNN,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for input_ids, targets in loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            probabilities = torch.sigmoid(logits).cpu().numpy().tolist()
            scores.extend(float(value) for value in probabilities)
            labels.extend(int(value) for value in targets.tolist())
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def _train_epoch(model: TextCNN, loader: DataLoader, criterion, optimizer, *, device: torch.device, grad_clip: float) -> float:
    model.train()
    losses = []
    for input_ids, targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def fit_binary_ai_textcnn(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    config: BinaryAITextCNNConfig | None = None,
) -> BinaryAITextCNNArtifacts:
    config = config or BinaryAITextCNNConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    _seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")

    vocab = build_vocabulary(train_frame["text"].fillna("").astype(str).tolist(), max_size=config.vocab_size)
    train_dataset = BinaryAIDataset(train_frame, vocab, max_length=config.max_length)
    validation_dataset = BinaryAIDataset(validation_frame, vocab, max_length=config.max_length)
    test_dataset = BinaryAIDataset(test_frame, vocab, max_length=config.max_length)
    train_loader = _build_loader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = _build_loader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = _build_loader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        num_filters=config.num_filters,
        kernel_sizes=config.kernel_sizes,
        dense_dim=config.dense_dim,
        dropout=config.dropout,
    ).to(device)

    y_train = train_frame["is_ai"].astype(int).to_numpy()
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_state: dict[str, Any] | None = None
    best_threshold = 0.5
    patience = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device=device, grad_clip=config.grad_clip)
        validation_labels, validation_scores = _evaluate_model(model, validation_loader, device=device)
        threshold_result = optimize_binary_threshold(
            validation_labels,
            validation_scores,
            mode=config.threshold_mode,
            min_recall=config.min_recall,
        )
        validation_report = evaluate_binary_classification(
            validation_labels,
            validation_scores,
            threshold=threshold_result.threshold,
            model_name="binary_ai_textcnn",
        )
        metric = validation_report.metrics.get("pr_auc")
        if metric is None:
            metric = validation_report.metrics.get("f1_ia") or 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_pr_auc": validation_report.metrics.get("pr_auc"),
                "validation_f1_ia": validation_report.metrics.get("f1_ia"),
                "threshold": threshold_result.threshold,
            }
        )
        if metric > best_metric:
            best_metric = float(metric)
            patience = 0
            best_threshold = float(threshold_result.threshold)
            best_state = {
                "model_state_dict": model.state_dict(),
                "threshold": best_threshold,
                "config": asdict(config),
                "vocab": vocab,
            }
        else:
            patience += 1
            if patience >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    train_labels, train_scores = _evaluate_model(model, train_loader, device=device)
    validation_labels, validation_scores = _evaluate_model(model, validation_loader, device=device)
    test_labels, test_scores = _evaluate_model(model, test_loader, device=device)

    threshold_result = optimize_binary_threshold(
        validation_labels,
        validation_scores,
        mode=config.threshold_mode,
        min_recall=config.min_recall,
    )
    best_threshold = float(threshold_result.threshold)

    train_report = evaluate_binary_classification(train_labels, train_scores, threshold=best_threshold, model_name="binary_ai_textcnn")
    validation_report = evaluate_binary_classification(validation_labels, validation_scores, threshold=best_threshold, model_name="binary_ai_textcnn")
    test_report = evaluate_binary_classification(test_labels, test_scores, threshold=best_threshold, model_name="binary_ai_textcnn")

    training_time_ms = (time.perf_counter() - start_time) * 1000.0
    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    model_path = output_dir / "model.pt"
    vocab_path = output_dir / "vocabulary.json"
    config_path = output_dir / "config.json"
    thresholds_path = output_dir / "thresholds.json"
    metadata_path = output_dir / "metadata.json"
    history_path = output_dir / "training_history.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold": best_threshold,
            "config": asdict(config),
            "vocab": vocab,
            "model_version": model_version,
            "training_history": history,
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "test_rows": int(len(test_frame)),
            "pretrained_model": False,
            "pretrained_embeddings": False,
            "random_initialization": True,
            "training_time_ms": training_time_ms,
        },
        model_path,
    )
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    save_thresholds_json(
        thresholds_path,
        threshold_result,
        model_name="binary_ai_textcnn",
        version=model_version,
        metric=config.threshold_mode,
    )
    metadata = {
        "model_name": "binary_ai_textcnn",
        "model_version": model_version,
        "pretrained_model": False,
        "pretrained_embeddings": False,
        "random_initialization": True,
        "seed": config.seed,
        "vocab_size": len(vocab),
        "max_length": config.max_length,
        "embedding_dim": config.embedding_dim,
        "num_filters": config.num_filters,
        "kernel_sizes": list(config.kernel_sizes),
        "dense_dim": config.dense_dim,
        "dropout": config.dropout,
        "threshold": best_threshold,
        "threshold_result": asdict(threshold_result),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "train_report": train_report.to_dict(),
        "validation_report": validation_report.to_dict(),
        "test_report": test_report.to_dict(),
        "device": str(device),
        "training_time_ms": training_time_ms,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    return BinaryAITextCNNArtifacts(
        model_dir=str(output_dir),
        model_name="binary_ai_textcnn",
        model_version=metadata["model_version"],
        threshold=best_threshold,
        train_report=train_report.to_dict(),
        validation_report=validation_report.to_dict(),
        test_report=test_report.to_dict(),
        training_history=history,
    )


def load_binary_ai_textcnn(model_dir: str | Path) -> tuple[TextCNN, dict[str, Any], dict[str, int], torch.device]:
    model_dir = Path(model_dir)
    payload = torch.load(model_dir / "model.pt", map_location="cpu")
    metadata = {}
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    history_path = model_dir / "training_history.json"
    training_history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else payload.get("training_history", [])
    payload = {**payload, **metadata, "training_history": training_history}
    config = payload["config"]
    vocab = payload["vocab"]
    device = torch.device("cpu")
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        num_filters=config["num_filters"],
        kernel_sizes=config["kernel_sizes"],
        dense_dim=config["dense_dim"],
        dropout=config["dropout"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload, vocab, device


def predict_binary_ai_textcnn(text: str, *, model_dir: str | Path) -> dict[str, Any]:
    model, payload, vocab, device = load_binary_ai_textcnn(model_dir)
    start = time.perf_counter()
    encoded = torch.tensor([encode_text(text, vocab, max_length=payload["config"]["max_length"])], dtype=torch.long)
    with torch.no_grad():
        logits = model(encoded.to(device))
        probability = float(torch.sigmoid(logits)[0].item())
    threshold = float(payload.get("threshold", 0.5))
    label = "IA" if probability >= threshold else "non-IA"
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "label": label,
        "probability_ai": probability,
        "threshold": threshold,
        "model_name": "binary_ai_textcnn",
        "model_version": payload.get("model_version", ""),
        "pretrained": False,
        "latency_ms": latency_ms,
    }

