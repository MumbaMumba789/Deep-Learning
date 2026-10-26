"""Train and save the IMDb review sentiment classifier."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torch import nn

from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pth"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.joblib"


def train(args: argparse.Namespace) -> None:
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    print(f"Reading dataset: {args.data}", flush=True)
    frame = pd.read_csv(args.data)
    required = {"review", "sentiment"}
    if not required.issubset(frame.columns):
        raise ValueError("Dataset must contain 'review' and 'sentiment' columns.")
    frame = frame.dropna(subset=["sentiment"])
    texts = frame["review"].fillna("").astype(str)
    labels = frame["sentiment"].astype(str).str.strip().str.lower().map(
        {"negative": 0, "positive": 1}
    )
    valid = labels.notna()
    texts, labels = texts[valid], labels[valid].astype(np.int64)
    if labels.nunique() != 2:
        raise ValueError("Dataset must include both positive and negative labels.")

    vectorizer = CountVectorizer(max_features=args.max_features)
    features = vectorizer.fit_transform(texts)
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    y = labels.to_numpy()

    torch.manual_seed(args.seed)
    model = build_model(features.shape[1], args.hidden_units)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(
        f"Training on {len(train_indices):,} reviews; validating on "
        f"{len(test_indices):,}; vocabulary size {features.shape[1]:,}.",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        shuffled = rng.permutation(train_indices)
        total_loss = 0.0
        batches = 0

        for start in range(0, len(shuffled), args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            batch_x = torch.from_numpy(features[batch_indices].toarray()).float()
            batch_y = torch.from_numpy(y[batch_indices]).float()
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(1)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        accuracy = evaluate(model, features, y, test_indices, args.batch_size)
        print(
            f"Epoch {epoch}/{args.epochs} | loss {total_loss / batches:.4f} "
            f"| validation accuracy {accuracy:.2%}",
            flush=True,
        )

    save_artifacts(model, vectorizer, args.hidden_units)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved vectorizer: {VECTORIZER_PATH}")


def evaluate(
    model: nn.Module,
    features,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> float:
    model.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_x = torch.from_numpy(features[batch_indices].toarray()).float()
            predicted = (model(batch_x).squeeze(1) >= 0).long().numpy()
            correct += int((predicted == labels[batch_indices]).sum())
    return correct / len(indices)


def save_artifacts(model: nn.Module, vectorizer: CountVectorizer, hidden_units: int) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_temp = None
    vectorizer_temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, suffix=".pth", delete=False) as f:
            model_temp = Path(f.name)
        with tempfile.NamedTemporaryFile(dir=VECTORIZER_PATH.parent, suffix=".joblib", delete=False) as f:
            vectorizer_temp = Path(f.name)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_features": len(vectorizer.vocabulary_),
                "hidden_units": hidden_units,
            },
            model_temp,
        )
        joblib.dump(vectorizer, vectorizer_temp)
        os.replace(model_temp, MODEL_PATH)
        os.replace(vectorizer_temp, VECTORIZER_PATH)
    finally:
        for temp in (model_temp, vectorizer_temp):
            if temp is not None and temp.exists():
                temp.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the IMDb sentiment classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CSV with review and sentiment columns")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_features, args.hidden_units) < 1:
        parser.error("epochs, batch-size, max-features and hidden-units must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    return args


if __name__ == "__main__":
    try:
        train(parse_args())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
