"""Predict the sentiment of one movie review."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import torch

from model import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pth"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.joblib"


def load_classifier():
    missing = [path for path in (MODEL_PATH, VECTORIZER_PATH) if not path.is_file()]
    if missing:
        names = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing trained artifacts:\n{names}\nRun `python src/train.py` first.")

    vectorizer = joblib.load(VECTORIZER_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Model file is from an older notebook run. Retrain with `python src/train.py`.")
    input_features = int(checkpoint["input_features"])
    hidden_units = int(checkpoint["hidden_units"])
    if input_features != len(vectorizer.vocabulary_):
        raise ValueError("Saved model and vectorizer do not match. Retrain with `python src/train.py`.")

    model = build_model(input_features, hidden_units)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vectorizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify an IMDb review as positive or negative.")
    parser.add_argument("review", nargs="*", help="Review text; omit to be prompted.")
    args = parser.parse_args()
    review = " ".join(args.review).strip() or input("Enter a movie review: ").strip()
    if not review:
        parser.error("review text must not be empty")

    try:
        model, vectorizer = load_classifier()
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")

    vector = vectorizer.transform([review]).toarray()
    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.from_numpy(vector).float())).item()
    sentiment = "Positive" if probability >= 0.5 else "Negative"
    print(f"Sentiment: {sentiment}")
    print(f"Positive probability: {probability:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
