"""Model definition for the IMDb review sentiment classifier."""

from torch import nn


def build_model(input_features: int, hidden_units: int = 64) -> nn.Sequential:
    """Create a binary classifier for bag-of-words feature vectors."""
    return nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 1),
    )
