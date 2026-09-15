"""Blending ensemble utilities."""


def blend_predictions(predictions, weights=None):
    """Return a simple weighted average if weights are supplied."""
    if weights is None:
        return predictions
    if len(predictions) != len(weights):
        raise ValueError("predictions and weights must have the same length.")
    return sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
