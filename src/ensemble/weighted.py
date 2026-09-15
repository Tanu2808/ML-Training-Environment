"""Weighted ensemble utilities."""


def weighted_average(values, weights):
    """Compute a weighted average over numeric values."""
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length.")
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)
