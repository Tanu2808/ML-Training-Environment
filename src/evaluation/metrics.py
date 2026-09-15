"""Evaluation metrics."""


def accuracy_score(y_true, y_pred):
    """Return accuracy score for classification outputs."""
    if len(y_true) == 0:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
