"""Cross-validation helpers."""


def cross_validate(model, X, y, folds=3):
    """Return a placeholder validation payload."""
    return {"model": model, "n_folds": folds, "rows": len(X), "labels": len(y)}
