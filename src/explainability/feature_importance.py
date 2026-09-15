"""Feature importance helper placeholders."""


def explain_feature_importance(model, X):
    """Return a placeholder feature importance payload."""
    return {"model": model, "n_features": X.shape[1] if hasattr(X, "shape") else 0}
