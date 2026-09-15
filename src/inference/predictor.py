"""Prediction helpers."""


def predict(model, X):
    """Predict using a model if it exposes a predict method."""
    if hasattr(model, "predict"):
        return model.predict(X)
    return []
