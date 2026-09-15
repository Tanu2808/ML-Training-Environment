"""Model evaluation helpers."""


def evaluate_model(model, X_test, y_test):
    """Return a small evaluation payload for a trained model."""
    predictions = model.predict(X_test) if hasattr(model, "predict") else []
    return {
        "n_samples": len(X_test),
        "predictions": predictions,
    }
