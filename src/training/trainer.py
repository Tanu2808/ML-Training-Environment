"""Training helper interfaces."""


def train_model(model, X_train, y_train):
    """Return the model object after a placeholder training pass."""
    if hasattr(model, "fit"):
        model.fit(X_train, y_train)
    return model
