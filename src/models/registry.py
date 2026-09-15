"""Model registry utilities."""

MODEL_REGISTRY = {}


def register_model(name: str, model):
    """Register a model by name."""
    MODEL_REGISTRY[name] = model
    return model


def get_model(name: str):
    """Retrieve a registered model."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered.")
    return MODEL_REGISTRY[name]
