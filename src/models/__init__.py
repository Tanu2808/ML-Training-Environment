"""Model registry and factory utilities."""

from src.models.registry import ModelRegistry
from src.models.factory import ModelFactory

# Import submodules to trigger registrations
import src.models.classification
import src.models.regression

__all__ = ["ModelRegistry", "ModelFactory"]
