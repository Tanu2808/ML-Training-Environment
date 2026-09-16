"""Kernel models for classification."""

from sklearn.svm import SVC
from src.models.registry import model_registry

model_registry.register(
    name="svm_classifier",
    task="classification",
    constructor=SVC,
    description="Support Vector Machine classifier.",
    aliases=["svm"]
)
