"""Neighbor models for classification."""

from sklearn.neighbors import KNeighborsClassifier
from src.models.registry import model_registry

model_registry.register(
    name="knn_classifier",
    task="classification",
    constructor=KNeighborsClassifier,
    description="K-Nearest Neighbors classifier.",
    aliases=["knn"]
)
