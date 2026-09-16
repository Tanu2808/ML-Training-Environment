"""Neighbor models for regression."""

from sklearn.neighbors import KNeighborsRegressor
from src.models.registry import model_registry

model_registry.register(
    name="knn_regressor",
    task="regression",
    constructor=KNeighborsRegressor,
    description="K-Nearest Neighbors regressor.",
    aliases=["knn_reg"]
)
