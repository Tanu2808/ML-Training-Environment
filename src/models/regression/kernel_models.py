"""Kernel models for regression."""

from sklearn.svm import SVR
from src.models.registry import model_registry

model_registry.register(
    name="svm_regressor",
    task="regression",
    constructor=SVR,
    description="Support Vector Machine regressor.",
    aliases=["svr"]
)
