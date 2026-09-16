"""Ensemble models for classification."""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.models.registry import model_registry

model_registry.register(
    name="random_forest_classifier",
    task="classification",
    constructor=RandomForestClassifier,
    description="Random Forest classifier.",
    aliases=["rf_classifier", "random_forest"]
)

model_registry.register(
    name="gradient_boosting_classifier",
    task="classification",
    constructor=GradientBoostingClassifier,
    description="Gradient Boosting classifier.",
    aliases=["gb_classifier"]
)
