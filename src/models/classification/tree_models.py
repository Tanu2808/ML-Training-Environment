"""Decision Tree models for classification."""

from sklearn.tree import DecisionTreeClassifier
from src.models.registry import model_registry

model_registry.register(
    name="decision_tree_classifier",
    task="classification",
    constructor=DecisionTreeClassifier,
    description="Decision Tree classifier.",
    aliases=["dt_classifier"]
)
