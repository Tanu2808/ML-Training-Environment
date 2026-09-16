"""Decision Tree models for regression."""

from sklearn.tree import DecisionTreeRegressor
from src.models.registry import model_registry

model_registry.register(
    name="decision_tree_regressor",
    task="regression",
    constructor=DecisionTreeRegressor,
    description="Decision Tree regressor.",
    aliases=["dt_regressor"]
)
