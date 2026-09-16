"""Ensemble models for regression."""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.models.registry import model_registry

model_registry.register(
    name="random_forest_regressor",
    task="regression",
    constructor=RandomForestRegressor,
    description="Random Forest regressor.",
    aliases=["rf_regressor"]
)

model_registry.register(
    name="gradient_boosting_regressor",
    task="regression",
    constructor=GradientBoostingRegressor,
    description="Gradient Boosting regressor.",
    aliases=["gb_regressor"]
)
