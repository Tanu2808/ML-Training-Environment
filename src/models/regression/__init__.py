"""Regression models registration."""

from sklearn.linear_model import LinearRegression

from src.models.registry import model_registry


model_registry.register(
    name="linear_regression",
    task="regression",
    constructor=LinearRegression,
    description="Standard Linear Regression model.",
    aliases=["linreg"]
)

import src.models.regression.linear_models
import src.models.regression.tree_models
import src.models.regression.ensemble_models
import src.models.regression.kernel_models
import src.models.regression.neighbor_models


