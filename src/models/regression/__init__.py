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

