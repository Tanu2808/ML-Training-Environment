"""Classification models registration."""

from sklearn.linear_model import LogisticRegression

from src.models.registry import ModelRegistry


ModelRegistry.register(
    name="logistic_regression",
    task="classification",
    constructor=LogisticRegression,
    description="Standard Logistic Regression model.",
    aliases=["logreg", "logistic"]
)
