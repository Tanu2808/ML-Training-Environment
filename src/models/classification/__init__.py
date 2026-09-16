"""Classification models registration."""

from sklearn.linear_model import LogisticRegression

from src.models.registry import model_registry


model_registry.register(
    name="logistic_regression",
    task="classification",
    constructor=LogisticRegression,
    description="Standard Logistic Regression model.",
    aliases=["logreg", "logistic"]
)

import src.models.classification.tree_models
import src.models.classification.ensemble_models
import src.models.classification.kernel_models
import src.models.classification.neighbor_models
import src.models.classification.probabilistic_models


