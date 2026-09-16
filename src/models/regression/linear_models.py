"""Linear models for regression."""

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from src.models.registry import model_registry

model_registry.register(
    name="ridge_regression",
    task="regression",
    constructor=Ridge,
    description="Ridge regression model.",
    aliases=["ridge"]
)

model_registry.register(
    name="lasso_regression",
    task="regression",
    constructor=Lasso,
    description="Lasso regression model.",
    aliases=["lasso"]
)

model_registry.register(
    name="elastic_net_regression",
    task="regression",
    constructor=ElasticNet,
    description="Elastic Net regression model.",
    aliases=["elastic_net"]
)
