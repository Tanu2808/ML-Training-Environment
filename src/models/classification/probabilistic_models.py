"""Probabilistic models for classification."""

from sklearn.naive_bayes import GaussianNB
from src.models.registry import model_registry

model_registry.register(
    name="naive_bayes_classifier",
    task="classification",
    constructor=GaussianNB,
    description="Gaussian Naive Bayes classifier.",
    aliases=["gaussian_nb", "nb_classifier"]
)
