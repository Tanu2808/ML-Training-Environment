import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.evaluation.metrics import classification_metrics, regression_metrics


def test_classification_metrics_binary():
    X, y = make_classification(n_samples=100, n_classes=2, random_state=42)
    model = LogisticRegression().fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    # Without prob
    metrics = classification_metrics(y, y_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "log_loss" not in metrics
    assert "roc_auc" not in metrics
    assert 0 <= metrics["accuracy"] <= 1.0
    
    # With prob
    metrics_prob = classification_metrics(y, y_pred, y_prob)
    assert "log_loss" in metrics_prob
    assert "roc_auc" in metrics_prob
    assert 0 <= metrics_prob["roc_auc"] <= 1.0


def test_classification_metrics_multiclass():
    X, y = make_classification(n_samples=100, n_classes=3, n_informative=3, random_state=42)
    model = LogisticRegression().fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    
    metrics = classification_metrics(y, y_pred, y_prob, average="macro")
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "log_loss" in metrics


def test_classification_metrics_invalid():
    # Empty inputs
    with pytest.raises(ValueError, match="empty input"):
        classification_metrics([], [])
        
    # Mismatched probability array
    with pytest.raises(ValueError, match="does not match"):
        classification_metrics([0, 1], [0, 1], [0.1])


def test_regression_metrics():
    res = make_regression(n_samples=100, random_state=42)
    X, y = res[0], res[1]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    metrics = regression_metrics(y, y_pred)
    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics
    
    assert metrics["mse"] >= 0
    assert metrics["mae"] >= 0


def test_regression_metrics_invalid():
    with pytest.raises(ValueError, match="empty input"):
        regression_metrics([], [])
