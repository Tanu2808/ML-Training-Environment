import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.evaluation.evaluator import ClassificationEvaluator, RegressionEvaluator, EvaluationResult


class DummyModel:
    pass


def test_classification_evaluator():
    X, y = make_classification(n_samples=100, n_classes=2, random_state=42)
    model = LogisticRegression().fit(X, y)
    
    evaluator = ClassificationEvaluator(model)
    result = evaluator.evaluate(X, y)
    
    assert isinstance(result, EvaluationResult)
    assert result.y_prob is not None
    assert "accuracy" in result.metrics
    assert "log_loss" in result.metrics
    assert result.to_dict() == result.metrics
    
    # Check probabilities are bound correctly
    assert result.y_prob.shape[0] == 100


def test_classification_evaluator_no_proba():
    X, y = make_classification(n_samples=100, n_classes=2, random_state=42)
    # A model without predict_proba but with decision_function
    from sklearn.svm import SVC
    model = SVC().fit(X, y)
    
    evaluator = ClassificationEvaluator(model)
    result = evaluator.evaluate(X, y)
    
    assert isinstance(result, EvaluationResult)
    assert result.y_prob is not None
    # SVC has decision_function which evaluates to 1D array for binary
    assert result.y_prob.shape == (100,)
    assert "roc_auc" in result.metrics


def test_regression_evaluator():
    res = make_regression(n_samples=100, random_state=42)
    X, y = res[0], res[1]
    model = LinearRegression().fit(X, y)
    
    evaluator = RegressionEvaluator(model)
    result = evaluator.evaluate(X, y)
    
    assert isinstance(result, EvaluationResult)
    assert result.y_prob is None
    assert "mse" in result.metrics
    assert result.to_dict() == result.metrics


def test_evaluator_invalid_model():
    model = DummyModel()
    
    with pytest.raises(TypeError, match="must implement `predict`"):
        ClassificationEvaluator(model).evaluate(np.array([[0]]), np.array([0]))
        
    with pytest.raises(TypeError, match="must implement `predict`"):
        RegressionEvaluator(model).evaluate(np.array([[0]]), np.array([0]))
