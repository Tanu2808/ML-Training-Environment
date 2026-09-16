"""Evaluator classes for computing metrics from fitted models."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .metrics import classification_metrics, regression_metrics


@dataclass
class EvaluationResult:
    """Structured result containing metrics and predictions."""
    metrics: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Return metrics as a dictionary for easy logging."""
        return self.metrics.copy()


class ClassificationEvaluator:
    """Evaluates classification models."""
    
    def __init__(self, model: Any):
        """
        Parameters
        ----------
        model : object
            A fitted model implementing `predict`.
            Optionally implements `predict_proba` or `decision_function`.
        """
        self.model = model

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray, list],
        average: str = "macro"
    ) -> EvaluationResult:
        """
        Generate predictions and calculate metrics.
        
        Parameters
        ----------
        X : array-like
            Input features.
        y_true : array-like
            True labels.
        average : str
            Averaging method for metrics.
            
        Returns
        -------
        EvaluationResult
        """
        y_true = np.asarray(y_true)
        
        if not hasattr(self.model, "predict"):
            raise TypeError("Model must implement `predict` method.")
            
        y_pred = self.model.predict(X)
        
        # Try to get probabilities or scores
        y_prob = None
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            y_prob = self.model.decision_function(X)
            
        metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, average=average)
        
        return EvaluationResult(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )


class RegressionEvaluator:
    """Evaluates regression models."""
    
    def __init__(self, model: Any):
        """
        Parameters
        ----------
        model : object
            A fitted model implementing `predict`.
        """
        self.model = model

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_true: Union[pd.Series, np.ndarray, list]
    ) -> EvaluationResult:
        """
        Generate predictions and calculate metrics.
        
        Parameters
        ----------
        X : array-like
            Input features.
        y_true : array-like
            True targets.
            
        Returns
        -------
        EvaluationResult
        """
        y_true = np.asarray(y_true)
        
        if not hasattr(self.model, "predict"):
            raise TypeError("Model must implement `predict` method.")
            
        y_pred = self.model.predict(X)
        
        metrics = regression_metrics(y_true, y_pred)
        
        return EvaluationResult(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=None
        )
