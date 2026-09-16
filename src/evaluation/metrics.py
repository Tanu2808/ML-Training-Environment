"""Metrics library for model evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from typing import Optional, Union, Dict


def classification_metrics(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    y_prob: Optional[Union[pd.DataFrame, np.ndarray, list]] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """Calculate standard classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    y_prob : array-like, optional
        Predicted probabilities. Shape should be (n_samples, n_classes) or (n_samples,) for binary.
        If provided, log_loss and roc_auc will be calculated.
    average : str
        Averaging strategy for precision/recall/f1 (e.g. 'macro', 'micro', 'weighted', 'binary').
        
    Returns
    -------
    dict
        Dictionary containing the computed metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on empty input.")
        
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),  # type: ignore[arg-type]
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),  # type: ignore[arg-type]
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),  # type: ignore[arg-type]
    }
    
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if len(y_prob) != len(y_true):
            raise ValueError(f"y_prob length ({len(y_prob)}) does not match y_true length ({len(y_true)})")
            
        # Compute log_loss only if values look like probabilities [0, 1]
        if np.all((y_prob >= 0) & (y_prob <= 1)):
            try:
                metrics["log_loss"] = float(log_loss(y_true, y_prob))
            except ValueError:
                pass
            
        # Compute ROC AUC
        try:
            if len(np.unique(y_true)) > 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovo", average=average))  # type: ignore[arg-type]
            else:
                # Binary classification, check shape
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_prob_auc = y_prob[:, 1]
                else:
                    y_prob_auc = y_prob
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob_auc))
        except ValueError as e:
            # E.g. only one class present in y_true
            pass
            
    return metrics


def regression_metrics(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
) -> Dict[str, float]:
    """Calculate standard regression metrics.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
        
    Returns
    -------
    dict
        Dictionary containing the computed metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on empty input.")
        
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    
    # MAPE can raise division by zero issues in older sklearn, but is generally safe now
    # We add a tiny epsilon to prevent true zero division if users have exact 0 targets
    # Actually, mean_absolute_percentage_error handles epsilon internally in recent sklearn
    metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    
    return metrics
