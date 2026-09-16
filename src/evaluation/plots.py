"""Visualization utilities for evaluation metrics."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from typing import Optional, Union, Tuple, List, Any


def plot_confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    labels: Optional[List[Any]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a confusion matrix.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        List of labels to index the matrix.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        labels_list = np.unique(np.concatenate([y_true, y_pred])).tolist()
    else:
        labels_list = list(labels)
        
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    ax.set_xticks(np.arange(len(labels_list)))
    ax.set_yticks(np.arange(len(labels_list)))
    ax.set_xticklabels(labels_list, rotation=45, ha="left")
    ax.set_yticklabels(labels_list)
    
    # Text annotations
    for i in range(len(labels_list)):
        for j in range(len(labels_list)):
            ax.text(j, i, str(cm[i, j]), va="center", ha="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
            
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix", pad=20)
    fig.tight_layout()
    
    return fig, ax


def plot_roc_curve(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list]
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Receiver Operating Characteristic (ROC) curve for binary classification.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Handle passing full predict_proba output (n_samples, 2)
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
        
    if len(np.unique(y_true)) > 2:
        raise ValueError("plot_roc_curve currently supports only binary classification.")
        
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    
    return fig, ax


def plot_precision_recall_curve(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list]
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Precision-Recall curve for binary classification.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    # Handle passing full predict_proba output (n_samples, 2)
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
        
    if len(np.unique(y_true)) > 2:
        raise ValueError("plot_precision_recall_curve currently supports only binary classification.")
        
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color="blue", lw=2, label=f"AP = {ap:.2f}")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    
    return fig, ax


def plot_actual_vs_predicted(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list]
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Actual vs Predicted values for regression.
    
    Parameters
    ----------
    y_true : array-like
        True targets.
    y_pred : array-like
        Predicted targets.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="blue", edgecolors="k")
    
    # Add identity line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
    
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    
    return fig, ax


def plot_residuals(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list]
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Residuals vs Predicted values for regression.
    
    Parameters
    ----------
    y_true : array-like
        True targets.
    y_pred : array-like
        Predicted targets.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.5, color="green", edgecolors="k")
    ax.axhline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    
    return fig, ax


def plot_residual_distribution(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list]
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot histogram of residuals for regression.
    
    Parameters
    ----------
    y_true : array-like
        True targets.
    y_pred : array-like
        Predicted targets.
        
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30, alpha=0.7, color="purple", edgecolor="black")
    ax.axvline(0, color='r', linestyle='--', lw=2)
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    
    return fig, ax
