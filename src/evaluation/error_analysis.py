"""Utilities for detailed error analysis."""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict


def get_classification_errors(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Return a DataFrame of misclassified samples.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    df : DataFrame, optional
        Original features. If provided, misclassified rows are returned alongside features.
        
    Returns
    -------
    DataFrame
        A DataFrame containing 'y_true', 'y_pred', and optionally original features
        for misclassified samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
        
    mask = y_true != y_pred
    
    if df is not None:
        if len(df) != len(y_true):
            raise ValueError("df must have the same length as y_true.")
        out = df[mask].copy()
        out["y_true"] = y_true[mask]
        out["y_pred"] = y_pred[mask]
    else:
        out = pd.DataFrame({
            "y_true": y_true[mask],
            "y_pred": y_pred[mask]
        })
        
    return out


def class_error_rates(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list]
) -> pd.DataFrame:
    """Calculate error rates per class.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
        
    Returns
    -------
    DataFrame
        DataFrame with columns 'total', 'errors', 'error_rate' indexed by class label.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
        
    classes = np.unique(y_true)
    results = []
    
    for c in classes:
        mask = y_true == c
        total = int(mask.sum())
        errors = int((y_pred[mask] != c).sum())
        rate = float(errors / total) if total > 0 else 0.0
        results.append({"class": c, "total": total, "errors": errors, "error_rate": rate})
        
    df = pd.DataFrame(results).set_index("class")
    return df


def get_regression_errors(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    df: Optional[pd.DataFrame] = None,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """Return a DataFrame with regression errors (residuals, abs error, sq error).
    
    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    df : DataFrame, optional
        Original features. If provided, they are appended to the output.
    top_n : int, optional
        If provided, returns only the top N largest absolute errors.
        
    Returns
    -------
    DataFrame
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
        
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)
    sq_errors = residuals ** 2
    
    if df is not None:
        if len(df) != len(y_true):
            raise ValueError("df must have the same length as y_true.")
        out = df.copy()
    else:
        out = pd.DataFrame()
        
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["residual"] = residuals
    out["abs_error"] = abs_errors
    out["sq_error"] = sq_errors
    
    if top_n is not None:
        out = out.sort_values(by="abs_error", ascending=False).head(top_n)
        
    return out


def residual_statistics(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list]
) -> Dict[str, float]:
    """Calculate summary statistics for regression residuals.
    
    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
        
    Returns
    -------
    dict
        Dictionary with residual mean, std, min, max, etc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
        
    if len(y_true) == 0:
        return {}
        
    residuals = y_true - y_pred
    
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "25%": float(np.percentile(residuals, 25)),
        "median": float(np.median(residuals)),
        "75%": float(np.percentile(residuals, 75)),
        "max": float(np.max(residuals))
    }
