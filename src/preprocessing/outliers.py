"""Outlier detection and handling utilities.

Implements IQR-based outlier detection with explicit, user-controlled
handling strategies.  Data is **never** silently removed — the user must
explicitly call :func:`remove_outliers` or pass ``action="remove"`` to the
dispatcher.

Key functions
-------------
``detect_outliers``  Return a boolean mask of outlier positions.
``get_outlier_stats`` Return IQR bounds and counts per column.
``clip_outliers``    Clip values to IQR-derived bounds.
``remove_outliers``  Remove rows containing outliers (explicit opt-in).
``handle_outliers``  Dispatcher combining detection + chosen action.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Action = Literal["clip", "remove", "none"]
DEFAULT_IQR_MULTIPLIER = 1.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_numeric_columns(
    df: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    if columns is None:
        return df.select_dtypes(include="number").columns.tolist()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    non_numeric = [
        c for c in columns
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(
            f"Outlier handling requires numeric columns. "
            f"Non-numeric columns: {non_numeric}"
        )
    return list(columns)


def _compute_bounds(
    series: pd.Series,
    multiplier: float,
) -> tuple[float, float]:
    """Return (lower, upper) IQR bounds for *series*."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_outlier_stats(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> dict[str, dict[str, float | int]]:
    """Compute IQR-based outlier statistics for each selected column.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric columns to analyse.  ``None`` → all numeric.
    multiplier:
        IQR multiplier for the fence.  Default ``1.5`` (Tukey's rule).

    Returns
    -------
    dict
        Per-column dict with keys ``"q1"``, ``"q3"``, ``"iqr"``,
        ``"lower_bound"``, ``"upper_bound"``, ``"n_outliers"``,
        ``"pct_outliers"``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if multiplier <= 0:
        raise ValueError(f"'multiplier' must be > 0, got {multiplier!r}.")

    cols = _resolve_numeric_columns(df, columns)
    stats: dict[str, dict[str, float | int]] = {}

    for col in cols:
        series = df[col].dropna()
        lower, upper = _compute_bounds(series, multiplier)
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        stats[col] = {
            "q1": float(series.quantile(0.25)),
            "q3": float(series.quantile(0.75)),
            "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
            "lower_bound": lower,
            "upper_bound": upper,
            "n_outliers": n_out,
            "pct_outliers": round(n_out / len(df) * 100, 4) if len(df) else 0.0,
        }

    return stats


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """Return a boolean mask DataFrame indicating outlier positions.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric columns to check.  ``None`` → all numeric.
    multiplier:
        IQR fence multiplier.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with the same shape as *df[columns]*.
        ``True`` where a value is an outlier.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if multiplier <= 0:
        raise ValueError(f"'multiplier' must be > 0, got {multiplier!r}.")

    cols = _resolve_numeric_columns(df, columns)
    mask = pd.DataFrame(False, index=df.index, columns=cols)

    for col in cols:
        series = df[col]
        lower, upper = _compute_bounds(series.dropna(), multiplier)
        mask[col] = (series < lower) | (series > upper)

    return mask


def clip_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> pd.DataFrame:
    """Clip outliers to IQR-derived (or manually specified) bounds.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    columns:
        Numeric columns to clip.  ``None`` → all numeric.
    multiplier:
        IQR fence multiplier (used when bounds are not manually specified).
    lower_bound:
        Override lower clip bound for all columns.
    upper_bound:
        Override upper clip bound for all columns.

    Returns
    -------
    pd.DataFrame
        New DataFrame with outliers clipped.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    result = df.copy()
    cols = _resolve_numeric_columns(df, columns)

    for col in cols:
        if lower_bound is not None or upper_bound is not None:
            lo = lower_bound
            hi = upper_bound
        else:
            lo, hi = _compute_bounds(df[col].dropna(), multiplier)
        result[col] = result[col].clip(lower=lo, upper=hi)
        logger.debug("clip_outliers: '%s' clipped to [%s, %s]", col, lo, hi)

    return result


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """Remove rows that contain outliers in any of the selected columns.

    **This is an explicit opt-in function.**  Prefer :func:`clip_outliers`
    when you want to preserve all rows.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    columns:
        Numeric columns to check.  ``None`` → all numeric.
    multiplier:
        IQR fence multiplier.

    Returns
    -------
    pd.DataFrame
        New DataFrame with outlier rows removed and the index reset.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    mask = detect_outliers(df, columns=columns, multiplier=multiplier)
    is_outlier_row = mask.any(axis=1)
    n_removed = int(is_outlier_row.sum())

    if n_removed > 0:
        logger.warning(
            "remove_outliers: removed %d / %d rows (%.1f%%).",
            n_removed, len(df), 100 * n_removed / len(df),
        )

    return df[~is_outlier_row].reset_index(drop=True)


def handle_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    action: Action = "clip",
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """Detect outliers and apply *action*.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric columns to process.  ``None`` → all numeric.
    action:
        ``"clip"``   — clip to IQR bounds (default, no rows removed).
        ``"remove"`` — remove outlier rows (data loss — use deliberately).
        ``"none"``   — detect only, return *df* unchanged.
    multiplier:
        IQR fence multiplier.

    Returns
    -------
    pd.DataFrame
    """
    if action == "clip":
        return clip_outliers(df, columns=columns, multiplier=multiplier)
    if action == "remove":
        return remove_outliers(df, columns=columns, multiplier=multiplier)
    if action == "none":
        return df.copy()
    raise ValueError(f"Unknown action '{action}'. Choose from: 'clip', 'remove', 'none'.")
