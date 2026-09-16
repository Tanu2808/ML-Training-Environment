"""Numerical feature engineering utilities.

Provides vectorised, leakage-safe functions for constructing new numerical
features from existing DataFrame columns.  All functions are stateless
(they do not learn parameters from the data), so fit/transform semantics are
not required.

Functions
---------
add_features            col_a + col_b
subtract_features       col_a - col_b
multiply_features       col_a * col_b
ratio_feature           col_a / col_b  (configurable zero-denom behaviour)
absolute_difference     |col_a - col_b|
percentage_difference   (col_a - col_b) / col_b * 100
aggregate_features      row-wise agg (sum/mean/min/max/std/range)
bin_numeric_feature     pd.cut / pd.qcut discretisation
log_feature             log1p of a single column
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ZeroDenomBehavior = Literal["raise", "nan", "fill"]
AggMethod = Literal["sum", "mean", "min", "max", "std", "range"]
BinStrategy = Literal["uniform", "quantile"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, *columns: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _check_output_col(df: pd.DataFrame, name: str, overwrite: bool) -> None:
    if name in df.columns and not overwrite:
        raise ValueError(
            f"Output column '{name}' already exists. "
            "Pass overwrite=True to replace it."
        )


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    on_zero_denom: ZeroDenomBehavior,
    fill_value: float,
) -> pd.Series:
    zero_mask = denominator == 0
    n_zeros = int(zero_mask.sum())  # type: ignore[redundant-cast]
    if n_zeros > 0:
        if on_zero_denom == "raise":
            raise ValueError(
                f"Division by zero: {n_zeros} row(s) have denominator == 0. "
                "Use on_zero_denom='nan' or 'fill' to handle them."
            )
        if on_zero_denom == "nan":
            logger.warning(
                "ratio_feature: %d zero denominator(s) → NaN.", n_zeros
            )
        else:
            logger.warning(
                "ratio_feature: %d zero denominator(s) → %s.", n_zeros, fill_value
            )

    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator

    if n_zeros > 0 and on_zero_denom == "nan":
        # numpy gives inf for finite/0; replace zero-denom positions with NaN
        result = result.where(~zero_mask, other=np.nan)
    elif n_zeros > 0 and on_zero_denom == "fill":
        result = result.where(~zero_mask, other=fill_value)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_features(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    *,
    output_col: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add *col_a* and *col_b*, storing the result in *output_col*.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    col_a, col_b:
        Columns to add.  Must be numeric.
    output_col:
        Name for the new column.  Defaults to ``"{col_a}_plus_{col_b}"``.
    overwrite:
        Allow overwriting an existing column.

    Returns
    -------
    pd.DataFrame
        New DataFrame with the added column appended.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    _validate_columns(df, col_a, col_b)
    name = output_col or f"{col_a}_plus_{col_b}"
    _check_output_col(df, name, overwrite)
    result = df.copy()
    result[name] = df[col_a] + df[col_b]
    return result


def subtract_features(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    *,
    output_col: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Subtract *col_b* from *col_a*, storing the result in *output_col*.

    Parameters
    ----------
    df:
        Input DataFrame.
    col_a, col_b:
        Columns to subtract.
    output_col:
        Name for the new column.  Defaults to ``"{col_a}_minus_{col_b}"``.
    overwrite:
        Allow overwriting an existing column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    _validate_columns(df, col_a, col_b)
    name = output_col or f"{col_a}_minus_{col_b}"
    _check_output_col(df, name, overwrite)
    result = df.copy()
    result[name] = df[col_a] - df[col_b]
    return result


def multiply_features(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    *,
    output_col: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Multiply *col_a* by *col_b*, storing the result in *output_col*.

    Parameters
    ----------
    df:
        Input DataFrame.
    col_a, col_b:
        Columns to multiply.
    output_col:
        Name for the new column.  Defaults to ``"{col_a}_times_{col_b}"``.
    overwrite:
        Allow overwriting an existing column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    _validate_columns(df, col_a, col_b)
    name = output_col or f"{col_a}_times_{col_b}"
    _check_output_col(df, name, overwrite)
    result = df.copy()
    result[name] = df[col_a] * df[col_b]
    return result


def ratio_feature(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    *,
    output_col: str | None = None,
    on_zero_denom: ZeroDenomBehavior = "raise",
    fill_value: float = 0.0,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Compute *numerator_col* / *denominator_col*.

    Parameters
    ----------
    df:
        Input DataFrame.
    numerator_col, denominator_col:
        Columns for the ratio.
    output_col:
        Name for the new column.
    on_zero_denom:
        Behaviour when the denominator is zero:
        ``"raise"`` (default) — raise :exc:`ValueError`.
        ``"nan"``   — replace result with ``NaN``.
        ``"fill"``  — replace result with *fill_value*.
    fill_value:
        Value used when ``on_zero_denom="fill"``.
    overwrite:
        Allow overwriting an existing column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    _validate_columns(df, numerator_col, denominator_col)
    name = output_col or f"{numerator_col}_div_{denominator_col}"
    _check_output_col(df, name, overwrite)
    result = df.copy()
    result[name] = _safe_divide(
        df[numerator_col], df[denominator_col], on_zero_denom, fill_value
    )
    return result


def absolute_difference(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    *,
    output_col: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Compute |*col_a* - *col_b*|.

    Parameters
    ----------
    df:
        Input DataFrame.
    col_a, col_b:
        Columns to subtract.
    output_col:
        Name for the new column.  Defaults to ``"abs_{col_a}_minus_{col_b}"``.
    overwrite:
        Allow overwriting an existing column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    _validate_columns(df, col_a, col_b)
    name = output_col or f"abs_{col_a}_minus_{col_b}"
    _check_output_col(df, name, overwrite)
    result = df.copy()
    result[name] = (df[col_a] - df[col_b]).abs()
    return result


def percentage_difference(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    *,
    output_col: str | None = None,
    on_zero_denom: ZeroDenomBehavior = "nan",
    fill_value: float = 0.0,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Compute ``(col_a - col_b) / col_b * 100``.

    Parameters
    ----------
    df:
        Input DataFrame.
    col_a, col_b:
        Reference columns.  *col_b* is the denominator.
    output_col:
        Name for the new column.
    on_zero_denom:
        Behaviour when *col_b* is zero.  Defaults to ``"nan"``.
    fill_value:
        Fill value used when ``on_zero_denom="fill"``.
    overwrite:
        Allow overwriting an existing column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    _validate_columns(df, col_a, col_b)
    name = output_col or f"pct_diff_{col_a}_{col_b}"
    _check_output_col(df, name, overwrite)
    result = df.copy()
    diff = df[col_a] - df[col_b]
    result[name] = _safe_divide(diff, df[col_b], on_zero_denom, fill_value) * 100.0
    return result


def aggregate_features(
    df: pd.DataFrame,
    columns: list[str],
    method: AggMethod = "sum",
    *,
    output_col: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Compute a row-wise aggregation across *columns*.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric columns to aggregate across rows.
    method:
        One of ``"sum"``, ``"mean"``, ``"min"``, ``"max"``,
        ``"std"``, ``"range"`` (max - min).
    output_col:
        Name for the new column.
    overwrite:
        Allow overwriting an existing column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if not columns:
        raise ValueError("'columns' must be a non-empty list.")
    _validate_columns(df, *columns)
    valid_methods = {"sum", "mean", "min", "max", "std", "range"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {sorted(valid_methods)}"
        )
    name = output_col or f"{'_'.join(columns)}_{method}"
    _check_output_col(df, name, overwrite)

    result = df.copy()
    sub = df[columns]
    if method == "sum":
        result[name] = sub.sum(axis=1)
    elif method == "mean":
        result[name] = sub.mean(axis=1)
    elif method == "min":
        result[name] = sub.min(axis=1)
    elif method == "max":
        result[name] = sub.max(axis=1)
    elif method == "std":
        result[name] = sub.std(axis=1)
    elif method == "range":
        result[name] = sub.max(axis=1) - sub.min(axis=1)
    return result


def bin_numeric_feature(
    df: pd.DataFrame,
    column: str,
    n_bins: int = 5,
    *,
    strategy: BinStrategy = "uniform",
    output_col: str | None = None,
    labels: list | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Discretise a numeric column into *n_bins* bins.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Column to bin.
    n_bins:
        Number of bins.  Must be >= 2.
    strategy:
        ``"uniform"`` — equal-width bins (``pd.cut``).
        ``"quantile"`` — equal-frequency bins (``pd.qcut``).
    output_col:
        Name for the binned column.  Defaults to ``"{column}_bin"``.
    labels:
        Optional list of labels for the bins.
    overwrite:
        Allow overwriting an existing column.

    Returns
    -------
    pd.DataFrame
        New DataFrame with a categorical (integer-coded) bin column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}.")
    _validate_columns(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric.")
    name = output_col or f"{column}_bin"
    _check_output_col(df, name, overwrite)

    result = df.copy()
    try:
        if strategy == "uniform":
            result[name] = pd.cut(
                df[column], bins=n_bins, labels=labels,
                include_lowest=True,
            )
        elif strategy == "quantile":
            result[name] = pd.qcut(
                df[column], q=n_bins, labels=labels,
                duplicates="drop",
            )
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose 'uniform' or 'quantile'."
            )
    except ValueError as exc:
        raise ValueError(
            f"Binning column '{column}' failed: {exc}"
        ) from exc

    return result


def log_feature(
    df: pd.DataFrame,
    column: str,
    *,
    output_col: str | None = None,
    on_invalid: Literal["raise", "nan", "clip"] = "raise",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Apply ``log1p(x)`` to *column*.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Numeric column to transform.
    output_col:
        Name for the new column.  Defaults to ``"log1p_{column}"``.
    on_invalid:
        How to handle x < 0: ``"raise"`` | ``"nan"`` | ``"clip"``.
    overwrite:
        Allow overwriting an existing column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    _validate_columns(df, column)
    name = output_col or f"log1p_{column}"
    _check_output_col(df, name, overwrite)

    result = df.copy()
    series = df[column].copy()
    invalid = series < 0

    if invalid.any():
        n = int(invalid.sum())  # type: ignore[redundant-cast]
        if on_invalid == "raise":
            raise ValueError(
                f"log_feature: {n} negative value(s) in '{column}'. "
                "Use on_invalid='nan' or 'clip'."
            )
        if on_invalid == "clip":
            series = series.clip(lower=0)
            logger.warning("log_feature: clipped %d negative value(s) in '%s'.", n, column)
        else:
            logger.warning("log_feature: %d negative → NaN in '%s'.", n, column)
            series = series.where(~invalid, other=np.nan)

    result[name] = np.log1p(series)
    return result


# ---------------------------------------------------------------------------
# Backward-compatible stub
# ---------------------------------------------------------------------------

def add_binned_features(df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    """Backward-compatible stub.  Use :func:`bin_numeric_feature` instead."""
    return df.copy()
