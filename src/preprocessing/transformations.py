"""Numerical transformation utilities.

Implements common mathematical transformations for numeric features.
All functions are stateless pure transformations — they require no fitting.
Invalid values are detected and result in clear errors or configurable
fill behaviour rather than silent NaN/inf generation.

Supported transforms
--------------------
``log1p``    log(1 + x)              requires x >= 0 (x > -1 strictly)
``sqrt``     sqrt(x)                 requires x >= 0
``power``    x ** exponent           configurable
``reciprocal``  1 / x               requires x != 0
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OnInvalid = Literal["raise", "nan", "clip"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_numeric_columns(
    df: pd.DataFrame,
    columns: list[str] | None,
) -> list[str]:
    """Return target numeric columns, validating against *df*."""
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
            f"Transformation requires numeric columns. "
            f"Non-numeric columns: {non_numeric}"
        )
    return list(columns)


def _handle_invalid(
    series: pd.Series,
    mask: pd.Series,
    on_invalid: OnInvalid,
    transform_name: str,
    clip_value: float | None = None,
) -> None:
    """Raise, warn, or silently continue based on *on_invalid*."""
    n_invalid = int(mask.sum())
    if n_invalid == 0:
        return
    if on_invalid == "raise":
        raise ValueError(
            f"'{transform_name}' received {n_invalid} invalid value(s) in "
            f"column '{series.name}'. Use on_invalid='nan' or 'clip' to "
            "handle them."
        )
    if on_invalid == "nan":
        logger.warning(
            "'%s': %d invalid value(s) in '%s' replaced with NaN.",
            transform_name, n_invalid, series.name,
        )
    if on_invalid == "clip":
        logger.warning(
            "'%s': %d invalid value(s) in '%s' clipped to %s.",
            transform_name, n_invalid, series.name, clip_value,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_log1p(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    on_invalid: OnInvalid = "raise",
) -> pd.DataFrame:
    """Apply ``log(1 + x)`` to selected numeric columns.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    columns:
        Columns to transform.  ``None`` → all numeric.
    on_invalid:
        How to handle values where ``x < 0`` (invalid for ``log1p``):
        ``"raise"`` (default) — raise :exc:`ValueError`.
        ``"nan"``  — replace invalid values with ``NaN``.
        ``"clip"`` — clip values to ``0`` before applying.

    Returns
    -------
    pd.DataFrame
        New DataFrame with transformed columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    result = df.copy()
    cols = _resolve_numeric_columns(df, columns)

    for col in cols:
        invalid_mask = result[col] < 0
        _handle_invalid(result[col], invalid_mask, on_invalid, "log1p", clip_value=0)
        if on_invalid == "clip":
            result[col] = result[col].clip(lower=0)
        elif on_invalid == "nan":
            result.loc[invalid_mask, col] = np.nan
        result[col] = np.log1p(result[col])

    return result


def apply_sqrt(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    on_invalid: OnInvalid = "raise",
) -> pd.DataFrame:
    """Apply ``sqrt(x)`` to selected numeric columns.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    columns:
        Columns to transform.  ``None`` → all numeric.
    on_invalid:
        How to handle negative values:
        ``"raise"`` — raise :exc:`ValueError`.
        ``"nan"``  — replace with ``NaN``.
        ``"clip"`` — clip to ``0`` before applying.

    Returns
    -------
    pd.DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    result = df.copy()
    cols = _resolve_numeric_columns(df, columns)

    for col in cols:
        invalid_mask = result[col] < 0
        _handle_invalid(result[col], invalid_mask, on_invalid, "sqrt", clip_value=0)
        if on_invalid == "clip":
            result[col] = result[col].clip(lower=0)
        elif on_invalid == "nan":
            result.loc[invalid_mask, col] = np.nan
        result[col] = np.sqrt(result[col])

    return result


def apply_power(
    df: pd.DataFrame,
    exponent: float = 2.0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Raise selected numeric columns to *exponent*.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    exponent:
        Power to raise values to.  Default ``2.0`` (squaring).
    columns:
        Columns to transform.  ``None`` → all numeric.

    Returns
    -------
    pd.DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    result = df.copy()
    cols = _resolve_numeric_columns(df, columns)
    for col in cols:
        result[col] = np.power(result[col].astype(float), exponent)
    return result


def apply_reciprocal(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    on_invalid: OnInvalid = "raise",
) -> pd.DataFrame:
    """Apply ``1 / x`` to selected numeric columns.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    columns:
        Columns to transform.  ``None`` → all numeric.
    on_invalid:
        How to handle zero values (undefined for reciprocal):
        ``"raise"`` — raise :exc:`ValueError`.
        ``"nan"``  — replace with ``NaN``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        When *on_invalid* is ``"raise"`` and zero values are present.
        Note: ``"clip"`` is not meaningful for reciprocal and treated as
        ``"nan"``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    result = df.copy()
    cols = _resolve_numeric_columns(df, columns)

    for col in cols:
        zero_mask = result[col] == 0
        if on_invalid == "clip":
            logger.warning(
                "apply_reciprocal: 'clip' is not meaningful for reciprocal; "
                "zeros in '%s' will be replaced with NaN.", col
            )
            on_invalid_effective: OnInvalid = "nan"
        else:
            on_invalid_effective = on_invalid

        _handle_invalid(result[col], zero_mask, on_invalid_effective, "reciprocal")
        if on_invalid_effective == "nan":
            result.loc[zero_mask, col] = np.nan
        result[col] = 1.0 / result[col]

    return result


# ---------------------------------------------------------------------------
# Backward-compatible stub
# ---------------------------------------------------------------------------

def log_transform(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Apply ``log1p`` to numeric columns.

    Backward-compatible alias for :func:`apply_log1p` with
    ``on_invalid='nan'`` so the old no-op-style call won't raise.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to transform.
    """
    return apply_log1p(df, columns=columns, on_invalid="nan")
