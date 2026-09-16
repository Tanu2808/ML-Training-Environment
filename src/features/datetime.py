"""Datetime feature extraction utilities.

Extracts calendar and cyclical features from datetime columns.
All functions are stateless — no fitting required.

Functions
---------
extract_datetime_features    Extract calendar components into new columns.
add_cyclical_features        sin/cos cyclical encoding of periodic values.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OnError = Literal["raise", "coerce"]

# Calendar component → extractor attribute name on pd.Series.dt
_COMPONENT_MAP: dict[str, str] = {
    "year": "year",
    "month": "month",
    "day": "day",
    "day_of_week": "dayofweek",      # Monday=0, Sunday=6
    "day_of_month": "day",
    "day_of_year": "dayofyear",
    "week_of_year": "isocalendar",   # handled specially
    "quarter": "quarter",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
}

_FLAG_COMPONENTS = {
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
}

ALL_COMPONENTS = (
    list(_COMPONENT_MAP.keys()) + list(_FLAG_COMPONENTS)
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_datetime(series: pd.Series, on_error: OnError) -> pd.Series:
    """Convert *series* to datetime, respecting *on_error*."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    converted = pd.to_datetime(series, errors=on_error)  # type: ignore[arg-type]
    n_failed = converted.isna().sum() - series.isna().sum()
    if n_failed > 0:
        if on_error == "raise":
            raise ValueError(
                f"Could not parse {n_failed} value(s) as datetime in '{series.name}'."
            )
        logger.warning(
            "datetime: %d value(s) in '%s' could not be parsed → NaT.",
            n_failed, series.name,
        )
    return converted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_datetime_features(
    df: pd.DataFrame,
    column: str,
    components: list[str] | None = None,
    *,
    prefix: str | None = None,
    on_error: OnError = "coerce",
    drop_original: bool = False,
) -> pd.DataFrame:
    """Extract calendar components from a datetime column.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    column:
        The datetime column to decompose.
    components:
        List of components to extract.  ``None`` extracts a sensible default
        set (year, month, day, day_of_week, quarter, hour, is_weekend).
        Pass ``"all"`` as a list element — or the constant ``ALL_COMPONENTS``
        — for every supported component.
    prefix:
        Prefix for generated column names.  Defaults to *column*.
    on_error:
        ``"coerce"`` — unparseable values become ``NaT`` (default).
        ``"raise"``  — raise :exc:`ValueError` on bad values.
    drop_original:
        If ``True``, drop the original datetime column from the result.

    Returns
    -------
    pd.DataFrame
        New DataFrame with extracted feature columns appended.

    Supported components
    --------------------
    year, month, day, day_of_week, day_of_month, day_of_year,
    week_of_year, quarter, hour, minute, second,
    is_weekend, is_month_start, is_month_end,
    is_quarter_start, is_quarter_end.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    default_components = [
        "year", "month", "day", "day_of_week", "quarter", "hour", "is_weekend",
    ]
    comps = components if components is not None else default_components
    unknown = [c for c in comps if c not in ALL_COMPONENTS]
    if unknown:
        raise ValueError(
            f"Unknown components: {unknown}. "
            f"Supported: {sorted(ALL_COMPONENTS)}"
        )

    pfx = prefix if prefix is not None else column
    result = df.copy()
    dt = _to_datetime(result[column], on_error)

    for comp in comps:
        out_col = f"{pfx}_{comp}"
        if comp == "week_of_year":
            # isocalendar().week returns a DataFrame
            result[out_col] = dt.dt.isocalendar().week.astype("Int64")  # type: ignore
        elif comp == "day_of_month":
            result[out_col] = dt.dt.day  # type: ignore
        elif comp == "day_of_week":
            result[out_col] = dt.dt.dayofweek  # type: ignore
        elif comp == "is_weekend":
            result[out_col] = dt.dt.dayofweek.isin([5, 6]).astype("Int8")  # type: ignore
        elif comp == "is_month_start":
            result[out_col] = dt.dt.is_month_start.astype("Int8")  # type: ignore
        elif comp == "is_month_end":
            result[out_col] = dt.dt.is_month_end.astype("Int8")  # type: ignore
        elif comp == "is_quarter_start":
            result[out_col] = dt.dt.is_quarter_start.astype("Int8")  # type: ignore
        elif comp == "is_quarter_end":
            result[out_col] = dt.dt.is_quarter_end.astype("Int8")  # type: ignore
        else:
            attr = _COMPONENT_MAP[comp]
            result[out_col] = getattr(dt.dt, attr)

        logger.debug("extract_datetime_features: added '%s'", out_col)

    if drop_original:
        result = result.drop(columns=[column])

    return result


def add_cyclical_features(
    df: pd.DataFrame,
    column: str,
    period: float,
    *,
    prefix: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Encode a periodic numeric column as sin/cos pair.

    Cyclical encoding preserves the circular topology of periodic values
    (e.g., hour 23 and hour 0 are adjacent).

    The two new columns are named:
        ``{prefix}_sin`` and ``{prefix}_cos``

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    column:
        Numeric column containing the periodic values (e.g., hour 0–23).
    period:
        The full period of the variable (e.g., 24 for hours, 7 for weekdays,
        12 for months).
    prefix:
        Prefix for the output column names.  Defaults to *column*.
    overwrite:
        Allow overwriting existing columns.

    Returns
    -------
    pd.DataFrame
        New DataFrame with two appended columns: ``{prefix}_sin`` and
        ``{prefix}_cos``.

    Examples
    --------
    >>> df = add_cyclical_features(df, column="hour", period=24)
    >>> df = add_cyclical_features(df, column="month", period=12)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric for cyclical encoding.")
    if period <= 0:
        raise ValueError(f"'period' must be > 0, got {period!r}.")

    pfx = prefix if prefix is not None else column
    sin_col = f"{pfx}_sin"
    cos_col = f"{pfx}_cos"

    if sin_col in df.columns and not overwrite:
        raise ValueError(
            f"Output column '{sin_col}' already exists. Pass overwrite=True."
        )
    if cos_col in df.columns and not overwrite:
        raise ValueError(
            f"Output column '{cos_col}' already exists. Pass overwrite=True."
        )

    result = df.copy()
    angle = 2.0 * np.pi * df[column] / period
    result[sin_col] = np.sin(angle)
    result[cos_col] = np.cos(angle)
    return result


# ---------------------------------------------------------------------------
# Backward-compatible stub
# ---------------------------------------------------------------------------

def extract_datetime_components(df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    """Backward-compatible stub.  Use :func:`extract_datetime_features` instead."""
    return df.copy()
