"""Dataset profiling utilities.

Provides a structured summary of a :class:`pandas.DataFrame` suitable for
quick EDA, logging, and JSON serialisation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a comprehensive profile of *df* as a plain Python dictionary.

    The returned dictionary is JSON-serialisable (all values are Python
    primitives or nested dicts/lists).

    Parameters
    ----------
    df:
        The DataFrame to profile.

    Returns
    -------
    dict
        A nested dictionary with the following top-level keys:

        - ``shape`` – ``(n_rows, n_cols)``
        - ``n_rows`` – number of rows
        - ``n_cols`` – number of columns
        - ``columns`` – list of column names
        - ``dtypes`` – ``{column: dtype_str}``
        - ``missing`` – per-column missing-value stats
        - ``unique`` – per-column unique-value counts
        - ``column_types`` – categorised column lists
        - ``duplicates`` – number of fully-duplicate rows
        - ``numeric_stats`` – basic descriptive stats for numeric columns

    Raises
    ------
    TypeError
        When *df* is not a :class:`pandas.DataFrame`.
    ValueError
        When *df* is empty (zero rows **and** zero columns).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.shape == (0, 0):
        raise ValueError("Cannot profile an empty DataFrame with no rows and no columns.")

    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols if n_cols else 0

    # ----- Missing values ---------------------------------------------------
    missing_counts: dict[str, int] = df.isna().sum().to_dict()
    missing_pct: dict[str, float] = {
        col: round(count / n_rows * 100, 4) if n_rows else 0.0
        for col, count in missing_counts.items()
    }

    # ----- Unique values ----------------------------------------------------
    unique_counts: dict[str, int] = df.nunique().to_dict()

    # ----- Column type classification ---------------------------------------
    numeric_cols: list[str] = df.select_dtypes(include="number").columns.tolist()
    categorical_cols: list[str] = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols: list[str] = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    bool_cols: list[str] = df.select_dtypes(include="bool").columns.tolist()
    other_cols: list[str] = [
        c for c in df.columns
        if c not in numeric_cols + categorical_cols + datetime_cols + bool_cols
    ]

    # ----- Duplicate rows ---------------------------------------------------
    n_duplicates = int(df.duplicated().sum())

    # ----- Numeric statistics -----------------------------------------------
    numeric_stats: dict[str, dict[str, float]] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        for col, stats in desc.items():
            numeric_stats[col] = {k: (float(v) if not np.isnan(v) else None) for k, v in stats.items()}  # type: ignore[arg-type]

    profile: dict[str, Any] = {
        "shape": [n_rows, n_cols],
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": {
            "counts": missing_counts,
            "percentages": missing_pct,
            "total_missing_cells": sum(missing_counts.values()),
            "total_cells": total_cells,
        },
        "unique": unique_counts,
        "column_types": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "boolean": bool_cols,
            "other": other_cols,
        },
        "duplicates": n_duplicates,
        "numeric_stats": numeric_stats,
    }

    logger.debug("Profiled DataFrame: %d rows × %d cols", n_rows, n_cols)
    return profile


def describe_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Lightweight alias that returns a minimal summary dict.

    Preserved for backward compatibility with any existing callers.
    For the full profile use :func:`profile_dataset`.

    Parameters
    ----------
    df:
        The DataFrame to describe.

    Returns
    -------
    dict
        ``{"rows": int, "columns": list[str], "dtypes": dict}``
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
