"""Missing-value handling utilities.

Provides both stateless helper functions and a sklearn-compatible
``MissingValueImputer`` transformer that learns statistics from training data
and applies them to unseen data without leakage.

Supported strategies
--------------------
``"mean"``        Numeric columns only.
``"median"``      Numeric columns only.
``"mode"``        Numeric and categorical columns.
``"constant"``    Any column — fills with *fill_value*.
``"ffill"``       Forward fill (temporal/ordered data).
``"bfill"``       Backward fill (temporal/ordered data).
``"drop_rows"``   Drop rows that contain any missing value in *columns*.
``"drop_cols"``   Drop columns whose missing rate exceeds *threshold*.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

Strategy = Literal[
    "mean", "median", "mode", "constant", "ffill", "bfill",
    "drop_rows", "drop_cols",
]

_NUMERIC_ONLY_STRATEGIES = {"mean", "median"}
_STATELESS_STRATEGIES = {"ffill", "bfill", "drop_rows", "drop_cols"}
_STATEFUL_STRATEGIES = {"mean", "median", "mode", "constant"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_columns(
    df: pd.DataFrame,
    columns: list[str] | None,
    strategy: str,
) -> list[str]:
    """Return the list of target columns, validating them against *df*."""
    if columns is None:
        if strategy in _NUMERIC_ONLY_STRATEGIES:
            cols = df.select_dtypes(include="number").columns.tolist()
        else:
            cols = df.columns.tolist()
    else:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Requested columns not found in DataFrame: {missing}. "
                f"Available: {list(df.columns)}"
            )
        if strategy in _NUMERIC_ONLY_STRATEGIES:
            non_numeric = [
                c for c in columns
                if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
            ]
            if non_numeric:
                raise ValueError(
                    f"Strategy '{strategy}' requires numeric columns. "
                    f"Non-numeric columns requested: {non_numeric}"
                )
        cols = columns
    return cols


# ---------------------------------------------------------------------------
# Stateless functional API
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    strategy: Strategy = "mean",
    *,
    columns: list[str] | None = None,
    fill_value: Any = 0,
    drop_threshold: float = 1.0,
) -> pd.DataFrame:
    """Apply a missing-value strategy to *df* and return a new DataFrame.

    This function is stateless — it computes statistics from *df* itself.
    For leakage-safe ML workflows use :class:`MissingValueImputer` instead.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    strategy:
        One of ``"mean"``, ``"median"``, ``"mode"``, ``"constant"``,
        ``"ffill"``, ``"bfill"``, ``"drop_rows"``, ``"drop_cols"``.
    columns:
        Columns to operate on.  ``None`` means all eligible columns.
    fill_value:
        Value used when ``strategy="constant"``.
    drop_threshold:
        Used only with ``"drop_cols"``.  Columns whose fraction of missing
        values is **>= drop_threshold** are dropped.  Default is ``1.0``
        (only drop all-NaN columns).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with missing values handled.

    Raises
    ------
    TypeError
        When *df* is not a DataFrame.
    ValueError
        On invalid strategy, columns, or strategy/dtype mismatch.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        return df.copy()

    valid = {
        "mean", "median", "mode", "constant", "ffill", "bfill",
        "drop_rows", "drop_cols",
    }
    if strategy not in valid:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {sorted(valid)}")

    result = df.copy()

    if strategy == "drop_rows":
        cols = _resolve_columns(df, columns, strategy)
        before = len(result)
        result = result.dropna(subset=cols).reset_index(drop=True)
        logger.debug("drop_rows: removed %d rows", before - len(result))
        return result

    if strategy == "drop_cols":
        frac_missing = result.isna().mean()
        to_drop = frac_missing[frac_missing >= drop_threshold].index.tolist()
        if columns is not None:
            to_drop = [c for c in to_drop if c in columns]
        result = result.drop(columns=to_drop)
        logger.debug("drop_cols: dropped %d columns: %s", len(to_drop), to_drop)
        return result

    cols = _resolve_columns(df, columns, strategy)

    for col in cols:
        if result[col].isna().sum() == 0:
            continue
        if strategy == "mean":
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == "median":
            result[col] = result[col].fillna(result[col].median())
        elif strategy == "mode":
            mode_vals = result[col].mode()
            if len(mode_vals) > 0:
                result[col] = result[col].fillna(mode_vals.iloc[0])
        elif strategy == "constant":
            result[col] = result[col].fillna(fill_value)
        elif strategy == "ffill":
            result[col] = result[col].ffill()
        elif strategy == "bfill":
            result[col] = result[col].bfill()

    return result


# ---------------------------------------------------------------------------
# Sklearn-compatible stateful Transformer
# ---------------------------------------------------------------------------

class MissingValueImputer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible imputer that learns statistics from training data.

    Prevents leakage: fit statistics are computed **only** from the training
    split and applied unchanged to validation/test data.

    Parameters
    ----------
    strategy:
        Imputation strategy.  One of ``"mean"``, ``"median"``, ``"mode"``,
        ``"constant"``.  Forward/backward fill and drop strategies are not
        supported here because they cannot be meaningfully "fit" on training
        data and then applied to a different set.
    columns:
        Columns to impute.  ``None`` means all eligible columns.
    fill_value:
        Value for ``strategy="constant"``.

    Examples
    --------
    >>> imputer = MissingValueImputer(strategy="mean")
    >>> X_train_imp = imputer.fit_transform(X_train)
    >>> X_test_imp = imputer.transform(X_test)  # uses train statistics
    """

    def __init__(
        self,
        strategy: Literal["mean", "median", "mode", "constant"] = "mean",
        *,
        columns: list[str] | None = None,
        fill_value: Any = 0,
    ) -> None:
        self.strategy = strategy
        self.columns = columns
        self.fill_value = fill_value

    # Internal state (set during fit)
    def _validate_strategy(self) -> None:
        valid = {"mean", "median", "mode", "constant"}
        if self.strategy not in valid:
            raise ValueError(
                f"MissingValueImputer strategy must be one of {sorted(valid)}, "
                f"got '{self.strategy}'."
            )

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MissingValueImputer":
        """Learn imputation statistics from *X* (training data).

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored.  Present for sklearn API compatibility.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        self._validate_strategy()

        cols = _resolve_columns(X, self.columns, self.strategy)
        self._fitted_columns: list[str] = cols
        self._fill_map: dict[str, Any] = {}

        for col in cols:
            if self.strategy == "mean":
                self._fill_map[col] = X[col].mean()
            elif self.strategy == "median":
                self._fill_map[col] = X[col].median()
            elif self.strategy == "mode":
                mode_vals = X[col].mode()
                self._fill_map[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else np.nan
            elif self.strategy == "constant":
                self._fill_map[col] = self.fill_value

        logger.debug(
            "MissingValueImputer.fit: strategy='%s', %d columns",
            self.strategy, len(cols),
        )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Apply fitted imputation statistics to *X*.

        Parameters
        ----------
        X:
            DataFrame to transform.  May contain unseen rows.
        y:
            Ignored.
        """
        if not hasattr(self, "_fill_map"):
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")

        result = X.copy()
        for col, fill_val in self._fill_map.items():
            if col not in result.columns:
                logger.warning("Column '%s' not found during transform — skipping.", col)
                continue
            result[col] = result[col].fillna(fill_val).infer_objects(copy=False)
        return result
