"""Numerical feature scaling utilities.

Wraps sklearn's StandardScaler, MinMaxScaler, and RobustScaler with a
pandas-friendly interface that preserves column names and DataFrame structure.

All scalers follow fit/transform semantics so that scaling parameters are
learned **only** from training data, preventing leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MinMaxScaler as _MinMaxScaler,
    RobustScaler as _RobustScaler,
    StandardScaler as _StandardScaler,
)

logger = logging.getLogger(__name__)

ScalerType = Literal["standard", "minmax", "robust"]

_SCALER_MAP = {
    "standard": _StandardScaler,
    "minmax": _MinMaxScaler,
    "robust": _RobustScaler,
}


# ---------------------------------------------------------------------------
# Unified DataFrame scaler
# ---------------------------------------------------------------------------

class DataFrameScaler(BaseEstimator, TransformerMixin):
    """Pandas-friendly wrapper around sklearn scalers.

    Fits on numeric columns and returns a DataFrame with the same structure
    (same column names, original non-numeric columns preserved).

    Parameters
    ----------
    scaler_type:
        One of ``"standard"``, ``"minmax"``, ``"robust"``.
    columns:
        Numeric columns to scale.  ``None`` means all numeric columns.
    with_std:
        Only for ``"standard"`` — whether to scale to unit variance.
    feature_range:
        Only for ``"minmax"`` — target range, default ``(0, 1)``.

    Examples
    --------
    >>> scaler = DataFrameScaler("standard")
    >>> X_train_sc = scaler.fit_transform(X_train)
    >>> X_test_sc  = scaler.transform(X_test)
    """

    def __init__(
        self,
        scaler_type: ScalerType = "standard",
        *,
        columns: list[str] | None = None,
        with_std: bool = True,
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.scaler_type = scaler_type
        self.columns = columns
        self.with_std = with_std
        self.feature_range = feature_range

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DataFrameScaler":
        """Compute scaling parameters from *X* (training data).

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        if self.scaler_type not in _SCALER_MAP:
            raise ValueError(
                f"Unknown scaler_type '{self.scaler_type}'. "
                f"Choose from: {list(_SCALER_MAP)}"
            )

        self._fitted_columns = self._resolve_columns(X)
        if not self._fitted_columns:
            raise ValueError("No numeric columns found to scale.")

        # Build sklearn scaler
        if self.scaler_type == "standard":
            self._scaler = _StandardScaler(with_std=self.with_std)
        elif self.scaler_type == "minmax":
            self._scaler = _MinMaxScaler(feature_range=self.feature_range)
        else:
            self._scaler = _RobustScaler()

        self._scaler.fit(X[self._fitted_columns])
        logger.debug("DataFrameScaler.fit: type='%s', %d columns",
                     self.scaler_type, len(self._fitted_columns))
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Scale *X* using fitted parameters.

        Parameters
        ----------
        X:
            DataFrame to scale.
        """
        if not hasattr(self, "_scaler"):
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")

        missing_cols = [c for c in self._fitted_columns if c not in X.columns]
        if missing_cols:
            raise ValueError(
                f"Columns missing from transform input: {missing_cols}"
            )

        result = X.copy()
        scaled = self._scaler.transform(X[self._fitted_columns])
        for i, col in enumerate(self._fitted_columns):
            result[col] = scaled[:, i]
        return result

    def _resolve_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is None:
            return X.select_dtypes(include="number").columns.tolist()
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        non_numeric = [
            c for c in self.columns
            if c in X.columns and not pd.api.types.is_numeric_dtype(X[c])
        ]
        if non_numeric:
            raise ValueError(
                f"Scaling requires numeric columns. Non-numeric: {non_numeric}"
            )
        return list(self.columns)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def make_standard_scaler(
    columns: list[str] | None = None,
    *,
    with_std: bool = True,
) -> DataFrameScaler:
    """Return a :class:`DataFrameScaler` configured with ``StandardScaler``.

    Parameters
    ----------
    columns:
        Columns to scale.  ``None`` → all numeric.
    with_std:
        Whether to scale to unit variance.
    """
    return DataFrameScaler("standard", columns=columns, with_std=with_std)


def make_minmax_scaler(
    columns: list[str] | None = None,
    *,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> DataFrameScaler:
    """Return a :class:`DataFrameScaler` configured with ``MinMaxScaler``.

    Parameters
    ----------
    columns:
        Columns to scale.  ``None`` → all numeric.
    feature_range:
        Desired range for the scaled values.
    """
    return DataFrameScaler("minmax", columns=columns, feature_range=feature_range)


def make_robust_scaler(
    columns: list[str] | None = None,
) -> DataFrameScaler:
    """Return a :class:`DataFrameScaler` configured with ``RobustScaler``.

    Parameters
    ----------
    columns:
        Columns to scale.  ``None`` → all numeric.
    """
    return DataFrameScaler("robust", columns=columns)


# ---------------------------------------------------------------------------
# Backward-compatible stub
# ---------------------------------------------------------------------------

def standardize(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Stateless standard scaling — fits and transforms *df* in one step.

    For leakage-safe ML use :class:`DataFrameScaler` instead.

    Parameters
    ----------
    df:
        DataFrame to scale.
    columns:
        Numeric columns to scale.  ``None`` → all numeric.
    """
    scaler = DataFrameScaler("standard", columns=columns)
    return scaler.fit_transform(df)
