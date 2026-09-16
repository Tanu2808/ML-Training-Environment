"""Categorical feature engineering utilities.

Implements leakage-safe categorical transformations.  All stateful encoders
follow sklearn-compatible fit/transform semantics: statistics are learned
**only** from training data and applied unchanged to validation/test data.

Classes
-------
FrequencyEncoder      category → frequency in training data (float)
CountEncoder          category → count in training data (int)
RareCategoryGrouper   low-frequency categories → "__RARE__"

Functions
---------
normalize_categories  string cleaning (strip, lowercase, type conversion)
frequency_encode      stateless functional alias (fits on df itself)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

RARE_LABEL = "__RARE__"
UNKNOWN_LABEL = "__UNKNOWN__"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if columns is None:
        return df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    return list(columns)


# ---------------------------------------------------------------------------
# Frequency Encoder
# ---------------------------------------------------------------------------

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categories by their frequency in the training data.

    Each category is replaced with:
        count(category) / total_non_null_count

    Parameters
    ----------
    columns:
        Categorical columns to encode.  ``None`` → all object/category cols.
    unknown_value:
        Frequency assigned to unseen categories at transform time.
        Default ``0.0``.
    handle_nan:
        How to treat NaN in the column:
        ``"encode"`` → treat NaN as its own category.
        ``"ignore"`` → leave NaN as NaN (default).

    Examples
    --------
    >>> enc = FrequencyEncoder(columns=["city"])
    >>> enc.fit(X_train)
    >>> X_train_enc = enc.transform(X_train)
    >>> X_test_enc  = enc.transform(X_test)   # unknown city → 0.0
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        unknown_value: float = 0.0,
        handle_nan: Literal["encode", "ignore"] = "ignore",
    ) -> None:
        self.columns = columns
        self.unknown_value = unknown_value
        self.handle_nan = handle_nan

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FrequencyEncoder":
        """Learn frequency maps from *X* (training data).

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        cols = _resolve_columns(X, self.columns)
        self._fitted_columns: list[str] = cols
        self._freq_maps: dict[str, dict] = {}

        for col in cols:
            series = X[col]
            if self.handle_nan == "encode":
                total = len(series)
                freq = series.value_counts(dropna=False, normalize=True)
            else:
                total = series.notna().sum()
                freq = series.value_counts(dropna=True, normalize=True)
            self._freq_maps[col] = freq.to_dict()

        logger.debug("FrequencyEncoder.fit: %d columns", len(cols))
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Apply frequency encoding using training statistics.

        Parameters
        ----------
        X:
            DataFrame to transform.
        """
        if not hasattr(self, "_freq_maps"):
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        result = X.copy()
        for col in self._fitted_columns:
            if col not in result.columns:
                logger.warning("Column '%s' missing from transform input — skipping.", col)
                continue
            result[col] = result[col].map(self._freq_maps[col]).fillna(self.unknown_value)
        return result


# ---------------------------------------------------------------------------
# Count Encoder
# ---------------------------------------------------------------------------

class CountEncoder(BaseEstimator, TransformerMixin):
    """Encode categories by their count in the training data.

    Parameters
    ----------
    columns:
        Columns to encode.  ``None`` → all object/category cols.
    unknown_value:
        Count assigned to unseen categories.  Default ``0``.

    Examples
    --------
    >>> enc = CountEncoder(columns=["product"])
    >>> enc.fit(X_train)
    >>> X_test_enc = enc.transform(X_test)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        unknown_value: int = 0,
    ) -> None:
        self.columns = columns
        self.unknown_value = unknown_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CountEncoder":
        """Learn count maps from *X*.

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        cols = _resolve_columns(X, self.columns)
        self._fitted_columns: list[str] = cols
        self._count_maps: dict[str, dict] = {}
        for col in cols:
            self._count_maps[col] = X[col].value_counts(dropna=True).to_dict()
        logger.debug("CountEncoder.fit: %d columns", len(cols))
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Apply count encoding using training statistics.

        Parameters
        ----------
        X:
            DataFrame to transform.
        """
        if not hasattr(self, "_count_maps"):
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        result = X.copy()
        for col in self._fitted_columns:
            if col not in result.columns:
                logger.warning("Column '%s' missing — skipping.", col)
                continue
            result[col] = result[col].map(self._count_maps[col]).fillna(self.unknown_value)
        return result


# ---------------------------------------------------------------------------
# Rare Category Grouper
# ---------------------------------------------------------------------------

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Replace low-frequency categories with ``"__RARE__"``.

    Parameters
    ----------
    columns:
        Columns to process.  ``None`` → all object/category cols.
    min_frequency:
        Minimum *count* for a category to be kept.  Mutually exclusive
        with *min_frequency_frac*.  Default ``None``.
    min_frequency_frac:
        Minimum *fraction* of rows for a category to be kept.
        Mutually exclusive with *min_frequency*.  Default ``0.01``.
    rare_label:
        Label used for rare categories.  Default ``"__RARE__"``.

    Examples
    --------
    >>> grouper = RareCategoryGrouper(columns=["city"], min_frequency=5)
    >>> grouper.fit(X_train)
    >>> X_test_enc = grouper.transform(X_test)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        min_frequency: int | None = None,
        min_frequency_frac: float | None = 0.01,
        rare_label: str = RARE_LABEL,
    ) -> None:
        # Only conflict if BOTH are explicitly non-None
        if min_frequency is not None and min_frequency_frac is not None:
            raise ValueError(
                "Specify only one of 'min_frequency' or 'min_frequency_frac'."
            )
        # When min_frequency is provided, clear the frac default
        if min_frequency is not None:
            min_frequency_frac = None
        self.columns = columns
        self.min_frequency = min_frequency
        self.min_frequency_frac = min_frequency_frac
        self.rare_label = rare_label

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RareCategoryGrouper":
        """Learn which categories are frequent from *X*.

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        cols = _resolve_columns(X, self.columns)
        self._fitted_columns: list[str] = cols
        self._frequent_cats: dict[str, set] = {}

        n = len(X)
        for col in cols:
            counts = X[col].value_counts(dropna=True)
            if self.min_frequency is not None:
                threshold = self.min_frequency
                keep = counts[counts >= threshold].index.tolist()
            else:
                frac = self.min_frequency_frac if self.min_frequency_frac is not None else 0.01
                threshold = max(1, int(n * frac))
                keep = counts[counts >= threshold].index.tolist()
            self._frequent_cats[col] = set(keep)

        logger.debug("RareCategoryGrouper.fit: %d columns", len(cols))
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Replace rare categories with ``rare_label``.

        Parameters
        ----------
        X:
            DataFrame to transform.
        """
        if not hasattr(self, "_frequent_cats"):
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        result = X.copy()
        for col in self._fitted_columns:
            if col not in result.columns:
                logger.warning("Column '%s' missing — skipping.", col)
                continue
            freq_set = self._frequent_cats[col]
            result[col] = result[col].apply(
                lambda v: v if (pd.isna(v) or v in freq_set) else self.rare_label
            )
        return result


# ---------------------------------------------------------------------------
# Category normalisation (stateless)
# ---------------------------------------------------------------------------

def normalize_categories(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    strip: bool = True,
    lowercase: bool = False,
    to_string: bool = True,
) -> pd.DataFrame:
    """Normalise string values in categorical columns.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    columns:
        Columns to normalise.  ``None`` → all object/category columns.
    strip:
        Strip leading/trailing whitespace.  Default ``True``.
    lowercase:
        Convert to lowercase.  Default ``False`` — opt-in to avoid
        accidental normalisation.
    to_string:
        Convert non-null values to ``str`` before normalising.

    Returns
    -------
    pd.DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    result = df.copy()
    cols = _resolve_columns(df, columns)
    for col in cols:
        s = result[col]
        if to_string:
            s = s.where(s.isna(), s.astype(str))
        if strip:
            s = s.str.strip()
        if lowercase:
            s = s.str.lower()
        result[col] = s
    return result


# ---------------------------------------------------------------------------
# Backward-compatible functional alias
# ---------------------------------------------------------------------------

def frequency_encode(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    unknown_value: float = 0.0,
) -> pd.DataFrame:
    """Stateless frequency encoding (fits on *df* itself).

    For leakage-safe ML use :class:`FrequencyEncoder` instead.

    Parameters
    ----------
    df:
        DataFrame to encode.
    columns:
        Columns to encode.  ``None`` → all object/category cols.
    unknown_value:
        Not applicable for the stateless version (included for API symmetry).
    """
    enc = FrequencyEncoder(columns=columns, unknown_value=unknown_value)
    return enc.fit(df).transform(df)
