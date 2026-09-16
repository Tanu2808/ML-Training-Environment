"""Categorical encoding utilities.

Provides sklearn-compatible transformers for one-hot encoding and ordinal
encoding of categorical features in pandas DataFrames.

Both encoders follow fit/transform semantics so that encoding mappings are
learned **only** from training data and applied consistently to unseen data,
preventing train/test leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One-hot encoder
# ---------------------------------------------------------------------------

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Leakage-safe one-hot encoder for categorical DataFrame columns.

    Learns the category vocabulary from training data.  During transform,
    unknown categories are handled gracefully (encoded as all-zeros) instead
    of raising.

    Parameters
    ----------
    columns:
        Categorical columns to encode.  ``None`` means all ``object`` and
        ``category`` dtype columns.
    drop_first:
        When ``True`` drop the first dummy column to avoid the dummy-variable
        trap in linear models.
    handle_unknown:
        ``"ignore"`` — unknown categories produce an all-zero row (default).
        ``"error"``  — raise :exc:`ValueError` on unknown categories.
    sparse_output:
        When ``True`` the returned DataFrame uses ``pd.SparseDtype``.
        Default ``False`` (dense).

    Examples
    --------
    >>> enc = OneHotEncoder(columns=["city", "color"])
    >>> X_train_enc = enc.fit_transform(X_train)
    >>> X_test_enc  = enc.transform(X_test)   # unknown cities → all zeros
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        drop_first: bool = False,
        handle_unknown: Literal["ignore", "error"] = "ignore",
        sparse_output: bool = False,
    ) -> None:
        self.columns = columns
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OneHotEncoder":
        """Learn category vocabularies from *X*.

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored (sklearn API compatibility).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")

        cols = self._resolve_columns(X)
        self._fitted_columns: list[str] = cols
        # Map column → sorted list of categories seen during training
        self._categories: dict[str, list] = {}
        for col in cols:
            cats = sorted(X[col].dropna().unique().tolist(), key=str)
            self._categories[col] = cats

        # Pre-compute the exact dummy column names for consistent ordering
        self._dummy_columns: dict[str, list[str]] = {}
        for col, cats in self._categories.items():
            if self.drop_first and len(cats) > 1:
                effective_cats = cats[1:]
            else:
                effective_cats = cats
            self._dummy_columns[col] = [f"{col}_{cat}" for cat in effective_cats]

        logger.debug("OneHotEncoder.fit: %d columns, handle_unknown='%s'",
                     len(cols), self.handle_unknown)
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Encode *X* using the vocabulary learned during ``fit``.

        Parameters
        ----------
        X:
            DataFrame to encode.  May contain unseen categories.
        """
        if not hasattr(self, "_categories"):
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")

        # Check for unknown categories before transforming
        if self.handle_unknown == "error":
            for col in self._fitted_columns:
                if col not in X.columns:
                    raise ValueError(
                        f"Column '{col}' was present at fit time but is missing "
                        "from the transform input."
                    )
                known = set(self._categories[col])
                unseen = set(X[col].dropna().unique()) - known
                if unseen:
                    raise ValueError(
                        f"Unknown categories in column '{col}': {unseen}. "
                        "Set handle_unknown='ignore' to silently encode them as zeros."
                    )

        # Keep non-encoded columns
        other_cols = [c for c in X.columns if c not in self._fitted_columns]
        result = X[other_cols].copy()

        for col in self._fitted_columns:
            if col not in X.columns:
                # Column missing at transform — fill all dummies with zero
                for dummy_col in self._dummy_columns[col]:
                    result[dummy_col] = 0
                continue

            known_cats = self._categories[col]
            dummy_names = self._dummy_columns[col]

            # Build boolean masks for each expected category
            for dummy_col, cat in zip(
                dummy_names,
                known_cats[1:] if self.drop_first else known_cats,
            ):
                result[dummy_col] = (X[col] == cat).astype(int)

        if self.sparse_output:
            for col in result.select_dtypes(include="number").columns:
                result[col] = result[col].astype(pd.SparseDtype("int", fill_value=0))

        return result

    def get_feature_names_out(self) -> list[str]:
        """Return the output feature names after encoding."""
        if not hasattr(self, "_dummy_columns"):
            raise RuntimeError("Call fit() first.")
        names: list[str] = []
        for dummies in self._dummy_columns.values():
            names.extend(dummies)
        return names

    def _resolve_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is None:
            return X.select_dtypes(include=["object", "category"]).columns.tolist()
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        return list(self.columns)


# ---------------------------------------------------------------------------
# Ordinal encoder
# ---------------------------------------------------------------------------

class OrdinalCategoryEncoder(BaseEstimator, TransformerMixin):
    """Ordinal encoder that maps categories to integers.

    Wraps sklearn's ``OrdinalEncoder`` with a pandas-friendly interface.
    Unknown categories at transform time are replaced with ``-1`` by default.

    Parameters
    ----------
    columns:
        Columns to encode.  ``None`` means all ``object``/``category`` cols.
    categories:
        Optional explicit category ordering per column, as a dict mapping
        ``{column_name: [cat1, cat2, ...]}`` in the desired ordinal order.
        Columns not present in this dict are sorted lexicographically.
    handle_unknown:
        ``"use_encoded_value"`` (default) → unknown categories → ``unknown_value``.
        ``"error"`` → raise on unknown categories.
    unknown_value:
        Integer to use for unknown categories when
        ``handle_unknown="use_encoded_value"``.  Default ``-1``.

    Examples
    --------
    >>> enc = OrdinalCategoryEncoder(
    ...     columns=["size"],
    ...     categories={"size": ["S", "M", "L", "XL"]},
    ... )
    >>> X_train_enc = enc.fit_transform(X_train)
    >>> X_test_enc  = enc.transform(X_test)
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        *,
        categories: dict[str, list] | None = None,
        handle_unknown: Literal["use_encoded_value", "error"] = "use_encoded_value",
        unknown_value: int = -1,
    ) -> None:
        self.columns = columns
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OrdinalCategoryEncoder":
        """Fit the encoder on *X* (training data).

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")

        cols = self._resolve_columns(X)
        self._fitted_columns: list[str] = cols

        # Build category lists per column
        sk_categories: list[list] = []
        for col in cols:
            if self.categories and col in self.categories:
                sk_categories.append(list(self.categories[col]))
            else:
                sk_categories.append(sorted(X[col].dropna().unique().tolist(), key=str))

        sk_handle = (
            "use_encoded_value"
            if self.handle_unknown == "use_encoded_value"
            else "error"
        )
        self._encoder = OrdinalEncoder(
            categories=sk_categories,
            handle_unknown=sk_handle,
            unknown_value=self.unknown_value if sk_handle == "use_encoded_value" else None,
            dtype=np.int64,
        )
        self._encoder.fit(X[cols].astype(str))

        logger.debug("OrdinalCategoryEncoder.fit: %d columns", len(cols))
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Encode *X* using fitted ordinal mapping.

        Parameters
        ----------
        X:
            DataFrame to transform.
        """
        if not hasattr(self, "_encoder"):
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")

        result = X.copy()
        encoded = self._encoder.transform(X[self._fitted_columns].astype(str))
        for i, col in enumerate(self._fitted_columns):
            result[col] = encoded[:, i].astype(np.int64)
        return result

    def _resolve_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is None:
            return X.select_dtypes(include=["object", "category"]).columns.tolist()
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        return list(self.columns)


# ---------------------------------------------------------------------------
# Backward-compatible functional stub
# ---------------------------------------------------------------------------

def one_hot_encode(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    drop_first: bool = False,
) -> pd.DataFrame:
    """Functional one-hot encoding (stateless, fits on *df* itself).

    For leakage-safe ML use :class:`OneHotEncoder` instead.

    Parameters
    ----------
    df:
        DataFrame to encode.
    columns:
        Columns to encode.  ``None`` means all object/category columns.
    drop_first:
        Drop the first dummy column per categorical feature.

    Returns
    -------
    pd.DataFrame
    """
    enc = OneHotEncoder(columns=columns, drop_first=drop_first)
    return enc.fit_transform(df)
