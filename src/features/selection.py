"""Feature selection utilities.

Provides leakage-safe, sklearn-compatible feature selectors.  All selectors
follow fit/transform semantics: selection criteria are computed **only** from
training data.

Classes
-------
VarianceSelector        Remove low-variance features.
CorrelationSelector     Remove highly-correlated features.
MutualInfoSelector      Select top-k features by mutual information.
ModelBasedSelector      Select features by model feature importance.

Functions
---------
select_features         Simple column subsetting (stateless).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

Task = Literal["classification", "regression"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_fitted(obj: object, attr: str = "_selected_columns") -> None:
    if not hasattr(obj, attr):
        raise RuntimeError("Call fit() before transform().")


def _numeric_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")
        return [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    return df.select_dtypes(include="number").columns.tolist()


# ---------------------------------------------------------------------------
# Variance selector
# ---------------------------------------------------------------------------

class VarianceSelector(BaseEstimator, TransformerMixin):
    """Remove numeric features whose variance falls below *threshold*.

    Parameters
    ----------
    threshold:
        Minimum variance a feature must have to be retained.  Default ``0.0``
        removes only constant (zero-variance) features.
    columns:
        Columns to consider.  ``None`` → all numeric columns.

    Examples
    --------
    >>> sel = VarianceSelector(threshold=0.01)
    >>> X_train_sel = sel.fit_transform(X_train)
    >>> X_test_sel  = sel.transform(X_test)
    """

    def __init__(
        self,
        threshold: float = 0.0,
        *,
        columns: list[str] | None = None,
    ) -> None:
        self.threshold = threshold
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "VarianceSelector":
        """Compute variances from *X* (training data).

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        cols = _numeric_columns(X, self.columns)
        variances = X[cols].var()
        self._selected_columns: list[str] = variances[variances > self.threshold].index.tolist()
        self._all_numeric_cols = cols
        logger.debug(
            "VarianceSelector.fit: %d / %d features retained",
            len(self._selected_columns), len(cols),
        )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Return *X* with low-variance columns removed.

        Parameters
        ----------
        X:
            DataFrame to transform.
        """
        _assert_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        non_numeric = [c for c in X.columns if c not in self._all_numeric_cols]
        return X[non_numeric + self._selected_columns]

    @property
    def selected_features(self) -> list[str]:
        """List of features retained after fitting."""
        _assert_fitted(self)
        return self._selected_columns


# ---------------------------------------------------------------------------
# Correlation selector
# ---------------------------------------------------------------------------

class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Remove features that are highly correlated with another feature.

    When two features have correlation >= *threshold*, the one appearing
    later in the column order is dropped.

    Parameters
    ----------
    threshold:
        Absolute correlation threshold.  Default ``0.95``.
    columns:
        Columns to check.  ``None`` → all numeric columns.
    method:
        Correlation method: ``"pearson"``, ``"spearman"``, ``"kendall"``.

    Examples
    --------
    >>> sel = CorrelationSelector(threshold=0.9)
    >>> X_train_sel = sel.fit_transform(X_train)
    >>> X_test_sel  = sel.transform(X_test)
    """

    def __init__(
        self,
        threshold: float = 0.95,
        *,
        columns: list[str] | None = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
    ) -> None:
        self.threshold = threshold
        self.columns = columns
        self.method = method

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CorrelationSelector":
        """Identify correlated features from *X* (training data).

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        cols = _numeric_columns(X, self.columns)
        if len(cols) < 2:
            self._selected_columns = cols
            self._all_numeric_cols = cols
            return self

        corr = X[cols].corr(method=self.method).abs()
        # Upper triangle mask
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] >= self.threshold)]
        self._selected_columns = [c for c in cols if c not in to_drop]
        self._all_numeric_cols = cols
        logger.debug(
            "CorrelationSelector.fit: dropped %d feature(s): %s",
            len(to_drop), to_drop,
        )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Return *X* with correlated columns removed.

        Parameters
        ----------
        X:
            DataFrame to transform.
        """
        _assert_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        non_numeric = [c for c in X.columns if c not in self._all_numeric_cols]
        return X[non_numeric + self._selected_columns]

    @property
    def selected_features(self) -> list[str]:
        """Features retained after fitting."""
        _assert_fitted(self)
        return self._selected_columns

    @property
    def dropped_features(self) -> list[str]:
        """Features removed as highly correlated."""
        _assert_fitted(self)
        return [c for c in self._all_numeric_cols if c not in self._selected_columns]


# ---------------------------------------------------------------------------
# Mutual information selector
# ---------------------------------------------------------------------------

class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """Select top-*k* features by mutual information score.

    Parameters
    ----------
    k:
        Number of features to select.  Default ``10``.
    task:
        ``"classification"`` or ``"regression"``.
    columns:
        Columns to consider.  ``None`` → all numeric columns.
    random_state:
        Random seed for reproducibility.

    Examples
    --------
    >>> sel = MutualInfoSelector(k=5, task="regression")
    >>> X_train_sel = sel.fit_transform(X_train, y_train)
    >>> X_test_sel  = sel.transform(X_test)
    """

    def __init__(
        self,
        k: int = 10,
        *,
        task: Task = "regression",
        columns: list[str] | None = None,
        random_state: int = 0,
    ) -> None:
        self.k = k
        self.task = task
        self.columns = columns
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MutualInfoSelector":
        """Compute mutual information scores from *X* and *y*.

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Target series (required for mutual information).
        """
        if y is None:
            raise ValueError("MutualInfoSelector.fit() requires a target 'y'.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")

        try:
            from sklearn.feature_selection import (
                mutual_info_classif,
                mutual_info_regression,
            )
        except ImportError as exc:
            raise ImportError("MutualInfoSelector requires scikit-learn.") from exc

        cols = _numeric_columns(X, self.columns)
        if self.task == "classification":
            scores = mutual_info_classif(
                X[cols].fillna(0), y,
                random_state=self.random_state,
            )
        else:
            scores = mutual_info_regression(
                X[cols].fillna(0), y,
                random_state=self.random_state,
            )

        k = min(self.k, len(cols))
        top_idx = np.argsort(scores)[::-1][:k]
        self._selected_columns = [cols[i] for i in top_idx]
        self._all_numeric_cols = cols
        self._scores = dict(zip(cols, scores))
        logger.debug(
            "MutualInfoSelector.fit: selected %d / %d features",
            len(self._selected_columns), len(cols),
        )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Return *X* with only the top-k features retained.

        Parameters
        ----------
        X:
            DataFrame to transform.
        """
        _assert_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        non_numeric = [c for c in X.columns if c not in self._all_numeric_cols]
        return X[non_numeric + self._selected_columns]

    @property
    def scores(self) -> dict[str, float]:
        """Mutual information scores per feature."""
        _assert_fitted(self, "_scores")
        return self._scores


# ---------------------------------------------------------------------------
# Model-based selector
# ---------------------------------------------------------------------------

class ModelBasedSelector(BaseEstimator, TransformerMixin):
    """Select features by model feature importance.

    Parameters
    ----------
    model:
        A fitted or unfitted sklearn estimator with a ``feature_importances_``
        attribute (e.g., ``RandomForestClassifier``).
    k:
        Number of top features to retain.  Default ``10``.
    columns:
        Columns to consider.  ``None`` → all numeric columns.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> sel = ModelBasedSelector(RandomForestClassifier(n_estimators=50), k=5)
    >>> X_train_sel = sel.fit_transform(X_train, y_train)
    >>> X_test_sel  = sel.transform(X_test)
    """

    def __init__(
        self,
        model: Any,
        k: int = 10,
        *,
        columns: list[str] | None = None,
    ) -> None:
        self.model = model
        self.k = k
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ModelBasedSelector":
        """Fit the model and identify important features.

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Target (required for supervised models).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        cols = _numeric_columns(X, self.columns)
        self.model.fit(X[cols].fillna(0), y)
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError(
                "The provided model does not expose 'feature_importances_'. "
                "Use a tree-based model such as RandomForestClassifier."
            )
        importances = self.model.feature_importances_
        k = min(self.k, len(cols))
        top_idx = np.argsort(importances)[::-1][:k]
        self._selected_columns = [cols[i] for i in top_idx]
        self._all_numeric_cols = cols
        self._importances = dict(zip(cols, importances))
        logger.debug(
            "ModelBasedSelector.fit: selected %d / %d features",
            len(self._selected_columns), len(cols),
        )
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Return *X* with only the selected features.

        Parameters
        ----------
        X:
            DataFrame to transform.
        """
        _assert_fitted(self)
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        non_numeric = [c for c in X.columns if c not in self._all_numeric_cols]
        return X[non_numeric + self._selected_columns]

    @property
    def importances(self) -> dict[str, float]:
        """Feature importances from the fitted model."""
        _assert_fitted(self, "_importances")
        return self._importances


# ---------------------------------------------------------------------------
# Backward-compatible stateless function
# ---------------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Return *df* with only the requested *columns*.

    Backward-compatible stateless subsetter.  For leakage-safe selection use
    :class:`VarianceSelector`, :class:`CorrelationSelector`,
    :class:`MutualInfoSelector`, or :class:`ModelBasedSelector`.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Columns to keep.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    return df[list(columns)].copy()
