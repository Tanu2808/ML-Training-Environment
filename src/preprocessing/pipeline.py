"""Reusable preprocessing pipeline.

Provides:

1. :class:`PreprocessingPipeline` — the original simple class (preserved for
   backward compatibility with ``test_framework.py`` and any other callers).

2. :func:`build_preprocessing_pipeline` — factory that builds a full
   ``sklearn.pipeline.Pipeline`` + ``ColumnTransformer`` combining numeric
   and categorical preprocessing.  Learns all parameters from training data
   only, preventing leakage.

3. :class:`FullPreprocessingPipeline` — a pandas-friendly wrapper around the
   sklearn pipeline that returns DataFrames rather than numpy arrays.

Architecture
------------
The pipeline produced by ``build_preprocessing_pipeline`` follows::

    ColumnTransformer
    ├── numeric_pipeline:
    │     SimpleImputer(strategy) → StandardScaler / MinMaxScaler / RobustScaler
    └── categorical_pipeline:
          SimpleImputer(strategy="most_frequent") → OneHotEncoder / OrdinalEncoder
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder as SklearnOHE,
    OrdinalEncoder as SklearnOrdinal,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)

ScalerChoice = Literal["standard", "minmax", "robust", "none"]
CatEncoderChoice = Literal["onehot", "ordinal"]


# ---------------------------------------------------------------------------
# Backward-compatible simple pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """Simple no-op preprocessing pipeline.

    Preserved for backward compatibility.  Accepts any DataFrame via
    ``fit`` / ``transform`` / ``fit_transform`` and returns it unchanged.
    Use :class:`FullPreprocessingPipeline` or :func:`build_preprocessing_pipeline`
    for real preprocessing.
    """

    def fit(self, df: pd.DataFrame, y: Any = None) -> "PreprocessingPipeline":
        """Record column names (no-op).

        Parameters
        ----------
        df:
            Training DataFrame.
        y:
            Ignored.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
        self.columns = list(df.columns)
        return self

    def transform(self, df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Return a copy of *df* (no-op transform).

        Parameters
        ----------
        df:
            DataFrame to transform.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
        return df.copy()

    def fit_transform(self, df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit then transform (no-op).

        Parameters
        ----------
        df:
            DataFrame to fit and transform.
        """
        self.fit(df, y)
        return self.transform(df)


# ---------------------------------------------------------------------------
# Sklearn pipeline factory
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    numeric_impute_strategy: str = "median",
    categorical_impute_strategy: str = "most_frequent",
    scaler: ScalerChoice = "standard",
    cat_encoder: CatEncoderChoice = "onehot",
    handle_unknown: Literal["ignore", "error"] = "ignore",
) -> ColumnTransformer:
    """Build a leakage-safe sklearn ``ColumnTransformer`` preprocessing pipeline.

    Parameters
    ----------
    numeric_features:
        List of numeric column names.
    categorical_features:
        List of categorical column names.
    numeric_impute_strategy:
        Imputation strategy for numeric columns.  One of ``"mean"``,
        ``"median"``, ``"most_frequent"``, ``"constant"``.
    categorical_impute_strategy:
        Imputation strategy for categorical columns.
    scaler:
        Scaling method: ``"standard"``, ``"minmax"``, ``"robust"``,
        or ``"none"`` to skip scaling.
    cat_encoder:
        Categorical encoding: ``"onehot"`` or ``"ordinal"``.
    handle_unknown:
        How to handle unseen categories during transform:
        ``"ignore"`` (default) or ``"error"``.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A fitted-ready transformer.  Call ``.fit(X_train)`` then
        ``.transform(X_test)`` to prevent leakage.

    Examples
    --------
    >>> ct = build_preprocessing_pipeline(
    ...     numeric_features=["age", "income"],
    ...     categorical_features=["city", "gender"],
    ... )
    >>> ct.fit(X_train)
    >>> X_train_processed = ct.transform(X_train)
    >>> X_test_processed  = ct.transform(X_test)
    """
    # --- Numeric sub-pipeline ---
    num_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=numeric_impute_strategy)),
    ]
    if scaler != "none":
        _scaler_map: dict[str, Any] = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        if scaler not in _scaler_map:
            raise ValueError(
                f"Unknown scaler '{scaler}'. Choose from: {list(_scaler_map)} or 'none'."
            )
        num_steps.append(("scaler", _scaler_map[scaler]))

    numeric_pipeline = Pipeline(steps=num_steps)

    # --- Categorical sub-pipeline ---
    if cat_encoder == "onehot":
        cat_enc: Any = SklearnOHE(
            handle_unknown="ignore" if handle_unknown == "ignore" else "error",
            sparse_output=False,
        )
    elif cat_encoder == "ordinal":
        cat_enc = SklearnOrdinal(
            handle_unknown="use_encoded_value" if handle_unknown == "ignore" else "error",
            unknown_value=-1,
        )
    else:
        raise ValueError(
            f"Unknown cat_encoder '{cat_encoder}'. Choose from: 'onehot', 'ordinal'."
        )

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
        ("encoder", cat_enc),
    ])

    # --- Assemble ColumnTransformer ---
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError(
            "At least one of 'numeric_features' or 'categorical_features' "
            "must be non-empty."
        )

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    logger.debug(
        "build_preprocessing_pipeline: %d numeric, %d categorical features",
        len(numeric_features), len(categorical_features),
    )
    return ct


# ---------------------------------------------------------------------------
# Pandas-friendly full pipeline wrapper
# ---------------------------------------------------------------------------

class FullPreprocessingPipeline:
    """Pandas-friendly wrapper around the sklearn ``ColumnTransformer``.

    Fits on training data only and returns DataFrames (not numpy arrays)
    from ``transform``.  Column names are preserved where possible.

    Parameters
    ----------
    numeric_features:
        Numeric column names.
    categorical_features:
        Categorical column names.
    **pipeline_kwargs:
        Forwarded to :func:`build_preprocessing_pipeline`.

    Examples
    --------
    >>> pipe = FullPreprocessingPipeline(
    ...     numeric_features=["age", "salary"],
    ...     categorical_features=["city"],
    ... )
    >>> X_train_out = pipe.fit_transform(X_train)
    >>> X_test_out  = pipe.transform(X_test)
    """

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        **pipeline_kwargs: Any,
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.pipeline_kwargs = pipeline_kwargs
        self._ct: ColumnTransformer | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "FullPreprocessingPipeline":
        """Fit the transformer on *X* (training data).

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored.
        """
        self._ct = build_preprocessing_pipeline(
            self.numeric_features,
            self.categorical_features,
            **self.pipeline_kwargs,
        )
        self._ct.fit(X)
        # Capture output feature names
        try:
            self._out_columns = list(self._ct.get_feature_names_out())
        except AttributeError:
            self._out_columns = None
        logger.debug("FullPreprocessingPipeline.fit complete.")
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Transform *X* using fitted parameters.

        Parameters
        ----------
        X:
            DataFrame to transform.
        """
        if self._ct is None:
            raise RuntimeError("Call fit() before transform().")
        arr = self._ct.transform(X)
        columns = self._out_columns if self._out_columns else [
            f"feature_{i}" for i in range(arr.shape[1])
        ]
        return pd.DataFrame(arr, columns=columns, index=X.index)

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step.

        Parameters
        ----------
        X:
            Training DataFrame.
        y:
            Ignored.
        """
        self.fit(X, y)
        return self.transform(X)

    @property
    def column_transformer(self) -> ColumnTransformer | None:
        """Expose the underlying sklearn ``ColumnTransformer``."""
        return self._ct
