"""Dataset splitting utilities.

Provides random, stratified, group-aware, and time-series splits for
:class:`pandas.DataFrame` objects.  All splits strictly prevent data leakage
between train and test sets.

The original ``train_test_split_frame`` function is preserved with a fixed
implementation (it previously used positional slicing and ignored
``random_state``).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split

logger = logging.getLogger(__name__)

# Type alias for the common 4-tuple return value
SplitResult = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_test_split_frame(
    df: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
    stratify: bool = False,
) -> SplitResult:
    """Split a DataFrame into train/test arrays for features and target.

    This function replaces the original positional-slice implementation which
    did not respect ``random_state``.  The API surface is preserved.

    Parameters
    ----------
    df:
        Input DataFrame containing both features and target.
    target_column:
        Name of the target column.  Must be present in *df*.
    test_size:
        Proportion of rows to include in the test set.  Must be ``(0, 1)``.
    random_state:
        Seed for the random-number generator.  Ensures reproducibility.
    stratify:
        When ``True`` the split preserves the class distribution of
        *target_column* (suitable for classification tasks).

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple[DataFrame, DataFrame, Series, Series]

    Raises
    ------
    ValueError
        When *target_column* is absent, *test_size* is invalid, or the
        DataFrame is too small.
    """
    _validate_split_inputs(df, target_column, test_size)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_arr = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arr,
    )

    logger.info(
        "Split: %d train / %d test (stratify=%s, seed=%s)",
        len(X_train), len(X_test), stratify, random_state,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def stratified_split(
    df: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> SplitResult:
    """Stratified train/test split that preserves class proportions.

    Convenience wrapper around :func:`train_test_split_frame` with
    ``stratify=True``.

    Parameters
    ----------
    df:
        Input DataFrame.
    target_column:
        Name of the categorical/class target column.
    test_size:
        Fraction of data for the test set.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split_frame(
        df,
        target_column,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )


def group_split(
    df: pd.DataFrame,
    target_column: str,
    group_column: str,
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> SplitResult:
    """Group-aware train/test split.

    Ensures that all rows sharing the same value of *group_column* end up in
    the same split.  Prevents leakage when rows are correlated within groups
    (e.g. multiple measurements for the same entity).

    Parameters
    ----------
    df:
        Input DataFrame.
    target_column:
        Name of the target column.
    group_column:
        Column whose values define the groups.  Each unique value appears in
        exactly one split.
    test_size:
        Fraction of *groups* to allocate to the test set.
    random_state:
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test

    Raises
    ------
    ValueError
        On invalid parameters.
    KeyError
        When *group_column* is not in *df*.
    """
    _validate_split_inputs(df, target_column, test_size)
    if group_column not in df.columns:
        raise KeyError(f"Group column '{group_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    groups = df[group_column]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    logger.info(
        "Group split: %d train / %d test rows (%d unique groups)",
        len(train_idx), len(test_idx), groups.nunique(),
    )
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


def time_series_split(
    df: pd.DataFrame,
    target_column: str,
    *,
    test_size: float = 0.2,
) -> SplitResult:
    """Temporal train/test split for time-series data.

    Data is **not** shuffled.  The first ``(1 - test_size)`` fraction of rows
    form the training set and the remaining rows form the test set.  This
    preserves the temporal ordering and prevents future data from leaking
    into the past.

    Parameters
    ----------
    df:
        Input DataFrame.  Must already be sorted in chronological order.
    target_column:
        Name of the target column.
    test_size:
        Fraction of rows to include in the test set.

    Returns
    -------
    X_train, X_test, y_train, y_test

    Raises
    ------
    ValueError
        On invalid parameters.
    """
    _validate_split_inputs(df, target_column, test_size)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    n_test = max(1, int(round(len(df) * test_size)))
    n_train = len(df) - n_test

    logger.info("Time-series split: %d train / %d test (no shuffle)", n_train, n_test)
    return (
        X.iloc[:n_train].copy(),
        X.iloc[n_train:].copy(),
        y.iloc[:n_train].copy(),
        y.iloc[n_train:].copy(),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_split_inputs(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
) -> None:
    """Validate common split parameters."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if len(df) < 2:
        raise ValueError(f"DataFrame must contain at least 2 rows, got {len(df)}.")
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    if not (0 < test_size < 1):
        raise ValueError(f"'test_size' must be in (0, 1), got {test_size!r}.")
