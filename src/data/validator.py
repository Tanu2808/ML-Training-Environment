"""Dataset validation utilities.

Functions for validating :class:`pandas.DataFrame` objects before they enter
a training or preprocessing pipeline.  All functions raise descriptive
exceptions rather than silently returning ``False``.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_dataframe(
    df: pd.DataFrame,
    *,
    min_rows: int = 1,
    min_cols: int = 1,
    allow_empty: bool = False,
) -> None:
    """Assert that *df* is a non-empty, well-formed DataFrame.

    Parameters
    ----------
    df:
        The object to validate.
    min_rows:
        Minimum acceptable number of rows.
    min_cols:
        Minimum acceptable number of columns.
    allow_empty:
        When ``True`` skip the row/column size checks (useful in testing).

    Raises
    ------
    TypeError
        When *df* is not a :class:`pandas.DataFrame`.
    ValueError
        When the DataFrame is empty or below the minimum size thresholds.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}."
        )
    if allow_empty:
        return
    if len(df) < min_rows:
        raise ValueError(
            f"DataFrame must have at least {min_rows} row(s), got {len(df)}."
        )
    if len(df.columns) < min_cols:
        raise ValueError(
            f"DataFrame must have at least {min_cols} column(s), got {len(df.columns)}."
        )


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str] | tuple[str, ...],
) -> None:
    """Assert that all *required_columns* are present in *df*.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    required_columns:
        Column names that must be present.

    Raises
    ------
    TypeError
        When *df* is not a DataFrame.
    ValueError
        When one or more required columns are missing.
    """
    validate_dataframe(df, allow_empty=True)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Available: {list(df.columns)}"
        )


def validate_target_column(
    df: pd.DataFrame,
    target_column: str,
    *,
    allow_missing_values: bool = False,
) -> None:
    """Assert that *target_column* exists and contains no missing values.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    target_column:
        Name of the target column.
    allow_missing_values:
        When ``True`` the missing-value check is skipped.

    Raises
    ------
    ValueError
        When the column is absent or contains ``NaN`` values (unless
        *allow_missing_values* is ``True``).
    """
    validate_columns(df, [target_column])

    if not allow_missing_values:
        n_missing = int(df[target_column].isna().sum())
        if n_missing > 0:
            raise ValueError(
                f"Target column '{target_column}' contains {n_missing} missing "
                f"value(s). Handle missing targets before training."
            )


def validate_no_duplicates(
    df: pd.DataFrame,
    *,
    subset: list[str] | None = None,
    raise_on_duplicates: bool = True,
) -> int:
    """Check for duplicate rows in *df*.

    Parameters
    ----------
    df:
        The DataFrame to check.
    subset:
        Optional list of column names to consider for duplicate detection.
        When ``None`` all columns are used.
    raise_on_duplicates:
        When ``True`` (default) raise :exc:`ValueError` if duplicates are
        found.  When ``False`` return the duplicate count without raising.

    Returns
    -------
    int
        The number of duplicate rows found.

    Raises
    ------
    ValueError
        When duplicates are found and *raise_on_duplicates* is ``True``.
    """
    validate_dataframe(df, allow_empty=True)
    n_dups = int(df.duplicated(subset=subset).sum())

    if n_dups > 0 and raise_on_duplicates:
        raise ValueError(
            f"Found {n_dups} duplicate row(s) in the DataFrame."
        )

    if n_dups > 0:
        logger.warning("DataFrame contains %d duplicate row(s).", n_dups)

    return n_dups


def validate_dataset(
    df: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
    target_column: str | None = None,
    min_rows: int = 1,
    allow_missing_target: bool = False,
    check_duplicates: bool = False,
) -> dict[str, object]:
    """Run a full validation suite on *df* and return a summary report.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    required_columns:
        Column names that must be present.
    target_column:
        If provided, validate the target column exists and has no missing
        values (unless *allow_missing_target* is ``True``).
    min_rows:
        Minimum acceptable row count.
    allow_missing_target:
        Skip missing-value check on the target column.
    check_duplicates:
        When ``True`` count (but do not raise on) duplicate rows.

    Returns
    -------
    dict
        A report with keys ``"valid"`` (bool), ``"n_rows"``, ``"n_cols"``,
        ``"n_duplicates"`` (when checked), and ``"warnings"`` (list of str).

    Raises
    ------
    TypeError / ValueError
        On validation failures.
    """
    validate_dataframe(df, min_rows=min_rows)

    warnings: list[str] = []
    report: dict[str, object] = {
        "valid": True,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "warnings": warnings,
    }

    if required_columns:
        validate_columns(df, required_columns)

    if target_column:
        validate_target_column(df, target_column, allow_missing_values=allow_missing_target)

    # Warn about all-NaN columns
    all_nan_cols = [col for col in df.columns if df[col].isna().all()]
    if all_nan_cols:
        msg = f"Columns with all missing values: {all_nan_cols}"
        warnings.append(msg)
        logger.warning(msg)

    if check_duplicates:
        n_dups = validate_no_duplicates(df, raise_on_duplicates=False)
        report["n_duplicates"] = n_dups

    logger.debug("Validation passed: %d rows × %d cols", len(df), len(df.columns))
    return report
