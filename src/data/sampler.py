"""Dataset sampling utilities.

Provides reproducible random, fraction-based, count-based, and stratified
sampling for :class:`pandas.DataFrame` objects.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_random(
    df: pd.DataFrame,
    *,
    n: int | None = None,
    frac: float | None = None,
    random_state: int | None = None,
    replace: bool = False,
) -> pd.DataFrame:
    """Return a random sample of rows from *df*.

    Exactly one of *n* or *frac* must be provided.

    Parameters
    ----------
    df:
        Source DataFrame.
    n:
        Absolute number of rows to sample.  Must be ``>= 1``.  When
        ``replace=False`` (the default) *n* must not exceed ``len(df)``.
    frac:
        Fraction of rows to sample in the range ``(0, 1]``.
    random_state:
        Seed for the random-number generator.  Pass an integer for
        reproducible results.
    replace:
        Whether to sample with replacement.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the sampled rows (copy).

    Raises
    ------
    TypeError
        When *df* is not a :class:`pandas.DataFrame`.
    ValueError
        When neither or both of *n* / *frac* are provided, or when the
        requested sample size is out of range.
    """
    _validate_df(df, min_rows=1)

    if n is not None and frac is not None:
        raise ValueError("Provide either 'n' or 'frac', not both.")
    if n is None and frac is None:
        raise ValueError("Provide either 'n' (count) or 'frac' (fraction).")

    if frac is not None:
        if not (0 < frac <= 1):
            raise ValueError(f"'frac' must be in the range (0, 1], got {frac!r}.")
        effective_n = max(1, int(round(len(df) * frac)))
    else:
        effective_n = int(n)  # type: ignore[arg-type]
        if effective_n < 1:
            raise ValueError(f"'n' must be >= 1, got {n!r}.")
        if not replace and effective_n > len(df):
            raise ValueError(
                f"'n' ({effective_n}) exceeds DataFrame length ({len(df)}) "
                "when replace=False."
            )

    result = df.sample(n=effective_n, random_state=random_state, replace=replace).copy()
    logger.debug("Sampled %d rows from %d (frac=%.4f)", len(result), len(df), len(result) / len(df))
    return result


def sample_stratified(
    df: pd.DataFrame,
    stratify_col: str,
    *,
    n: int | None = None,
    frac: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Return a stratified sample that preserves class proportions.

    Rows are sampled from each stratum in *stratify_col* in proportion to
    that stratum's relative frequency in *df*.

    Parameters
    ----------
    df:
        Source DataFrame.
    stratify_col:
        Name of the column to stratify by.  Must be present in *df*.
    n:
        Total number of rows to sample (distributed proportionally).
    frac:
        Fraction of rows to sample from each stratum.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with stratified rows (shuffled, copy).

    Raises
    ------
    ValueError
        On invalid parameters or if a stratum would have 0 samples.
    KeyError
        When *stratify_col* is not in *df*.
    """
    _validate_df(df, min_rows=2)

    if stratify_col not in df.columns:
        raise KeyError(f"Column '{stratify_col}' not found in DataFrame.")

    if n is not None and frac is not None:
        raise ValueError("Provide either 'n' or 'frac', not both.")
    if n is None and frac is None:
        raise ValueError("Provide either 'n' (count) or 'frac' (fraction).")

    if frac is not None:
        if not (0 < frac <= 1):
            raise ValueError(f"'frac' must be in (0, 1], got {frac!r}.")

    parts: list[pd.DataFrame] = []
    value_counts = df[stratify_col].value_counts(normalize=False)

    for stratum_val, stratum_count in value_counts.items():
        stratum_df = df[df[stratify_col] == stratum_val]

        if frac is not None:
            stratum_n = max(1, int(round(stratum_count * frac)))
        else:
            # Proportional allocation
            proportion = stratum_count / len(df)
            stratum_n = max(1, int(round(n * proportion)))  # type: ignore[operator]

        stratum_n = min(stratum_n, len(stratum_df))
        parts.append(stratum_df.sample(n=stratum_n, random_state=random_state))

    result = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    logger.debug(
        "Stratified sample: %d rows from %d (%d strata)",
        len(result), len(df), len(value_counts),
    )
    return result


def sample_rows(df: pd.DataFrame, sample_size: int | None = None) -> pd.DataFrame:
    """Backward-compatible helper: sample *sample_size* rows or return full frame.

    This function is preserved for callers that relied on the original API.
    For new code, prefer :func:`sample_random`.

    Parameters
    ----------
    df:
        Source DataFrame.
    sample_size:
        Number of rows to sample.  When ``None`` the full DataFrame is
        returned unchanged.

    Returns
    -------
    pd.DataFrame
    """
    if sample_size is None:
        return df
    _validate_df(df, min_rows=1)
    effective = min(sample_size, len(df))
    if effective < 1:
        raise ValueError(f"'sample_size' must be >= 1, got {sample_size!r}.")
    return df.sample(n=effective).copy()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_df(df: pd.DataFrame, min_rows: int = 1) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if len(df) < min_rows:
        raise ValueError(
            f"DataFrame must have at least {min_rows} row(s), got {len(df)}."
        )
