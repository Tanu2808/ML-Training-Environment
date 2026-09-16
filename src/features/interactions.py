"""Controlled feature interaction utilities.

Creates new features by combining pairs of existing columns.  Feature
explosion is explicitly controlled — interactions are only generated for
the columns you specify.

Functions
---------
create_interaction_features    Pairwise interactions for specified columns.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Operation = Literal["multiply", "add", "subtract", "ratio"]
ALL_OPERATIONS: list[Operation] = ["multiply", "add", "subtract", "ratio"]

ZeroDenomBehavior = Literal["raise", "nan", "fill"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")


def _check_output(df: pd.DataFrame, name: str, overwrite: bool) -> None:
    if name in df.columns and not overwrite:
        raise ValueError(
            f"Output column '{name}' already exists. Pass overwrite=True."
        )


def _safe_ratio(
    a: pd.Series,
    b: pd.Series,
    on_zero_denom: ZeroDenomBehavior,
    fill_value: float,
) -> pd.Series:
    zero_mask = b == 0
    n = int(zero_mask.sum())  # type: ignore[redundant-cast]
    if n > 0:
        if on_zero_denom == "raise":
            raise ValueError(
                f"Ratio interaction: {n} zero denominator(s). "
                "Use on_zero_denom='nan' or 'fill'."
            )
        logger.warning("Interaction ratio: %d zero denominator(s).", n)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b

    if n > 0 and on_zero_denom == "nan":
        # numpy gives inf for finite/0; replace those positions with NaN
        result = result.where(~zero_mask, other=np.nan)
    elif n > 0 and on_zero_denom == "fill":
        result = result.where(~zero_mask, other=fill_value)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_interaction_features(
    df: pd.DataFrame,
    columns: list[str],
    operations: list[Operation] | None = None,
    *,
    on_zero_denom: ZeroDenomBehavior = "nan",
    fill_value: float = 0.0,
    overwrite: bool = False,
    name_sep: str = "__",
) -> pd.DataFrame:
    """Create pairwise interaction features for the given *columns*.

    For each pair (col_a, col_b) in ``combinations(columns, 2)`` and each
    requested operation, a new column is created with a predictable name.

    Output column naming
    --------------------
    ``multiply``  → ``{col_a}{sep}x{sep}{col_b}``
    ``add``       → ``{col_a}{sep}plus{sep}{col_b}``
    ``subtract``  → ``{col_a}{sep}minus{sep}{col_b}``
    ``ratio``     → ``{col_a}{sep}div{sep}{col_b}``

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    columns:
        Explicit list of columns to interact.  Must have at least 2 entries.
        All must be present in *df*.
    operations:
        List of operations to apply.  Defaults to ``["multiply"]``.
        Valid options: ``"multiply"``, ``"add"``, ``"subtract"``, ``"ratio"``.
    on_zero_denom:
        Behaviour for zero denominators in ``"ratio"`` interactions.
    fill_value:
        Fill value when ``on_zero_denom="fill"``.
    overwrite:
        Allow overwriting existing columns.
    name_sep:
        Separator used in output column names.  Default ``"__"``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with interaction columns appended.

    Raises
    ------
    ValueError
        If fewer than 2 columns are provided, invalid columns are requested,
        invalid operations are specified, or a name collision occurs without
        ``overwrite=True``.

    Examples
    --------
    >>> result = create_interaction_features(
    ...     df,
    ...     columns=["age", "income", "score"],
    ...     operations=["multiply", "ratio"],
    ... )
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if len(columns) < 2:
        raise ValueError(
            f"'columns' must have at least 2 entries, got {len(columns)}."
        )
    _validate_columns(df, columns)

    ops = operations if operations is not None else ["multiply"]
    unknown_ops = [o for o in ops if o not in ALL_OPERATIONS]
    if unknown_ops:
        raise ValueError(
            f"Unknown operations: {unknown_ops}. "
            f"Choose from: {ALL_OPERATIONS}"
        )

    result = df.copy()

    for col_a, col_b in combinations(columns, 2):
        for op in ops:
            if op == "multiply":
                name = f"{col_a}{name_sep}x{name_sep}{col_b}"
                _check_output(result, name, overwrite)
                result[name] = df[col_a] * df[col_b]
            elif op == "add":
                name = f"{col_a}{name_sep}plus{name_sep}{col_b}"
                _check_output(result, name, overwrite)
                result[name] = df[col_a] + df[col_b]
            elif op == "subtract":
                name = f"{col_a}{name_sep}minus{name_sep}{col_b}"
                _check_output(result, name, overwrite)
                result[name] = df[col_a] - df[col_b]
            elif op == "ratio":
                name = f"{col_a}{name_sep}div{name_sep}{col_b}"
                _check_output(result, name, overwrite)
                result[name] = _safe_ratio(
                    df[col_a], df[col_b], on_zero_denom, fill_value
                )

    n_new = len(result.columns) - len(df.columns)
    logger.debug(
        "create_interaction_features: %d new columns from %d cols × %d ops",
        n_new, len(columns), len(ops),
    )
    return result


# ---------------------------------------------------------------------------
