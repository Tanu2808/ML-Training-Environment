"""Dataset loading utilities.

Supports CSV, JSON, and optionally Parquet (requires ``pyarrow`` or
``fastparquet``).  All loaders validate path existence and file extension
before attempting to read, and surface actionable error messages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CSV_EXTENSIONS = {".csv"}
_PARQUET_EXTENSIONS = {".parquet", ".pq"}
_JSON_EXTENSIONS = {".json", ".jsonl", ".ndjson"}


def _resolve_path(path: str | Path) -> Path:
    """Resolve and validate that *path* exists and is a file."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a regular file: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_csv(
    path: str | Path,
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    dtype: dict[str, Any] | None = None,
    usecols: list[str] | None = None,
    nrows: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a CSV file into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path:
        Path to the CSV file.  Accepts both ``str`` and :class:`pathlib.Path`.
    sep:
        Column delimiter.  Defaults to ``","``; use ``"\\t"`` for TSV files.
    encoding:
        File encoding.  Defaults to ``"utf-8"``.
    dtype:
        Optional dict mapping column names to dtypes, forwarded to
        :func:`pandas.read_csv`.
    usecols:
        Optional list of column names to read.  Avoids loading unused columns.
    nrows:
        Optional maximum number of rows to read.
    **kwargs:
        Any additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    ValueError
        When the file extension is not recognised as CSV/TSV or the file is
        empty.
    """
    resolved = _resolve_path(path)
    if resolved.suffix.lower() not in _CSV_EXTENSIONS and sep == ",":
        # Warn but still attempt – the user may have a .csv renamed .txt etc.
        logger.warning(
            "File '%s' has extension '%s', not a recognised CSV extension.",
            resolved,
            resolved.suffix,
        )

    logger.debug("Loading CSV: %s", resolved)
    df = pd.read_csv(
        resolved,
        sep=sep,
        encoding=encoding,
        dtype=dtype,
        usecols=usecols,
        nrows=nrows,
        **kwargs,
    )
    if df.empty:
        raise ValueError(f"CSV file loaded as an empty DataFrame: {resolved}")
    logger.info("Loaded CSV '%s': %d rows × %d cols", resolved.name, len(df), len(df.columns))
    return df


def load_parquet(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a Parquet file into a :class:`pandas.DataFrame`.

    Requires either ``pyarrow`` or ``fastparquet`` to be installed.

    Parameters
    ----------
    path:
        Path to the Parquet file.
    columns:
        Optional list of columns to read.
    **kwargs:
        Additional keyword arguments forwarded to :func:`pandas.read_parquet`.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    ImportError
        When neither ``pyarrow`` nor ``fastparquet`` is installed.
    """
    resolved = _resolve_path(path)
    if resolved.suffix.lower() not in _PARQUET_EXTENSIONS:
        raise ValueError(
            f"Expected a Parquet file (.parquet / .pq), got '{resolved.suffix}'."
        )

    try:
        import pyarrow  # type: ignore[import-not-found]  # noqa: F401 – optional dep
    except ImportError:
        try:
            import fastparquet  # type: ignore[import-not-found]  # noqa: F401 – optional dep
        except ImportError as exc:
            raise ImportError(
                "Loading Parquet files requires 'pyarrow' or 'fastparquet'. "
                "Install one with: pip install pyarrow"
            ) from exc

    logger.debug("Loading Parquet: %s", resolved)
    df = pd.read_parquet(resolved, columns=columns, **kwargs)
    if df.empty:
        raise ValueError(f"Parquet file loaded as an empty DataFrame: {resolved}")
    logger.info(
        "Loaded Parquet '%s': %d rows × %d cols", resolved.name, len(df), len(df.columns)
    )
    return df


def load_json(
    path: str | Path,
    *,
    orient: str | None = None,
    lines: bool = False,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a JSON (or JSONL/NDJSON) file into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path:
        Path to the JSON file.
    orient:
        Indication of expected JSON string format.  Forwarded to
        :func:`pandas.read_json`.  Pass ``None`` for pandas default inference.
    lines:
        When ``True``, reads the file as newline-delimited JSON (JSONL).
        Automatically set to ``True`` for ``.jsonl`` / ``.ndjson`` extensions.
    encoding:
        File encoding.
    **kwargs:
        Additional keyword arguments forwarded to :func:`pandas.read_json`.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    ValueError
        When the file extension is not recognised as JSON.
    """
    resolved = _resolve_path(path)
    if resolved.suffix.lower() not in _JSON_EXTENSIONS:
        raise ValueError(
            f"Expected a JSON file ({', '.join(_JSON_EXTENSIONS)}), got '{resolved.suffix}'."
        )
    # Force lines=True for JSONL/NDJSON
    if resolved.suffix.lower() in {".jsonl", ".ndjson"}:
        lines = True

    read_kwargs: dict[str, Any] = {"encoding": encoding, "lines": lines, **kwargs}
    if orient is not None:
        read_kwargs["orient"] = orient

    logger.debug("Loading JSON: %s", resolved)
    df = pd.read_json(resolved, **read_kwargs)
    if df.empty:
        raise ValueError(f"JSON file loaded as an empty DataFrame: {resolved}")
    logger.info(
        "Loaded JSON '%s': %d rows × %d cols", resolved.name, len(df), len(df.columns)
    )
    return df


def load_dataset(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Auto-detect file format from extension and load into a DataFrame.

    Supports CSV, Parquet, and JSON.  Pass format-specific keyword arguments
    directly; they are forwarded to the appropriate loader.

    Parameters
    ----------
    path:
        Path to the dataset file.
    **kwargs:
        Forwarded to the underlying loader (:func:`load_csv`,
        :func:`load_parquet`, or :func:`load_json`).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    ValueError
        When the file extension is not supported.
    """
    resolved = _resolve_path(path)
    suffix = resolved.suffix.lower()

    if suffix in _CSV_EXTENSIONS:
        return load_csv(resolved, **kwargs)
    if suffix in _PARQUET_EXTENSIONS:
        return load_parquet(resolved, **kwargs)
    if suffix in _JSON_EXTENSIONS:
        return load_json(resolved, **kwargs)

    raise ValueError(
        f"Unsupported file extension '{suffix}' for path '{resolved}'. "
        f"Supported: {sorted(_CSV_EXTENSIONS | _PARQUET_EXTENSIONS | _JSON_EXTENSIONS)}"
    )
