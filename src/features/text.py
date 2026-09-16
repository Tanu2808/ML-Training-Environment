"""Text feature extraction utilities.

Provides lightweight, vectorised text features without requiring heavy NLP
dependencies.  All basic features work with pandas string operations.
The optional TF-IDF transformer uses sklearn (already a project dependency).

Functions
---------
extract_text_features    Compute statistical text features for a column.

Classes
-------
TfidfTransformer         Leakage-safe TF-IDF wrapper (requires sklearn).
"""

from __future__ import annotations

import logging
import string
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_str_series(series: pd.Series) -> pd.Series:
    """Convert non-null values to str, leaving NaN as NaN."""
    return series.where(series.isna(), series.astype(str))


# ---------------------------------------------------------------------------
# Stateless text feature extractor
# ---------------------------------------------------------------------------

def extract_text_features(
    df: pd.DataFrame,
    column: str,
    *,
    prefix: str | None = None,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Compute statistical text features for a text column.

    Parameters
    ----------
    df:
        Input DataFrame.  Never mutated.
    column:
        The text column to process.
    prefix:
        Prefix for generated column names.  Defaults to *column*.
    features:
        Specific features to compute.  ``None`` → all features.
        Available: ``"char_count"``, ``"word_count"``, ``"whitespace_count"``,
        ``"digit_count"``, ``"uppercase_count"``, ``"lowercase_count"``,
        ``"punctuation_count"``, ``"newline_count"``, ``"avg_word_length"``,
        ``"unique_word_count"``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with text feature columns appended.

    Notes
    -----
    - ``NaN`` / ``None`` values produce ``NaN`` in all output columns.
    - Non-string values are coerced to ``str`` before processing.
    - Empty strings produce ``0`` for count features and ``NaN`` for averages.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    all_features = [
        "char_count", "word_count", "whitespace_count", "digit_count",
        "uppercase_count", "lowercase_count", "punctuation_count",
        "newline_count", "avg_word_length", "unique_word_count",
    ]
    requested = features if features is not None else all_features
    unknown = [f for f in requested if f not in all_features]
    if unknown:
        raise ValueError(
            f"Unknown features: {unknown}. Supported: {all_features}"
        )

    pfx = prefix if prefix is not None else column
    result = df.copy()

    # Coerce non-null values to str; keep NaN as NaN
    s = _to_str_series(df[column])

    punct_set = set(string.punctuation)

    def _count_punct(text: str) -> float:
        return sum(1 for c in text if c in punct_set)

    def _avg_word_len(text: str) -> float:
        words = text.split()
        if not words:
            return float("nan")
        return sum(len(w) for w in words) / len(words)

    def _unique_words(text: str) -> int:
        return len(set(text.lower().split()))

    for feat in requested:
        col_name = f"{pfx}_{feat}"
        if feat == "char_count":
            result[col_name] = s.str.len()
        elif feat == "word_count":
            result[col_name] = s.str.split().str.len()
        elif feat == "whitespace_count":
            result[col_name] = s.apply(
                lambda x: sum(1 for c in x if c.isspace()) if isinstance(x, str) else float("nan")
            )
        elif feat == "digit_count":
            result[col_name] = s.apply(
                lambda x: sum(1 for c in x if c.isdigit()) if isinstance(x, str) else float("nan")
            )
        elif feat == "uppercase_count":
            result[col_name] = s.apply(
                lambda x: sum(1 for c in x if c.isupper()) if isinstance(x, str) else float("nan")
            )
        elif feat == "lowercase_count":
            result[col_name] = s.apply(
                lambda x: sum(1 for c in x if c.islower()) if isinstance(x, str) else float("nan")
            )
        elif feat == "punctuation_count":
            result[col_name] = s.apply(
                lambda x: _count_punct(x) if isinstance(x, str) else float("nan")
            )
        elif feat == "newline_count":
            result[col_name] = s.str.count(r"\n")
        elif feat == "avg_word_length":
            result[col_name] = s.apply(
                lambda x: _avg_word_len(x) if isinstance(x, str) else float("nan")
            )
        elif feat == "unique_word_count":
            result[col_name] = s.apply(
                lambda x: _unique_words(x) if isinstance(x, str) else float("nan")
            )

    return result


# ---------------------------------------------------------------------------
# TF-IDF transformer (leakage-safe, requires sklearn)
# ---------------------------------------------------------------------------

class TfidfTransformer:
    """Leakage-safe TF-IDF wrapper.

    Learns the vocabulary and IDF weights **only** from training text, then
    applies them to validation/test text without refitting.

    Parameters
    ----------
    column:
        The text column to vectorise.
    max_features:
        Maximum number of TF-IDF features.  Default ``500``.
    ngram_range:
        Range of n-gram sizes.  Default ``(1, 1)`` (unigrams only).
    min_df:
        Minimum document frequency.  Default ``1``.
    max_df:
        Maximum document frequency (float = fraction, int = count).
        Default ``1.0``.
    prefix:
        Prefix for output column names.  Defaults to ``"tfidf_{column}"``.

    Examples
    --------
    >>> tfidf = TfidfTransformer(column="review_text", max_features=200)
    >>> train_feats = tfidf.fit_transform(X_train)
    >>> test_feats  = tfidf.transform(X_test)
    """

    def __init__(
        self,
        column: str,
        *,
        max_features: int = 500,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        prefix: str | None = None,
    ) -> None:
        self.column = column
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.prefix = prefix
        self._vectorizer: Any = None

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TfidfTransformer":
        """Fit the TF-IDF vocabulary on *X* (training data).

        Parameters
        ----------
        X:
            Training DataFrame containing the text column.
        y:
            Ignored.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError(
                "TfidfTransformer requires scikit-learn. "
                "Install it with: pip install scikit-learn"
            ) from exc

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")

        text = X[self.column].fillna("").astype(str)
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self._vectorizer.fit(text)
        pfx = self.prefix or f"tfidf_{self.column}"
        self._feature_names = [
            f"{pfx}_{name}" for name in self._vectorizer.get_feature_names_out()
        ]
        logger.debug("TfidfTransformer.fit: %d features", len(self._feature_names))
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Transform *X* using the fitted vocabulary.

        Parameters
        ----------
        X:
            DataFrame to transform.
        """
        if self._vectorizer is None:
            raise RuntimeError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(X).__name__}.")
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")

        text = X[self.column].fillna("").astype(str)
        arr = self._vectorizer.transform(text).toarray()
        tfidf_df = pd.DataFrame(arr, columns=self._feature_names, index=X.index)
        # Append to input (preserving non-text columns)
        return pd.concat([X.drop(columns=[self.column]), tfidf_df], axis=1)

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit and transform in one step.

        Parameters
        ----------
        X:
            Training DataFrame.
        """
        self.fit(X, y)
        return self.transform(X)

    @property
    def feature_names(self) -> list[str]:
        """Names of the TF-IDF output columns."""
        if self._vectorizer is None:
            raise RuntimeError("Call fit() first.")
        return self._feature_names


# ---------------------------------------------------------------------------
# Backward-compatible stub
# ---------------------------------------------------------------------------

def basic_text_features(df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    """Backward-compatible stub.  Use :func:`extract_text_features` instead."""
    return df.copy()
