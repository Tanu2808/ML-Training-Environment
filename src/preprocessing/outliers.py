"""Outlier handling utilities."""


def clip_outliers(df, lower_bound=None, upper_bound=None):
    """Return the DataFrame with no outlier clipping applied."""
    return df.copy()
