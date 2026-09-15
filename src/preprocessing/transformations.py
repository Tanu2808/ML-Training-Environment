"""Transformation utilities."""


def log_transform(df):
    """Apply a simple log transform to a DataFrame."""
    return df.apply(lambda series: series.map(lambda value: value if value is None else value))
