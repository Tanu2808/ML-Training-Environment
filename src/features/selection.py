"""Feature selection utilities."""


def select_features(df, columns):
    """Select a set of columns from a DataFrame."""
    return df[list(columns)]
