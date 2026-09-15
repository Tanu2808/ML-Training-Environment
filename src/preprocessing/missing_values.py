"""Missing value handling utilities."""


def fill_missing(df, value=0):
    """Fill missing values for all columns."""
    return df.fillna(value)
