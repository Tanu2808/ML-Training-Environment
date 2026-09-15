"""Validators for input datasets."""


def validate_columns(df, required_columns):
    """Check that required columns are present."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True
