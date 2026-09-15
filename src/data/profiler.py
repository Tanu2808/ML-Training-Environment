"""Dataset profiling utilities."""


def describe_dataset(df):
    """Return a lightweight summary for a DataFrame."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
