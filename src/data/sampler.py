"""Sampling helpers."""


def sample_rows(df, sample_size=None):
    """Return a sampled DataFrame or the full frame when no size is requested."""
    if sample_size is None:
        return df
    return df.sample(n=min(sample_size, len(df))).copy()
