import pandas as pd


def train_test_split_frame(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int | None = 42):
    """Split a DataFrame into train/test arrays for features and target."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if len(df) < 2:
        raise ValueError("DataFrame must contain at least 2 rows.")

    test_count = max(1, int(round(len(df) * test_size)))
    train_count = len(df) - test_count

    X_train = X.iloc[:train_count].copy()
    X_test = X.iloc[train_count:].copy()
    y_train = y.iloc[:train_count].copy()
    y_test = y.iloc[train_count:].copy()

    return X_train, X_test, y_train, y_test
