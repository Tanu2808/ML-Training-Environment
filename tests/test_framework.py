import pandas as pd

from src.data.loader import load_csv
from src.data.splitter import train_test_split_frame
from src.preprocessing.pipeline import PreprocessingPipeline


def test_load_csv_reads_dataframe(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(path, index=False)

    df = load_csv(path)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["A", "B"]


def test_train_test_split_frame_returns_expected_shapes():
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "target": [0, 1, 0, 1]})

    X_train, X_test, y_train, y_test = train_test_split_frame(df, target_column="target", test_size=0.5)

    assert len(X_train) == 2
    assert len(X_test) == 2
    assert len(y_train) == 2
    assert len(y_test) == 2


def test_preprocessing_pipeline_runs():
    df = pd.DataFrame({"num": [1.0, None, 3.0], "cat": ["a", "b", "a"]})

    pipeline = PreprocessingPipeline()
    result = pipeline.fit_transform(df)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["num", "cat"]
