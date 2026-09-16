"""Phase 1A tests: data loading, profiling, validation, sampling, splitting.

All tests are deterministic and self-contained (no external files required
beyond what pytest's tmp_path fixture provides).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_a": rng.standard_normal(n),
            "feature_b": rng.integers(0, 10, n),
            "category": rng.choice(["cat", "dog", "bird"], n),
            "target": rng.integers(0, 2, n),
        }
    )


# ===========================================================================
# loader.py
# ===========================================================================


class TestLoadCsv:
    def test_loads_dataframe(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(path, index=False)
        from src.data.loader import load_csv

        df = load_csv(path)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["A", "B"]
        assert len(df) == 2

    def test_columns_match(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"X": [10, 20, 30]}).to_csv(path, index=False)
        from src.data.loader import load_csv

        df = load_csv(path)
        assert "X" in df.columns

    def test_invalid_path_raises(self):
        from src.data.loader import load_csv

        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/file.csv")

    def test_usecols(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"A": [1], "B": [2], "C": [3]}).to_csv(path, index=False)
        from src.data.loader import load_csv

        df = load_csv(path, usecols=["A", "C"])
        assert list(df.columns) == ["A", "C"]

    def test_nrows(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"A": range(100)}).to_csv(path, index=False)
        from src.data.loader import load_csv

        df = load_csv(path, nrows=10)
        assert len(df) == 10


class TestLoadJson:
    def test_loads_dataframe(self, tmp_path):
        path = tmp_path / "data.json"
        pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_json(path, orient="records")
        from src.data.loader import load_json

        df = load_json(path, orient="records")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"A", "B"}

    def test_wrong_extension_raises(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("hello")
        from src.data.loader import load_json

        with pytest.raises(ValueError, match="Expected a JSON file"):
            load_json(path)


class TestLoadDataset:
    def test_auto_detects_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"A": [1, 2]}).to_csv(path, index=False)
        from src.data.loader import load_dataset

        df = load_dataset(path)
        assert isinstance(df, pd.DataFrame)

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "data.xyz"
        path.write_text("whatever")
        from src.data.loader import load_dataset

        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_dataset(path)


# ===========================================================================
# profiler.py
# ===========================================================================


class TestProfileDataset:
    def test_returns_dict(self):
        from src.data.profiler import profile_dataset

        df = _make_df()
        profile = profile_dataset(df)
        assert isinstance(profile, dict)

    def test_shape(self):
        from src.data.profiler import profile_dataset

        df = _make_df(30)
        profile = profile_dataset(df)
        assert profile["n_rows"] == 30
        assert profile["n_cols"] == 4

    def test_missing_values_counted(self):
        from src.data.profiler import profile_dataset

        df = pd.DataFrame({"A": [1.0, None, 3.0], "B": [4, 5, 6]})
        profile = profile_dataset(df)
        assert profile["missing"]["counts"]["A"] == 1
        assert profile["missing"]["counts"]["B"] == 0

    def test_missing_pct(self):
        from src.data.profiler import profile_dataset

        df = pd.DataFrame({"A": [None, None, 3.0, 4.0]})
        profile = profile_dataset(df)
        assert abs(profile["missing"]["percentages"]["A"] - 50.0) < 0.01

    def test_duplicate_count(self):
        from src.data.profiler import profile_dataset

        df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
        profile = profile_dataset(df)
        assert profile["duplicates"] == 1

    def test_column_type_classification(self):
        from src.data.profiler import profile_dataset

        df = pd.DataFrame(
            {
                "num": [1.0, 2.0],
                "cat": ["a", "b"],
                "dt": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            }
        )
        profile = profile_dataset(df)
        assert "num" in profile["column_types"]["numeric"]
        assert "cat" in profile["column_types"]["categorical"]
        assert "dt" in profile["column_types"]["datetime"]

    def test_json_serialisable(self):
        from src.data.profiler import profile_dataset

        df = _make_df()
        profile = profile_dataset(df)
        # Should not raise
        json.dumps(profile)

    def test_raises_on_non_dataframe(self):
        from src.data.profiler import profile_dataset

        with pytest.raises(TypeError):
            profile_dataset([1, 2, 3])  # type: ignore[arg-type]

    def test_empty_dataframe_raises(self):
        from src.data.profiler import profile_dataset

        with pytest.raises(ValueError):
            profile_dataset(pd.DataFrame())

    def test_describe_dataset_backward_compat(self):
        from src.data.profiler import describe_dataset

        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = describe_dataset(df)
        assert result["rows"] == 2
        assert "columns" in result
        assert "dtypes" in result


# ===========================================================================
# sampler.py
# ===========================================================================


class TestSampleRandom:
    def test_count_based(self):
        from src.data.sampler import sample_random

        df = _make_df(50)
        result = sample_random(df, n=10)
        assert len(result) == 10

    def test_fraction_based(self):
        from src.data.sampler import sample_random

        df = _make_df(100)
        result = sample_random(df, frac=0.2)
        assert len(result) == 20

    def test_reproducible_with_seed(self):
        from src.data.sampler import sample_random

        df = _make_df(50)
        r1 = sample_random(df, n=10, random_state=0)
        r2 = sample_random(df, n=10, random_state=0)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_different_seeds_differ(self):
        from src.data.sampler import sample_random

        df = _make_df(100)
        r1 = sample_random(df, n=50, random_state=0)
        r2 = sample_random(df, n=50, random_state=99)
        # It's astronomically unlikely they produce the same order
        assert not r1.index.equals(r2.index)

    def test_invalid_n_too_large(self):
        from src.data.sampler import sample_random

        df = _make_df(5)
        with pytest.raises(ValueError):
            sample_random(df, n=100)

    def test_invalid_frac(self):
        from src.data.sampler import sample_random

        df = _make_df(10)
        with pytest.raises(ValueError):
            sample_random(df, frac=1.5)

    def test_both_n_and_frac_raises(self):
        from src.data.sampler import sample_random

        df = _make_df(10)
        with pytest.raises(ValueError):
            sample_random(df, n=5, frac=0.5)

    def test_neither_raises(self):
        from src.data.sampler import sample_random

        df = _make_df(10)
        with pytest.raises(ValueError):
            sample_random(df)

    def test_does_not_mutate_original(self):
        from src.data.sampler import sample_random

        df = _make_df(20)
        original_len = len(df)
        sample_random(df, n=5)
        assert len(df) == original_len


class TestSampleStratified:
    def test_proportions_preserved(self):
        from src.data.sampler import sample_stratified

        df = pd.DataFrame(
            {
                "feature": range(100),
                "label": ["A"] * 70 + ["B"] * 30,
            }
        )
        result = sample_stratified(df, "label", frac=0.5)
        counts = result["label"].value_counts(normalize=True)
        assert abs(counts["A"] - 0.7) < 0.1
        assert abs(counts["B"] - 0.3) < 0.1

    def test_missing_column_raises(self):
        from src.data.sampler import sample_stratified

        df = _make_df(20)
        with pytest.raises(KeyError):
            sample_stratified(df, "nonexistent_col", n=10)

    def test_both_n_and_frac_raises(self):
        from src.data.sampler import sample_stratified

        df = _make_df(20)
        with pytest.raises(ValueError):
            sample_stratified(df, "category", n=5, frac=0.5)


class TestSampleRows:
    def test_none_returns_full(self):
        from src.data.sampler import sample_rows

        df = _make_df(10)
        result = sample_rows(df, None)
        assert len(result) == 10

    def test_count_capped_at_len(self):
        from src.data.sampler import sample_rows

        df = _make_df(5)
        result = sample_rows(df, 100)
        assert len(result) == 5


# ===========================================================================
# splitter.py
# ===========================================================================


class TestTrainTestSplitFrame:
    def test_basic_shapes(self):
        from src.data.splitter import train_test_split_frame

        df = _make_df(100)
        X_train, X_test, y_train, y_test = train_test_split_frame(
            df, "target", test_size=0.2, random_state=42
        )
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_target_not_in_features(self):
        from src.data.splitter import train_test_split_frame

        df = _make_df(50)
        X_train, X_test, _, _ = train_test_split_frame(df, "target")
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns

    def test_random_state_is_respected(self):
        """Different seeds must produce different splits."""
        from src.data.splitter import train_test_split_frame

        df = _make_df(100)
        _, _, y1, _ = train_test_split_frame(df, "target", random_state=0)
        _, _, y2, _ = train_test_split_frame(df, "target", random_state=999)
        # It is vanishingly unlikely that 80 labels match in the same order
        assert not (y1.values == y2.values).all()

    def test_reproducibility(self):
        from src.data.splitter import train_test_split_frame

        df = _make_df(100)
        X1, _, _, _ = train_test_split_frame(df, "target", random_state=7)
        X2, _, _, _ = train_test_split_frame(df, "target", random_state=7)
        pd.testing.assert_frame_equal(X1, X2)

    def test_invalid_test_size_raises(self):
        from src.data.splitter import train_test_split_frame

        df = _make_df(10)
        with pytest.raises(ValueError):
            train_test_split_frame(df, "target", test_size=1.5)
        with pytest.raises(ValueError):
            train_test_split_frame(df, "target", test_size=0.0)

    def test_missing_target_raises(self):
        from src.data.splitter import train_test_split_frame

        df = _make_df(10)
        with pytest.raises(ValueError, match="not found"):
            train_test_split_frame(df, "nonexistent")

    def test_too_small_dataframe_raises(self):
        from src.data.splitter import train_test_split_frame

        df = pd.DataFrame({"A": [1], "target": [0]})
        with pytest.raises(ValueError):
            train_test_split_frame(df, "target")

    def test_no_data_leakage(self):
        from src.data.splitter import train_test_split_frame

        df = _make_df(50)
        X_train, X_test, _, _ = train_test_split_frame(df, "target")
        # Convert each row to a hashable tuple and check there is no overlap
        train_rows = set(map(tuple, X_train.itertuples(index=False, name=None)))
        test_rows = set(map(tuple, X_test.itertuples(index=False, name=None)))
        assert train_rows.isdisjoint(test_rows)


class TestStratifiedSplit:
    def test_class_balance_maintained(self):
        from src.data.splitter import stratified_split

        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "feat": rng.standard_normal(200),
                "target": ["A"] * 100 + ["B"] * 60 + ["C"] * 40,
            }
        )
        X_train, X_test, y_train, y_test = stratified_split(df, "target", test_size=0.2)
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        for cls in ["A", "B", "C"]:
            assert abs(train_dist[cls] - test_dist[cls]) < 0.1


class TestGroupSplit:
    def test_groups_not_shared(self):
        from src.data.splitter import group_split

        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "group": [i // 5 for i in range(100)],
                "feat": rng.standard_normal(100),
                "target": rng.integers(0, 2, 100),
            }
        )
        X_train, X_test, y_train, y_test = group_split(
            df, "target", "group", test_size=0.2, random_state=42
        )
        # "group" column is in X since it's not the target
        train_groups = set(X_train["group"].unique())
        test_groups = set(X_test["group"].unique())
        assert train_groups.isdisjoint(test_groups)

    def test_missing_group_col_raises(self):
        from src.data.splitter import group_split

        df = _make_df(20)
        with pytest.raises(KeyError):
            group_split(df, "target", "nonexistent_group")


class TestTimeSeriesSplit:
    def test_no_shuffle(self):
        from src.data.splitter import time_series_split

        df = pd.DataFrame({"time": range(100), "target": range(100)})
        X_train, X_test, y_train, y_test = time_series_split(df, "target", test_size=0.2)
        # Train values must all be < test values (temporal order preserved)
        assert y_train.max() < y_test.min()

    def test_shapes(self):
        from src.data.splitter import time_series_split

        df = pd.DataFrame({"feat": range(100), "target": range(100)})
        X_train, X_test, y_train, y_test = time_series_split(df, "target", test_size=0.2)
        assert len(X_train) + len(X_test) == 100


# ===========================================================================
# validator.py
# ===========================================================================


class TestValidateDataframe:
    def test_valid_df_passes(self):
        from src.data.validator import validate_dataframe

        df = _make_df(10)
        validate_dataframe(df)  # should not raise

    def test_non_df_raises_type_error(self):
        from src.data.validator import validate_dataframe

        with pytest.raises(TypeError):
            validate_dataframe([1, 2, 3])  # type: ignore[arg-type]

    def test_empty_df_raises(self):
        from src.data.validator import validate_dataframe

        with pytest.raises(ValueError):
            validate_dataframe(pd.DataFrame())

    def test_min_rows_respected(self):
        from src.data.validator import validate_dataframe

        df = pd.DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ValueError):
            validate_dataframe(df, min_rows=10)

    def test_allow_empty_skips_check(self):
        from src.data.validator import validate_dataframe

        validate_dataframe(pd.DataFrame(), allow_empty=True)  # should not raise


class TestValidateColumns:
    def test_all_present_passes(self):
        from src.data.validator import validate_columns

        df = pd.DataFrame({"A": [1], "B": [2]})
        validate_columns(df, ["A", "B"])  # should not raise

    def test_missing_column_raises(self):
        from src.data.validator import validate_columns

        df = pd.DataFrame({"A": [1]})
        with pytest.raises(ValueError, match="Missing required column"):
            validate_columns(df, ["A", "B"])


class TestValidateTargetColumn:
    def test_clean_target_passes(self):
        from src.data.validator import validate_target_column

        df = pd.DataFrame({"feat": [1, 2], "target": [0, 1]})
        validate_target_column(df, "target")  # should not raise

    def test_missing_target_values_raise(self):
        from src.data.validator import validate_target_column

        df = pd.DataFrame({"feat": [1, 2], "target": [0, None]})
        with pytest.raises(ValueError, match="missing value"):
            validate_target_column(df, "target")

    def test_allow_missing_suppresses_error(self):
        from src.data.validator import validate_target_column

        df = pd.DataFrame({"feat": [1], "target": [None]})
        validate_target_column(df, "target", allow_missing_values=True)

    def test_absent_column_raises(self):
        from src.data.validator import validate_target_column

        df = pd.DataFrame({"feat": [1]})
        with pytest.raises(ValueError):
            validate_target_column(df, "target")


class TestValidateNoDuplicates:
    def test_no_dups_returns_zero(self):
        from src.data.validator import validate_no_duplicates

        df = pd.DataFrame({"A": [1, 2, 3]})
        n = validate_no_duplicates(df, raise_on_duplicates=False)
        assert n == 0

    def test_dup_raises_by_default(self):
        from src.data.validator import validate_no_duplicates

        df = pd.DataFrame({"A": [1, 1, 2]})
        with pytest.raises(ValueError, match="duplicate"):
            validate_no_duplicates(df)

    def test_dup_no_raise_returns_count(self):
        from src.data.validator import validate_no_duplicates

        df = pd.DataFrame({"A": [1, 1, 2, 2]})
        n = validate_no_duplicates(df, raise_on_duplicates=False)
        assert n == 2


class TestValidateDataset:
    def test_valid_dataset_passes(self):
        from src.data.validator import validate_dataset

        df = _make_df(20)
        report = validate_dataset(df, required_columns=["feature_a"], target_column="target")
        assert report["valid"] is True

    def test_warns_on_all_nan_col(self):
        from src.data.validator import validate_dataset

        df = pd.DataFrame({"A": [1, 2], "B": [None, None]})
        report = validate_dataset(df)
        assert any("B" in w for w in report["warnings"])  # type: ignore[operator]

    def test_duplicate_check(self):
        from src.data.validator import validate_dataset

        df = pd.DataFrame({"A": [1, 1, 2]})
        report = validate_dataset(df, check_duplicates=True)
        assert report.get("n_duplicates") == 1
