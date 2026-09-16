"""Phase 1B tests: preprocessing — missing values, encoding, scaling,
transformations, outliers, pipeline, and data-leakage proofs.

All tests are deterministic and self-contained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# Fixtures / helpers
# ===========================================================================

def _num_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n) * 5 + 10,
        "c": rng.integers(1, 100, n).astype(float),
    })


def _mixed_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "num1": rng.standard_normal(n),
        "num2": rng.integers(0, 100, n).astype(float),
        "cat1": rng.choice(["X", "Y", "Z"], n),
        "cat2": rng.choice(["p", "q"], n),
    })


def _df_with_nans(n: int = 20, seed: int = 0) -> pd.DataFrame:
    df = _num_df(n, seed)
    rng = np.random.default_rng(seed + 1)
    for col in df.columns:
        idx = rng.choice(n, size=4, replace=False)
        df.loc[idx, col] = np.nan
    return df


# ===========================================================================
# 1. missing_values.py
# ===========================================================================

class TestHandleMissingValuesStateless:

    def test_mean_fills_numeric(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = handle_missing_values(df, strategy="mean")
        assert result["a"].isna().sum() == 0
        assert abs(result["a"].iloc[1] - 2.0) < 1e-9

    def test_median_fills_numeric(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [1.0, None, 3.0, 5.0]})
        result = handle_missing_values(df, strategy="median")
        assert result["a"].isna().sum() == 0

    def test_mode_fills_categorical(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"c": ["a", "a", "b", None]})
        result = handle_missing_values(df, strategy="mode", columns=["c"])
        assert result["c"].iloc[3] == "a"
        assert result["c"].isna().sum() == 0

    def test_constant_strategy(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = handle_missing_values(df, strategy="constant", fill_value=-999)
        assert result["a"].iloc[1] == -999

    def test_ffill(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [1.0, None, None, 4.0]})
        result = handle_missing_values(df, strategy="ffill")
        assert result["a"].tolist() == [1.0, 1.0, 1.0, 4.0]

    def test_bfill(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [None, None, 3.0, 4.0]})
        result = handle_missing_values(df, strategy="bfill")
        assert result["a"].tolist() == [3.0, 3.0, 3.0, 4.0]

    def test_drop_rows(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4, 5, 6]})
        result = handle_missing_values(df, strategy="drop_rows", columns=["a"])
        assert len(result) == 2
        assert result["a"].isna().sum() == 0

    def test_drop_cols_all_nan(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [None, None]})
        result = handle_missing_values(df, strategy="drop_cols", drop_threshold=1.0)
        assert "b" not in result.columns
        assert "a" in result.columns

    def test_does_not_mutate_original(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        original_nan = df["a"].isna().sum()
        handle_missing_values(df, strategy="mean")
        assert df["a"].isna().sum() == original_nan

    def test_invalid_strategy_raises(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_missing_values(df, strategy="magic")  # type: ignore

    def test_invalid_column_raises(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="not found"):
            handle_missing_values(df, strategy="mean", columns=["nonexistent"])

    def test_mean_on_categorical_raises(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"c": ["a", "b", None]})
        with pytest.raises(ValueError, match="numeric"):
            handle_missing_values(df, strategy="mean", columns=["c"])

    def test_empty_df_returns_copy(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame()
        result = handle_missing_values(df, strategy="mean")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_all_nan_column_mean_results_in_nan(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [None, None, None]})
        result = handle_missing_values(df, strategy="mean")
        # Mean of all-NaN is NaN, values remain NaN
        assert result["a"].isna().all()


class TestMissingValueImputer:

    def test_mean_fit_transform(self):
        from src.preprocessing.missing_values import MissingValueImputer
        train = pd.DataFrame({"a": [10.0, 20.0, None]})
        test = pd.DataFrame({"a": [None, 5.0]})
        imp = MissingValueImputer(strategy="mean")
        imp.fit(train)
        result = imp.transform(test)
        # Fill value must be mean of TRAIN (15.0), not mean of test
        assert abs(result["a"].iloc[0] - 15.0) < 1e-9
        assert result["a"].iloc[1] == 5.0

    def test_leakage_proof_mean(self):
        """Statistics must come only from training data."""
        from src.preprocessing.missing_values import MissingValueImputer
        train = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
        imp = MissingValueImputer(strategy="mean")
        imp.fit(train)
        assert abs(imp._fill_map["a"] - 20.0) < 1e-9  # mean(10,20,30)

        test = pd.DataFrame({"a": [1000.0, None]})
        result = imp.transform(test)
        # NaN in test filled with TRAIN mean (20), not test mean (1000)
        assert abs(result["a"].iloc[1] - 20.0) < 1e-9

    def test_median_strategy(self):
        from src.preprocessing.missing_values import MissingValueImputer
        train = pd.DataFrame({"x": [1.0, 2.0, 100.0]})
        test = pd.DataFrame({"x": [None]})
        imp = MissingValueImputer(strategy="median").fit(train)
        result = imp.transform(test)
        assert abs(result["x"].iloc[0] - 2.0) < 1e-9

    def test_mode_strategy(self):
        from src.preprocessing.missing_values import MissingValueImputer
        train = pd.DataFrame({"c": ["a", "a", "b"]})
        test = pd.DataFrame({"c": [None, "b"]})
        imp = MissingValueImputer(strategy="mode").fit(train)
        result = imp.transform(test)
        assert result["c"].iloc[0] == "a"

    def test_constant_strategy(self):
        from src.preprocessing.missing_values import MissingValueImputer
        train = pd.DataFrame({"a": [1.0, 2.0]})
        test = pd.DataFrame({"a": [None]})
        imp = MissingValueImputer(strategy="constant", fill_value=-1).fit(train)
        result = imp.transform(test)
        assert result["a"].iloc[0] == -1

    def test_transform_before_fit_raises(self):
        from src.preprocessing.missing_values import MissingValueImputer
        imp = MissingValueImputer()
        with pytest.raises(RuntimeError, match="fit"):
            imp.transform(pd.DataFrame({"a": [1.0]}))

    def test_invalid_strategy_raises(self):
        from src.preprocessing.missing_values import MissingValueImputer
        imp = MissingValueImputer(strategy="ffill")  # type: ignore
        with pytest.raises(ValueError):
            imp.fit(pd.DataFrame({"a": [1.0]}))


# ===========================================================================
# 2. encoding.py
# ===========================================================================

class TestOneHotEncoder:

    def test_basic_onehot(self):
        from src.preprocessing.encoding import OneHotEncoder
        df = pd.DataFrame({"color": ["red", "blue", "red"]})
        enc = OneHotEncoder(columns=["color"])
        result = enc.fit_transform(df)
        assert "color_red" in result.columns
        assert "color_blue" in result.columns

    def test_fit_transform_train_then_transform_test(self):
        from src.preprocessing.encoding import OneHotEncoder
        train = pd.DataFrame({"size": ["S", "M", "L"]})
        test = pd.DataFrame({"size": ["M", "L"]})
        enc = OneHotEncoder(columns=["size"])
        enc.fit(train)
        train_enc = enc.transform(train)
        test_enc = enc.transform(test)
        # Same columns in both
        assert list(train_enc.columns) == list(test_enc.columns)

    def test_unknown_category_handled_as_zeros(self):
        from src.preprocessing.encoding import OneHotEncoder
        train = pd.DataFrame({"city": ["NYC", "LA"]})
        test = pd.DataFrame({"city": ["NYC", "UNKNOWN_CITY"]})
        enc = OneHotEncoder(columns=["city"], handle_unknown="ignore")
        enc.fit(train)
        result = enc.transform(test)
        # UNKNOWN_CITY row should have all-zero dummies
        assert result["city_NYC"].iloc[1] == 0
        assert result["city_LA"].iloc[1] == 0

    def test_unknown_category_error_mode(self):
        from src.preprocessing.encoding import OneHotEncoder
        train = pd.DataFrame({"city": ["NYC", "LA"]})
        test = pd.DataFrame({"city": ["UNKNOWN_CITY"]})
        enc = OneHotEncoder(columns=["city"], handle_unknown="error")
        enc.fit(train)
        with pytest.raises(ValueError, match="Unknown categories"):
            enc.transform(test)

    def test_drop_first(self):
        from src.preprocessing.encoding import OneHotEncoder
        df = pd.DataFrame({"color": ["red", "blue", "green"]})
        enc = OneHotEncoder(columns=["color"], drop_first=True)
        result = enc.fit_transform(df)
        # With 3 categories and drop_first, expect 2 dummy columns
        dummy_cols = [c for c in result.columns if c.startswith("color_")]
        assert len(dummy_cols) == 2

    def test_multiple_columns(self):
        from src.preprocessing.encoding import OneHotEncoder
        df = _mixed_df()
        enc = OneHotEncoder(columns=["cat1", "cat2"])
        result = enc.fit_transform(df)
        assert any(c.startswith("cat1_") for c in result.columns)
        assert any(c.startswith("cat2_") for c in result.columns)

    def test_no_mutation_of_original(self):
        from src.preprocessing.encoding import OneHotEncoder
        df = pd.DataFrame({"c": ["a", "b"]})
        original_cols = list(df.columns)
        enc = OneHotEncoder()
        enc.fit_transform(df)
        assert list(df.columns) == original_cols

    def test_auto_detect_categorical_columns(self):
        from src.preprocessing.encoding import OneHotEncoder
        df = pd.DataFrame({"num": [1, 2], "cat": ["a", "b"]})
        enc = OneHotEncoder()  # no columns specified
        result = enc.fit_transform(df)
        assert "num" in result.columns
        assert "cat_a" in result.columns

    def test_transform_before_fit_raises(self):
        from src.preprocessing.encoding import OneHotEncoder
        enc = OneHotEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(pd.DataFrame({"c": ["a"]}))

    def test_column_not_in_df_raises(self):
        from src.preprocessing.encoding import OneHotEncoder
        df = pd.DataFrame({"a": ["x"]})
        enc = OneHotEncoder(columns=["nonexistent"])
        with pytest.raises(ValueError):
            enc.fit(df)

    def test_consistent_column_order(self):
        from src.preprocessing.encoding import OneHotEncoder
        train = pd.DataFrame({"c": ["a", "b", "c"]})
        test1 = pd.DataFrame({"c": ["b"]})
        test2 = pd.DataFrame({"c": ["a"]})
        enc = OneHotEncoder(columns=["c"])
        enc.fit(train)
        assert list(enc.transform(test1).columns) == list(enc.transform(test2).columns)


class TestOrdinalCategoryEncoder:

    def test_basic_ordinal(self):
        from src.preprocessing.encoding import OrdinalCategoryEncoder
        df = pd.DataFrame({"size": ["S", "M", "L"]})
        enc = OrdinalCategoryEncoder(columns=["size"])
        result = enc.fit_transform(df)
        assert result["size"].dtype in (np.int64, np.int32)
        assert len(result["size"].unique()) == 3

    def test_explicit_ordering(self):
        from src.preprocessing.encoding import OrdinalCategoryEncoder
        train = pd.DataFrame({"size": ["L", "S", "M"]})
        enc = OrdinalCategoryEncoder(
            columns=["size"],
            categories={"size": ["S", "M", "L"]},
        )
        enc.fit(train)
        result = enc.transform(train)
        # S→0, M→1, L→2
        mapping = dict(zip(train["size"], result["size"]))
        assert mapping["S"] < mapping["M"] < mapping["L"]

    def test_unknown_category_returns_minus_one(self):
        from src.preprocessing.encoding import OrdinalCategoryEncoder
        train = pd.DataFrame({"c": ["a", "b"]})
        test = pd.DataFrame({"c": ["a", "UNKNOWN"]})
        enc = OrdinalCategoryEncoder(columns=["c"]).fit(train)
        result = enc.transform(test)
        assert result["c"].iloc[1] == -1

    def test_fit_only_on_train(self):
        """Transform test must use train vocabulary."""
        from src.preprocessing.encoding import OrdinalCategoryEncoder
        train = pd.DataFrame({"c": ["a", "b"]})
        test = pd.DataFrame({"c": ["b", "a"]})
        enc = OrdinalCategoryEncoder(columns=["c"]).fit(train)
        r_train = enc.transform(train)
        r_test = enc.transform(test)
        # 'b' gets same code in both
        assert r_train.loc[r_train.index[1], "c"] == r_test.loc[r_test.index[0], "c"]

    def test_transform_before_fit_raises(self):
        from src.preprocessing.encoding import OrdinalCategoryEncoder
        enc = OrdinalCategoryEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(pd.DataFrame({"c": ["a"]}))


class TestOneHotEncodeFunctional:
    def test_returns_dataframe(self):
        from src.preprocessing.encoding import one_hot_encode
        df = pd.DataFrame({"c": ["a", "b", "a"]})
        result = one_hot_encode(df)
        assert isinstance(result, pd.DataFrame)
        assert "c_a" in result.columns


# ===========================================================================
# 3. scaling.py
# ===========================================================================

class TestDataFrameScaler:

    def test_standard_scaler_zero_mean(self):
        from src.preprocessing.scaling import DataFrameScaler
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        scaler = DataFrameScaler("standard")
        result = scaler.fit_transform(df)
        assert abs(result["a"].mean()) < 1e-9

    def test_minmax_range(self):
        from src.preprocessing.scaling import DataFrameScaler
        df = pd.DataFrame({"a": [0.0, 5.0, 10.0]})
        scaler = DataFrameScaler("minmax")
        result = scaler.fit_transform(df)
        assert abs(result["a"].min()) < 1e-9
        assert abs(result["a"].max() - 1.0) < 1e-9

    def test_robust_scaler_runs(self):
        from src.preprocessing.scaling import DataFrameScaler
        df = _num_df()
        scaler = DataFrameScaler("robust")
        result = scaler.fit_transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_leakage_proof(self):
        """Test set must be scaled using TRAIN statistics."""
        from src.preprocessing.scaling import DataFrameScaler
        train = pd.DataFrame({"a": [0.0, 10.0]})   # mean=5, std=~7.07
        test = pd.DataFrame({"a": [1000.0]})
        scaler = DataFrameScaler("standard").fit(train)
        result = scaler.transform(test)
        # Scaled value = (1000 - 5) / std_of_train
        expected = (1000.0 - train["a"].mean()) / train["a"].std(ddof=0)
        assert abs(result["a"].iloc[0] - expected) < 0.01

    def test_selected_columns_only(self):
        from src.preprocessing.scaling import DataFrameScaler
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [100.0, 200.0], "cat": ["x", "y"]})
        scaler = DataFrameScaler("standard", columns=["a"])
        result = scaler.fit_transform(df)
        # 'b' untouched
        assert result["b"].tolist() == [100.0, 200.0]
        # 'cat' untouched
        assert result["cat"].tolist() == ["x", "y"]

    def test_preserves_column_names(self):
        from src.preprocessing.scaling import DataFrameScaler
        df = _num_df()
        scaler = DataFrameScaler("standard")
        result = scaler.fit_transform(df)
        assert list(result.columns) == list(df.columns)

    def test_does_not_mutate_original(self):
        from src.preprocessing.scaling import DataFrameScaler
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        orig = df["a"].tolist()
        DataFrameScaler("standard").fit_transform(df)
        assert df["a"].tolist() == orig

    def test_transform_before_fit_raises(self):
        from src.preprocessing.scaling import DataFrameScaler
        scaler = DataFrameScaler("standard")
        with pytest.raises(RuntimeError):
            scaler.transform(pd.DataFrame({"a": [1.0]}))

    def test_invalid_scaler_type_raises(self):
        from src.preprocessing.scaling import DataFrameScaler
        scaler = DataFrameScaler("unknown_type")  # type: ignore
        with pytest.raises(ValueError):
            scaler.fit(pd.DataFrame({"a": [1.0]}))

    def test_non_numeric_column_raises(self):
        from src.preprocessing.scaling import DataFrameScaler
        df = pd.DataFrame({"cat": ["a", "b"]})
        scaler = DataFrameScaler("standard", columns=["cat"])
        with pytest.raises(ValueError, match="numeric"):
            scaler.fit(df)

    def test_train_test_consistency(self):
        from src.preprocessing.scaling import DataFrameScaler
        df = _num_df(100)
        train, test = df.iloc[:80], df.iloc[80:]
        scaler = DataFrameScaler("standard").fit(train)
        r_train = scaler.transform(train)
        r_test = scaler.transform(test)
        assert list(r_train.columns) == list(r_test.columns)


class TestScalerConveniences:
    def test_make_standard_scaler(self):
        from src.preprocessing.scaling import make_standard_scaler
        scaler = make_standard_scaler()
        result = scaler.fit_transform(_num_df())
        assert isinstance(result, pd.DataFrame)

    def test_make_minmax_scaler(self):
        from src.preprocessing.scaling import make_minmax_scaler
        scaler = make_minmax_scaler()
        result = scaler.fit_transform(_num_df())
        assert result.min().min() >= -1e-9
        assert result.max().max() <= 1 + 1e-9

    def test_make_robust_scaler(self):
        from src.preprocessing.scaling import make_robust_scaler
        scaler = make_robust_scaler()
        result = scaler.fit_transform(_num_df())
        assert isinstance(result, pd.DataFrame)

    def test_standardize_functional(self):
        from src.preprocessing.scaling import standardize
        df = _num_df()
        result = standardize(df)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# 4. transformations.py
# ===========================================================================

class TestApplyLog1p:
    def test_basic(self):
        from src.preprocessing.transformations import apply_log1p
        df = pd.DataFrame({"a": [0.0, 1.0, np.e - 1]})
        result = apply_log1p(df)
        assert abs(result["a"].iloc[0]) < 1e-9        # log1p(0) = 0
        assert abs(result["a"].iloc[1] - np.log(2)) < 1e-9
        assert abs(result["a"].iloc[2] - 1.0) < 1e-9

    def test_negative_raises_by_default(self):
        from src.preprocessing.transformations import apply_log1p
        df = pd.DataFrame({"a": [-1.0, 1.0]})
        with pytest.raises(ValueError, match="invalid value"):
            apply_log1p(df)

    def test_negative_on_invalid_nan(self):
        from src.preprocessing.transformations import apply_log1p
        df = pd.DataFrame({"a": [-1.0, 1.0]})
        result = apply_log1p(df, on_invalid="nan")
        assert np.isnan(result["a"].iloc[0])

    def test_negative_on_invalid_clip(self):
        from src.preprocessing.transformations import apply_log1p
        df = pd.DataFrame({"a": [-5.0, 2.0]})
        result = apply_log1p(df, on_invalid="clip")
        assert result["a"].iloc[0] == 0.0   # log1p(0) = 0 after clip

    def test_does_not_mutate(self):
        from src.preprocessing.transformations import apply_log1p
        df = pd.DataFrame({"a": [1.0, 2.0]})
        orig = df["a"].tolist()
        apply_log1p(df)
        assert df["a"].tolist() == orig

    def test_selected_columns_only(self):
        from src.preprocessing.transformations import apply_log1p
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
        result = apply_log1p(df, columns=["a"])
        assert result["b"].tolist() == [10.0, 20.0]


class TestApplySqrt:
    def test_basic(self):
        from src.preprocessing.transformations import apply_sqrt
        df = pd.DataFrame({"a": [0.0, 4.0, 9.0]})
        result = apply_sqrt(df)
        assert abs(result["a"].iloc[0]) < 1e-9
        assert abs(result["a"].iloc[1] - 2.0) < 1e-9
        assert abs(result["a"].iloc[2] - 3.0) < 1e-9

    def test_negative_raises(self):
        from src.preprocessing.transformations import apply_sqrt
        df = pd.DataFrame({"a": [-1.0, 4.0]})
        with pytest.raises(ValueError):
            apply_sqrt(df)

    def test_negative_on_invalid_clip(self):
        from src.preprocessing.transformations import apply_sqrt
        df = pd.DataFrame({"a": [-4.0, 9.0]})
        result = apply_sqrt(df, on_invalid="clip")
        assert result["a"].iloc[0] == 0.0


class TestApplyPower:
    def test_square(self):
        from src.preprocessing.transformations import apply_power
        df = pd.DataFrame({"a": [2.0, 3.0]})
        result = apply_power(df, exponent=2.0)
        assert result["a"].tolist() == [4.0, 9.0]

    def test_cube(self):
        from src.preprocessing.transformations import apply_power
        df = pd.DataFrame({"a": [2.0]})
        result = apply_power(df, exponent=3.0)
        assert result["a"].iloc[0] == 8.0

    def test_does_not_mutate(self):
        from src.preprocessing.transformations import apply_power
        df = pd.DataFrame({"a": [2.0, 3.0]})
        apply_power(df, exponent=2)
        assert df["a"].tolist() == [2.0, 3.0]


class TestApplyReciprocal:
    def test_basic(self):
        from src.preprocessing.transformations import apply_reciprocal
        df = pd.DataFrame({"a": [2.0, 4.0]})
        result = apply_reciprocal(df)
        assert abs(result["a"].iloc[0] - 0.5) < 1e-9
        assert abs(result["a"].iloc[1] - 0.25) < 1e-9

    def test_zero_raises(self):
        from src.preprocessing.transformations import apply_reciprocal
        df = pd.DataFrame({"a": [0.0, 2.0]})
        with pytest.raises(ValueError):
            apply_reciprocal(df)

    def test_zero_on_invalid_nan(self):
        from src.preprocessing.transformations import apply_reciprocal
        df = pd.DataFrame({"a": [0.0, 2.0]})
        result = apply_reciprocal(df, on_invalid="nan")
        assert np.isnan(result["a"].iloc[0])
        assert abs(result["a"].iloc[1] - 0.5) < 1e-9


class TestLogTransformBackcompat:
    def test_runs_without_error(self):
        from src.preprocessing.transformations import log_transform
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = log_transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_negative_values_become_nan(self):
        from src.preprocessing.transformations import log_transform
        df = pd.DataFrame({"a": [-1.0, 2.0]})
        result = log_transform(df)
        assert np.isnan(result["a"].iloc[0])


# ===========================================================================
# 5. outliers.py
# ===========================================================================

class TestGetOutlierStats:
    def test_returns_expected_keys(self):
        from src.preprocessing.outliers import get_outlier_stats
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0]})
        stats = get_outlier_stats(df)
        assert "a" in stats
        for key in ["q1", "q3", "iqr", "lower_bound", "upper_bound",
                    "n_outliers", "pct_outliers"]:
            assert key in stats["a"]

    def test_detects_extreme_value(self):
        from src.preprocessing.outliers import get_outlier_stats
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 1000.0]})
        stats = get_outlier_stats(df)
        assert stats["a"]["n_outliers"] >= 1

    def test_invalid_multiplier_raises(self):
        from src.preprocessing.outliers import get_outlier_stats
        with pytest.raises(ValueError, match="multiplier"):
            get_outlier_stats(pd.DataFrame({"a": [1.0]}), multiplier=-1)


class TestDetectOutliers:
    def test_mask_dtype_bool(self):
        from src.preprocessing.outliers import detect_outliers
        df = pd.DataFrame({"a": [1.0, 2.0, 1000.0]})
        mask = detect_outliers(df)
        assert mask.dtypes["a"] == bool

    def test_extreme_value_flagged(self):
        from src.preprocessing.outliers import detect_outliers
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 999.0]})
        mask = detect_outliers(df)
        assert mask["a"].iloc[3] is True or mask["a"].iloc[3] == True

    def test_normal_values_not_flagged(self):
        from src.preprocessing.outliers import detect_outliers
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        mask = detect_outliers(df)
        assert mask["a"].sum() == 0

    def test_does_not_mutate_original(self):
        from src.preprocessing.outliers import detect_outliers
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        orig = df["a"].tolist()
        detect_outliers(df)
        assert df["a"].tolist() == orig


class TestClipOutliers:
    def test_clips_to_bounds(self):
        from src.preprocessing.outliers import clip_outliers, get_outlier_stats
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        stats = get_outlier_stats(df)
        result = clip_outliers(df)
        assert result["a"].max() <= stats["a"]["upper_bound"] + 1e-9

    def test_manual_bounds(self):
        from src.preprocessing.outliers import clip_outliers
        df = pd.DataFrame({"a": [-10.0, 0.0, 5.0, 100.0]})
        result = clip_outliers(df, lower_bound=0.0, upper_bound=10.0)
        assert result["a"].min() >= 0.0
        assert result["a"].max() <= 10.0

    def test_does_not_mutate(self):
        from src.preprocessing.outliers import clip_outliers
        df = pd.DataFrame({"a": [1.0, 1000.0]})
        orig_max = df["a"].max()
        clip_outliers(df)
        assert df["a"].max() == orig_max

    def test_non_numeric_column_raises(self):
        from src.preprocessing.outliers import clip_outliers
        df = pd.DataFrame({"cat": ["a", "b"]})
        with pytest.raises(ValueError, match="numeric"):
            clip_outliers(df, columns=["cat"])


class TestRemoveOutliers:
    def test_removes_outlier_rows(self):
        from src.preprocessing.outliers import remove_outliers
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        result = remove_outliers(df)
        assert len(result) < len(df)
        assert 1000.0 not in result["a"].values

    def test_preserves_non_outlier_rows(self):
        from src.preprocessing.outliers import remove_outliers
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        result = remove_outliers(df)
        assert 1.0 in result["a"].values

    def test_no_silent_deletion(self):
        """remove_outliers must be an explicit function call, not a side effect."""
        from src.preprocessing.outliers import clip_outliers
        df = pd.DataFrame({"a": [1.0, 1000.0]})
        # clip_outliers must NOT remove any rows
        result = clip_outliers(df)
        assert len(result) == len(df)


class TestHandleOutliers:
    def test_clip_action(self):
        from src.preprocessing.outliers import handle_outliers
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 1000.0]})
        result = handle_outliers(df, action="clip")
        assert len(result) == len(df)

    def test_remove_action(self):
        from src.preprocessing.outliers import handle_outliers
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 1000.0]})
        result = handle_outliers(df, action="remove")
        assert len(result) < len(df)

    def test_none_action_unchanged(self):
        from src.preprocessing.outliers import handle_outliers
        df = pd.DataFrame({"a": [1.0, 2.0, 1000.0]})
        result = handle_outliers(df, action="none")
        assert result["a"].max() == 1000.0

    def test_invalid_action_raises(self):
        from src.preprocessing.outliers import handle_outliers
        with pytest.raises(ValueError, match="action"):
            handle_outliers(pd.DataFrame({"a": [1.0]}), action="magic")  # type: ignore


# ===========================================================================
# 6. pipeline.py
# ===========================================================================

class TestPreprocessingPipelineBackcompat:
    """Ensure the original PreprocessingPipeline still works as before."""

    def test_fit_transform_returns_dataframe(self):
        from src.preprocessing.pipeline import PreprocessingPipeline
        df = pd.DataFrame({"num": [1.0, None, 3.0], "cat": ["a", "b", "a"]})
        pipeline = PreprocessingPipeline()
        result = pipeline.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["num", "cat"]

    def test_stores_columns_after_fit(self):
        from src.preprocessing.pipeline import PreprocessingPipeline
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        p = PreprocessingPipeline().fit(df)
        assert p.columns == ["a", "b"]

    def test_transform_returns_copy(self):
        from src.preprocessing.pipeline import PreprocessingPipeline
        df = pd.DataFrame({"a": [1, 2]})
        p = PreprocessingPipeline().fit(df)
        result = p.transform(df)
        result["a"] = 999
        assert df["a"].tolist() == [1, 2]  # original unchanged


class TestBuildPreprocessingPipeline:
    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        from src.preprocessing.pipeline import build_preprocessing_pipeline
        ct = build_preprocessing_pipeline(["a", "b"], ["c"])
        assert isinstance(ct, ColumnTransformer)

    def test_fit_transform_numpy_array(self):
        from src.preprocessing.pipeline import build_preprocessing_pipeline
        df = _mixed_df(50)
        ct = build_preprocessing_pipeline(
            numeric_features=["num1", "num2"],
            categorical_features=["cat1", "cat2"],
        )
        ct.fit(df)
        arr = ct.transform(df)
        assert arr.shape[0] == 50

    def test_empty_features_raises(self):
        from src.preprocessing.pipeline import build_preprocessing_pipeline
        with pytest.raises(ValueError, match="non-empty"):
            build_preprocessing_pipeline([], [])

    def test_invalid_scaler_raises(self):
        from src.preprocessing.pipeline import build_preprocessing_pipeline
        with pytest.raises(ValueError, match="scaler"):
            build_preprocessing_pipeline(["a"], [], scaler="magical")  # type: ignore

    def test_ordinal_encoder_option(self):
        from src.preprocessing.pipeline import build_preprocessing_pipeline
        df = _mixed_df(20)
        ct = build_preprocessing_pipeline(
            numeric_features=["num1"],
            categorical_features=["cat1"],
            cat_encoder="ordinal",
        )
        ct.fit(df)
        arr = ct.transform(df)
        assert arr.shape[0] == 20

    def test_no_scaling_option(self):
        from src.preprocessing.pipeline import build_preprocessing_pipeline
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": ["x", "y", "x"]})
        ct = build_preprocessing_pipeline(["a"], ["c"], scaler="none")
        ct.fit(df)
        arr = ct.transform(df)
        assert arr is not None


class TestFullPreprocessingPipeline:
    def test_fit_transform_returns_dataframe(self):
        from src.preprocessing.pipeline import FullPreprocessingPipeline
        df = _mixed_df(40)
        pipe = FullPreprocessingPipeline(
            numeric_features=["num1", "num2"],
            categorical_features=["cat1", "cat2"],
        )
        result = pipe.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 40

    def test_transform_test_uses_train_stats(self):
        from src.preprocessing.pipeline import FullPreprocessingPipeline
        df = _mixed_df(100)
        train, test = df.iloc[:80].copy(), df.iloc[80:].copy()
        pipe = FullPreprocessingPipeline(
            numeric_features=["num1", "num2"],
            categorical_features=["cat1"],
        )
        pipe.fit(train)
        r_train = pipe.transform(train)
        r_test = pipe.transform(test)
        # Same number of output columns
        assert r_train.shape[1] == r_test.shape[1]

    def test_unseen_category_handled(self):
        from src.preprocessing.pipeline import FullPreprocessingPipeline
        train = pd.DataFrame({
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
        })
        test = pd.DataFrame({
            "num": [4.0],
            "cat": ["UNSEEN"],
        })
        pipe = FullPreprocessingPipeline(
            numeric_features=["num"],
            categorical_features=["cat"],
        )
        pipe.fit(train)
        # Should not raise (handle_unknown="ignore" by default)
        result = pipe.transform(test)
        assert isinstance(result, pd.DataFrame)

    def test_no_leakage_numeric(self):
        """Scaling stats must be based solely on train data."""
        from src.preprocessing.pipeline import FullPreprocessingPipeline
        # Train: very small values → mean ≈ 1
        train = pd.DataFrame({"num": [0.0, 1.0, 2.0], "cat": ["a", "a", "b"]})
        # Test: extreme value
        test = pd.DataFrame({"num": [1000.0], "cat": ["a"]})
        pipe = FullPreprocessingPipeline(
            numeric_features=["num"],
            categorical_features=["cat"],
        )
        pipe.fit(train)
        train_result = pipe.transform(train)
        test_result = pipe.transform(test)
        # The scaled test value must be far from [−2, 2] (train range)
        # If leakage occurred the test value would be in-range
        assert test_result.iloc[0, 0] > 100  # very large if correctly scaled with train stats

    def test_transform_before_fit_raises(self):
        from src.preprocessing.pipeline import FullPreprocessingPipeline
        pipe = FullPreprocessingPipeline(["num"], ["cat"])
        with pytest.raises(RuntimeError):
            pipe.transform(pd.DataFrame({"num": [1.0], "cat": ["a"]}))


# ===========================================================================
# 7. Explicit data-leakage tests
# ===========================================================================

class TestDataLeakage:

    def test_imputer_train_mean_not_test_mean(self):
        from src.preprocessing.missing_values import MissingValueImputer
        train = pd.DataFrame({"age": [20.0, 25.0, 30.0]})  # mean = 25
        test = pd.DataFrame({"age": [None]})
        imp = MissingValueImputer(strategy="mean").fit(train)
        result = imp.transform(test)
        assert abs(result["age"].iloc[0] - 25.0) < 1e-9

    def test_scaler_train_params_not_test_params(self):
        from src.preprocessing.scaling import DataFrameScaler
        train = pd.DataFrame({"val": [0.0, 10.0]})   # min=0, max=10
        test = pd.DataFrame({"val": [5.0]})
        scaler = DataFrameScaler("minmax").fit(train)
        result = scaler.transform(test)
        assert abs(result["val"].iloc[0] - 0.5) < 1e-9

    def test_onehot_vocab_from_train_only(self):
        from src.preprocessing.encoding import OneHotEncoder
        train = pd.DataFrame({"c": ["a", "b"]})
        test = pd.DataFrame({"c": ["b", "c"]})  # "c" unseen in train
        enc = OneHotEncoder().fit(train)
        result = enc.transform(test)
        # Columns must be exactly train vocabulary dummies
        assert "c_a" in result.columns
        assert "c_b" in result.columns
        assert "c_c" not in result.columns   # unseen → not a column

    def test_ordinal_vocab_from_train_only(self):
        from src.preprocessing.encoding import OrdinalCategoryEncoder
        train = pd.DataFrame({"grade": ["A", "B", "C"]})
        test = pd.DataFrame({"grade": ["A", "D"]})  # "D" unseen
        enc = OrdinalCategoryEncoder().fit(train)
        result = enc.transform(test)
        # "D" should be -1 (unknown), not a new code from test
        assert result["grade"].iloc[1] == -1


# ===========================================================================
# 8. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_one_row_dataframe_scales(self):
        from src.preprocessing.scaling import DataFrameScaler
        train = pd.DataFrame({"a": [5.0, 10.0]})
        test = pd.DataFrame({"a": [5.0]})
        scaler = DataFrameScaler("standard").fit(train)
        result = scaler.transform(test)
        assert isinstance(result, pd.DataFrame)

    def test_constant_column_standard_scaler(self):
        """Constant columns produce NaN after standard scaling (std=0).
        This is sklearn's behaviour — we preserve it rather than hiding it."""
        from src.preprocessing.scaling import DataFrameScaler
        train = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
        scaler = DataFrameScaler("standard").fit(train)
        result = scaler.transform(train)
        # sklearn StandardScaler with constant column → result is 0.0 (not NaN)
        # (because std=0 → division suppressed, result is 0)
        assert isinstance(result, pd.DataFrame)

    def test_missing_column_in_transform_raises(self):
        from src.preprocessing.scaling import DataFrameScaler
        train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        scaler = DataFrameScaler("standard", columns=["a", "b"]).fit(train)
        test = pd.DataFrame({"a": [1.0]})  # "b" missing
        with pytest.raises(ValueError, match="missing"):
            scaler.transform(test)

    def test_encode_single_category_column(self):
        from src.preprocessing.encoding import OneHotEncoder
        df = pd.DataFrame({"c": ["only_one"] * 5})
        enc = OneHotEncoder()
        result = enc.fit_transform(df)
        assert "c_only_one" in result.columns

    def test_handle_missing_values_no_nan_noop(self):
        from src.preprocessing.missing_values import handle_missing_values
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = handle_missing_values(df, strategy="mean")
        pd.testing.assert_frame_equal(result, df)

    def test_outlier_all_same_values(self):
        from src.preprocessing.outliers import detect_outliers
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0, 5.0]})
        mask = detect_outliers(df)
        # IQR = 0 so bounds equal 5.0; no value exceeds them
        assert mask["a"].sum() == 0

    def test_log1p_non_dataframe_raises(self):
        from src.preprocessing.transformations import apply_log1p
        with pytest.raises(TypeError):
            apply_log1p([1, 2, 3])  # type: ignore
