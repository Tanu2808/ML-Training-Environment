"""Phase 1C tests: feature engineering — numerical, categorical, datetime,
interactions, text, selection, leakage proofs, and edge cases.
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
        "b": rng.standard_normal(n) * 2 + 5,
        "c": rng.integers(1, 50, n).astype(float),
    })


def _cat_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "city": rng.choice(["NYC", "LA", "CHI"], n),
        "grade": rng.choice(["A", "B", "C", "D"], n),
        "num": rng.standard_normal(n),
    })


# ===========================================================================
# 1. numerical.py
# ===========================================================================

class TestAddFeatures:
    def test_result_correct(self):
        from src.features.numerical import add_features
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = add_features(df, "a", "b")
        assert result["a_plus_b"].tolist() == [4.0, 6.0]

    def test_default_output_name(self):
        from src.features.numerical import add_features
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        result = add_features(df, "x", "y")
        assert "x_plus_y" in result.columns

    def test_custom_output_name(self):
        from src.features.numerical import add_features
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        result = add_features(df, "x", "y", output_col="sum_xy")
        assert "sum_xy" in result.columns

    def test_does_not_mutate(self):
        from src.features.numerical import add_features
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        orig_cols = list(df.columns)
        add_features(df, "a", "b")
        assert list(df.columns) == orig_cols

    def test_missing_column_raises(self):
        from src.features.numerical import add_features
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="not found"):
            add_features(df, "a", "missing")

    def test_overwrite_guard(self):
        from src.features.numerical import add_features
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "a_plus_b": [99.0]})
        with pytest.raises(ValueError, match="already exists"):
            add_features(df, "a", "b")

    def test_overwrite_allowed(self):
        from src.features.numerical import add_features
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "a_plus_b": [99.0]})
        result = add_features(df, "a", "b", overwrite=True)
        assert result["a_plus_b"].iloc[0] == 3.0


class TestSubtractFeatures:
    def test_result_correct(self):
        from src.features.numerical import subtract_features
        df = pd.DataFrame({"a": [5.0, 3.0], "b": [2.0, 1.0]})
        result = subtract_features(df, "a", "b")
        assert result["a_minus_b"].tolist() == [3.0, 2.0]

    def test_does_not_mutate(self):
        from src.features.numerical import subtract_features
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        subtract_features(df, "a", "b")
        assert "a_minus_b" not in df.columns


class TestMultiplyFeatures:
    def test_result_correct(self):
        from src.features.numerical import multiply_features
        df = pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]})
        result = multiply_features(df, "a", "b")
        assert result["a_times_b"].tolist() == [8.0, 15.0]


class TestRatioFeature:
    def test_basic_division(self):
        from src.features.numerical import ratio_feature
        df = pd.DataFrame({"a": [10.0, 6.0], "b": [2.0, 3.0]})
        result = ratio_feature(df, "a", "b")
        assert result["a_div_b"].tolist() == [5.0, 2.0]

    def test_zero_denom_raises(self):
        from src.features.numerical import ratio_feature
        df = pd.DataFrame({"a": [1.0], "b": [0.0]})
        with pytest.raises(ValueError, match="zero"):
            ratio_feature(df, "a", "b", on_zero_denom="raise")

    def test_zero_denom_nan(self):
        from src.features.numerical import ratio_feature
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 2.0]})
        result = ratio_feature(df, "a", "b", on_zero_denom="nan")
        assert np.isnan(result["a_div_b"].iloc[0])
        assert result["a_div_b"].iloc[1] == 1.0

    def test_zero_denom_fill(self):
        from src.features.numerical import ratio_feature
        df = pd.DataFrame({"a": [1.0], "b": [0.0]})
        result = ratio_feature(df, "a", "b", on_zero_denom="fill", fill_value=-99.0)
        assert result["a_div_b"].iloc[0] == -99.0


class TestAbsoluteDifference:
    def test_result_correct(self):
        from src.features.numerical import absolute_difference
        df = pd.DataFrame({"a": [1.0, 5.0], "b": [3.0, 2.0]})
        result = absolute_difference(df, "a", "b")
        assert result["abs_a_minus_b"].tolist() == [2.0, 3.0]


class TestPercentageDifference:
    def test_result_correct(self):
        from src.features.numerical import percentage_difference
        df = pd.DataFrame({"a": [110.0], "b": [100.0]})
        result = percentage_difference(df, "a", "b")
        assert abs(result["pct_diff_a_b"].iloc[0] - 10.0) < 1e-9

    def test_zero_denom_default_nan(self):
        from src.features.numerical import percentage_difference
        df = pd.DataFrame({"a": [5.0], "b": [0.0]})
        result = percentage_difference(df, "a", "b")
        assert np.isnan(result["pct_diff_a_b"].iloc[0])


class TestAggregateFeatures:
    def test_sum(self):
        from src.features.numerical import aggregate_features
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = aggregate_features(df, ["x", "y"], method="sum")
        assert result.iloc[0]["x_y_sum"] == 4.0

    def test_mean(self):
        from src.features.numerical import aggregate_features
        df = pd.DataFrame({"x": [2.0, 4.0], "y": [4.0, 6.0]})
        result = aggregate_features(df, ["x", "y"], method="mean")
        assert result.iloc[0]["x_y_mean"] == 3.0

    def test_range(self):
        from src.features.numerical import aggregate_features
        df = pd.DataFrame({"x": [1.0], "y": [5.0]})
        result = aggregate_features(df, ["x", "y"], method="range")
        assert result["x_y_range"].iloc[0] == 4.0

    def test_invalid_method_raises(self):
        from src.features.numerical import aggregate_features
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        with pytest.raises(ValueError, match="method"):
            aggregate_features(df, ["x", "y"], method="median")  # type: ignore

    def test_empty_columns_raises(self):
        from src.features.numerical import aggregate_features
        df = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError, match="non-empty"):
            aggregate_features(df, [], method="sum")


class TestBinNumericFeature:
    def test_uniform_bins(self):
        from src.features.numerical import bin_numeric_feature
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = bin_numeric_feature(df, "a", n_bins=5, strategy="uniform")
        assert "a_bin" in result.columns
        assert result["a_bin"].notna().all()

    def test_quantile_bins(self):
        from src.features.numerical import bin_numeric_feature
        df = pd.DataFrame({"a": list(range(1, 11))})
        result = bin_numeric_feature(df, "a", n_bins=2, strategy="quantile")
        assert "a_bin" in result.columns

    def test_n_bins_less_than_2_raises(self):
        from src.features.numerical import bin_numeric_feature
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError):
            bin_numeric_feature(df, "a", n_bins=1)

    def test_does_not_mutate(self):
        from src.features.numerical import bin_numeric_feature
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        orig_cols = list(df.columns)
        bin_numeric_feature(df, "a", n_bins=2)
        assert list(df.columns) == orig_cols


class TestLogFeature:
    def test_basic(self):
        from src.features.numerical import log_feature
        df = pd.DataFrame({"a": [0.0, 1.0]})
        result = log_feature(df, "a")
        assert abs(result["log1p_a"].iloc[0]) < 1e-9
        assert abs(result["log1p_a"].iloc[1] - np.log(2)) < 1e-9

    def test_negative_raises(self):
        from src.features.numerical import log_feature
        df = pd.DataFrame({"a": [-1.0, 2.0]})
        with pytest.raises(ValueError, match="negative"):
            log_feature(df, "a", on_invalid="raise")

    def test_negative_nan(self):
        from src.features.numerical import log_feature
        df = pd.DataFrame({"a": [-1.0, 2.0]})
        result = log_feature(df, "a", on_invalid="nan")
        assert np.isnan(result["log1p_a"].iloc[0])

    def test_does_not_mutate(self):
        from src.features.numerical import log_feature
        df = pd.DataFrame({"a": [1.0, 2.0]})
        orig = df["a"].tolist()
        log_feature(df, "a")
        assert df["a"].tolist() == orig


# ===========================================================================
# 2. categorical.py
# ===========================================================================

class TestFrequencyEncoder:
    def test_basic_fit_transform(self):
        from src.features.categorical import FrequencyEncoder
        df = pd.DataFrame({"c": ["a", "a", "b"]})
        enc = FrequencyEncoder()
        result = enc.fit(df).transform(df)
        assert abs(result["c"].iloc[0] - 2/3) < 1e-9
        assert abs(result["c"].iloc[2] - 1/3) < 1e-9

    def test_leakage_proof(self):
        """Frequencies must come from training data only."""
        from src.features.categorical import FrequencyEncoder
        train = pd.DataFrame({"c": ["A", "A", "B"]})   # A=2/3, B=1/3
        test = pd.DataFrame({"c": ["A", "C"]})          # C is unseen
        enc = FrequencyEncoder(unknown_value=0.0).fit(train)
        result = enc.transform(test)
        # A → train freq (2/3)
        assert abs(result["c"].iloc[0] - 2/3) < 1e-9
        # C → unknown_value (0.0)
        assert result["c"].iloc[1] == 0.0

    def test_unknown_category_fill(self):
        from src.features.categorical import FrequencyEncoder
        train = pd.DataFrame({"c": ["a", "b"]})
        test = pd.DataFrame({"c": ["a", "UNSEEN"]})
        enc = FrequencyEncoder(unknown_value=-1.0).fit(train)
        result = enc.transform(test)
        assert result["c"].iloc[1] == -1.0

    def test_does_not_mutate(self):
        from src.features.categorical import FrequencyEncoder
        df = pd.DataFrame({"c": ["a", "b"]})
        orig = df["c"].tolist()
        FrequencyEncoder().fit(df).transform(df)
        assert df["c"].tolist() == orig

    def test_transform_before_fit_raises(self):
        from src.features.categorical import FrequencyEncoder
        enc = FrequencyEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(pd.DataFrame({"c": ["a"]}))

    def test_multiple_columns(self):
        from src.features.categorical import FrequencyEncoder
        df = _cat_df()
        enc = FrequencyEncoder(columns=["city", "grade"])
        result = enc.fit_transform(df)
        assert result["city"].dtype == float
        assert result["grade"].dtype == float
        assert result["num"].dtype == float  # preserved


class TestCountEncoder:
    def test_basic(self):
        from src.features.categorical import CountEncoder
        train = pd.DataFrame({"c": ["a", "a", "b"]})
        enc = CountEncoder().fit(train)
        result = enc.transform(train)
        assert result["c"].iloc[0] == 2
        assert result["c"].iloc[2] == 1

    def test_unknown_returns_zero(self):
        from src.features.categorical import CountEncoder
        train = pd.DataFrame({"c": ["a", "b"]})
        test = pd.DataFrame({"c": ["a", "UNSEEN"]})
        enc = CountEncoder().fit(train)
        result = enc.transform(test)
        assert result["c"].iloc[1] == 0

    def test_train_counts_not_test(self):
        from src.features.categorical import CountEncoder
        train = pd.DataFrame({"c": ["A", "A", "A"]})  # A count = 3
        test = pd.DataFrame({"c": ["A", "A"]})        # A count in test = 2 (irrelevant)
        enc = CountEncoder().fit(train)
        result = enc.transform(test)
        assert result["c"].iloc[0] == 3  # train count

    def test_transform_before_fit_raises(self):
        from src.features.categorical import CountEncoder
        enc = CountEncoder()
        with pytest.raises(RuntimeError):
            enc.transform(pd.DataFrame({"c": ["a"]}))


class TestRareCategoryGrouper:
    def test_rare_replaced(self):
        from src.features.categorical import RareCategoryGrouper, RARE_LABEL
        train = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 10 + ["rare_val"]})
        grouper = RareCategoryGrouper(min_frequency=2, min_frequency_frac=None).fit(train)
        result = grouper.transform(train)
        assert RARE_LABEL in result["c"].values
        assert "rare_val" not in result["c"].values

    def test_frequent_categories_preserved(self):
        from src.features.categorical import RareCategoryGrouper
        train = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 5 + ["c"] * 1})
        grouper = RareCategoryGrouper(min_frequency=3, min_frequency_frac=None).fit(train)
        result = grouper.transform(train)
        assert "a" in result["c"].values
        assert "b" in result["c"].values

    def test_frequency_frac_threshold(self):
        from src.features.categorical import RareCategoryGrouper, RARE_LABEL
        train = pd.DataFrame({"c": ["a"] * 90 + ["b"] * 10})
        # min_frequency_frac=0.15 → threshold = 15 → b is rare
        grouper = RareCategoryGrouper(
            min_frequency=None, min_frequency_frac=0.15
        ).fit(train)
        result = grouper.transform(train)
        assert RARE_LABEL in result["c"].values

    def test_train_statistics_only(self):
        from src.features.categorical import RareCategoryGrouper, RARE_LABEL
        train = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 1})
        test = pd.DataFrame({"c": ["a", "b", "NEW"]})
        grouper = RareCategoryGrouper(min_frequency=3, min_frequency_frac=None).fit(train)
        result = grouper.transform(test)
        assert result["c"].iloc[0] == "a"
        assert result["c"].iloc[1] == RARE_LABEL  # b was rare in train
        assert result["c"].iloc[2] == RARE_LABEL  # NEW is unknown → rare

    def test_does_not_mutate(self):
        from src.features.categorical import RareCategoryGrouper
        df = pd.DataFrame({"c": ["a", "b", "c"]})
        orig = df["c"].tolist()
        RareCategoryGrouper(min_frequency=2, min_frequency_frac=None).fit(df).transform(df)
        assert df["c"].tolist() == orig


class TestNormalizeCategories:
    def test_strip_whitespace(self):
        from src.features.categorical import normalize_categories
        df = pd.DataFrame({"c": ["  hello  ", "world "]})
        result = normalize_categories(df, columns=["c"])
        assert result["c"].iloc[0] == "hello"

    def test_lowercase(self):
        from src.features.categorical import normalize_categories
        df = pd.DataFrame({"c": ["Hello", "WORLD"]})
        result = normalize_categories(df, columns=["c"], lowercase=True)
        assert result["c"].tolist() == ["hello", "world"]

    def test_nan_preserved(self):
        from src.features.categorical import normalize_categories
        df = pd.DataFrame({"c": ["  hello  ", None]})
        result = normalize_categories(df, columns=["c"])
        assert pd.isna(result["c"].iloc[1])

    def test_does_not_mutate(self):
        from src.features.categorical import normalize_categories
        df = pd.DataFrame({"c": ["  hello  "]})
        normalize_categories(df)
        assert df["c"].iloc[0] == "  hello  "


class TestFrequencyEncodeFunctional:
    def test_returns_dataframe(self):
        from src.features.categorical import frequency_encode
        df = pd.DataFrame({"c": ["a", "a", "b"]})
        result = frequency_encode(df)
        assert isinstance(result, pd.DataFrame)
        assert result["c"].dtype == float


# ===========================================================================
# 3. datetime.py
# ===========================================================================

class TestExtractDatetimeFeatures:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ts": pd.to_datetime([
                "2023-06-15 14:30:45",
                "2023-01-01 00:00:00",
                "2023-12-31 23:59:59",
            ])
        })

    def test_year(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(self._df(), "ts", ["year"])
        assert result["ts_year"].tolist() == [2023, 2023, 2023]

    def test_month(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(self._df(), "ts", ["month"])
        assert result["ts_month"].tolist() == [6, 1, 12]

    def test_day(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(self._df(), "ts", ["day"])
        assert result["ts_day"].tolist() == [15, 1, 31]

    def test_day_of_week(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(self._df(), "ts", ["day_of_week"])
        # 2023-06-15 is Thursday (3)
        assert result["ts_day_of_week"].iloc[0] == 3

    def test_hour(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(self._df(), "ts", ["hour"])
        assert result["ts_hour"].tolist() == [14, 0, 23]

    def test_quarter(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(self._df(), "ts", ["quarter"])
        assert result["ts_quarter"].tolist() == [2, 1, 4]

    def test_is_weekend(self):
        from src.features.datetime import extract_datetime_features
        # 2023-01-01 is Sunday (6) → weekend
        result = extract_datetime_features(self._df(), "ts", ["is_weekend"])
        assert result["ts_is_weekend"].iloc[1] == 1

    def test_is_month_start(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(self._df(), "ts", ["is_month_start"])
        assert result["ts_is_month_start"].iloc[1] == 1  # Jan 1

    def test_is_month_end(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(self._df(), "ts", ["is_month_end"])
        assert result["ts_is_month_end"].iloc[2] == 1  # Dec 31

    def test_week_of_year(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(self._df(), "ts", ["week_of_year"])
        assert "ts_week_of_year" in result.columns

    def test_custom_prefix(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(self._df(), "ts", ["year"], prefix="dt")
        assert "dt_year" in result.columns

    def test_drop_original(self):
        from src.features.datetime import extract_datetime_features
        result = extract_datetime_features(
            self._df(), "ts", ["year"], drop_original=True
        )
        assert "ts" not in result.columns

    def test_string_dates_parsed(self):
        from src.features.datetime import extract_datetime_features
        df = pd.DataFrame({"d": ["2023-03-15", "2023-07-04"]})
        result = extract_datetime_features(df, "d", ["month"])
        assert result["d_month"].tolist() == [3, 7]

    def test_invalid_date_coerced_to_nat(self):
        from src.features.datetime import extract_datetime_features
        df = pd.DataFrame({"d": ["2023-01-01", "not-a-date"]})
        result = extract_datetime_features(df, "d", ["year"], on_error="coerce")
        assert pd.isna(result["d_year"].iloc[1])

    def test_unknown_component_raises(self):
        from src.features.datetime import extract_datetime_features
        df = pd.DataFrame({"d": ["2023-01-01"]})
        with pytest.raises(ValueError, match="Unknown components"):
            extract_datetime_features(df, "d", ["nonexistent_field"])

    def test_does_not_mutate(self):
        from src.features.datetime import extract_datetime_features
        df = self._df()
        orig_cols = list(df.columns)
        extract_datetime_features(df, "ts", ["year"])
        assert list(df.columns) == orig_cols


class TestAddCyclicalFeatures:
    def test_hour_encoding(self):
        from src.features.datetime import add_cyclical_features
        df = pd.DataFrame({"hour": [0, 6, 12, 18]})
        result = add_cyclical_features(df, "hour", period=24)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        # hour=0 → sin=0, cos=1
        assert abs(result["hour_sin"].iloc[0]) < 1e-9
        assert abs(result["hour_cos"].iloc[0] - 1.0) < 1e-9

    def test_month_encoding(self):
        from src.features.datetime import add_cyclical_features
        df = pd.DataFrame({"month": [1, 7, 12]})
        result = add_cyclical_features(df, "month", period=12)
        assert "month_sin" in result.columns

    def test_invalid_period_raises(self):
        from src.features.datetime import add_cyclical_features
        df = pd.DataFrame({"h": [0, 6]})
        with pytest.raises(ValueError, match="period"):
            add_cyclical_features(df, "h", period=0)

    def test_overwrite_guard(self):
        from src.features.datetime import add_cyclical_features
        df = pd.DataFrame({"h": [0], "h_sin": [0.0]})
        with pytest.raises(ValueError, match="already exists"):
            add_cyclical_features(df, "h", period=24)

    def test_custom_prefix(self):
        from src.features.datetime import add_cyclical_features
        df = pd.DataFrame({"hour": [6]})
        result = add_cyclical_features(df, "hour", period=24, prefix="hr")
        assert "hr_sin" in result.columns


# ===========================================================================
# 4. interactions.py
# ===========================================================================

class TestCreateInteractionFeatures:
    def test_multiply(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [2.0], "b": [3.0]})
        result = create_interaction_features(df, ["a", "b"], ["multiply"])
        assert result["a__x__b"].iloc[0] == 6.0

    def test_add(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [2.0], "b": [3.0]})
        result = create_interaction_features(df, ["a", "b"], ["add"])
        assert result["a__plus__b"].iloc[0] == 5.0

    def test_subtract(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [5.0], "b": [3.0]})
        result = create_interaction_features(df, ["a", "b"], ["subtract"])
        assert result["a__minus__b"].iloc[0] == 2.0

    def test_ratio(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [6.0], "b": [3.0]})
        result = create_interaction_features(df, ["a", "b"], ["ratio"])
        assert result["a__div__b"].iloc[0] == 2.0

    def test_three_columns_all_pairs(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        result = create_interaction_features(df, ["a", "b", "c"], ["multiply"])
        # pairs: (a,b), (a,c), (b,c) → 3 new columns
        new_cols = [c for c in result.columns if c not in df.columns]
        assert len(new_cols) == 3

    def test_single_column_raises(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="at least 2"):
            create_interaction_features(df, ["a"], ["multiply"])

    def test_invalid_operation_raises(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(ValueError, match="Unknown operations"):
            create_interaction_features(df, ["a", "b"], ["power"])  # type: ignore

    def test_name_collision_raises(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "a__x__b": [0.0]})
        with pytest.raises(ValueError, match="already exists"):
            create_interaction_features(df, ["a", "b"], ["multiply"])

    def test_zero_ratio_nan(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [1.0], "b": [0.0]})
        result = create_interaction_features(
            df, ["a", "b"], ["ratio"], on_zero_denom="nan"
        )
        assert np.isnan(result["a__div__b"].iloc[0])

    def test_does_not_mutate(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        orig_cols = list(df.columns)
        create_interaction_features(df, ["a", "b"], ["multiply"])
        assert list(df.columns) == orig_cols

    def test_multiple_operations(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"x": [2.0], "y": [3.0]})
        result = create_interaction_features(
            df, ["x", "y"], ["multiply", "add", "subtract"]
        )
        new_cols = [c for c in result.columns if c not in df.columns]
        assert len(new_cols) == 3  # one pair × 3 ops


# ===========================================================================
# 5. text.py
# ===========================================================================

class TestExtractTextFeatures:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "text": [
                "Hello World 123",
                "  spaces  ",
                "",
                None,
                "ONE TWO three",
            ]
        })

    def test_char_count(self):
        from src.features.text import extract_text_features
        result = extract_text_features(self._df(), "text", features=["char_count"])
        assert result["text_char_count"].iloc[0] == 15
        assert result["text_char_count"].iloc[2] == 0
        assert pd.isna(result["text_char_count"].iloc[3])

    def test_word_count(self):
        from src.features.text import extract_text_features
        result = extract_text_features(self._df(), "text", features=["word_count"])
        assert result["text_word_count"].iloc[0] == 3
        assert result["text_word_count"].iloc[2] == 0

    def test_digit_count(self):
        from src.features.text import extract_text_features
        result = extract_text_features(self._df(), "text", features=["digit_count"])
        assert result["text_digit_count"].iloc[0] == 3

    def test_uppercase_count(self):
        from src.features.text import extract_text_features
        result = extract_text_features(self._df(), "text", features=["uppercase_count"])
        assert result["text_uppercase_count"].iloc[0] == 2  # H, W

    def test_lowercase_count(self):
        from src.features.text import extract_text_features
        result = extract_text_features(self._df(), "text", features=["lowercase_count"])
        # "Hello World 123": 'ello' (4) + 'orld' (4) = 8 lowercase letters
        assert result["text_lowercase_count"].iloc[0] == 8

    def test_nan_produces_nan(self):
        from src.features.text import extract_text_features
        result = extract_text_features(self._df(), "text", features=["word_count"])
        assert pd.isna(result["text_word_count"].iloc[3])

    def test_empty_string_word_count_zero(self):
        from src.features.text import extract_text_features
        result = extract_text_features(self._df(), "text", features=["word_count"])
        assert result["text_word_count"].iloc[2] == 0

    def test_avg_word_length(self):
        from src.features.text import extract_text_features
        df = pd.DataFrame({"t": ["ab cde"]})
        result = extract_text_features(df, "t", features=["avg_word_length"])
        # words: ["ab", "cde"] → (2+3)/2 = 2.5
        assert abs(result["t_avg_word_length"].iloc[0] - 2.5) < 1e-9

    def test_unique_word_count(self):
        from src.features.text import extract_text_features
        df = pd.DataFrame({"t": ["the cat sat on the mat"]})
        result = extract_text_features(df, "t", features=["unique_word_count"])
        assert result["t_unique_word_count"].iloc[0] == 5  # the/cat/sat/on/mat

    def test_custom_prefix(self):
        from src.features.text import extract_text_features
        df = pd.DataFrame({"body": ["hello world"]})
        result = extract_text_features(df, "body", prefix="msg", features=["word_count"])
        assert "msg_word_count" in result.columns

    def test_unknown_feature_raises(self):
        from src.features.text import extract_text_features
        df = pd.DataFrame({"t": ["hello"]})
        with pytest.raises(ValueError, match="Unknown features"):
            extract_text_features(df, "t", features=["sentiment"])

    def test_does_not_mutate(self):
        from src.features.text import extract_text_features
        df = pd.DataFrame({"t": ["hello world"]})
        orig = df["t"].iloc[0]
        extract_text_features(df, "t")
        assert df["t"].iloc[0] == orig


class TestTfidfTransformer:
    def test_fit_transform_shape(self):
        from src.features.text import TfidfTransformer
        train = pd.DataFrame({"text": ["hello world", "foo bar", "hello foo"]})
        tfidf = TfidfTransformer("text", max_features=10)
        result = tfidf.fit_transform(train)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3
        assert result.shape[1] <= 10

    def test_transform_test_uses_train_vocab(self):
        from src.features.text import TfidfTransformer
        train = pd.DataFrame({"text": ["hello world", "foo bar"]})
        test = pd.DataFrame({"text": ["hello unknown_word"]})
        tfidf = TfidfTransformer("text", max_features=50)
        tfidf.fit(train)
        r_train = tfidf.transform(train)
        r_test = tfidf.transform(test)
        assert list(r_train.columns) == list(r_test.columns)

    def test_transform_before_fit_raises(self):
        from src.features.text import TfidfTransformer
        tfidf = TfidfTransformer("text")
        with pytest.raises(RuntimeError):
            tfidf.transform(pd.DataFrame({"text": ["hello"]}))

    def test_nan_handled(self):
        from src.features.text import TfidfTransformer
        train = pd.DataFrame({"text": ["hello world", None]})
        tfidf = TfidfTransformer("text", max_features=10)
        result = tfidf.fit_transform(train)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# 6. selection.py
# ===========================================================================

class TestVarianceSelector:
    def test_removes_constant(self):
        from src.features.selection import VarianceSelector
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "const": [5.0, 5.0, 5.0, 5.0],
        })
        sel = VarianceSelector(threshold=0.0)
        result = sel.fit(df).transform(df)
        assert "const" not in result.columns
        assert "a" in result.columns

    def test_threshold_filtering(self):
        from src.features.selection import VarianceSelector
        df = pd.DataFrame({
            "high_var": [0.0, 10.0, 0.0, 10.0],
            "low_var":  [1.0, 1.01, 0.99, 1.0],
        })
        sel = VarianceSelector(threshold=1.0).fit(df)
        result = sel.transform(df)
        assert "high_var" in result.columns
        assert "low_var" not in result.columns

    def test_train_test_consistency(self):
        from src.features.selection import VarianceSelector
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"a": rng.standard_normal(20), "b": [1.0] * 20})
        test = pd.DataFrame({"a": rng.standard_normal(10), "b": [1.0] * 10})
        sel = VarianceSelector(threshold=0.0).fit(train)
        r_test = sel.transform(test)
        assert "b" not in r_test.columns

    def test_transform_before_fit_raises(self):
        from src.features.selection import VarianceSelector
        sel = VarianceSelector()
        with pytest.raises(RuntimeError):
            sel.transform(pd.DataFrame({"a": [1.0]}))

    def test_selected_features_property(self):
        from src.features.selection import VarianceSelector
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 1.0, 1.0]})
        sel = VarianceSelector().fit(df)
        assert "a" in sel.selected_features
        assert "b" not in sel.selected_features


class TestCorrelationSelector:
    def test_removes_correlated(self):
        from src.features.selection import CorrelationSelector
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        df = pd.DataFrame({"a": x, "b": x * 1.001})  # almost perfectly correlated
        sel = CorrelationSelector(threshold=0.99).fit(df)
        result = sel.transform(df)
        assert len([c for c in ["a", "b"] if c in result.columns]) == 1

    def test_low_correlation_kept(self):
        from src.features.selection import CorrelationSelector
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a": rng.standard_normal(50),
            "b": rng.standard_normal(50),
        })
        sel = CorrelationSelector(threshold=0.95).fit(df)
        result = sel.transform(df)
        # Both should survive (random normals typically have low correlation)
        assert "a" in result.columns
        assert "b" in result.columns

    def test_train_test_consistency(self):
        from src.features.selection import CorrelationSelector
        rng = np.random.default_rng(0)
        x = rng.standard_normal(40)
        train = pd.DataFrame({"a": x, "b": x})
        test = pd.DataFrame({"a": rng.standard_normal(10), "b": rng.standard_normal(10)})
        sel = CorrelationSelector(threshold=0.99).fit(train)
        r_train = sel.transform(train)
        r_test = sel.transform(test)
        assert list(r_train.columns) == list(r_test.columns)

    def test_dropped_features_property(self):
        from src.features.selection import CorrelationSelector
        rng = np.random.default_rng(0)
        x = rng.standard_normal(50)
        df = pd.DataFrame({"a": x, "b": x})
        sel = CorrelationSelector(threshold=0.99).fit(df)
        assert len(sel.dropped_features) > 0


class TestMutualInfoSelector:
    def test_regression_selects_k(self):
        from src.features.selection import MutualInfoSelector
        rng = np.random.default_rng(0)
        X = pd.DataFrame({f"f{i}": rng.standard_normal(50) for i in range(5)})
        y = X["f0"] + rng.standard_normal(50) * 0.1  # f0 is the informative feature
        sel = MutualInfoSelector(k=2, task="regression", random_state=42).fit(X, y)
        result = sel.transform(X)
        assert result.shape[1] == 2

    def test_classification_selects_k(self):
        from src.features.selection import MutualInfoSelector
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"a": rng.standard_normal(40), "b": rng.standard_normal(40)})
        y = pd.Series(rng.integers(0, 2, 40))
        sel = MutualInfoSelector(k=1, task="classification").fit(X, y)
        result = sel.transform(X)
        assert result.shape[1] == 1

    def test_no_target_raises(self):
        from src.features.selection import MutualInfoSelector
        X = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="target"):
            MutualInfoSelector().fit(X, y=None)

    def test_train_test_consistency(self):
        from src.features.selection import MutualInfoSelector
        rng = np.random.default_rng(0)
        X = pd.DataFrame({f"f{i}": rng.standard_normal(40) for i in range(4)})
        y = X["f0"]
        train, test = X.iloc[:30], X.iloc[30:]
        sel = MutualInfoSelector(k=2, random_state=0).fit(train, y.iloc[:30])
        r_train = sel.transform(train)
        r_test = sel.transform(test)
        assert list(r_train.columns) == list(r_test.columns)


class TestModelBasedSelector:
    def test_selects_top_k(self):
        from sklearn.ensemble import RandomForestClassifier
        from src.features.selection import ModelBasedSelector
        rng = np.random.default_rng(0)
        X = pd.DataFrame({
            "signal": rng.standard_normal(80),
            "noise1": rng.standard_normal(80),
            "noise2": rng.standard_normal(80),
        })
        y = pd.Series((X["signal"] > 0).astype(int))
        sel = ModelBasedSelector(RandomForestClassifier(n_estimators=20, random_state=0), k=1)
        result = sel.fit(X, y).transform(X)
        assert result.shape[1] == 1

    def test_transform_before_fit_raises(self):
        from sklearn.ensemble import RandomForestRegressor
        from src.features.selection import ModelBasedSelector
        sel = ModelBasedSelector(RandomForestRegressor())
        with pytest.raises(RuntimeError):
            sel.transform(pd.DataFrame({"a": [1.0]}))


class TestSelectFeatures:
    def test_subsets_columns(self):
        from src.features.selection import select_features
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = select_features(df, ["a", "c"])
        assert list(result.columns) == ["a", "c"]

    def test_missing_column_raises(self):
        from src.features.selection import select_features
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            select_features(df, ["a", "z"])


# ===========================================================================
# 7. Explicit leakage tests
# ===========================================================================

class TestFeatureLeakage:

    def test_frequency_encoder_uses_train_freq_only(self):
        from src.features.categorical import FrequencyEncoder
        # Train: A appears 2/3, B appears 1/3
        train = pd.DataFrame({"c": ["A", "A", "B"]})
        # Test: A appears only once out of 2 → freq would be 1/2 if computed on test
        test = pd.DataFrame({"c": ["A", "C"]})
        enc = FrequencyEncoder().fit(train)
        result = enc.transform(test)
        # A must use TRAIN freq (2/3), not test freq (1/2)
        assert abs(result["c"].iloc[0] - 2/3) < 1e-9

    def test_count_encoder_uses_train_count_only(self):
        from src.features.categorical import CountEncoder
        train = pd.DataFrame({"c": ["A", "A", "A"]})  # A count = 3 in train
        test = pd.DataFrame({"c": ["A"]})              # A count = 1 in test
        enc = CountEncoder().fit(train)
        result = enc.transform(test)
        assert result["c"].iloc[0] == 3  # train count, not test count

    def test_variance_selector_uses_train_variance(self):
        from src.features.selection import VarianceSelector
        # Constant in train → dropped even if variable in test
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
        test = pd.DataFrame({"a": [1.0], "b": [99.0]})  # b varies in test
        sel = VarianceSelector(threshold=0.0).fit(train)
        result = sel.transform(test)
        assert "b" not in result.columns  # dropped based on TRAIN variance

    def test_mutual_info_uses_train_only(self):
        from src.features.selection import MutualInfoSelector
        rng = np.random.default_rng(0)
        X_train = pd.DataFrame({"a": rng.standard_normal(30), "b": rng.standard_normal(30)})
        y_train = X_train["a"] + rng.standard_normal(30) * 0.01
        X_test = pd.DataFrame({"a": rng.standard_normal(10), "b": rng.standard_normal(10)})
        sel = MutualInfoSelector(k=1, random_state=0).fit(X_train, y_train)
        r_train = sel.transform(X_train)
        r_test = sel.transform(X_test)
        # Column selection must be identical
        assert list(r_train.columns) == list(r_test.columns)


# ===========================================================================
# 8. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_dataframe_add(self):
        from src.features.numerical import add_features
        df = pd.DataFrame({"a": pd.Series([], dtype=float),
                           "b": pd.Series([], dtype=float)})
        result = add_features(df, "a", "b")
        assert len(result) == 0
        assert "a_plus_b" in result.columns

    def test_one_row_multiply(self):
        from src.features.numerical import multiply_features
        df = pd.DataFrame({"a": [3.0], "b": [4.0]})
        result = multiply_features(df, "a", "b")
        assert result["a_times_b"].iloc[0] == 12.0

    def test_all_nan_frequency_encode(self):
        from src.features.categorical import FrequencyEncoder
        train = pd.DataFrame({"c": [None, None]})
        enc = FrequencyEncoder(handle_nan="encode").fit(train)
        result = enc.transform(train)
        assert isinstance(result, pd.DataFrame)

    def test_rare_grouper_no_rare_categories(self):
        from src.features.categorical import RareCategoryGrouper, RARE_LABEL
        train = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 10})
        grouper = RareCategoryGrouper(
            min_frequency=2, min_frequency_frac=None
        ).fit(train)
        result = grouper.transform(train)
        assert RARE_LABEL not in result["c"].values

    def test_datetime_missing_values(self):
        from src.features.datetime import extract_datetime_features
        df = pd.DataFrame({"ts": pd.to_datetime(["2023-01-01", None])})  # type: ignore[arg-type]
        result = extract_datetime_features(df, "ts", ["year"])
        assert pd.isna(result["ts_year"].iloc[1])

    def test_interaction_preserves_non_interacted_cols(self):
        from src.features.interactions import create_interaction_features
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "cat": ["x"]})
        result = create_interaction_features(df, ["a", "b"], ["multiply"])
        assert "cat" in result.columns

    def test_text_features_non_string_values(self):
        from src.features.text import extract_text_features
        df = pd.DataFrame({"t": [42, 3.14, None, "hello"]})
        result = extract_text_features(df, "t", features=["char_count"])
        assert isinstance(result, pd.DataFrame)

    def test_variance_selector_single_feature(self):
        from src.features.selection import VarianceSelector
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        sel = VarianceSelector(threshold=0.0).fit(df)
        result = sel.transform(df)
        assert "a" in result.columns

    def test_ratio_no_zero_denom(self):
        from src.features.numerical import ratio_feature
        df = pd.DataFrame({"a": [6.0, 8.0], "b": [2.0, 4.0]})
        result = ratio_feature(df, "a", "b")
        assert result["a_div_b"].tolist() == [3.0, 2.0]
