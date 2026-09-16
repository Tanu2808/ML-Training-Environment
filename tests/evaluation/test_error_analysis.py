import pytest
import numpy as np
import pandas as pd

from src.evaluation.error_analysis import (
    get_classification_errors,
    class_error_rates,
    get_regression_errors,
    residual_statistics
)


def test_get_classification_errors():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    
    # Simple arrays
    df_err = get_classification_errors(y_true, y_pred)
    assert len(df_err) == 1
    assert df_err.iloc[0]["y_true"] == 1
    assert df_err.iloc[0]["y_pred"] == 0
    
    # With features
    df = pd.DataFrame({"feat": [10, 20, 30, 40]})
    df_err_features = get_classification_errors(y_true, y_pred, df)
    assert len(df_err_features) == 1
    assert "feat" in df_err_features.columns
    assert df_err_features.iloc[0]["feat"] == 20
    
    # Invalid length
    with pytest.raises(ValueError, match="length"):
        get_classification_errors(y_true, [0, 1])


def test_class_error_rates():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    
    rates = class_error_rates(y_true, y_pred)
    assert len(rates) == 3
    assert rates.loc[0, "errors"] == 1
    assert rates.loc[0, "total"] == 2
    assert rates.loc[0, "error_rate"] == 0.5
    
    assert rates.loc[1, "errors"] == 0
    assert rates.loc[1, "error_rate"] == 0.0
    
    assert rates.loc[2, "errors"] == 1
    assert rates.loc[2, "error_rate"] == 1.0


def test_get_regression_errors():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 30])
    
    errs = get_regression_errors(y_true, y_pred)
    assert len(errs) == 3
    np.testing.assert_array_equal(errs["residual"].values, np.array([-2, 2, 0]))
    np.testing.assert_array_equal(errs["abs_error"].values, np.array([2, 2, 0]))
    np.testing.assert_array_equal(errs["sq_error"].values, np.array([4, 4, 0]))
    
    # Top N
    errs_top = get_regression_errors(y_true, y_pred, top_n=1)
    assert len(errs_top) == 1
    assert errs_top.iloc[0]["abs_error"] == 2
    
    # With features
    df = pd.DataFrame({"feat": [1, 2, 3]})
    errs_feat = get_regression_errors(y_true, y_pred, df)
    assert "feat" in errs_feat.columns


def test_residual_statistics():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 30])
    
    stats = residual_statistics(y_true, y_pred)
    assert "mean" in stats
    assert "std" in stats
    assert stats["mean"] == 0.0  # -2, 2, 0
    assert stats["min"] == -2.0
    assert stats["max"] == 2.0
    
    # Empty inputs
    assert residual_statistics([], []) == {}
