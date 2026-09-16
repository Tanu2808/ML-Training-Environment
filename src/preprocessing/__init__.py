"""Preprocessing utilities.

Public API for the preprocessing module.
"""

from .encoding import (
    OneHotEncoder,
    OrdinalCategoryEncoder,
    one_hot_encode,
)
from .missing_values import (
    MissingValueImputer,
    handle_missing_values,
)
from .outliers import (
    clip_outliers,
    detect_outliers,
    get_outlier_stats,
    handle_outliers,
    remove_outliers,
)
from .pipeline import (
    FullPreprocessingPipeline,
    PreprocessingPipeline,
    build_preprocessing_pipeline,
)
from .scaling import (
    DataFrameScaler,
    make_minmax_scaler,
    make_robust_scaler,
    make_standard_scaler,
    standardize,
)
from .transformations import (
    apply_log1p,
    apply_power,
    apply_reciprocal,
    apply_sqrt,
    log_transform,
)

__all__ = [
    # missing_values
    "handle_missing_values",
    "MissingValueImputer",
    # encoding
    "OneHotEncoder",
    "OrdinalCategoryEncoder",
    "one_hot_encode",
    # scaling
    "DataFrameScaler",
    "make_standard_scaler",
    "make_minmax_scaler",
    "make_robust_scaler",
    "standardize",
    # transformations
    "apply_log1p",
    "apply_sqrt",
    "apply_power",
    "apply_reciprocal",
    "log_transform",
    # outliers
    "get_outlier_stats",
    "detect_outliers",
    "clip_outliers",
    "remove_outliers",
    "handle_outliers",
    # pipeline
    "PreprocessingPipeline",
    "build_preprocessing_pipeline",
    "FullPreprocessingPipeline",
]
