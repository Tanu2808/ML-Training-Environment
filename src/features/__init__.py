"""Feature engineering utilities.

Public API for the features module.
"""

# Numerical
from .numerical import (
    absolute_difference,
    add_binned_features,
    add_features,
    aggregate_features,
    bin_numeric_feature,
    log_feature,
    multiply_features,
    percentage_difference,
    ratio_feature,
    subtract_features,
)

# Categorical
from .categorical import (
    CountEncoder,
    FrequencyEncoder,
    RareCategoryGrouper,
    frequency_encode,
    normalize_categories,
)

# Datetime
from .datetime import (
    add_cyclical_features,
    extract_datetime_components,
    extract_datetime_features,
)

# Interactions
from .interactions import (
    create_interaction_features,
    create_interaction_terms,
)

# Text
from .text import (
    TfidfTransformer,
    basic_text_features,
    extract_text_features,
)

# Selection
from .selection import (
    CorrelationSelector,
    ModelBasedSelector,
    MutualInfoSelector,
    VarianceSelector,
    select_features,
)

__all__ = [
    # numerical
    "add_features",
    "subtract_features",
    "multiply_features",
    "ratio_feature",
    "absolute_difference",
    "percentage_difference",
    "aggregate_features",
    "bin_numeric_feature",
    "log_feature",
    "add_binned_features",
    # categorical
    "FrequencyEncoder",
    "CountEncoder",
    "RareCategoryGrouper",
    "normalize_categories",
    "frequency_encode",
    # datetime
    "extract_datetime_features",
    "add_cyclical_features",
    "extract_datetime_components",
    # interactions
    "create_interaction_features",
    "create_interaction_terms",
    # text
    "extract_text_features",
    "TfidfTransformer",
    "basic_text_features",
    # selection
    "VarianceSelector",
    "CorrelationSelector",
    "MutualInfoSelector",
    "ModelBasedSelector",
    "select_features",
]
