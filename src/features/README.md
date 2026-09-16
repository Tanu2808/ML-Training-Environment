# Feature Engineering Module

The `src/features/` module provides a comprehensive, reusable, and leakage-safe collection of feature engineering utilities designed for tabular ML tasks.

## Design Principles

1. **Leakage Prevention**: All stateful transformations (e.g., frequency encoding, variance selection) use `scikit-learn`-compatible `fit`/`transform` semantics. They calculate statistics **only** on the training data and apply those learned parameters to validation/test sets without recalculating.
2. **Immutability**: Input pandas DataFrames are never mutated silently. New columns are appended to a copy of the DataFrame.
3. **Explicit Targeting**: Column names are always explicitly provided to transformations. There is no implicit mutation of existing features.
4. **Resilience**: Tools handle edge cases like zero division, unparsed dates, empty dataframes, and missing values safely without crashing. 

## Currently Available

### Numerical Features (`numerical.py`)
- Basic Arithmetic: `add_features`, `subtract_features`, `multiply_features`
- Ratios & Differences: `ratio_feature` (with zero-denominator handling), `absolute_difference`, `percentage_difference`
- Aggregation: `aggregate_features` (sum, mean, max, min, median, std, range across multiple columns)
- Binning: `bin_numeric_feature` (uniform or quantile binning)
- Transformations: `log_feature` (safe `log1p` handling)

### Categorical Features (`categorical.py`)
- String Normalization: `normalize_categories` (strip whitespace, lowercase)
- Stateful Encoders (Leakage-Safe):
  - `FrequencyEncoder`: Replaces categories with their relative frequency (calculated strictly from the training set). Unseen categories in test sets default to 0 (or a configurable fallback).
  - `CountEncoder`: Replaces categories with their absolute count in the training set.
  - `RareCategoryGrouper`: Groups low-frequency categories into a single `__RARE__` label. Configurable via absolute minimum counts or percentage thresholds.

### Datetime Features (`datetime.py`)
- Calendar Components: `extract_datetime_features` (extract year, month, day, day_of_week, quarter, hour, is_weekend, is_month_start, etc.)
- Periodic Encoding: `add_cyclical_features` (transforms periodic values like hour or month into sine/cosine pairs to preserve circular topology).

### Feature Interactions (`interactions.py`)
- Controlled Pairwise Combinations: `create_interaction_features` (generates specified interactions — multiply, add, subtract, ratio — strictly for a provided list of columns to avoid feature explosion).

### Text Features (`text.py`)
- Vectorized Statistics: `extract_text_features` (char count, word count, digit count, uppercase count, punctuation count, average word length, unique word count).
- Stateful NLP: `TfidfTransformer` (Leakage-safe wrapper over `scikit-learn`'s TF-IDF vectorizer that preserves non-text columns).

### Feature Selection (`selection.py`)
All selectors inherit from `BaseEstimator, TransformerMixin` to guarantee leakage safety:
- `VarianceSelector`: Removes features with low variance.
- `CorrelationSelector`: Drops highly correlated features to reduce multicollinearity.
- `MutualInfoSelector`: Retains the top-k features based on mutual information with the target (classification or regression).
- `ModelBasedSelector`: Selects features using `feature_importances_` from a tree-based estimator (e.g., Random Forest).
- Stateless Subsetting: `select_features` (simple column subsetting).

## Examples

### Leakage-Safe Categorical Encoding

```python
from src.features import FrequencyEncoder

# Fit ONLY on training data
encoder = FrequencyEncoder(columns=["city", "grade"], unknown_value=0.0)
encoder.fit(X_train)

# Transform both safely
X_train_enc = encoder.transform(X_train)
X_test_enc = encoder.transform(X_test)  # Unseen categories get 0.0
```

### Chained Numerical & Interaction Engineering

```python
from src.features import ratio_feature, create_interaction_features

# Safe ratio (no division by zero issues)
df = ratio_feature(df, numerator="clicks", denominator="impressions", on_zero_denom="nan")

# Explicit pairwise interactions (multiply & add)
df = create_interaction_features(
    df, 
    columns=["age", "income", "credit_score"], 
    operations=["multiply", "add"]
)
```

### Feature Selection in a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.features import VarianceSelector, MutualInfoSelector

pipeline = Pipeline([
    ('drop_constant', VarianceSelector(threshold=0.0)),
    ('top_10_mi', MutualInfoSelector(k=10, task="classification")),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```
