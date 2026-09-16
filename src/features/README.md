# Feature Engineering Layer

## Status
Phase 1C — COMPLETE

## Purpose
The feature engineering layer is responsible for creating new, predictive signals from existing data and selecting the most valuable features for modeling. It handles numerical manipulation, datetime extraction, cyclical encoding, text statistics, interaction terms, and statistical/model-based feature selection.

## Current Capabilities
- **Numerical**: Arithmetic operations, ratios, absolute/percentage differences, binning, log transformations.
- **Categorical**: Frequency encoding, count encoding, rare category grouping.
- **Datetime**: Extraction of year, month, day, hour, day of week, weekend flags; cyclical encoding for periodic time features.
- **Text**: Statistics extraction (char count, word count, digit count, average length) and TF-IDF matrix generation.
- **Interactions**: Automated pairwise interactions (multiply, divide, add, subtract).
- **Selection**: Filter-based (variance, correlation, mutual info) and wrapper-based (model feature importances).

## Module Structure

```text
src/features/
├── __init__.py
├── categorical.py
├── datetime.py
├── interactions.py
├── numerical.py
├── selection.py
├── text.py
└── README.md
```

## Public API

### `numerical.py`

#### `add_features(df: pd.DataFrame, col1: str, col2: str, name: str, ...) -> pd.DataFrame`
Adds two numeric columns.

#### `subtract_features(df: pd.DataFrame, col1: str, col2: str, name: str, ...) -> pd.DataFrame`
Subtracts `col2` from `col1`.

#### `multiply_features(df: pd.DataFrame, col1: str, col2: str, name: str, ...) -> pd.DataFrame`
Multiplies two numeric columns.

#### `ratio_feature(df: pd.DataFrame, num_col: str, den_col: str, name: str, ...) -> pd.DataFrame`
Computes the ratio `num_col / den_col`, safely handling zero division.

#### `absolute_difference(...)` / `percentage_difference(...)`
Calculates delta metrics between columns.

#### `aggregate_features(df: pd.DataFrame, columns: list[str], agg_func: str, name: str, ...) -> pd.DataFrame`
Aggregates multiple columns row-wise (e.g. `sum`, `mean`, `max`).

#### `bin_numeric_feature(df: pd.DataFrame, col: str, bins: int, name: str, ...) -> pd.DataFrame`
Discretizes a numeric feature into equal-width bins using `pd.cut`.

### `categorical.py`

#### `FrequencyEncoder(columns: list[str] | None = None)`
Leakage-safe encoder (`BaseEstimator`, `TransformerMixin`) that encodes categories by their normalized frequency (proportion) in the training set.

#### `CountEncoder(columns: list[str] | None = None)`
Leakage-safe encoder that encodes categories by their raw count in the training set.

#### `RareCategoryGrouper(columns: list[str] | None, threshold: float = 0.05, replace_with: str = "Rare")`
Leakage-safe transformer that groups rare categories (appearing less than `threshold` frequency) into a single category.

### `datetime.py`

#### `extract_datetime_features(df: pd.DataFrame, column: str, features: list[str], ...) -> pd.DataFrame`
Extracts components such as `year`, `month`, `day`, `dayofweek`, `hour`, `is_weekend` from a datetime column.

#### `add_cyclical_features(df: pd.DataFrame, column: str, max_val: float) -> pd.DataFrame`
Creates `sin` and `cos` transformations of periodic time features (e.g. month, hour) to preserve cyclical distance.

### `interactions.py`

#### `create_interaction_features(df: pd.DataFrame, columns: list[str], operations: list[str], ...) -> pd.DataFrame`
Creates pairwise combinations of the specified `columns` using operations from `['multiply', 'add', 'subtract', 'divide']`.

### `text.py`

#### `extract_text_features(df: pd.DataFrame, column: str, features: list[str], ...) -> pd.DataFrame`
Extracts text statistics: `char_count`, `word_count`, `digit_count`, `uppercase_count`, `lowercase_count`, `avg_word_length`, `unique_word_count`.

#### `TfidfTransformer(column: str, max_features: int = 100, ...)`
Leakage-safe TF-IDF vectorizer that expands a text column into multiple TF-IDF score columns.

### `selection.py`

#### `VarianceSelector(threshold: float = 0.0)`
Leakage-safe selector that drops numeric features with variance below `threshold`.

#### `CorrelationSelector(threshold: float = 0.9)`
Leakage-safe selector that drops highly collinear features (keeping the first one seen).

#### `MutualInfoSelector(k: int = 10, task: str = "classification")`
Leakage-safe selector that keeps the top `k` features based on Mutual Information with the target. Requires calling `fit(X, y)`.

#### `ModelBasedSelector(estimator, max_features: int | None = None)`
Leakage-safe selector utilizing an estimator's `feature_importances_` or `coef_` attribute to select the top features.

#### `select_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame`
Stateless convenience function to subset columns.

## Usage Examples

```python
from src.features.selection import MutualInfoSelector
from src.features.categorical import RareCategoryGrouper
from src.features.numerical import ratio_feature

# Create simple ratio feature
df = ratio_feature(df, num_col="income", den_col="debt", name="income_debt_ratio")

# Group rare categories safely based on training data
grouper = RareCategoryGrouper(columns=["city"], threshold=0.01)
train_df = grouper.fit_transform(train_df)
test_df = grouper.transform(test_df)

# Select top 5 features based on mutual info
selector = MutualInfoSelector(k=5, task="classification")
train_X_selected = selector.fit_transform(train_X, train_y)
test_X_selected = selector.transform(test_X)
```

## Dependencies
- `pandas`
- `numpy`
- `scikit-learn` (for TF-IDF, BaseEstimator/TransformerMixin, mutual_info)

## Testing
Tested via `tests/test_phase1c_features.py`. Validates numerical math, category frequency leakage avoidance, time feature calculations, cyclical math, text parsing, and correct subsetting in selectors.

## Not Implemented
- Deep NLP features (e.g. Word2Vec, BERT embeddings)
- Auto-Feature engineering frameworks (e.g. Featuretools)
- Sequence/Time-series lagging utilities
