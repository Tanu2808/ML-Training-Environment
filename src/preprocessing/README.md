# Preprocessing Layer

## Status
Phase 1B — COMPLETE

## Purpose
The preprocessing layer is responsible for preparing raw data for machine learning models. It handles missing values, encodes categorical variables, scales numerical features, detects/handles outliers, and applies mathematical transformations.

## Current Capabilities
- Imputation of missing values via mean, median, most_frequent, or constant strategies.
- One-hot and ordinal encoding of categorical features.
- Standardization, min-max scaling, and robust scaling.
- Outlier detection via IQR or Z-score methods, supporting clipping or removal.
- Mathematical transformations (log, square root, power, reciprocal).
- Extensible `sklearn`-compatible transformers (`BaseEstimator`, `TransformerMixin`) for train/test data leakage safety.
- Preprocessing pipelines (`PreprocessingPipeline`) to string steps together.

## Module Structure

```text
src/preprocessing/
├── __init__.py
├── encoding.py
├── missing_values.py
├── outliers.py
├── pipeline.py
├── scaling.py
├── transformations.py
└── README.md
```

## Public API

### `missing_values.py`

#### `MissingValueImputer(strategy: str = "mean", fill_value: Any = None, columns: list[str] | None = None)`
Leakage-safe imputer that learns statistics (e.g., mean) on the training set during `fit()` and applies them to both train and test during `transform()`.

#### `handle_missing_values(df: pd.DataFrame, strategy: str = "mean", ...)`
Stateless convenience function to fill missing values (caution: not strictly leakage-safe for test sets if using mean/median without prior fitting).

### `encoding.py`

#### `OneHotEncoder(columns: list[str] | None = None, drop_first: bool = False, handle_unknown: str = "ignore")`
Leakage-safe one-hot encoder that learns categories during `fit()` and gracefully handles unknown categories during `transform()`.

#### `OrdinalCategoryEncoder(columns: list[str] | None = None, handle_unknown: str = "use_encoded_value", unknown_value: int = -1)`
Leakage-safe ordinal integer encoder.

#### `one_hot_encode(df: pd.DataFrame, columns: list[str], drop_first: bool = False) -> pd.DataFrame`
Stateless convenience function (wraps `pd.get_dummies`).

### `scaling.py`

#### `DataFrameScaler(scaler, columns: list[str] | None = None)`
A wrapper that applies a given sklearn scaler (like `StandardScaler`) to a pandas DataFrame and returns a DataFrame (preserving columns and indices).

#### `make_standard_scaler(columns: list[str] | None = None) -> DataFrameScaler`
Creates a `DataFrameScaler` backed by `sklearn.preprocessing.StandardScaler`.

#### `make_minmax_scaler(columns: list[str] | None = None) -> DataFrameScaler`
Creates a `DataFrameScaler` backed by `sklearn.preprocessing.MinMaxScaler`.

#### `make_robust_scaler(columns: list[str] | None = None) -> DataFrameScaler`
Creates a `DataFrameScaler` backed by `sklearn.preprocessing.RobustScaler`.

#### `standardize(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame`
Stateless standard scaler (Z-score normalization).

### `transformations.py`

#### `apply_log1p(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame`
Applies `log(1 + x)` to numeric columns.

#### `apply_sqrt(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame`
Applies square root transformation.

#### `apply_power(df: pd.DataFrame, columns: list[str], power: float) -> pd.DataFrame`
Applies power transformation (`x ** power`).

#### `apply_reciprocal(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame`
Applies `1 / x` transformation (handling zero division).

#### `log_transform(df: pd.DataFrame, columns: list[str], offset: float = 1.0) -> pd.DataFrame`
Applies `log(x + offset)` transformation.

### `outliers.py`

#### `get_outlier_stats(df: pd.DataFrame, columns: list[str] | None = None, method: str = "iqr", threshold: float = 1.5) -> dict`
Calculates upper and lower boundaries based on IQR or Z-score.

#### `detect_outliers(df: pd.DataFrame, columns: list[str], method: str = "iqr", threshold: float = 1.5) -> pd.Series`
Returns a boolean mask of rows that contain at least one outlier.

#### `clip_outliers(df: pd.DataFrame, columns: list[str], lower: dict, upper: dict) -> pd.DataFrame`
Clips values to the provided pre-computed boundaries.

#### `remove_outliers(df: pd.DataFrame, columns: list[str], method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame`
Removes rows containing outliers.

#### `handle_outliers(df: pd.DataFrame, columns: list[str] | None = None, method: str = "iqr", threshold: float = 1.5, action: str = "clip") -> pd.DataFrame`
A convenience wrapper to compute bounds and apply either clipping or dropping.

### `pipeline.py`

#### `PreprocessingPipeline(steps: list[tuple[str, TransformerMixin]])`
A pipeline builder acting as a wrapper over `sklearn.pipeline.Pipeline` but guaranteeing `pd.DataFrame` output.

#### `build_preprocessing_pipeline(num_cols: list[str], cat_cols: list[str], impute_num: str = "mean", impute_cat: str = "most_frequent", scale: bool = True) -> BaseEstimator`
Builds a standard pipeline (imputation + scaling + encoding) mapped to numerical and categorical column subsets.

## Usage Examples

```python
from src.preprocessing.pipeline import build_preprocessing_pipeline

# Create a leakage-safe preprocessing pipeline
pipeline = build_preprocessing_pipeline(
    num_cols=["age", "balance"], 
    cat_cols=["education", "job"], 
    impute_num="median", 
    scale=True
)

# Fit on training data
train_processed = pipeline.fit_transform(train_df)

# Transform test data (using parameters learned from train_df)
test_processed = pipeline.transform(test_df)
```

## Dependencies
- `pandas`
- `numpy`
- `scikit-learn`

## Testing
Tested via `tests/test_phase1b_preprocessing.py`. Covers data leakage prevention for imputers and encoders, index/column preservation, and outlier threshold math.

## Not Implemented
- Target encoding (e.g. Mean Encoding)
- Discretization (K-Bins)
- Advanced imputation (KNNImputer, IterativeImputer)
- Feature extraction via PCA/SVD
