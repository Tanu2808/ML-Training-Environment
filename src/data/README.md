# Data Layer

## Status
Phase 1A — COMPLETE

## Purpose
The data layer is responsible for loading datasets from disk, profiling their statistical properties, sampling them effectively, validating their structure, and splitting them for training, validation, and testing.

## Current Capabilities
- Loading CSV, Parquet, and JSON formats.
- Generating statistical profiles of datasets.
- Random and stratified data sampling.
- Standard, stratified, grouped, and time-series train/test splitting.
- Validating DataFrames for structure, expected columns, missing values, target columns, and duplicate rows.

## Module Structure

```text
src/data/
├── __init__.py
├── loader.py
├── profiler.py
├── sampler.py
├── splitter.py
├── validator.py
└── README.md
```

## Public API

### `loader.py`

#### `load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame`
Loads a CSV file into a pandas DataFrame.

#### `load_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame`
Loads a Parquet file into a pandas DataFrame.

#### `load_json(path: str | Path, orient: str = "records", **kwargs: Any) -> pd.DataFrame`
Loads a JSON file into a pandas DataFrame.

#### `load_dataset(path: str | Path, **kwargs: Any) -> pd.DataFrame`
Automatically detects file extension (`.csv`, `.parquet`, `.json`) and routes to the correct loader.

### `profiler.py`

#### `profile_dataset(df: pd.DataFrame) -> dict[str, Any]`
Generates a comprehensive statistical profile containing: shape, memory usage, duplicates, missing values, and column-wise statistics.

#### `describe_dataset(df: pd.DataFrame) -> dict[str, Any]`
Provides a simplified structural description of the dataset focusing on features, targets, categorical vs. numerical column counts.

### `sampler.py`

#### `sample_random(df: pd.DataFrame, n: int | None = None, frac: float | None = None, random_state: int | None = None) -> pd.DataFrame`
Returns a random subset of rows using either an exact count (`n`) or a fraction (`frac`).

#### `sample_stratified(df: pd.DataFrame, stratify_col: str, n: int | None = None, frac: float | None = None, random_state: int | None = None) -> pd.DataFrame`
Returns a random subset of rows while preserving the class distribution of `stratify_col`.

#### `sample_rows(df: pd.DataFrame, sample_size: int | None = None) -> pd.DataFrame`
Simplistic wrapper to sample the top `sample_size` rows from the DataFrame.

### `splitter.py`

#### `train_test_split_frame(df: pd.DataFrame, test_size: float = 0.2, random_state: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]`
Splits the DataFrame randomly into train and test sets.

#### `stratified_split(df: pd.DataFrame, stratify_col: str, test_size: float = 0.2, random_state: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]`
Splits the DataFrame while maintaining the distribution of `stratify_col`.

#### `group_split(df: pd.DataFrame, group_col: str, test_size: float = 0.2, random_state: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]`
Splits the DataFrame such that all rows with the same group ID (in `group_col`) end up in the same split (train or test), avoiding data leakage across grouped records.

#### `time_series_split(df: pd.DataFrame, time_col: str, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]`
Sorts the DataFrame by `time_col` and splits chronologically (train comes strictly before test).

### `validator.py`

#### `validate_dataframe(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1) -> None`
Ensures the object is a pandas DataFrame and meets minimum shape requirements.

#### `validate_columns(df: pd.DataFrame, expected_columns: list[str], exact_match: bool = False) -> None`
Checks if `expected_columns` exist in the DataFrame.

#### `validate_target_column(df: pd.DataFrame, target_col: str, task: str = "classification") -> None`
Validates that the target column exists and contains valid types for the given ML task.

#### `validate_no_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> None`
Checks for exact duplicate rows.

#### `validate_dataset(df: pd.DataFrame, ...)`
Aggregates the above checks into a single comprehensive validation pipeline.

## Usage Examples

```python
from src.data.loader import load_dataset
from src.data.splitter import stratified_split
from src.data.validator import validate_dataset

# Load dataset
df = load_dataset("data/raw/data.csv")

# Validate
validate_dataset(df, expected_columns=["age", "income"], target_col="churn")

# Split securely while stratifying the target
train_df, test_df = stratified_split(df, stratify_col="churn", test_size=0.2, random_state=42)
```

## Dependencies
- `pandas`
- `numpy`
- `scikit-learn` (for splitting strategies in `splitter.py`)

## Testing
Tested via `tests/test_phase1a_data.py`. Covers format loading, validation checks, stratification ratios, time-series continuity, and leakage avoidance in grouped splits.

## Not Implemented
- Automatic dataset profiling reports (e.g. HTML generation like Pandas Profiling).
- Robust out-of-core data validation (e.g. Great Expectations).
- Loading directly from databases or object storage (S3/GCS).
