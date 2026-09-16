# `src/data` — Data Utilities

Dataset loading, profiling, validation, sampling, and splitting.
All utilities are reusable across projects and competition pipelines.

---

## Status: ✅ IMPLEMENTED (Phase 1A)

---

## Modules

### `loader.py`
Load datasets from disk into `pandas.DataFrame`.

| Function | Description |
|---|---|
| `load_csv(path, ...)` | Load a CSV/TSV file with configurable options |
| `load_parquet(path, ...)` | Load a Parquet file (requires `pyarrow` or `fastparquet`) |
| `load_json(path, ...)` | Load a JSON / JSONL / NDJSON file |
| `load_dataset(path, ...)` | Auto-detect format from extension and load |

All loaders validate path existence, file extension, and non-empty result.

---

### `profiler.py`
Generate structured EDA summaries.

| Function | Description |
|---|---|
| `profile_dataset(df)` | Full profile: shape, dtypes, missing values, unique counts, column types, duplicates, numeric stats — JSON-serialisable |
| `describe_dataset(df)` | Lightweight 3-key summary (backward-compatible) |

---

### `sampler.py`
Reproducible row sampling.

| Function | Description |
|---|---|
| `sample_random(df, n=, frac=, random_state=)` | Random count or fraction sample |
| `sample_stratified(df, stratify_col, n=, frac=)` | Stratified sample preserving class proportions |
| `sample_rows(df, sample_size=)` | Backward-compatible helper |

---

### `splitter.py`
Train/test splitting without data leakage.

| Function | Description |
|---|---|
| `train_test_split_frame(df, target, test_size=, random_state=, stratify=)` | Random split (fixed from positional-slice bug) |
| `stratified_split(df, target, ...)` | Stratified split preserving class balance |
| `group_split(df, target, group_col, ...)` | Group-aware split (no group spans both sets) |
| `time_series_split(df, target, test_size=)` | Temporal split — no shuffling |

---

### `validator.py`
Pre-pipeline dataset validation.

| Function | Description |
|---|---|
| `validate_dataframe(df, min_rows=, min_cols=)` | Assert DataFrame is non-empty and well-formed |
| `validate_columns(df, required_columns)` | Assert required columns are present |
| `validate_target_column(df, target, allow_missing_values=)` | Assert target exists and has no NaNs |
| `validate_no_duplicates(df, subset=, raise_on_duplicates=)` | Count / raise on duplicate rows |
| `validate_dataset(df, ...)` | Full validation suite returning a report dict |

---

## Design Notes

- No function mutates the input DataFrame in-place.
- All stateful transformations use fit/transform semantics where applicable.
- `random_state` is always honoured for reproducibility.
- Time-series splits never shuffle temporal data.
