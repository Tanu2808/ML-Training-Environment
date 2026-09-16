# `src/preprocessing` — Preprocessing Utilities

Missing-value imputation, categorical encoding, numerical scaling, transformations,
outlier handling, and reusable sklearn-compatible pipelines.

---

## Status: ✅ IMPLEMENTED (Phase 1B)

---

## Core Design Principles

- **Never mutates the input DataFrame** — all functions return a new DataFrame.
- **Leakage-safe** — stateful transformers expose `fit(train)` → `transform(test)`.  Statistics are **never** computed from test/validation data.
- **sklearn-compatible** — all stateful classes implement `BaseEstimator + TransformerMixin`.

---

## Modules

### `missing_values.py`

| API | Type | Description |
|---|---|---|
| `handle_missing_values(df, strategy, columns, fill_value, drop_threshold)` | Function | Stateless imputation (8 strategies) |
| `MissingValueImputer(strategy, columns, fill_value)` | Transformer | Leakage-safe fit/transform imputer |

**Strategies:** `"mean"`, `"median"`, `"mode"`, `"constant"`, `"ffill"`, `"bfill"`, `"drop_rows"`, `"drop_cols"`

```python
imp = MissingValueImputer(strategy="mean")
X_train_imp = imp.fit_transform(X_train)
X_test_imp  = imp.transform(X_test)   # uses train mean, not test mean
```

---

### `encoding.py`

| API | Type | Description |
|---|---|---|
| `OneHotEncoder(columns, drop_first, handle_unknown, sparse_output)` | Transformer | OHE with unknown-category handling |
| `OrdinalCategoryEncoder(columns, categories, handle_unknown, unknown_value)` | Transformer | Ordinal encoding with explicit ordering |
| `one_hot_encode(df, columns, drop_first)` | Function | Stateless OHE (fits on df itself) |

```python
enc = OneHotEncoder(columns=["city"], handle_unknown="ignore")
enc.fit(X_train)
X_train_enc = enc.transform(X_train)
X_test_enc  = enc.transform(X_test)  # unknown cities → all-zero row
```

---

### `scaling.py`

| API | Type | Description |
|---|---|---|
| `DataFrameScaler(scaler_type, columns)` | Transformer | Wraps StandardScaler / MinMaxScaler / RobustScaler |
| `make_standard_scaler(columns)` | Factory | `DataFrameScaler("standard")` |
| `make_minmax_scaler(columns, feature_range)` | Factory | `DataFrameScaler("minmax")` |
| `make_robust_scaler(columns)` | Factory | `DataFrameScaler("robust")` |
| `standardize(df, columns)` | Function | Stateless standard scaling |

```python
scaler = DataFrameScaler("standard")
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)  # uses train mean/std
```

---

### `transformations.py`

| API | Type | Description |
|---|---|---|
| `apply_log1p(df, columns, on_invalid)` | Function | `log(1+x)`, validates x ≥ 0 |
| `apply_sqrt(df, columns, on_invalid)` | Function | `√x`, validates x ≥ 0 |
| `apply_power(df, exponent, columns)` | Function | `xⁿ` |
| `apply_reciprocal(df, columns, on_invalid)` | Function | `1/x`, validates x ≠ 0 |
| `log_transform(df, columns)` | Function | Backward-compatible alias for `apply_log1p(..., on_invalid="nan")` |

`on_invalid` accepts: `"raise"` (default), `"nan"`, `"clip"`.

---

### `outliers.py`

| API | Type | Description |
|---|---|---|
| `get_outlier_stats(df, columns, multiplier)` | Function | IQR bounds + counts per column |
| `detect_outliers(df, columns, multiplier)` | Function | Boolean mask of outlier positions |
| `clip_outliers(df, columns, multiplier, lower_bound, upper_bound)` | Function | Clip to IQR bounds |
| `remove_outliers(df, columns, multiplier)` | Function | **Explicit opt-in** row removal |
| `handle_outliers(df, columns, action, multiplier)` | Function | Dispatcher: `"clip"`, `"remove"`, `"none"` |

Default IQR multiplier: **1.5** (Tukey's rule).

> **Important:** `remove_outliers` is always an **explicit** call — clipping never silently removes rows.

---

### `pipeline.py`

| API | Type | Description |
|---|---|---|
| `PreprocessingPipeline` | Class | Original no-op pipeline (backward compatible) |
| `build_preprocessing_pipeline(numeric_features, categorical_features, ...)` | Factory | Returns a sklearn `ColumnTransformer` |
| `FullPreprocessingPipeline(numeric_features, categorical_features, ...)` | Wrapper | Pandas-friendly fit/transform returning DataFrames |

```python
pipe = FullPreprocessingPipeline(
    numeric_features=["age", "salary"],
    categorical_features=["city", "gender"],
    scaler="standard",            # "standard" | "minmax" | "robust" | "none"
    cat_encoder="onehot",         # "onehot" | "ordinal"
    numeric_impute_strategy="median",
)
X_train_out = pipe.fit_transform(X_train)
X_test_out  = pipe.transform(X_test)    # no leakage
```

---

## Leakage Prevention

All stateful transformers:

1. Learn statistics **only** from training data during `fit()`.
2. Apply those statistics unchanged to validation/test data during `transform()`.
3. Never call `fit()` on test data.

Proven by dedicated `TestDataLeakage` test class covering imputation, scaling, and encoding.
