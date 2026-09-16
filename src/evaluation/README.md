# `src/evaluation` — Evaluation Module

A reusable library for evaluating Machine Learning models, producing structured metrics, and analyzing errors.

---

## Status: ✅ IMPLEMENTED (Phase 1D)

---

## Core Design Principles

1. **Structured Outputs**: Predictable dictionaries and dataclasses (`EvaluationResult`) instead of unstructured print statements.
2. **Duck Typing**: Evaluators work with any object implementing standard `predict`, `predict_proba`, or `decision_function` methods (e.g. `scikit-learn` estimators).
3. **No Mutations**: Error analysis and metric functions never mutate the provided arrays or DataFrames.
4. **Graceful Degradation**: If probabilities are not available, metrics like Accuracy and F1 still calculate successfully while ROC-AUC safely skips without crashing.

---

## Modules

### `metrics.py`

| API | Description |
|---|---|
| `classification_metrics(y_true, y_pred, y_prob, average)` | Calculates Accuracy, Precision, Recall, F1, and optionally Log Loss & ROC AUC. |
| `regression_metrics(y_true, y_pred)` | Calculates MAE, MSE, RMSE, R², and MAPE. |

### `evaluator.py`

| API | Description |
|---|---|
| `EvaluationResult` | Dataclass storing `metrics`, `y_true`, `y_pred`, and `y_prob`. |
| `ClassificationEvaluator(model)` | Generates classification predictions and returns an `EvaluationResult`. |
| `RegressionEvaluator(model)` | Generates regression predictions and returns an `EvaluationResult`. |

**Usage Example:**
```python
from src.evaluation import ClassificationEvaluator

# Assumes `model` is already fitted
evaluator = ClassificationEvaluator(model)
result = evaluator.evaluate(X_test, y_test)

print(result.metrics['accuracy'])
```

### `error_analysis.py`

| API | Description |
|---|---|
| `get_classification_errors(y_true, y_pred, df)` | Returns a DataFrame containing misclassified samples, optionally joining with original features. |
| `class_error_rates(y_true, y_pred)` | Returns a DataFrame of total counts and error rates per class. |
| `get_regression_errors(y_true, y_pred, df, top_n)` | Returns a DataFrame with residuals, absolute errors, and squared errors. |
| `residual_statistics(y_true, y_pred)` | Returns summary statistics (mean, std, min, max, percentiles) for residuals. |

### `plots.py`

Returns Matplotlib `Figure` and `Axes` objects without blocking execution (`plt.show()` is never called internally).

**Classification Plots:**
- `plot_confusion_matrix(y_true, y_pred, labels)`
- `plot_roc_curve(y_true, y_prob)` (Binary)
- `plot_precision_recall_curve(y_true, y_prob)` (Binary)

**Regression Plots:**
- `plot_actual_vs_predicted(y_true, y_pred)`
- `plot_residuals(y_true, y_pred)`
- `plot_residual_distribution(y_true, y_pred)`

---

## Planned Architecture

The following functionality is intended for future phases and is **NOT** currently implemented:
- Automated Model Comparison Engine
- Cross-Validation Report Generation
- HTML/PDF Report Exporting
- Explainability (SHAP, Permutation Importance) (See `src/explainability`)
