# Evaluation Layer

## Status
Phase 1D — COMPLETE

## Purpose
The evaluation layer computes metrics, packages results, enables detailed error analysis, and visualizes model performance for both classification and regression tasks.

## Current Capabilities
- **Metrics**: Standard arrays of classification and regression metrics.
- **Evaluators**: Unified evaluator classes that wrap predictions and metrics into a standardized `EvaluationResult`.
- **Error Analysis**: Extraction of top misclassified instances, per-class error rates, and regression residuals.
- **Plots**: Matplotlib-based visualizations including confusion matrices, ROC curves, PR curves, residual plots, and actual vs. predicted distributions.

## Module Structure

```text
src/evaluation/
├── __init__.py
├── error_analysis.py
├── evaluator.py
├── metrics.py
├── plots.py
└── README.md
```

## Public API

### `metrics.py`

#### `classification_metrics(y_true, y_pred, y_prob=None, average="macro") -> dict`
Calculates `accuracy`, `precision`, `recall`, `f1`, and `confusion_matrix`. If `y_prob` is provided, includes `roc_auc` and `log_loss`.

#### `regression_metrics(y_true, y_pred) -> dict`
Calculates `mae`, `mse`, `rmse`, `r2`, and `mape`.

### `evaluator.py`

#### `EvaluationResult`
A dataclass representing the output of an evaluation, storing `metrics` (dict), `y_true`, `y_pred`, and `y_prob`.

#### `ClassificationEvaluator.evaluate(model, X, y_true, average="macro") -> EvaluationResult`
Runs `predict` and (if available) `predict_proba` on the model, returning an `EvaluationResult`.

#### `RegressionEvaluator.evaluate(model, X, y_true) -> EvaluationResult`
Runs `predict` on the model, returning an `EvaluationResult`.

### `error_analysis.py`

#### `get_classification_errors(y_true, y_pred, X=None) -> pd.DataFrame`
Returns a DataFrame containing instances where `y_true != y_pred`.

#### `class_error_rates(y_true, y_pred) -> dict`
Calculates the error rate specific to each unique class.

#### `get_regression_errors(y_true, y_pred, X=None) -> pd.DataFrame`
Returns a DataFrame with columns for `y_true`, `y_pred`, `error` (actual - predicted), and `abs_error`.

#### `residual_statistics(y_true, y_pred) -> dict`
Returns statistical properties of the residuals (mean, std, min, max, median, skew).

### `plots.py`

#### `plot_confusion_matrix(y_true, y_pred, class_names=None) -> matplotlib.figure.Figure`
Generates a heatmap of the confusion matrix.

#### `plot_roc_curve(y_true, y_prob, class_names=None) -> matplotlib.figure.Figure`
Plots ROC curves for binary or multiclass classification.

#### `plot_precision_recall_curve(y_true, y_prob, class_names=None) -> matplotlib.figure.Figure`
Plots Precision-Recall curves.

#### `plot_actual_vs_predicted(y_true, y_pred) -> matplotlib.figure.Figure`
Plots a scatter of actuals vs predictions with a perfect-prediction reference line.

#### `plot_residuals(y_true, y_pred) -> matplotlib.figure.Figure`
Plots predictions on the x-axis and residuals on the y-axis to check for homoscedasticity.

#### `plot_residual_distribution(y_true, y_pred) -> matplotlib.figure.Figure`
Plots a histogram of regression residuals.

*Note: All plot functions return a `matplotlib.figure.Figure` object and do not call `plt.show()` internally.*

## Usage Examples

```python
from src.evaluation.evaluator import ClassificationEvaluator
from src.evaluation.plots import plot_confusion_matrix

# Evaluate
evaluator = ClassificationEvaluator()
result = evaluator.evaluate(model, X_test, y_test)

print(result.metrics['accuracy'])

# Visualize
fig = plot_confusion_matrix(result.y_true, result.y_pred)
fig.savefig("confusion_matrix.png")
```

## Dependencies
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn` (for confusion matrix heatmaps)

## Testing
Tested via `tests/test_phase1d_evaluation.py`. Validates metric logic, probability handling, multi-class support, evaluator integration, and plot figure generation.

## Not Implemented
- Interactive visualizations (e.g. Plotly/Bokeh)
- Automated model comparison reports
- Statistical significance tests (e.g. McNemar's test)
