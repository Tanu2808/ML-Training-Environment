# Machine Learning Training Environment

An extensible, modular machine learning training environment designed for clean data processing, robust feature engineering, reliable evaluation, and scalable model management.

## Project Status

**Phase 1 (Foundation):**
- 1A Data Layer — **COMPLETE**
- 1B Preprocessing — **COMPLETE**
- 1C Feature Engineering — **COMPLETE**
- 1D Evaluation — **COMPLETE**

**Phase 2 (Modeling):**
- 2A Model Registry + Factory — **COMPLETE**
- 2B Classical ML Models — **COMPLETE**
- 2C Training Infrastructure — **NOT IMPLEMENTED**
- 2D Cross-Validation — **NOT IMPLEMENTED**
- 2E Hyperparameter Tuning — **NOT IMPLEMENTED**
- 2F Ensemble Learning — **NOT IMPLEMENTED**
- 2G Explainability — **NOT IMPLEMENTED**
- 2H Inference/Persistence — **NOT IMPLEMENTED**
- 2I Integration/Audit — **NOT IMPLEMENTED**

## Current Capabilities

The framework currently provides the following functional capabilities:

* **Data Layer**: Loading (CSV, Parquet, JSON), statistical profiling, robust structural validation, random/stratified sampling, and secure splits (standard, stratified, grouped, time-series).
* **Preprocessing**: Leakage-safe missing value imputation, one-hot/ordinal encoding, scaling (Standard, MinMax, Robust), outlier management (clipping/removal), and mathematical transforms.
* **Feature Engineering**: Numerical operations, interaction generation, datetime cyclical encoding, text feature extraction (TF-IDF/statistics), and robust feature selection (variance, correlation, mutual info, model-based).
* **Evaluation**: Standard metrics computation for classification/regression, unified Evaluator classes, error analysis tools, and metric visualization plots (confusion matrix, ROC, PR, residuals).
* **Model Management**: A centralized singleton registry and factory for model lifecycle tracking and safe parameter forwarding to Scikit-Learn estimators.

## Repository Structure

```text
configs/              # Future integration configurations
data/                 # Raw/processed data directories
src/                  
├── data/             # Data loading, sampling, splitting, validation (Implemented)
├── evaluation/       # Metrics, evaluators, error analysis, plotting (Implemented)
├── features/         # Numerical, categorical, text, time, interactions, selection (Implemented)
├── models/           # Registry, factory, and classical models (Implemented)
└── preprocessing/    # Missing values, encoding, scaling, outliers, transformations (Implemented)
tests/                # Pytest suites corresponding to each implemented Phase
```

## Models

The Model Registry allows decoupling of algorithm selection from code logic.

### Classification (AVAILABLE NOW)
- `logistic_regression` (aliases: `logreg`, `logistic`)
- `decision_tree_classifier` (aliases: `dt_classifier`)
- `random_forest_classifier` (aliases: `rf_classifier`, `random_forest`)
- `gradient_boosting_classifier` (aliases: `gb_classifier`)
- `svm_classifier` (aliases: `svm`)
- `knn_classifier` (aliases: `knn`)
- `naive_bayes_classifier` (aliases: `gaussian_nb`, `nb_classifier`)

### Regression (AVAILABLE NOW)
- `linear_regression` (aliases: `linreg`)
- `ridge_regression` (aliases: `ridge`)
- `lasso_regression` (aliases: `lasso`)
- `elastic_net_regression` (aliases: `elastic_net`)
- `decision_tree_regressor` (aliases: `dt_regressor`)
- `random_forest_regressor` (aliases: `rf_regressor`)
- `gradient_boosting_regressor` (aliases: `gb_regressor`)
- `svm_regressor` (aliases: `svr`)
- `knn_regressor` (aliases: `knn_reg`)

### PLANNED / FUTURE
- Neural Networks (PyTorch/TensorFlow)
- Gradient Boosting frameworks (XGBoost, LightGBM, CatBoost)
- Clustering models
- Time-series models

## Training
**Training infrastructure is NOT yet implemented.**
While the data, preprocessing, evaluation, and model construction layers are complete, the unified training orchestrator, cross-validation handlers, and hyperparameter tuners remain strictly as future planned phases.

## Evaluation
Currently implements comprehensive `Phase 1D` capabilities:
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss, MAE, MSE, RMSE, R², MAPE.
- **Evaluators**: `ClassificationEvaluator`, `RegressionEvaluator`.
- **Error Analysis**: Misclassified sample extraction, class error rates, regression residuals.
- **Plotting**: Matplotlib integration for Confusion Matrix, ROC curves, PR curves, Actual vs. Predicted, and Residual plots.

## Testing

The framework relies on Pytest to enforce behavior integrity.
Run tests via:
```bash
pytest -q
```
*As of Phase 2B, the test suite contains ~358 tests verifying the functional contracts and data-leakage protections of all implemented modules.*
