# Machine Learning Training Environment

A reusable **Machine Learning Training Environment** designed to help start new ML projects quickly without rebuilding common infrastructure from scratch.

> **Build once → reuse across projects → improve with every project.**

---

## 1. What Is This Project?

This repository is a general-purpose ML framework that provides reusable components for common machine-learning workflows.

It is intended for:
* Machine learning projects
* Kaggle competitions
* Hackathons and ML challenges
* Rapid prototyping
* Model training and experimentation
* Research and learning
* Tabular ML

The goal is to handle repetitive engineering so that developers can focus on the **actual ML problem**:

```text
Problem Understanding
        ↓
Data Understanding
        ↓
Feature Engineering
        ↓
Model Selection
        ↓
Training
        ↓
Evaluation
        ↓
Optimization
        ↓
Final Prediction
```

Instead of rebuilding the same infrastructure for every project, reusable components are maintained in this repository.

---

# 2. Project / Branch Workflow

`main` contains the reusable and stable ML framework.

Each new project should normally have its own branch.

```text
                         main
                          │
             ┌────────────┼────────────┐
             ↓            ↓            ↓
      amazon-ml      project-02    project-03
       challenge
             │
             ↓
       Project-specific
       work & experiments
```

Create a project branch:
```bash
git checkout main
git pull
git checkout -b <project-name>
```

The project branch inherits the complete ML environment from `main`.

### Important Rule
If something is useful only for the current project, keep it in the project branch.
If something is useful for **future ML projects**, add it to the reusable framework under `src/`, add tests, document it, and merge it into `main`.

---

# 3. Repository Structure

```text
Machine Learning Training Environment/
│
├── configs/                         → ⭐ WORK HERE: PROJECT CONFIGURATION
│   ├── config.yaml
│   ├── data.yaml
│   ├── model.yaml
│   └── training.yaml
│
├── data/                            → ⭐ WORK HERE: PROJECT DATA
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/                       → ⭐ WORK HERE: EDA & EXPERIMENTATION
│
├── experiments/                     → ⭐ WORK HERE: EXPERIMENT RECORDS
│
├── outputs/                         → GENERATED PROJECT OUTPUTS
│   ├── models/
│   ├── predictions/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
├── src/                             → 🧠 REUSABLE ML FRAMEWORK
│   │
│   ├── data/                        → Data loading, validation, profiling (Implemented)
│   │
│   ├── preprocessing/               → Data preprocessing (Implemented)
│   │
│   ├── features/                    → Feature engineering (Implemented)
│   │
│   ├── models/                      → Model registry, factory, and estimators (Implemented)
│   │   ├── classification/
│   │   └── regression/
│   │
│   ├── evaluation/                  → Metrics and evaluation (Implemented)
│   │
│   ├── training/                    → Training, CV, tuning (Planned)
│   │
│   ├── ensemble/                    → Ensemble methods (Planned)
│   │
│   ├── explainability/              → Model explanations (Planned)
│   │
│   ├── inference/                   → Prediction / inference (Planned)
│   │
│   └── utils/                       → General utilities (Planned)
│
├── tests/                           → FRAMEWORK TESTS
│
├── pyproject.toml                   → Python project configuration
└── README.md                        → Documentation
```

---

# 4. Currently Available Capabilities

This section is a **living inventory of the framework**.
> **Only properly implemented functionality should be listed as available.**

---

## Data Layer (Phase 1A: Complete)
- Loading CSV, Parquet, and JSON formats
- Generating statistical profiles and descriptions
- Random and stratified data sampling
- Standard, stratified, grouped, and time-series train/test splitting
- Structural validations (expected columns, missing values, duplicate rows)

## Preprocessing (Phase 1B: Complete)
- Imputation of missing values (leakage-safe strategies)
- One-hot and ordinal encoding of categorical features
- Standardization, min-max scaling, and robust scaling
- Outlier handling (IQR/Z-score detection, clipping, removal)
- Mathematical transformations (log, square root, power, reciprocal)
- Automated pipelines (`PreprocessingPipeline`)

## Feature Engineering (Phase 1C: Complete)
- Numerical manipulations (arithmetic, ratios, aggregation, binning, log)
- Categorical features (leakage-safe frequency/count encoding, rare category grouping)
- Datetime (calendar extraction, cyclical encoding)
- Text statistics and TF-IDF
- Pairwise interactions
- Feature selection (variance, correlation, mutual information, model-based selectors)

## Evaluation (Phase 1D: Complete)
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss, MAE, MSE, RMSE, R², MAPE
- **Evaluators**: `ClassificationEvaluator`, `RegressionEvaluator` returning a structured `EvaluationResult`
- **Error Analysis**: Misclassified sample extraction, class error rates, regression residuals
- **Plotting**: Matplotlib-based confusion matrices, ROC/PR curves, actual vs. predicted, and residual plots

## Models (Phase 2A & 2B: Complete)
The model framework contains a centralized `ModelRegistry` and `ModelFactory`.
Currently available models (as raw scikit-learn estimators):

**Classification:**
- `logistic_regression` (aliases: `logreg`, `logistic`)
- `decision_tree_classifier` (aliases: `dt_classifier`)
- `random_forest_classifier` (aliases: `rf_classifier`, `random_forest`)
- `gradient_boosting_classifier` (aliases: `gb_classifier`)
- `svm_classifier` (aliases: `svm`)
- `knn_classifier` (aliases: `knn`)
- `naive_bayes_classifier` (aliases: `gaussian_nb`, `nb_classifier`)

**Regression:**
- `linear_regression` (aliases: `linreg`)
- `ridge_regression` (aliases: `ridge`)
- `lasso_regression` (aliases: `lasso`)
- `elastic_net_regression` (aliases: `elastic_net`)
- `decision_tree_regressor` (aliases: `dt_regressor`)
- `random_forest_regressor` (aliases: `rf_regressor`)
- `gradient_boosting_regressor` (aliases: `gb_regressor`)
- `svm_regressor` (aliases: `svr`)
- `knn_regressor` (aliases: `knn_reg`)

---

# 5. Planned / Future Layers (Not Implemented)

The following architectures remain strictly planned for Phase 2C and beyond:
- Neural Networks, XGBoost, LightGBM, CatBoost
- Clustering and Time-Series models
- Unified training orchestration engine
- Cross-validation and Hyperparameter tuning systems
- Ensembling (voting, stacking, blending)
- Explainability (SHAP, feature importance)
- Inference and Model Persistence layers
- Automated experiment tracking

---

# 6. Usage Examples

Whenever new reusable functionality is added, it is accessible via the standard Python imports.

Example pipeline usage:
```python
from src.data.loader import load_dataset
from src.data.splitter import stratified_split
from src.preprocessing.pipeline import build_preprocessing_pipeline
from src.models import ModelFactory

# 1. Load Data
df = load_dataset("data/raw/data.csv")
train_df, test_df = stratified_split(df, stratify_col="target")

# 2. Preprocess (Leakage Safe)
pipeline = build_preprocessing_pipeline(num_cols=["age"], cat_cols=["city"])
X_train = pipeline.fit_transform(train_df)
X_test = pipeline.transform(test_df)

# 3. Model
model = ModelFactory.create("classification", "random_forest", n_estimators=100)
model.fit(X_train, train_df["target"])
```

---

# 7. Testing

The framework relies heavily on automated test suites.
Run pytest to execute the test suite (currently passing ~358 tests checking logic, leakage prevention, and model parameter forwarding):

```bash
pytest -q
```

---

# 8. Long-Term Architecture

The intended ML workflow is:

```text
                         CONFIG
                           │
                           ↓
                          DATA
                           │
                           ↓
                       VALIDATION
                           │
                           ↓
                        PROFILING
                           │
                           ↓
                     PREPROCESSING
                           │
                           ↓
                   FEATURE ENGINEERING
                           │
                           ↓
                    MODEL CANDIDATES
                           │
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
            RF            XGB         CatBoost
             │             │             │
             └─────────────┼─────────────┘
                           ↓
                    CROSS VALIDATION
                           │
                           ↓
                  HYPERPARAMETER TUNING
                           │
                           ↓
                       ENSEMBLING
                           │
                           ↓
                       EVALUATION
                           │
                           ↓
                     ERROR ANALYSIS
                           │
                           ↓
                      FINAL MODEL
                           │
                           ↓
                       INFERENCE
```

The long-term goal is to make this pipeline **configurable, reproducible, and reusable**.

---

# 9. Roadmap Status

## Foundation (Phase 1) - COMPLETE
- [x] Data directory & format handling
- [x] Train/test splitting & data validation
- [x] Complete preprocessing pipelines
- [x] Extensive feature engineering
- [x] Evaluation metrics, error analysis, and plotting

## Modeling (Phase 2) - IN PROGRESS
- [x] Model registry and factory (Phase 2A)
- [x] Classical classification/regression model library (Phase 2B)
- [ ] Complete training engine
- [ ] Cross-validation & Hyperparameter tuning
- [ ] XGBoost, LightGBM, Deep Learning templates

## Advanced (Phase 3+)
- [ ] Ensemble system
- [ ] Explainability
- [ ] Inference & Model persistence
- [ ] One-command ML workflow

> **Phase 1 and Phase 2A/2B are COMPLETE.** New functionality will be added progressively, tested, documented, and incorporated into the reusable environment.
