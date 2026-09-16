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
* Time-series ML
* NLP
* Computer Vision
* Deep Learning

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

If something is useful for **future ML projects**, add it to the reusable framework:

```text
Project Branch
      ↓
Reusable functionality discovered
      ↓
Implement under src/
      ↓
Add tests
      ↓
Document
      ↓
Merge into main
```

This allows every project to improve the framework for the next project.

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
│   ├── data/                        → Data loading, validation, profiling
│   │
│   ├── preprocessing/               → Data preprocessing
│   │
│   ├── features/                    → Feature engineering
│   │
│   ├── models/                      → Model implementations (Planned)
│   │   ├── classification/
│   │   ├── regression/
│   │   ├── clustering/
│   │   ├── time_series/
│   │   └── deep_learning/
│   │
│   ├── training/                    → Training, CV, tuning (Planned)
│   │
│   ├── evaluation/                  → Metrics and evaluation (Planned)
│   │
│   ├── ensemble/                    → Ensemble methods (Planned)
│   │
│   ├── explainability/              → Model explanations (Planned)
│   │
│   ├── inference/                   → Prediction / inference (Planned)
│   │
│   └── utils/                       → General utilities (Planned)
│
├── pipelines/                       → REUSABLE ML WORKFLOWS
│   ├── classification.py
│   ├── regression.py
│   ├── clustering.py
│   ├── time_series.py
│   └── deep_learning.py
│
├── scripts/                         → COMMAND-LINE SCRIPTS
│   ├── profile.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   └── predict.py
│
├── tests/                           → FRAMEWORK TESTS
│
├── pyproject.toml                   → Python project configuration
├── requirements.txt                 → Dependencies
├── .gitignore                       → Git exclusions
└── README.md                        → Documentation
```

---

# 4. Where Should I Work?

When starting a new project, the main workflow is:

```text
                    NEW PROJECT
                         │
                         ↓
                    configs/
                         │
                         ↓
                      data/
                         │
                         ↓
                   notebooks/
                         │
                         ↓
                  experiments/
                         │
                         ↓
                    outputs/
```

### Use `src/` when:

* Adding reusable ML functionality
* Adding a new model
* Adding preprocessing utilities
* Adding feature-engineering methods
* Improving training infrastructure
* Adding evaluation metrics
* Adding reusable utilities

### Use `notebooks/` when:

* Performing EDA
* Exploring the dataset
* Testing ideas
* Comparing approaches
* Visualizing results

### Use `configs/` when:

* Changing dataset paths
* Changing target columns
* Selecting models
* Changing training parameters
* Changing experiment settings

---

# 5. Currently Available

This section is a **living inventory of the framework**.

Whenever a new reusable feature, model, or capability is implemented and tested, it should be added here.

> **Only properly implemented functionality should be listed as available.**

---

## Data

### Available

* CSV dataset loading
* DataFrame train/test splitting
* Basic dataset profiling

  * Number of rows
  * Column names
  * Data types

### Data module

```text
src/data/
├── loader.py
├── profiler.py
├── sampler.py
├── splitter.py
└── validator.py
```

---

# 6. Preprocessing

The preprocessing layer currently contains modules for:

```text
src/preprocessing/
├── encoding.py
├── missing_values.py
├── outliers.py
├── pipeline.py
├── scaling.py
└── transformations.py
```

### Currently available

* Missing-value handling (8 strategies + leakage-safe imputer)
* Categorical encoding (OHE, ordinal)
* Numerical scaling (Standard, MinMax, Robust)
* Outlier handling (detection, clipping, and removal)
* Numerical transformations (log1p, sqrt, reciprocal, power)
* Reusable preprocessing pipelines

> **See `src/preprocessing/README.md` for detailed documentation.**

---

# 7. Feature Engineering

The feature-engineering layer currently contains:

```text
src/features/
├── categorical.py
├── datetime.py
├── interactions.py
├── numerical.py
├── selection.py
└── text.py
```

### Currently available

* Numerical feature engineering (arithmetic, ratios, aggregation, binning, log)
* Categorical feature engineering (leakage-safe frequency/count encoding, rare category grouping)
* Datetime features (calendar extraction, cyclical encoding)
* Text features (vectorized statistics, TF-IDF)
* Feature interactions (controlled pairwise combinations)
* Feature selection (leakage-safe variance, correlation, mutual information, and model-based selectors)

> **See `src/features/README.md` for detailed documentation and examples.**

---

# 8. Models

The model framework contains:

```text
src/models/
├── registry.py
├── factory.py
├── classification/
├── regression/
├── clustering/
├── time_series/
└── deep_learning/
```

### Currently available

* Model registry structure
* Model factory structure
* Task-specific model organization

### Planned model categories

#### Classification

* Logistic Regression
* K-Nearest Neighbors
* Naive Bayes
* Support Vector Machine
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost
* Neural Networks

#### Regression

* Linear Regression
* Ridge
* Lasso
* Elastic Net
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost
* Neural Networks

#### Clustering

* K-Means
* DBSCAN
* HDBSCAN
* Agglomerative Clustering
* Gaussian Mixture Models

#### Dimensionality Reduction

* PCA
* Truncated SVD
* NMF
* UMAP
* t-SNE

#### Time Series

Time-series models will be added progressively.

#### Deep Learning

Reusable deep-learning templates will be added progressively.

> Models should be moved from the planned list to the **Currently Available** list only after implementation and testing.

---

# 9. Training

The training layer currently contains:

```text
src/training/
├── callbacks.py
├── checkpoints.py
├── cross_validation.py
├── trainer.py
└── tuning.py
```

The training system is intended to support:

* Model training
* Cross-validation
* Stratified cross-validation
* Group cross-validation
* Time-series validation
* Hyperparameter tuning
* Callbacks
* Checkpointing
* Reproducibility
* GPU/device handling

---

# 10. Evaluation

The evaluation layer contains:

```text
src/evaluation/
├── error_analysis.py
├── evaluator.py
├── metrics.py
└── plots.py
```

### Currently available

* **Metrics**: Classification (Accuracy, Precision, Recall, F1, Log Loss, ROC-AUC) and Regression (MAE, MSE, RMSE, R², MAPE).
* **Evaluators**: `ClassificationEvaluator` and `RegressionEvaluator` with structured `EvaluationResult` outputs.
* **Error Analysis**: Misclassified sample extraction, class error rates, regression residuals, and residual statistics.
* **Plotting**: Confusion matrix, ROC curve, PR curve, actual vs. predicted, and residual plots (Matplotlib).

> **See `src/evaluation/README.md` for detailed documentation and examples.**

---

# 11. Ensemble Learning

The framework contains an ensemble layer:

```text
src/ensemble/
```

Planned capabilities:

* Voting
* Weighted voting
* Probability blending
* Stacking
* Weighted averaging
* Rank averaging

---

# 12. Explainability

The framework contains:

```text
src/explainability/
```

Planned capabilities include:

* Feature importance
* Permutation importance
* SHAP
* Prediction explanations
* Model behavior analysis

---

# 13. Inference

The framework contains:

```text
src/inference/
```

The inference layer is intended to standardize:

* Loading trained models
* Batch prediction
* Single prediction
* Post-processing
* Saving predictions

---

# 14. Pipelines

Reusable high-level pipelines are organized under:

```text
pipelines/
├── classification.py
├── regression.py
├── clustering.py
├── time_series.py
└── deep_learning.py
```

The goal is eventually to allow a project to move from:

```text
Dataset
   ↓
Preprocessing
   ↓
Features
   ↓
Model
   ↓
Training
   ↓
Evaluation
```

with minimal custom code.

---

# 15. Configuration

Project-specific configuration should be kept in:

```text
configs/
├── config.yaml
├── data.yaml
├── model.yaml
└── training.yaml
```

The long-term goal is to minimize hardcoded project-specific values.

For example:

```yaml
task: classification

data:
  train: data/raw/train.csv
  test: data/raw/test.csv
  target: target

model:
  name: random_forest

training:
  test_size: 0.2
  random_state: 42
```

This allows the framework to be reused by changing configuration instead of rewriting infrastructure.

---

# 16. Experiment Management

Experiments should be recorded under:

```text
experiments/
```

Generated artifacts should be separated under:

```text
outputs/
├── models/
├── predictions/
├── metrics/
├── plots/
└── reports/
```

This keeps:

* Models
* Predictions
* Metrics
* Visualizations
* Reports

separate and organized.

---

# 17. Testing

The framework should remain reliable as new functionality is added.

Run:

```bash
pytest -q
```

Every significant reusable component should have corresponding tests.

Recommended development cycle:

```text
Implement
    ↓
Test
    ↓
Verify
    ↓
Document
    ↓
Merge
```

---

# 18. Adding a New Model / Feature

Whenever new reusable functionality is added:

### Step 1 — Implement

Put it in the appropriate `src/` directory.

### Step 2 — Test

Add tests under `tests/`.

### Step 3 — Verify

Make sure existing tests still pass.

### Step 4 — Document

Update the **Currently Available** section of this README.

### Step 5 — Merge

If it is genuinely reusable, merge it into `main`.

### Example

Suppose XGBoost classification support is implemented:

```text
src/models/classification/xgboost.py
```

Then update this README:

```text
Models
└── Classification
    └── XGBoost
```

The README should always describe the **actual capabilities of the framework**.

---

# 19. Development Philosophy

This repository is intentionally being built incrementally.

The goal is not to create a huge collection of placeholder functions.

Every reusable component should eventually be:

```text
Implemented
    ↓
Tested
    ↓
Documented
    ↓
Integrated
    ↓
Reusable
```

The framework should handle repetitive engineering while the developer focuses on:

* Problem understanding
* Data understanding
* Feature engineering
* Model selection
* Validation strategy
* Model optimization
* Error analysis
* Final solution

---

# 20. Long-Term Architecture

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

# 21. Roadmap

## Foundation

* [x] Repository structure
* [x] Python package structure
* [x] Configuration structure
* [x] Data directory structure
* [x] Experiment/output structure

## Data

* [x] CSV loading
* [x] Basic profiling
* [x] Basic train/test splitting
* [ ] Robust data validation
* [ ] Automatic dataset profiling
* [ ] Multiple file formats
* [ ] Stratified splitting
* [ ] Group splitting
* [ ] Time-series splitting

## Preprocessing

* [x] Complete missing-value system
* [x] Complete encoding system
* [x] Complete scaling system
* [x] Outlier handling
* [x] Transformation utilities
* [x] Automated preprocessing pipeline

## Feature Engineering

* [x] Numerical feature library
* [x] Categorical feature library
* [x] Datetime feature library
* [x] Text feature library
* [x] Interaction features
* [x] Feature selection
* [ ] Dimensionality reduction

## Models

* [ ] Classification model library
* [ ] Regression model library
* [ ] Clustering model library
* [ ] Time-series model library
* [ ] Deep-learning model templates
* [ ] Complete model registry
* [ ] Complete model factory

## Training

* [ ] Complete training engine
* [ ] Cross-validation engine
* [ ] Hyperparameter optimization
* [ ] Callbacks
* [ ] Checkpointing
* [ ] Reproducibility
* [ ] GPU/device management

## Evaluation

* [x] Complete metrics library
* [x] Unified evaluator
* [x] Visualization system
* [x] Error analysis
* [ ] Model comparison

## Advanced

* [ ] Experiment tracking
* [ ] Ensemble system
* [ ] Explainability
* [ ] Inference system
* [ ] Model persistence
* [ ] One-command ML workflow

---

# 22. Status

🚧 **Active Development (Phase 2)**

**Phase 1 (Foundation: Data, Preprocessing, Feature Engineering, Evaluation) is COMPLETE.**
The repository foundation and initial ML framework structure are established and fully tested.

New functionality will be added progressively, tested, documented, and incorporated into the reusable environment.

> **Build once. Reuse everywhere. Improve the framework with every project.**
