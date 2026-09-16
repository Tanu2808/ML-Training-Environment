# Model Layer

## Status
Phase 2B — COMPLETE

## Purpose
The model layer provides a centralized registry for machine learning models and a factory for instantiating them safely. It decouples the choice of algorithm from the codebase, enabling easy addition of models while ensuring unified parameter forwarding.

## Current Capabilities
- **Model Registry**: Centralized `ModelRegistry` class that tracks canonical names, aliases, descriptions, and the underlying callable constructor.
- **Model Factory**: `ModelFactory` providing a single `create` method that fetches models from the registry and forwards arbitrary parameters.
- **Classification Models**: 7 classical classification estimators (from `scikit-learn`).
- **Regression Models**: 9 classical regression estimators (from `scikit-learn`).

## Module Structure

```text
src/models/
├── __init__.py
├── factory.py
├── registry.py
├── classification/
│   ├── __init__.py
│   ├── ensemble_models.py
│   ├── kernel_models.py
│   ├── neighbor_models.py
│   ├── probabilistic_models.py
│   └── tree_models.py
├── regression/
│   ├── __init__.py
│   ├── ensemble_models.py
│   ├── kernel_models.py
│   ├── linear_models.py
│   ├── neighbor_models.py
│   └── tree_models.py
└── README.md
```

## Public API

### `registry.py`

#### `ModelRegistry` (and the `model_registry` singleton)
A class to store and manage models. A module-level singleton `model_registry` is exposed and used by default across the codebase.

#### `model_registry.register(name: str, task: str, constructor: Callable, description: str = "", aliases: list[str] = None)`
Registers a new model.

#### `model_registry.get(name: str) -> dict`
Retrieves model metadata, resolving aliases automatically. Raises `ValueError` if the model does not exist.

#### `model_registry.list(task: str = None) -> list[str]`
Returns a list of canonical names of registered models. If `task` is provided, filters the list.

#### `model_registry.exists(name: str) -> bool`
Checks if a canonical name or alias is registered.

#### `model_registry.get_snapshot() -> dict` / `model_registry.restore_snapshot(snapshot: dict)`
Public API for capturing and restoring the registry state (useful primarily for test isolation).

### `factory.py`

#### `ModelFactory.create(task: str, model_name: str, **kwargs) -> Any`
Retrieves the model constructor from the `model_registry` and instantiates it with `**kwargs`. Raises `ValueError` for unknown models or task mismatches. The returned object is the raw estimator (e.g. `sklearn.ensemble.RandomForestClassifier`), not a custom wrapper.

## Currently Registered Models

### Classification
- `logistic_regression` (aliases: `logreg`, `logistic`)
- `decision_tree_classifier` (aliases: `dt_classifier`)
- `random_forest_classifier` (aliases: `rf_classifier`, `random_forest`)
- `gradient_boosting_classifier` (aliases: `gb_classifier`)
- `svm_classifier` (aliases: `svm`)
- `knn_classifier` (aliases: `knn`)
- `naive_bayes_classifier` (aliases: `gaussian_nb`, `nb_classifier`)

### Regression
- `linear_regression` (aliases: `linreg`)
- `ridge_regression` (aliases: `ridge`)
- `lasso_regression` (aliases: `lasso`)
- `elastic_net_regression` (aliases: `elastic_net`)
- `decision_tree_regressor` (aliases: `dt_regressor`)
- `random_forest_regressor` (aliases: `rf_regressor`)
- `gradient_boosting_regressor` (aliases: `gb_regressor`)
- `svm_regressor` (aliases: `svr`)
- `knn_regressor` (aliases: `knn_reg`)

## Usage Examples

```python
from src.models import ModelFactory, model_registry

# List available classification models
classifiers = model_registry.list(task="classification")

# Create a random forest using the factory (supports aliases)
model = ModelFactory.create(
    task="classification", 
    model_name="rf_classifier", 
    n_estimators=100, 
    max_depth=5, 
    random_state=42
)

# The model is a raw sklearn estimator ready for training
model.fit(X_train, y_train)
```

## Dependencies
- `scikit-learn` (for the actual model implementations)

## Testing
Tested via `tests/test_phase2a_models.py` and `tests/test_phase2b_models.py`. Validates singleton registry state, test isolation, missing models, duplicate names, task filtering, instantiation matching `sklearn` types, and parameter forwarding.

## Not Implemented
As of Phase 2B, the following remain strictly **unimplemented**:
- Unified training orchestration or training loops
- Cross-validation framework
- Hyperparameter tuning framework
- Ensemble generation framework
- Explainability framework
- Inference/persistence pipelines
- Experiment tracking

(These features are slated for Phase 2C and beyond).
