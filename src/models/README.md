# Model layer

This directory contains the machine learning model architecture.

## Phase 2A Architecture

The architecture currently provides a clean, extensible **Model Registry** and **Model Factory**. The goal is to allow future components to construct models without knowing the underlying implementation details, while preventing scattered hard-coded model construction logic.

### Model Registry

The `ModelRegistry` acts as the central source of truth for available models. It stores metadata about models, such as their canonical name, task, description, aliases, and the constructor callable. 

Features:
- Models are registered by task (`classification`, `regression`, `clustering`, `time_series`, `deep_learning`).
- Prevents duplicate registrations.
- Safely retrieves metadata or lists available models for a given task.

### Model Factory

The `ModelFactory` provides a unified interface for instantiating models:
```python
from src.models import ModelFactory

clf = ModelFactory.create(
    task="classification",
    model_name="logistic_regression",
    C=1.0, 
    max_iter=1000
)
```

The factory guarantees that:
1. The requested task is valid.
2. The requested model is registered and supports the task.
3. Useful exceptions are raised for invalid inputs, without silently falling back.

### Currently Registered Models (Phase 2A)

- **Classification**: `logistic_regression` (aliases: `logreg`, `logistic`)
- **Regression**: `linear_regression` (aliases: `linreg`)

*These initial models use standard scikit-learn estimators to prove the architecture.*

### How to Register a Future Model

To add a new model, create a new file (e.g., `src/models/classification/my_model.py`) or add to the relevant `__init__.py`:

```python
from src.models.registry import ModelRegistry
from my_module import MyEstimator

ModelRegistry.register(
    name="my_model",
    task="classification",
    constructor=MyEstimator,
    description="A great new model.",
    aliases=["mm"]
)
```

Make sure the module is imported in `src/models/__init__.py` so that the registration is executed.

### Intentionally NOT Implemented Yet

As per Phase 2A requirements, the following are **not** yet implemented:
- Model training loops
- Cross-validation
- Hyperparameter tuning
- Ensembles
- Explainability
- Inference pipelines
- Experiment tracking
- Base model abstractions (standard `sklearn` estimators are used via duck typing)
