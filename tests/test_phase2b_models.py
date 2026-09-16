import pytest
import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from src.models import ModelFactory, model_registry

@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure the registry is clean before and after each test."""
    original_models = model_registry.get_snapshot()
    yield
    model_registry.restore_snapshot(original_models)

# Tiny synthetic datasets
X_clf = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [2.0, 3.0],
])
y_clf = np.array([0, 0, 1, 1, 1, 1])

X_reg = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0],
    [4.0, 4.0],
])
y_reg = np.array([0.0, 2.0, 4.0, 6.0, 8.0])


# Define expected models for Phase 2B verification
NEW_CLASSIFICATION_MODELS = {
    "decision_tree_classifier": DecisionTreeClassifier,
    "random_forest_classifier": RandomForestClassifier,
    "gradient_boosting_classifier": GradientBoostingClassifier,
    "svm_classifier": SVC,
    "knn_classifier": KNeighborsClassifier,
    "naive_bayes_classifier": GaussianNB,
}

NEW_REGRESSION_MODELS = {
    "ridge_regression": Ridge,
    "lasso_regression": Lasso,
    "elastic_net_regression": ElasticNet,
    "decision_tree_regressor": DecisionTreeRegressor,
    "random_forest_regressor": RandomForestRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
    "svm_regressor": SVR,
    "knn_regressor": KNeighborsRegressor,
}

def test_registry_contains_new_models():
    """Test that all required models from Phase 2B are registered."""
    clf_models = model_registry.list(task="classification")
    for name in NEW_CLASSIFICATION_MODELS.keys():
        assert name in clf_models
        
    reg_models = model_registry.list(task="regression")
    for name in NEW_REGRESSION_MODELS.keys():
        assert name in reg_models


def test_aliases_resolve_correctly():
    """Verify sensible aliases resolve but are not in the main list."""
    assert model_registry.exists("rf_classifier")
    assert model_registry.exists("ridge")
    
    clf_models = model_registry.list(task="classification")
    assert "rf_classifier" not in clf_models
    
    metadata = model_registry.get("rf_classifier")
    assert metadata["name"] == "random_forest_classifier"


def test_parameter_forwarding():
    """Verify that constructor arguments reach the sklearn estimator."""
    rf = ModelFactory.create(
        "classification", 
        "random_forest_classifier", 
        n_estimators=25, 
        max_depth=3, 
        random_state=42
    )
    assert rf.get_params()["n_estimators"] == 25
    assert rf.get_params()["max_depth"] == 3
    assert rf.get_params()["random_state"] == 42
    
    ridge = ModelFactory.create("regression", "ridge_regression", alpha=2.5)
    assert ridge.get_params()["alpha"] == 2.5
    
    svc = ModelFactory.create("classification", "svm_classifier", C=0.5)
    assert svc.get_params()["C"] == 0.5


@pytest.mark.parametrize("model_name, expected_class", NEW_CLASSIFICATION_MODELS.items())
def test_classification_fit_predict(model_name, expected_class):
    """Verify that all new classifiers can be instantiated and perform fit/predict."""
    # Provide necessary kwargs to avoid warnings or ensure determinism
    kwargs = {}
    if model_name in ["random_forest_classifier", "gradient_boosting_classifier", "decision_tree_classifier"]:
        kwargs["random_state"] = 42
    if model_name == "knn_classifier":
        kwargs["n_neighbors"] = 3  # Tiny dataset requires small n_neighbors
        
    clf = ModelFactory.create("classification", model_name, **kwargs)
    
    assert isinstance(clf, expected_class)
    
    clf.fit(X_clf, y_clf)
    preds = clf.predict(X_clf)
    assert len(preds) == len(X_clf)


@pytest.mark.parametrize("model_name, expected_class", NEW_REGRESSION_MODELS.items())
def test_regression_fit_predict(model_name, expected_class):
    """Verify that all new regressors can be instantiated and perform fit/predict."""
    # Provide necessary kwargs to avoid warnings or ensure determinism
    kwargs = {}
    if model_name in ["random_forest_regressor", "gradient_boosting_regressor", "decision_tree_regressor"]:
        kwargs["random_state"] = 42
    if model_name == "knn_regressor":
        kwargs["n_neighbors"] = 3 # Tiny dataset requires small n_neighbors
        
    reg = ModelFactory.create("regression", model_name, **kwargs)
    
    assert isinstance(reg, expected_class)
    
    reg.fit(X_reg, y_reg)
    preds = reg.predict(X_reg)
    assert len(preds) == len(X_reg)


def test_edge_cases():
    """Verify edge cases established in Phase 2A still hold."""
    with pytest.raises(ValueError, match="Unknown model"):
        ModelFactory.create("classification", "unknown_model_xyz")
        
    with pytest.raises(ValueError, match="Unsupported task"):
        ModelFactory.create("unsupported_task", "random_forest_classifier")
        
    with pytest.raises(ValueError, match="is registered for task"):
        ModelFactory.create("regression", "random_forest_classifier")
