import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.models import model_registry, ModelFactory


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure the registry is clean before and after each test."""
    # Save original state (with default registrations)
    original_models = model_registry.get_snapshot()
    
    # We don't clear here because some tests might want to use the default registrations.
    # Instead, we just let tests run, and if they mutate the registry, we restore it after.
    
    yield
    
    # Restore original state
    model_registry.restore_snapshot(original_models)


def test_registry_registration_and_retrieval():
    def mock_constructor():
        return "mock_model"
        
    model_registry.register(
        name="test_model", 
        task="classification", 
        constructor=mock_constructor,
        aliases=["tm"]
    )
    
    assert model_registry.exists("test_model")
    assert model_registry.exists("tm")
    
    metadata = model_registry.get("test_model")
    assert metadata["name"] == "test_model"
    assert metadata["task"] == "classification"
    assert metadata["constructor"]() == "mock_model"
    
    alias_metadata = model_registry.get("tm")
    assert metadata == alias_metadata


def test_registry_unsupported_task():
    with pytest.raises(ValueError, match="Unsupported task 'invalid_task'"):
        model_registry.register("test", "invalid_task", lambda: None)


def test_registry_duplicate_registration():
    model_registry.register("dup_model", "regression", lambda: None)
    
    with pytest.raises(ValueError, match="already registered"):
        model_registry.register("dup_model", "regression", lambda: None)


def test_registry_duplicate_alias():
    model_registry.register("m1", "classification", lambda: None, aliases=["a1"])
    
    with pytest.raises(ValueError, match="already registered"):
        model_registry.register("m2", "classification", lambda: None, aliases=["a1"])


def test_registry_get_unknown():
    with pytest.raises(ValueError, match="Unknown model"):
        model_registry.get("unknown_model")


def test_registry_list_and_filter():
    model_registry.clear() # start fresh for this test
    model_registry.register("m1", "classification", lambda: None, aliases=["a1"])
    model_registry.register("m2", "regression", lambda: None)
    model_registry.register("m3", "classification", lambda: None)
    
    all_models = model_registry.list()
    assert set(all_models) == {"m1", "m2", "m3"}
    assert "a1" not in all_models
    
    clf_models = model_registry.list(task="classification")
    assert set(clf_models) == {"m1", "m3"}
    
    reg_models = model_registry.list(task="regression")
    assert set(reg_models) == {"m2"}
    
    with pytest.raises(ValueError, match="Unsupported task"):
        model_registry.list(task="invalid_task")


def test_factory_create():
    model_registry.register("m1", "classification", lambda x: f"model_{x}")
    
    model = ModelFactory.create(task="classification", model_name="m1", x=10)
    assert model == "model_10"


def test_factory_invalid_task():
    with pytest.raises(ValueError, match="Unsupported task"):
        ModelFactory.create("invalid_task", "m1")


def test_factory_unknown_model():
    with pytest.raises(ValueError, match="Unknown model 'unknown'"):
        ModelFactory.create("classification", "unknown")


def test_factory_wrong_task_for_model():
    model_registry.register("m1", "regression", lambda: None)
    
    with pytest.raises(ValueError, match="is registered for task 'regression', but requested for task 'classification'"):
        ModelFactory.create("classification", "m1")


def test_default_registrations():
    # We shouldn't clear it here, let's just make sure the defaults are present
    assert model_registry.exists("logistic_regression")
    assert model_registry.exists("linear_regression")
    
    assert "logistic_regression" in model_registry.list("classification")
    assert "linear_regression" in model_registry.list("regression")


def test_logistic_regression_compatibility():
    # Instantiate
    clf = ModelFactory.create("classification", "logistic_regression", C=0.5, random_state=42)
    assert isinstance(clf, LogisticRegression)
    assert clf.get_params()["C"] == 0.5
    
    # Fit/Predict
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 2


def test_linear_regression_compatibility():
    # Instantiate
    reg = ModelFactory.create("regression", "linear_regression", fit_intercept=False)
    assert isinstance(reg, LinearRegression)
    assert reg.get_params()["fit_intercept"] is False
    
    # Fit/Predict
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    reg.fit(X, y)
    preds = reg.predict(np.array([[4]]))
    assert np.allclose(preds, [8])
