import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 1. Data
from src.data.splitter import train_test_split_frame
from src.data.validator import validate_dataset

# 2. Preprocessing
from src.preprocessing.pipeline import FullPreprocessingPipeline

# 3. Features
from src.features.numerical import add_features
from src.features.selection import select_features

# 4. Evaluation
from src.evaluation.evaluator import ClassificationEvaluator
from src.evaluation.error_analysis import get_classification_errors
from src.evaluation.plots import plot_confusion_matrix

def test_cross_module_integration():
    np.random.seed(42)
    df = pd.DataFrame({
        "num1": np.random.randn(100),
        "num2": np.random.randn(100) * 10 + 5,
        "cat1": np.random.choice(["A", "B", "C"], 100),
        "cat2": np.random.choice(["X", "Y"], 100),
        "target": np.random.choice([0, 1], 100)
    })
    
    # 1. Data layer
    assert validate_dataset(df, target_column="target")["valid"] is True
    
    X_train, X_test, y_train, y_test = train_test_split_frame(df, target_column="target", test_size=0.2, random_state=42)
    
    # 2. Preprocessing layer
    pipe = FullPreprocessingPipeline(
        numeric_features=["num1", "num2"],
        categorical_features=["cat1", "cat2"],
        scaler="standard",
        cat_encoder="onehot"
    )
    
    X_train_prep = pipe.fit_transform(X_train)
    X_test_prep = pipe.transform(X_test)
    
    # 3. Features layer
    X_train_feat = add_features(X_train_prep, "num1", "num2", output_col="num_sum")
    X_test_feat = add_features(X_test_prep, "num1", "num2", output_col="num_sum")
    
    # Selection
    X_train_final = select_features(X_train_feat, ["num_sum", "cat1_B", "cat1_C", "cat2_Y"])
    X_test_final = select_features(X_test_feat, ["num_sum", "cat1_B", "cat1_C", "cat2_Y"])
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_final, y_train)
    
    # 4. Evaluation layer
    evaluator = ClassificationEvaluator(model)
    res = evaluator.evaluate(X_test_final, y_test)
    assert "accuracy" in res.metrics
    assert "log_loss" in res.metrics
    assert res.metrics["accuracy"] >= 0.0
    
    # Error analysis
    errors = get_classification_errors(res.y_true, res.y_pred, df=X_test_final)
    assert isinstance(errors, pd.DataFrame)
    
    # Plot (just to ensure it doesn't crash)
    fig, ax = plot_confusion_matrix(res.y_true, res.y_pred)
    assert fig is not None
    assert ax is not None
    plt.close(fig)
