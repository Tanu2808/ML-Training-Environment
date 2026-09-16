"""Model evaluation utilities.

Public API for the evaluation module.
"""

# Metrics
from .metrics import (
    classification_metrics,
    regression_metrics,
)

# Evaluators
from .evaluator import (
    EvaluationResult,
    ClassificationEvaluator,
    RegressionEvaluator,
)

# Error Analysis
from .error_analysis import (
    get_classification_errors,
    class_error_rates,
    get_regression_errors,
    residual_statistics,
)

# Plots
from .plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_residual_distribution,
)

__all__ = [
    # Metrics
    "classification_metrics",
    "regression_metrics",
    
    # Evaluators
    "EvaluationResult",
    "ClassificationEvaluator",
    "RegressionEvaluator",
    
    # Error Analysis
    "get_classification_errors",
    "class_error_rates",
    "get_regression_errors",
    "residual_statistics",
    
    # Plots
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_actual_vs_predicted",
    "plot_residuals",
    "plot_residual_distribution",
]
