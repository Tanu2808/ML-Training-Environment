import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for testing
matplotlib.use("Agg")

from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_residual_distribution
)


def test_plot_confusion_matrix():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    
    fig, ax = plot_confusion_matrix(y_true, y_pred)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Confusion Matrix"
    plt.close(fig)


def test_plot_roc_curve():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    
    fig, ax = plot_roc_curve(y_true, y_prob)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Receiver Operating Characteristic"
    plt.close(fig)
    
    # Passing 2D proba array
    y_prob_2d = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    fig, ax = plot_roc_curve(y_true, y_prob_2d)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Multiclass unsupported
    y_true_mc = np.array([0, 1, 2])
    y_prob_mc = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    with pytest.raises(ValueError, match="supports only binary classification"):
        plot_roc_curve(y_true_mc, y_prob_mc)


def test_plot_precision_recall_curve():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    
    fig, ax = plot_precision_recall_curve(y_true, y_prob)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Precision-Recall Curve"
    plt.close(fig)
    
    # Multiclass unsupported
    y_true_mc = np.array([0, 1, 2])
    y_prob_mc = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    with pytest.raises(ValueError, match="supports only binary classification"):
        plot_precision_recall_curve(y_true_mc, y_prob_mc)


def test_plot_actual_vs_predicted():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 30])
    
    fig, ax = plot_actual_vs_predicted(y_true, y_pred)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Actual vs Predicted"
    plt.close(fig)


def test_plot_residuals():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 30])
    
    fig, ax = plot_residuals(y_true, y_pred)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Residual Plot"
    plt.close(fig)


def test_plot_residual_distribution():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 30])
    
    fig, ax = plot_residual_distribution(y_true, y_pred)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Residual Distribution"
    plt.close(fig)
