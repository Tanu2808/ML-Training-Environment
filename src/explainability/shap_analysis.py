"""SHAP analysis placeholders."""


def run_shap_analysis(model, X):
    """Return a placeholder SHAP analysis payload."""
    return {"model": model, "n_rows": len(X), "status": "not_implemented"}
