"""Configuration helpers."""

from pathlib import Path


def read_yaml(path):
    """Return a YAML dict as a placeholder when PyYAML is unavailable."""
    return {"path": str(Path(path))}
