"""Model registry utilities."""

from typing import Any, Callable, Dict, List, Optional, Set

SUPPORTED_TASKS = {
    "classification",
    "regression",
    "clustering",
    "time_series",
    "deep_learning",
}


class ModelRegistry:
    """Central registry for machine learning models."""

    def __init__(self):
        self._models: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        task: str,
        constructor: Callable[..., Any],
        description: str = "",
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a model by name and task.

        Args:
            name: Canonical name of the model.
            task: Task category (e.g., 'classification', 'regression').
            constructor: Callable that instantiates the model.
            description: Optional description of the model.
            aliases: Optional alternative names for the model.
        """
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{task}'. Supported tasks: {', '.join(sorted(SUPPORTED_TASKS))}"
            )

        if name in self._models:
            existing_task = self._models[name]["task"]
            raise ValueError(
                f"Model '{name}' is already registered for task '{existing_task}'."
            )

        aliases = aliases or []
        for alias in aliases:
            if alias in self._models:
                existing_task = self._models[alias]["task"]
                raise ValueError(
                    f"Alias '{alias}' for model '{name}' is already registered for task '{existing_task}'."
                )

        metadata = {
            "name": name,
            "task": task,
            "constructor": constructor,
            "description": description,
            "aliases": aliases,
        }

        # Register under canonical name and all aliases
        self._models[name] = metadata
        for alias in aliases:
            self._models[alias] = metadata

    def get(self, name: str) -> Dict[str, Any]:
        """Retrieve a registered model's metadata."""
        if name not in self._models:
            raise ValueError(f"Unknown model '{name}'.")
        return self._models[name]

    def exists(self, name: str) -> bool:
        """Check whether a model is registered."""
        return name in self._models

    def list(self, task: Optional[str] = None) -> List[str]:
        """List registered model names, optionally filtered by task.

        Returns only the canonical names, not aliases.
        """
        if task and task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{task}'. Supported tasks: {', '.join(sorted(SUPPORTED_TASKS))}"
            )

        canonical_names = set()
        for metadata in self._models.values():
            if task is None or metadata["task"] == task:
                canonical_names.add(metadata["name"])
        
        return sorted(list(canonical_names))

    def clear(self) -> None:
        """Clear the registry (useful for testing)."""
        self._models.clear()

    def get_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Snapshot registry state for test isolation."""
        return self._models.copy()

    def restore_snapshot(self, snapshot: Dict[str, Dict[str, Any]]) -> None:
        """Restore registry state for test isolation."""
        self._models.clear()
        self._models.update(snapshot)


# Module-level singleton
model_registry = ModelRegistry()
