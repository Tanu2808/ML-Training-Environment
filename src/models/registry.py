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

    _models: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
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

        if name in cls._models:
            existing_task = cls._models[name]["task"]
            raise ValueError(
                f"Model '{name}' is already registered for task '{existing_task}'."
            )

        aliases = aliases or []
        for alias in aliases:
            if alias in cls._models:
                existing_task = cls._models[alias]["task"]
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
        cls._models[name] = metadata
        for alias in aliases:
            cls._models[alias] = metadata

    @classmethod
    def get(cls, name: str) -> Dict[str, Any]:
        """Retrieve a registered model's metadata."""
        if name not in cls._models:
            raise ValueError(f"Unknown model '{name}'.")
        return cls._models[name]

    @classmethod
    def exists(cls, name: str) -> bool:
        """Check whether a model is registered."""
        return name in cls._models

    @classmethod
    def list(cls, task: Optional[str] = None) -> List[str]:
        """List registered model names, optionally filtered by task.

        Returns only the canonical names, not aliases.
        """
        if task and task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{task}'. Supported tasks: {', '.join(sorted(SUPPORTED_TASKS))}"
            )

        canonical_names = set()
        for metadata in cls._models.values():
            if task is None or metadata["task"] == task:
                canonical_names.add(metadata["name"])
        
        return sorted(list(canonical_names))

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (useful for testing)."""
        cls._models.clear()
