"""Factory helpers for creating models."""

from typing import Any

from src.models.registry import model_registry, SUPPORTED_TASKS


class ModelFactory:
    """Factory for creating machine learning models from the registry."""

    @staticmethod
    def create(task: str, model_name: str, **kwargs) -> Any:
        """Create an instance of a registered model.

        Args:
            task: The task category (e.g., 'classification', 'regression').
            model_name: The name or alias of the registered model.
            **kwargs: Parameters to pass to the model constructor.

        Returns:
            An instantiated model.

        Raises:
            ValueError: If the task is unsupported or the model is unknown for the task.
        """
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{task}'. Supported tasks: {', '.join(sorted(SUPPORTED_TASKS))}"
            )

        if not model_registry.exists(model_name):
            available = model_registry.list(task=task)
            raise ValueError(
                f"Unknown model '{model_name}' for task '{task}'.\n"
                f"Available models for '{task}': {', '.join(available)}"
            )

        metadata = model_registry.get(model_name)
        
        if metadata["task"] != task:
            available = model_registry.list(task=task)
            raise ValueError(
                f"Model '{model_name}' is registered for task '{metadata['task']}', "
                f"but requested for task '{task}'.\n"
                f"Available models for '{task}': {', '.join(available)}"
            )

        constructor = metadata["constructor"]
        return constructor(**kwargs)

