"""
Model registry for managing different model loaders.

This module provides a central registry for model loaders and
utilities for loading models by name.
"""

from typing import Dict, Type, Optional, List
from loguru import logger

from .base import BaseModelLoader
from .deepseek import DeepSeekLoader
from ..core.model_wrapper import ModelWrapper


class ModelRegistry:
    """
    Registry for model loaders.

    Provides centralized access to model loaders and simplifies
    model loading by name or pattern matching.
    """

    def __init__(self):
        """Initialize the model registry."""
        self._loaders: Dict[str, Type[BaseModelLoader]] = {}
        self._aliases: Dict[str, str] = {}

        # Register default loaders
        self._register_defaults()

    def _register_defaults(self):
        """Register default model loaders."""
        # DeepSeek models
        self.register("deepseek-r1-distill", DeepSeekLoader)
        self.register("deepseek", DeepSeekLoader)

        # Aliases for convenience
        self.add_alias("deepseek-1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.add_alias("deepseek-7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        self.add_alias("deepseek-8b", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

        logger.debug(f"Registered {len(self._loaders)} default model loaders")

    def register(self, key: str, loader_class: Type[BaseModelLoader]):
        """
        Register a model loader.

        Args:
            key: Unique key for the loader (used for pattern matching)
            loader_class: Loader class
        """
        self._loaders[key] = loader_class
        logger.debug(f"Registered loader '{key}': {loader_class.__name__}")

    def add_alias(self, alias: str, model_name: str):
        """
        Add an alias for a model name.

        Args:
            alias: Short alias (e.g., 'deepseek-1.5b')
            model_name: Full model name (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
        """
        self._aliases[alias] = model_name
        logger.debug(f"Added alias '{alias}' -> '{model_name}'")

    def get_loader(self, model_name: str) -> Type[BaseModelLoader]:
        """
        Get appropriate loader for a model name.

        Args:
            model_name: Model name or pattern

        Returns:
            Loader class

        Raises:
            ValueError: If no appropriate loader found
        """
        # Check for exact match
        if model_name in self._loaders:
            return self._loaders[model_name]

        # Check aliases
        if model_name in self._aliases:
            resolved_name = self._aliases[model_name]
            return self.get_loader(resolved_name)

        # Pattern matching
        model_lower = model_name.lower()
        for key, loader_class in self._loaders.items():
            if key in model_lower:
                logger.debug(f"Matched '{model_name}' to loader '{key}'")
                return loader_class

        raise ValueError(
            f"No loader found for model '{model_name}'. "
            f"Available loaders: {list(self._loaders.keys())}"
        )

    def load_model(
        self,
        model_name: str,
        device: str = "auto",
        quantization: Optional[str] = None,
        **kwargs
    ) -> ModelWrapper:
        """
        Load a model by name.

        Args:
            model_name: Model name, alias, or HuggingFace path
            device: Device to load on
            quantization: Quantization mode
            **kwargs: Additional arguments

        Returns:
            ModelWrapper instance
        """
        # Resolve alias if present
        if model_name in self._aliases:
            model_name = self._aliases[model_name]

        # Get appropriate loader
        loader_class = self.get_loader(model_name)

        # Instantiate and load
        loader = loader_class(model_name)
        wrapper = loader.load(device=device, quantization=quantization, **kwargs)

        logger.info(f"Loaded model '{model_name}' using {loader_class.__name__}")
        return wrapper

    def list_loaders(self) -> List[str]:
        """
        List all registered loader keys.

        Returns:
            List of loader keys
        """
        return list(self._loaders.keys())

    def list_aliases(self) -> Dict[str, str]:
        """
        List all aliases and their mappings.

        Returns:
            Dictionary of aliases
        """
        return self._aliases.copy()

    def __repr__(self) -> str:
        return f"ModelRegistry(loaders={len(self._loaders)}, aliases={len(self._aliases)})"


# Global registry instance
_global_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """
    Get the global model registry.

    Returns:
        Global ModelRegistry instance
    """
    return _global_registry


def load_model(
    model_name: str,
    device: str = "auto",
    quantization: Optional[str] = None,
    **kwargs
) -> ModelWrapper:
    """
    Convenience function to load a model.

    Args:
        model_name: Model name or alias
        device: Device to load on
        quantization: Quantization mode
        **kwargs: Additional arguments

    Returns:
        ModelWrapper instance

    Example:
        >>> from interpretability.models import load_model
        >>> model = load_model("deepseek-1.5b", device="cpu", quantization="4bit")
    """
    return _global_registry.load_model(
        model_name=model_name,
        device=device,
        quantization=quantization,
        **kwargs
    )
