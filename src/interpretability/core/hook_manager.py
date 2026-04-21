"""
Hook manager for PyTorch forward and backward hooks.

This module provides utilities for registering, managing, and cleaning up
PyTorch hooks on model modules.
"""

from typing import Dict, List, Callable, Optional, Any, Tuple
from contextlib import contextmanager
import torch
import torch.nn as nn
from loguru import logger

from .types import HookConfig, ComponentType
from .activation_cache import ActivationCache


class HookHandle:
    """
    Handle for a registered hook.

    Provides methods to remove the hook and check if it's active.
    """

    def __init__(self, handle: torch.utils.hooks.RemovableHandle, config: HookConfig):
        self._handle = handle
        self.config = config
        self._active = True

    def remove(self):
        """Remove the hook."""
        if self._active:
            self._handle.remove()
            self._active = False
            logger.debug(f"Removed hook from {self.config.module_name}")

    @property
    def active(self) -> bool:
        """Check if hook is still active."""
        return self._active

    def __repr__(self) -> str:
        status = "active" if self._active else "removed"
        return f"HookHandle({self.config.module_name}, {status})"


class HookManager:
    """
    Manager for PyTorch hooks on model modules.

    Provides methods to register hooks, accumulate activations,
    and manage hook lifecycle.

    Args:
        model: PyTorch model to hook
        cache: Optional activation cache for storing results
    """

    def __init__(self, model: nn.Module, cache: Optional[ActivationCache] = None):
        self.model = model
        self.cache = cache or ActivationCache()

        # Storage for hooks and activations
        self._hooks: Dict[str, HookHandle] = {}
        self._activations: Dict[str, List[torch.Tensor]] = {}
        self._module_map: Dict[str, nn.Module] = {}

        # Build module name mapping
        self._build_module_map()

        logger.info(f"Initialized HookManager with {len(self._module_map)} modules")

    def _build_module_map(self):
        """Build mapping from module names to module objects."""
        for name, module in self.model.named_modules():
            self._module_map[name] = module

    def register_hook(
        self,
        module_name: str,
        hook_fn: Optional[Callable] = None,
        hook_type: str = "forward",
        component_type: Optional[ComponentType] = None,
        store_key: Optional[str] = None
    ) -> HookHandle:
        """
        Register a hook on a specific module.

        Args:
            module_name: Name of the module to hook
            hook_fn: Optional custom hook function. If None, uses default accumulator
            hook_type: Type of hook ('forward' or 'backward')
            component_type: Type of component being hooked
            store_key: Optional key for storing activations

        Returns:
            HookHandle for managing the hook

        Raises:
            ValueError: If module_name not found or hook_type invalid
        """
        if module_name not in self._module_map:
            raise ValueError(f"Module '{module_name}' not found in model")

        module = self._module_map[module_name]
        config = HookConfig(
            module_name=module_name,
            hook_type=hook_type,
            component_type=component_type
        )

        # Use default hook if none provided
        if hook_fn is None:
            hook_fn = self._create_default_hook(module_name, store_key)

        # Register hook
        if hook_type == "forward":
            handle = module.register_forward_hook(hook_fn)
        elif hook_type == "backward":
            handle = module.register_full_backward_hook(hook_fn)
        else:
            raise ValueError(f"Invalid hook_type: {hook_type}")

        # Store hook handle
        hook_handle = HookHandle(handle, config)
        self._hooks[module_name] = hook_handle

        logger.debug(f"Registered {hook_type} hook on {module_name}")
        return hook_handle

    def _create_default_hook(
        self,
        module_name: str,
        store_key: Optional[str] = None
    ) -> Callable:
        """
        Create default hook function that accumulates activations.

        Args:
            module_name: Name of module
            store_key: Optional key for storing in cache

        Returns:
            Hook function
        """
        def hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
            # Store output activation
            if module_name not in self._activations:
                self._activations[module_name] = []

            # Detach and clone to prevent memory issues
            if isinstance(output, tuple):
                activation = tuple(o.detach().clone() for o in output)
            else:
                activation = output.detach().clone()

            self._activations[module_name].append(activation)

            # Optionally store in cache
            if store_key and self.cache:
                self.cache.store(f"{store_key}:{module_name}", activation)

        return hook

    def register_layer_hooks(
        self,
        layer_pattern: str,
        layer_indices: Optional[List[int]] = None,
        component_type: Optional[ComponentType] = None
    ) -> List[HookHandle]:
        """
        Register hooks on multiple layers matching a pattern.

        Args:
            layer_pattern: Pattern to match (e.g., 'model.layers.{i}')
            layer_indices: List of layer indices to hook (None for all)
            component_type: Type of component

        Returns:
            List of HookHandles
        """
        handles = []

        # Find matching modules
        for name in self._module_map.keys():
            # Check if matches pattern
            if layer_pattern.replace("{i}", "") in name:
                # Extract layer index if pattern contains {i}
                if "{i}" in layer_pattern:
                    try:
                        # Extract number from name
                        parts = name.split('.')
                        for part in parts:
                            if part.isdigit():
                                layer_idx = int(part)
                                if layer_indices is None or layer_idx in layer_indices:
                                    handle = self.register_hook(
                                        name,
                                        component_type=component_type
                                    )
                                    handles.append(handle)
                                break
                    except (ValueError, IndexError):
                        continue
                else:
                    handle = self.register_hook(name, component_type=component_type)
                    handles.append(handle)

        logger.info(f"Registered {len(handles)} hooks matching pattern '{layer_pattern}'")
        return handles

    def get_activations(self, module_name: str) -> List[torch.Tensor]:
        """
        Get accumulated activations for a module.

        Args:
            module_name: Name of module

        Returns:
            List of activation tensors
        """
        return self._activations.get(module_name, [])

    def get_all_activations(self) -> Dict[str, List[torch.Tensor]]:
        """Get all accumulated activations."""
        return self._activations.copy()

    def clear_activations(self):
        """Clear all accumulated activations."""
        self._activations.clear()
        logger.debug("Cleared all activations")

    def remove_hook(self, module_name: str):
        """
        Remove hook from a specific module.

        Args:
            module_name: Name of module
        """
        if module_name in self._hooks:
            self._hooks[module_name].remove()
            del self._hooks[module_name]

    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for hook_handle in self._hooks.values():
            hook_handle.remove()
        self._hooks.clear()
        logger.debug("Removed all hooks")

    def get_module(self, module_name: str) -> Optional[nn.Module]:
        """Get module by name."""
        return self._module_map.get(module_name)

    def list_modules(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all module names, optionally filtered by pattern.

        Args:
            pattern: Optional substring pattern to filter

        Returns:
            List of module names
        """
        if pattern:
            return [name for name in self._module_map.keys() if pattern in name]
        return list(self._module_map.keys())

    @contextmanager
    def temporary_hooks(
        self,
        module_names: List[str],
        hook_fn: Optional[Callable] = None,
        hook_type: str = "forward"
    ):
        """
        Context manager for temporary hooks.

        Args:
            module_names: List of module names to hook
            hook_fn: Optional hook function
            hook_type: Type of hook

        Yields:
            Dictionary of accumulated activations

        Example:
            >>> with hook_manager.temporary_hooks(['layer1', 'layer2']) as activations:
            ...     output = model(input)
            ...     layer1_acts = activations['layer1']
        """
        handles = []

        try:
            # Register hooks
            for name in module_names:
                handle = self.register_hook(name, hook_fn, hook_type)
                handles.append(handle)

            # Clear any existing activations
            self.clear_activations()

            # Yield control back to user
            yield self._activations

        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()

    @contextmanager
    def intervention_context(
        self,
        module_name: str,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor]
    ):
        """
        Context manager for intervening on activations.

        Args:
            module_name: Module to intervene on
            intervention_fn: Function to modify activations

        Example:
            >>> def zero_out(activations):
            ...     return torch.zeros_like(activations)
            >>> with hook_manager.intervention_context('layer1', zero_out):
            ...     output = model(input)  # layer1 activations are zeroed
        """
        def intervention_hook(module, input, output):
            return intervention_fn(output)

        handle = self.register_hook(module_name, intervention_hook, "forward")

        try:
            yield
        finally:
            handle.remove()

    def stats(self) -> Dict[str, Any]:
        """Get hook manager statistics."""
        return {
            "total_modules": len(self._module_map),
            "active_hooks": len([h for h in self._hooks.values() if h.active]),
            "total_hooks": len(self._hooks),
            "modules_with_activations": len(self._activations),
            "total_activations": sum(len(acts) for acts in self._activations.values()),
        }

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"HookManager("
            f"modules={stats['total_modules']}, "
            f"active_hooks={stats['active_hooks']}, "
            f"activations={stats['total_activations']})"
        )

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_all_hooks()
