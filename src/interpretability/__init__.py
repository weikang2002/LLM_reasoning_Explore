"""
Reasoning Interpretability Framework

A mechanistic interpretability system for exploring reasoning in language models.
"""

__version__ = "0.1.0"

# Core imports
from .core.model_wrapper import ModelWrapper
from .core.hook_manager import HookManager, HookHandle
from .core.activation_cache import ActivationCache
from .core.types import (
    ModelOutput,
    AttentionPatterns,
    HiddenStates,
    Component,
    Circuit,
    ComponentType,
)
from .core.platform_utils import (
    get_platform_info,
    get_recommended_device,
    get_recommended_settings,
    check_model_compatibility,
    print_platform_info,
)

# Model loading
from .models.registry import load_model, get_registry
from .models.base import BaseModelLoader
from .models.deepseek import DeepSeekLoader

__all__ = [
    # Core
    "ModelWrapper",
    "HookManager",
    "HookHandle",
    "ActivationCache",

    # Types
    "ModelOutput",
    "AttentionPatterns",
    "HiddenStates",
    "Component",
    "Circuit",
    "ComponentType",

    # Platform utilities
    "get_platform_info",
    "get_recommended_device",
    "get_recommended_settings",
    "check_model_compatibility",
    "print_platform_info",

    # Models
    "load_model",
    "get_registry",
    "BaseModelLoader",
    "DeepSeekLoader",
]
