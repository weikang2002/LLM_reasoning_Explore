"""Core interpretability framework components."""

from .model_wrapper import ModelWrapper
from .hook_manager import HookManager
from .activation_cache import ActivationCache
from .types import (
    ModelOutput,
    AttentionPatterns,
    HiddenStates,
    Component,
    Circuit,
    DeviceType,
)
from .platform_utils import (
    get_platform_info,
    get_recommended_device,
    get_recommended_settings,
    check_model_compatibility,
    print_platform_info,
)

__all__ = [
    "ModelWrapper",
    "HookManager",
    "ActivationCache",
    "ModelOutput",
    "AttentionPatterns",
    "HiddenStates",
    "Component",
    "Circuit",
    "DeviceType",
    "get_platform_info",
    "get_recommended_device",
    "get_recommended_settings",
    "check_model_compatibility",
    "print_platform_info",
]
