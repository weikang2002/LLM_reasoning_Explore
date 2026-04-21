"""
Platform detection and configuration utilities.

Provides helper functions to detect the current platform and suggest
appropriate model loading configurations.
"""

import platform
import sys
from typing import Dict, Any, Optional, Tuple
import torch
from loguru import logger


def get_platform_info() -> Dict[str, Any]:
    """
    Get information about the current platform.

    Returns:
        Dictionary with platform information including OS, architecture,
        Python version, and available acceleration options.
    """
    system = platform.system()
    machine = platform.machine()
    python_version = sys.version_info

    info = {
        "os": system,
        "architecture": machine,
        "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cpu_only": not torch.cuda.is_available(),
    }

    # Check for Apple Silicon
    info["apple_silicon"] = system == "Darwin" and machine in ["arm64", "aarch64"]

    # Check for bitsandbytes availability
    try:
        import bitsandbytes
        info["bitsandbytes_available"] = True
        info["bitsandbytes_version"] = getattr(bitsandbytes, "__version__", "unknown")
    except ImportError:
        info["bitsandbytes_available"] = False
        info["bitsandbytes_version"] = None

    return info


def get_recommended_device() -> str:
    """
    Get recommended device for model loading based on platform.

    Returns:
        Recommended device string ('cuda', 'mps', 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_recommended_settings(
    model_size: Optional[str] = None,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Get recommended model loading settings for the current platform.

    Args:
        model_size: Optional hint about model size ('small', 'medium', 'large')
                   or parameter count like '1.5b', '7b', '8b'

    Returns:
        Tuple of (device, quantization, additional_kwargs)
    """
    info = get_platform_info()
    device = get_recommended_device()
    quantization = None
    kwargs = {}

    # Parse model size if provided
    size_category = "medium"
    if model_size:
        model_size_lower = model_size.lower()
        if any(x in model_size_lower for x in ["small", "1.5b", "1b"]):
            size_category = "small"
        elif any(x in model_size_lower for x in ["large", "70b", "65b"]):
            size_category = "large"
        elif any(x in model_size_lower for x in ["7b", "8b", "13b"]):
            size_category = "medium"

    # Platform-specific recommendations
    if info["apple_silicon"]:
        # Apple Silicon Mac
        device = "mps"
        quantization = None  # No bitsandbytes support
        kwargs["torch_dtype"] = torch.float16

        if size_category == "large":
            logger.warning(
                "Large models may not fit in memory on macOS. "
                "Consider using a smaller model variant."
            )

    elif info["cuda_available"]:
        # NVIDIA GPU
        device = "cuda"

        if info["bitsandbytes_available"]:
            # Use quantization to save memory
            if size_category in ["medium", "large"]:
                quantization = "4bit"
            # Small models can run in full precision
        else:
            logger.warning(
                "bitsandbytes not available. Install it for memory-efficient quantization: "
                "pip install bitsandbytes"
            )
            kwargs["torch_dtype"] = torch.float16

    else:
        # CPU only
        device = "cpu"

        if info["bitsandbytes_available"] and size_category != "small":
            # Use quantization for medium/large models on CPU
            quantization = "4bit"
            logger.info("Using 4-bit quantization for CPU inference")
        else:
            if size_category != "small":
                logger.warning(
                    "Loading medium/large model on CPU without quantization. "
                    "This may require significant memory (8-32GB RAM). "
                    "Consider installing bitsandbytes for quantization support."
                )
            kwargs["torch_dtype"] = torch.float32

    return device, quantization, kwargs


def check_model_compatibility(
    device: str,
    quantization: Optional[str],
    raise_on_error: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Check if the requested configuration is compatible with the platform.

    Args:
        device: Device to use ('cpu', 'cuda', 'mps', 'auto')
        quantization: Quantization mode ('4bit', '8bit', None)
        raise_on_error: Whether to raise an exception on incompatibility

    Returns:
        Tuple of (is_compatible, error_message)
    """
    info = get_platform_info()

    # Check device compatibility
    if device == "cuda" and not info["cuda_available"]:
        msg = "CUDA requested but not available on this system"
        if raise_on_error:
            raise RuntimeError(msg)
        return False, msg

    if device == "mps" and not info["mps_available"]:
        msg = "MPS requested but not available on this system"
        if raise_on_error:
            raise RuntimeError(msg)
        return False, msg

    # Check quantization compatibility
    if quantization in ["4bit", "8bit"]:
        if not info["bitsandbytes_available"]:
            system = platform.system()
            if system == "Darwin":
                msg = (
                    f"{quantization} quantization requires bitsandbytes, "
                    "which does not support macOS. Use quantization=None instead."
                )
            else:
                msg = (
                    f"{quantization} quantization requires bitsandbytes. "
                    "Install with: pip install bitsandbytes"
                )

            if raise_on_error:
                raise ImportError(msg)
            return False, msg

    return True, None


def print_platform_info():
    """Print detailed platform information and recommendations."""
    info = get_platform_info()

    print("\n" + "="*60)
    print("Platform Information")
    print("="*60)
    print(f"OS: {info['os']}")
    print(f"Architecture: {info['architecture']}")
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {torch.__version__}")

    print("\nAcceleration:")
    print(f"  CUDA available: {info['cuda_available']}")
    print(f"  MPS available: {info['mps_available']}")
    print(f"  Apple Silicon: {info['apple_silicon']}")

    print("\nQuantization:")
    print(f"  bitsandbytes available: {info['bitsandbytes_available']}")
    if info['bitsandbytes_available']:
        print(f"  bitsandbytes version: {info['bitsandbytes_version']}")

    device, quantization, kwargs = get_recommended_settings()
    print("\nRecommended Settings:")
    print(f"  device: '{device}'")
    print(f"  quantization: {quantization if quantization else 'None'}")
    print(f"  torch_dtype: {kwargs.get('torch_dtype', 'auto')}")

    print("\nExample Usage:")
    if quantization:
        print(f'  model = load_model("deepseek-1.5b", device="{device}", quantization="{quantization}")')
    else:
        print(f'  model = load_model("deepseek-1.5b", device="{device}", quantization=None)')

    print("="*60 + "\n")


if __name__ == "__main__":
    print_platform_info()
