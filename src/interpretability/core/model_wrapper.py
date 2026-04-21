"""
Unified model wrapper for consistent interface across models.

This module provides a wrapper class that standardizes access to
transformer models from different sources (HuggingFace, custom, etc.).
"""

from typing import Optional, List, Dict, Any, Union
import sys
import platform
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from loguru import logger

from .types import ModelOutput, DeviceType
from .hook_manager import HookManager
from .activation_cache import ActivationCache

# Try to import BitsAndBytesConfig, but handle gracefully if not available
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None  # type: ignore
    logger.warning(
        "bitsandbytes not available - quantization will not work. "
        "This is expected on macOS. Install bitsandbytes on Linux/Windows for quantization support."
    )


class ModelWrapper:
    """
    Unified wrapper for transformer language models.

    Provides consistent interface for loading, inference, and
    internal access across different model architectures.

    Args:
        model_name_or_path: HuggingFace model name or local path
        device: Device to load model on ('cpu', 'cuda', 'mps', 'auto')
        quantization: Quantization mode ('4bit', '8bit', None)
        trust_remote_code: Whether to trust remote code execution
        torch_dtype: PyTorch dtype for model weights
        **kwargs: Additional arguments for model loading
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        quantization: Optional[str] = None,
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.device_str = device
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code

        # Set default dtype
        if torch_dtype is None:
            torch_dtype = torch.float16 if quantization else torch.float32
        self.torch_dtype = torch_dtype

        # Initialize components
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.hook_manager: Optional[HookManager] = None
        self.cache: Optional[ActivationCache] = None

        # Model metadata
        self.config: Optional[Any] = None
        self._layer_names: List[str] = []
        self._num_layers: int = 0
        self._num_heads: List[int] = []
        self._hidden_size: int = 0

        # Load model and tokenizer
        self._load_model(**kwargs)
        self._load_tokenizer()
        self._initialize_metadata()

        # Initialize hook manager and cache
        self.cache = ActivationCache()
        self.hook_manager = HookManager(self.model, self.cache)

        logger.info(f"Initialized ModelWrapper for {model_name_or_path}")

    def _get_quantization_config(self) -> Optional[Any]:
        """Create quantization configuration."""
        if self.quantization is None:
            return None

        # Check if quantization is requested but bitsandbytes is not available
        if not BITSANDBYTES_AVAILABLE:
            system = platform.system()
            error_msg = (
                f"Quantization requested ({self.quantization}) but bitsandbytes is not available. "
            )

            if system == "Darwin":  # macOS
                error_msg += (
                    "\n\nbitsandbytes does not support macOS. "
                    "To run models on macOS, use one of these options:\n"
                    "  1. Load without quantization: load_model(..., quantization=None)\n"
                    "  2. Use MPS acceleration: load_model(..., device='mps', quantization=None)\n"
                    "  3. Use a smaller model (e.g., deepseek-1.5b instead of 7b/8b)\n"
                    "  4. Consider using MLX framework for Apple Silicon optimization"
                )
            else:
                error_msg += (
                    "\n\nInstall bitsandbytes with: pip install bitsandbytes"
                )

            logger.error(error_msg)
            raise ImportError(error_msg)

        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        return None

    def _load_model(self, **kwargs):
        """Load the model from HuggingFace or local path."""
        logger.info(f"Loading model {self.model_name_or_path}...")

        quantization_config = self._get_quantization_config()

        # Determine device map
        device_map = self.device_str if self.device_str == "auto" else None
        if device_map is None and self.device_str != "auto":
            device_map = {"": self.device_str}

        try:
            # Force 'eager' attention implementation for interpretability
            # SDPA (scaled dot product attention) doesn't support output_attentions
            if 'attn_implementation' not in kwargs:
                kwargs['attn_implementation'] = 'eager'
                logger.debug("Using 'eager' attention implementation for interpretability")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )

            # Set to eval mode
            self.model.eval()

            # Ensure the model config supports attention output
            # Some models have output_attentions disabled by default
            if hasattr(self.model.config, 'output_attentions'):
                # Don't override if explicitly set in kwargs
                if 'output_attentions' not in kwargs:
                    self.model.config.output_attentions = True
                    logger.debug("Enabled output_attentions in model config")

            if hasattr(self.model.config, 'output_hidden_states'):
                if 'output_hidden_states' not in kwargs:
                    self.model.config.output_hidden_states = True
                    logger.debug("Enabled output_hidden_states in model config")

            logger.info(
                f"Loaded model on {self.device_str} "
                f"with {self.quantization or 'no'} quantization"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_tokenizer(self):
        """Load the tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=True
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.debug(f"Loaded tokenizer for {self.model_name_or_path}")

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _initialize_metadata(self):
        """Initialize model metadata (layers, heads, etc.)."""
        self.config = self.model.config

        # Extract architecture information
        self._hidden_size = self.config.hidden_size

        # Find transformer layers
        self._layer_names = []
        for name, module in self.model.named_modules():
            if "layer" in name.lower() and isinstance(module, nn.Module):
                if hasattr(module, "self_attn") or hasattr(module, "attention"):
                    self._layer_names.append(name)

        self._num_layers = len(self._layer_names) or getattr(
            self.config, "num_hidden_layers", 0
        )

        # Get number of attention heads per layer
        num_attention_heads = getattr(self.config, "num_attention_heads", 0)
        self._num_heads = [num_attention_heads] * self._num_layers

        logger.debug(
            f"Model metadata: {self._num_layers} layers, "
            f"{num_attention_heads} heads, "
            f"{self._hidden_size} hidden size"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs
    ) -> ModelOutput:
        """
        Run forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            output_hidden_states: Whether to return hidden states from each layer
                - Returns tuple of (num_layers + 1) tensors
                - Index 0: Token embeddings (before any transformer layers)
                - Index 1-N: Output of each transformer layer
                - Shape per tensor: [batch_size, seq_len, hidden_dim]
                - Example: hidden_states[10] = output after layer 9
            output_attentions: Whether to return attention weights from each layer
                - Returns tuple of num_layers tensors
                - Each tensor: [batch_size, num_heads, seq_len, seq_len]
                - attention[i][b,h,from,to] = probability that token 'from'
                  attends to token 'to' in head h of layer i
                - Each row sums to 1.0 (softmax normalized)
            **kwargs: Additional arguments for model.forward()

        Returns:
            ModelOutput with logits and optional hidden states/attentions

        Example:
            >>> inputs = model.tokenize("Hello world", move_to_device=True)
            >>> output = model.forward(**inputs, output_hidden_states=True)
            >>> print(len(output.hidden_states))  # num_layers + 1
            >>> print(output.hidden_states[0].shape)  # Embeddings
            >>> print(output.hidden_states[1].shape)  # After layer 0
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                **kwargs
            )

        # Debug attention output
        if output_attentions:
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                logger.debug(f"Attention output: {len(outputs.attentions)} layers")
            else:
                logger.warning(
                    "output_attentions=True but model returned no attention weights. "
                    "Check model configuration."
                )

        return ModelOutput(
            logits=outputs.logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        return outputs

    def tokenize(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        move_to_device: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text input.

        Args:
            text: Text string or list of strings
            add_special_tokens: Whether to add special tokens
            return_tensors: Return type ('pt' for PyTorch)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            move_to_device: Whether to move tensors to model device (useful for MPS/CUDA)
            **kwargs: Additional tokenizer arguments

        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        outputs = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs
        )

        # Optionally move to model device
        if move_to_device:
            outputs = {k: v.to(self.device) for k, v in outputs.items()}

        return outputs

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs tensor
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decoder arguments

        Returns:
            Decoded text string(s)
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )

    def get_layer_names(self) -> List[str]:
        """Get list of layer names."""
        return self._layer_names.copy()

    def get_layer_module(self, layer_idx: int) -> nn.Module:
        """
        Get module for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Layer module
        """
        if layer_idx >= self._num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range (0-{self._num_layers-1})")

        layer_name = self._layer_names[layer_idx]
        return self.hook_manager.get_module(layer_name)

    def get_attention_heads(self, layer_idx: int) -> int:
        """
        Get number of attention heads for a layer.

        Args:
            layer_idx: Layer index

        Returns:
            Number of attention heads
        """
        if layer_idx >= len(self._num_heads):
            return self._num_heads[0] if self._num_heads else 0
        return self._num_heads[layer_idx]

    def get_hidden_dim(self) -> int:
        """Get hidden dimension size."""
        return self._hidden_size

    def get_num_layers(self) -> int:
        """Get number of layers."""
        return self._num_layers

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.model.parameters()).device

    def to(self, device: Union[str, torch.device]):
        """
        Move model to device.

        Args:
            device: Target device
        """
        if self.quantization:
            logger.warning("Cannot move quantized model to different device")
            return self

        self.model = self.model.to(device)
        return self

    def memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        num_params = sum(p.numel() for p in self.model.parameters())
        param_size_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / 1024 / 1024

        stats = {
            "num_parameters": num_params,
            "num_parameters_m": num_params / 1e6,
            "param_size_mb": param_size_mb,
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
            "quantization": self.quantization or "none",
        }

        return stats

    def __repr__(self) -> str:
        mem_stats = self.memory_stats()
        return (
            f"ModelWrapper(\n"
            f"  model={self.model_name_or_path},\n"
            f"  layers={self._num_layers},\n"
            f"  params={mem_stats['num_parameters_m']:.1f}M,\n"
            f"  size={mem_stats['param_size_mb']:.1f}MB,\n"
            f"  device={mem_stats['device']},\n"
            f"  quantization={mem_stats['quantization']}\n"
            f")"
        )
