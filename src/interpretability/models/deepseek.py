"""
DeepSeek-R1-Distill model loader.

This module provides the loader for DeepSeek-R1-Distill models with
reasoning token support.
"""

from typing import Dict, List, Optional, Tuple
import re
from loguru import logger

from .base import BaseModelLoader
from ..core.model_wrapper import ModelWrapper
from ..core.platform_utils import get_platform_info


class DeepSeekLoader(BaseModelLoader):
    """
    Loader for DeepSeek-R1-Distill models.

    DeepSeek-R1-Distill models use explicit reasoning tokens to mark
    chain-of-thought reasoning sections.

    Supported models:
        - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
        - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
        - deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    """

    # Reasoning token markers
    REASONING_START_TOKENS = ["<think>", "<｜begin▁of▁sentence｜>"]
    REASONING_END_TOKENS = ["</think>", "<｜end▁of▁sentence｜>"]

    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """
        Initialize DeepSeek loader.

        Args:
            model_name: HuggingFace model name or path
        """
        super().__init__(model_name)
        self._reasoning_token_ids: Optional[Dict[str, List[int]]] = None

    def load(
        self,
        device: str = "auto",
        quantization: Optional[str] = None,
        **kwargs
    ) -> ModelWrapper:
        """
        Load DeepSeek-R1-Distill model.

        Args:
            device: Device to load on ('cpu', 'cuda', 'mps', 'auto')
            quantization: Quantization mode ('4bit', '8bit', None)
            **kwargs: Additional arguments

        Returns:
            ModelWrapper instance
        """
        logger.info(f"Loading DeepSeek model: {self.model_name}")

        # Use recommended quantization if not specified
        if quantization is None:
            quantization = self.get_recommended_quantization()

        # Load model via ModelWrapper
        self.wrapper = ModelWrapper(
            model_name_or_path=self.model_name,
            device=device,
            quantization=quantization,
            trust_remote_code=True,
            **kwargs
        )

        # Cache reasoning token IDs
        self._cache_reasoning_tokens()

        logger.info(f"Loaded DeepSeek model successfully")
        return self.wrapper

    def _cache_reasoning_tokens(self):
        """Cache reasoning token IDs for efficient lookup."""
        if self.wrapper is None or self.wrapper.tokenizer is None:
            return

        self._reasoning_token_ids = {
            "start": [],
            "end": []
        }

        # Tokenize reasoning markers
        for token_str in self.REASONING_START_TOKENS:
            try:
                token_ids = self.wrapper.tokenizer.encode(
                    token_str,
                    add_special_tokens=False
                )
                self._reasoning_token_ids["start"].extend(token_ids)
            except Exception as e:
                logger.debug(f"Could not tokenize '{token_str}': {e}")

        for token_str in self.REASONING_END_TOKENS:
            try:
                token_ids = self.wrapper.tokenizer.encode(
                    token_str,
                    add_special_tokens=False
                )
                self._reasoning_token_ids["end"].extend(token_ids)
            except Exception as e:
                logger.debug(f"Could not tokenize '{token_str}': {e}")

        logger.debug(
            f"Cached reasoning tokens: "
            f"start={self._reasoning_token_ids['start']}, "
            f"end={self._reasoning_token_ids['end']}"
        )

    def get_layer_mapping(self) -> Dict[str, str]:
        """
        Get layer name mapping for DeepSeek models.

        DeepSeek-R1-Distill models are based on Qwen/Llama architectures.
        """
        # Determine base architecture from model name
        if "Qwen" in self.model_name:
            return {
                "transformer.layers": "model.layers",
                "attention": "self_attn",
                "mlp": "mlp",
                "input_layernorm": "input_layernorm",
                "post_attention_layernorm": "post_attention_layernorm",
            }
        elif "Llama" in self.model_name:
            return {
                "transformer.layers": "model.layers",
                "attention": "self_attn",
                "mlp": "mlp",
                "input_layernorm": "input_layernorm",
                "post_attention_layernorm": "post_attention_layernorm",
            }
        else:
            # Default mapping
            return {
                "transformer.layers": "model.layers",
                "attention": "self_attn",
                "mlp": "mlp",
            }

    def identify_reasoning_span(
        self,
        token_ids: List[int]
    ) -> Optional[Tuple[int, int]]:
        """
        Identify span of reasoning tokens in sequence.

        Args:
            token_ids: List of token IDs

        Returns:
            Tuple of (start_idx, end_idx) or None if no reasoning span
        """
        if self._reasoning_token_ids is None:
            return None

        start_tokens = set(self._reasoning_token_ids["start"])
        end_tokens = set(self._reasoning_token_ids["end"])

        # Find first start token
        start_idx = None
        for i, token_id in enumerate(token_ids):
            if token_id in start_tokens:
                start_idx = i
                break

        if start_idx is None:
            return None

        # Find first end token after start
        end_idx = None
        for i in range(start_idx + 1, len(token_ids)):
            if token_ids[i] in end_tokens:
                end_idx = i
                break

        if end_idx is None:
            # No explicit end token, assume rest of sequence is reasoning
            end_idx = len(token_ids) - 1

        return (start_idx, end_idx)

    def get_attention_layer_pattern(self) -> str:
        """Get attention layer pattern for DeepSeek models."""
        return "model.layers.{i}.self_attn"

    def get_mlp_layer_pattern(self) -> str:
        """Get MLP layer pattern for DeepSeek models."""
        return "model.layers.{i}.mlp"

    def get_hidden_states_pattern(self) -> str:
        """Get hidden states pattern for DeepSeek models."""
        return "model.layers.{i}"

    def get_recommended_quantization(self) -> Optional[str]:
        """
        Get recommended quantization for DeepSeek models.

        Returns:
            '4bit' for efficient CPU inference on Linux/Windows with bitsandbytes,
            None for macOS (bitsandbytes not supported)
        """
        info = get_platform_info()

        # Don't recommend quantization on macOS (bitsandbytes not supported)
        if info["os"] == "Darwin":
            logger.debug("macOS detected - recommending no quantization")
            return None

        # Don't recommend quantization if bitsandbytes is not available
        if not info["bitsandbytes_available"]:
            logger.debug("bitsandbytes not available - recommending no quantization")
            return None

        # For Linux/Windows with bitsandbytes, use 4-bit quantization
        return "4bit"

    def get_special_tokens(self) -> Dict[str, List[str]]:
        """Get special tokens for DeepSeek models."""
        return {
            "reasoning_start": self.REASONING_START_TOKENS,
            "reasoning_end": self.REASONING_END_TOKENS,
        }

    def preprocess_prompt(self, prompt: str) -> str:
        """
        Preprocess prompt for DeepSeek models.

        DeepSeek models work best with explicit reasoning instructions.

        Args:
            prompt: Raw prompt

        Returns:
            Preprocessed prompt
        """
        # Check if prompt already has reasoning markers
        if any(marker in prompt for marker in self.REASONING_START_TOKENS):
            return prompt

        # Add reasoning instruction if not present
        reasoning_keywords = ["think", "reason", "step by step", "let me"]
        if not any(kw in prompt.lower() for kw in reasoning_keywords):
            prompt = prompt.rstrip() + " Let me think step by step."

        return prompt

    def extract_reasoning(self, output: str) -> Optional[str]:
        """
        Extract reasoning portion from model output.

        Args:
            output: Full model output

        Returns:
            Reasoning text or None
        """
        # Try to find reasoning between markers
        for start_marker in self.REASONING_START_TOKENS:
            for end_marker in self.REASONING_END_TOKENS:
                pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
                match = re.search(pattern, output, re.DOTALL)
                if match:
                    return match.group(1).strip()

        # If no explicit markers, try to identify reasoning heuristically
        # Look for common reasoning patterns
        reasoning_patterns = [
            r"Let me think\.\s*(.*?)(?:\n\n|$)",
            r"Let's think step by step\.\s*(.*?)(?:\n\n|$)",
            r"First,\s*(.*?)(?:Therefore|So|Thus|Hence)",
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def get_model_size(self) -> str:
        """
        Get human-readable model size.

        Returns:
            Size string (e.g., '1.5B', '7B')
        """
        if "1.5B" in self.model_name or "1_5B" in self.model_name:
            return "1.5B"
        elif "7B" in self.model_name:
            return "7B"
        elif "8B" in self.model_name:
            return "8B"
        else:
            return "unknown"

    def get_architecture_info(self) -> Dict[str, any]:
        """Get DeepSeek-specific architecture information."""
        base_info = super().get_architecture_info()
        base_info.update({
            "model_family": "DeepSeek-R1-Distill",
            "model_size": self.get_model_size(),
            "has_reasoning_tokens": True,
            "reasoning_start_tokens": self.REASONING_START_TOKENS,
            "reasoning_end_tokens": self.REASONING_END_TOKENS,
            "base_architecture": "Qwen" if "Qwen" in self.model_name else "Llama",
        })
        return base_info

    def __repr__(self) -> str:
        size = self.get_model_size()
        return f"DeepSeekLoader(model_name='{self.model_name}', size={size})"
