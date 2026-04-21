"""
Base model loader interface.

This module provides the abstract base class for model-specific loaders.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ..core.model_wrapper import ModelWrapper


class BaseModelLoader(ABC):
    """
    Abstract base class for model loaders.

    Model-specific loaders should inherit from this class and implement
    the required methods for handling architecture-specific details.
    """

    def __init__(self, model_name: str):
        """
        Initialize the model loader.

        Args:
            model_name: Name or path of the model
        """
        self.model_name = model_name
        self.wrapper: Optional[ModelWrapper] = None

    @abstractmethod
    def load(
        self,
        device: str = "auto",
        quantization: Optional[str] = None,
        **kwargs
    ) -> ModelWrapper:
        """
        Load the model and return a ModelWrapper.

        Args:
            device: Device to load model on
            quantization: Quantization mode
            **kwargs: Additional loading arguments

        Returns:
            ModelWrapper instance
        """
        pass

    @abstractmethod
    def get_layer_mapping(self) -> Dict[str, str]:
        """
        Get mapping from generic layer names to model-specific names.

        Returns:
            Dictionary mapping generic names to model-specific names

        Example:
            {
                'transformer.layers': 'model.layers',
                'attention': 'self_attn',
                'mlp': 'mlp',
            }
        """
        pass

    @abstractmethod
    def identify_reasoning_span(
        self,
        token_ids: List[int]
    ) -> Optional[Tuple[int, int]]:
        """
        Identify span of reasoning tokens in sequence.

        For models with explicit reasoning tokens (e.g., <think>...</think>),
        this method identifies their positions.

        Args:
            token_ids: List of token IDs

        Returns:
            Tuple of (start_idx, end_idx) or None if no reasoning span
        """
        pass

    def get_attention_layer_pattern(self) -> str:
        """
        Get pattern for matching attention layers.

        Returns:
            Pattern string (e.g., 'model.layers.{i}.self_attn')
        """
        return "layers.{i}.attention"

    def get_mlp_layer_pattern(self) -> str:
        """
        Get pattern for matching MLP layers.

        Returns:
            Pattern string (e.g., 'model.layers.{i}.mlp')
        """
        return "layers.{i}.mlp"

    def get_hidden_states_pattern(self) -> str:
        """
        Get pattern for matching hidden state layers.

        Returns:
            Pattern string
        """
        return "layers.{i}"

    def get_recommended_quantization(self) -> Optional[str]:
        """
        Get recommended quantization mode for this model.

        Returns:
            Quantization mode ('4bit', '8bit', None)
        """
        return None

    def get_special_tokens(self) -> Dict[str, List[str]]:
        """
        Get special tokens for this model.

        Returns:
            Dictionary of special token types and their strings

        Example:
            {
                'reasoning_start': ['<think>', '<reasoning>'],
                'reasoning_end': ['</think>', '</reasoning>'],
            }
        """
        return {}

    def preprocess_prompt(self, prompt: str) -> str:
        """
        Apply model-specific prompt preprocessing.

        Args:
            prompt: Raw prompt string

        Returns:
            Preprocessed prompt
        """
        return prompt

    def postprocess_output(self, output: str) -> str:
        """
        Apply model-specific output postprocessing.

        Args:
            output: Raw model output

        Returns:
            Postprocessed output
        """
        return output

    def extract_reasoning(self, output: str) -> Optional[str]:
        """
        Extract reasoning portion from model output.

        Args:
            output: Full model output

        Returns:
            Reasoning text or None
        """
        return None

    def get_architecture_info(self) -> Dict[str, any]:
        """
        Get architecture-specific information.

        Returns:
            Dictionary with architecture details
        """
        if self.wrapper is None:
            return {}

        return {
            "model_name": self.model_name,
            "num_layers": self.wrapper.get_num_layers(),
            "hidden_size": self.wrapper.get_hidden_dim(),
            "num_attention_heads": self.wrapper.get_attention_heads(0),
            "layer_names": self.wrapper.get_layer_names(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
