"""
Type definitions for the interpretability system.

This module contains all the data classes and type definitions used throughout
the interpretability framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import torch
import numpy as np


class ComponentType(Enum):
    """Types of model components that can be analyzed."""
    ATTENTION = "attention"
    MLP = "mlp"
    LAYER_NORM = "layer_norm"
    RESIDUAL = "residual"
    EMBEDDING = "embedding"


class DeviceType(Enum):
    """Device types for model execution."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


@dataclass
class ModelOutput:
    """Output from a model forward pass."""
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None


@dataclass
class AttentionPatterns:
    """
    Attention patterns extracted from a model.

    Attributes:
        patterns: Attention weights [num_layers, batch_size, num_heads, seq_len, seq_len]
        layer_names: Names of layers from which attention was extracted
        num_heads_per_layer: Number of attention heads in each layer
        tokens: Token strings corresponding to sequence positions
    """
    patterns: torch.Tensor
    layer_names: List[str]
    num_heads_per_layer: List[int]
    tokens: Optional[List[str]] = None

    def __post_init__(self):
        """Validate attention pattern dimensions."""
        if self.patterns.ndim != 5:
            raise ValueError(
                f"Attention patterns must be 5D [layers, batch, heads, seq, seq], "
                f"got shape {self.patterns.shape}"
            )

    @property
    def num_layers(self) -> int:
        return self.patterns.shape[0]

    @property
    def batch_size(self) -> int:
        return self.patterns.shape[1]

    @property
    def seq_len(self) -> int:
        return self.patterns.shape[3]

    def get_layer(self, layer_idx: int) -> torch.Tensor:
        """Get attention patterns for a specific layer."""
        return self.patterns[layer_idx]

    def get_head(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """Get attention pattern for a specific head."""
        return self.patterns[layer_idx, :, head_idx, :, :]


@dataclass
class HiddenStates:
    """
    Hidden states extracted from model layers.

    Attributes:
        states: Hidden state tensors [num_layers, batch_size, seq_len, hidden_dim]
        layer_names: Names of layers
        tokens: Token strings corresponding to sequence positions
    """
    states: torch.Tensor
    layer_names: List[str]
    tokens: Optional[List[str]] = None

    def __post_init__(self):
        """Validate hidden state dimensions."""
        if self.states.ndim != 4:
            raise ValueError(
                f"Hidden states must be 4D [layers, batch, seq, hidden], "
                f"got shape {self.states.shape}"
            )

    @property
    def num_layers(self) -> int:
        return self.states.shape[0]

    @property
    def batch_size(self) -> int:
        return self.states.shape[1]

    @property
    def seq_len(self) -> int:
        return self.states.shape[2]

    @property
    def hidden_dim(self) -> int:
        return self.states.shape[3]

    def get_layer(self, layer_idx: int) -> torch.Tensor:
        """Get hidden states for a specific layer."""
        return self.states[layer_idx]

    def get_token(self, token_idx: int) -> torch.Tensor:
        """Get hidden states for a specific token across all layers."""
        return self.states[:, :, token_idx, :]


@dataclass
class ResidualComponents:
    """
    Decomposition of residual stream into attention and MLP contributions.

    Attributes:
        attention_contribution: Contribution from attention mechanism
        mlp_contribution: Contribution from MLP
        layer_norm_scale: Layer normalization scaling factors
        residual: Full residual stream
    """
    attention_contribution: torch.Tensor
    mlp_contribution: torch.Tensor
    layer_norm_scale: torch.Tensor
    residual: torch.Tensor


@dataclass
class LogitLensResults:
    """
    Results from applying logit lens at each layer.

    Attributes:
        predictions: Top-k token predictions per layer [num_layers, batch, seq_len, top_k]
        probabilities: Probabilities for top-k predictions
        token_ids: Token IDs for top-k predictions
        token_strings: Token strings for top-k predictions
        layer_names: Names of layers
    """
    predictions: List[List[List[Tuple[str, float, int]]]]  # [layer][batch][position]
    layer_names: List[str]

    @property
    def num_layers(self) -> int:
        return len(self.predictions)

    def get_layer_predictions(self, layer_idx: int, batch_idx: int = 0) -> List[List[Tuple[str, float, int]]]:
        """Get predictions for a specific layer and batch."""
        return self.predictions[layer_idx][batch_idx]


@dataclass
class Component:
    """
    A model component (e.g., attention head, MLP layer).

    Attributes:
        component_type: Type of component (attention, MLP, etc.)
        layer: Layer index
        head: Head index (for attention components)
        importance: Importance score (e.g., from activation patching)
    """
    component_type: ComponentType
    layer: int
    head: Optional[int] = None
    importance: float = 0.0

    def __str__(self) -> str:
        if self.head is not None:
            return f"{self.component_type.value}_L{self.layer}H{self.head}"
        return f"{self.component_type.value}_L{self.layer}"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Circuit:
    """
    A minimal circuit of model components.

    Attributes:
        components: List of critical components
        performance: Performance metrics (e.g., accuracy on validation set)
        metadata: Additional metadata about circuit discovery
    """
    components: List[Component]
    performance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.components)

    def get_attention_heads(self) -> List[Component]:
        """Get all attention head components in the circuit."""
        return [c for c in self.components if c.component_type == ComponentType.ATTENTION]

    def get_mlp_layers(self) -> List[Component]:
        """Get all MLP layer components in the circuit."""
        return [c for c in self.components if c.component_type == ComponentType.MLP]


@dataclass
class PatchSpec:
    """
    Specification for activation patching.

    Attributes:
        layer: Layer index to patch
        component_type: Type of component to patch
        head: Head index (for attention)
        position: Token position to patch (None for all)
    """
    layer: int
    component_type: ComponentType
    head: Optional[int] = None
    position: Optional[int] = None


@dataclass
class PatchingResults:
    """
    Results from activation patching experiments.

    Attributes:
        patch_specs: List of patching specifications
        effects: Effect size for each patch (e.g., logit difference)
        clean_output: Output from clean (unpatched) run
        corrupted_output: Output from corrupted run
    """
    patch_specs: List[PatchSpec]
    effects: List[float]
    clean_output: torch.Tensor
    corrupted_output: torch.Tensor

    def get_top_patches(self, k: int = 10) -> List[Tuple[PatchSpec, float]]:
        """Get top-k patches by effect size."""
        sorted_patches = sorted(
            zip(self.patch_specs, self.effects),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_patches[:k]


@dataclass
class AblationResults:
    """
    Results from component ablation experiments.

    Attributes:
        components: List of ablated components
        performance_drops: Performance drop for each ablation
        baseline_performance: Performance without ablation
    """
    components: List[Component]
    performance_drops: List[float]
    baseline_performance: float


@dataclass
class InformationFlowGraph:
    """
    Graph representing information flow between tokens.

    Attributes:
        adjacency: Adjacency matrix [num_layers, seq_len, seq_len]
        tokens: Token strings
        layer_names: Names of layers
    """
    adjacency: np.ndarray
    tokens: List[str]
    layer_names: List[str]

    def get_layer_flow(self, layer_idx: int) -> np.ndarray:
        """Get information flow for a specific layer."""
        return self.adjacency[layer_idx]


@dataclass
class ReasoningPath:
    """
    A reasoning path from source to target tokens.

    Attributes:
        path: Sequence of (layer, token_idx) tuples
        strength: Cumulative attention strength along path
        source_token: Starting token
        target_token: Ending token
    """
    path: List[Tuple[int, int]]
    strength: float
    source_token: str
    target_token: str


@dataclass
class ExperimentConfig:
    """
    Configuration for an interpretability experiment.

    Attributes:
        name: Experiment name
        model_name: Model identifier
        prompts: List of prompts to analyze
        analyses: List of analysis types to run
        output_dir: Directory for saving results
        cache_activations: Whether to cache activations
    """
    name: str
    model_name: str
    prompts: List[str]
    analyses: List[str]
    output_dir: str = "data/experiments"
    cache_activations: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResults:
    """
    Results from an interpretability experiment.

    Attributes:
        config: Experiment configuration
        attention_patterns: Extracted attention patterns
        hidden_states: Extracted hidden states
        circuits: Discovered circuits
        metrics: Performance metrics
    """
    config: ExperimentConfig
    attention_patterns: Optional[AttentionPatterns] = None
    hidden_states: Optional[HiddenStates] = None
    logit_lens: Optional[LogitLensResults] = None
    circuits: Optional[List[Circuit]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResults:
    """
    Results from comparing multiple prompts or models.

    Attributes:
        prompts: List of compared prompts
        attention_similarity: Pairwise attention pattern similarity matrix
        hidden_state_divergence: Pairwise hidden state divergence
        circuit_overlap: Circuit component overlap statistics
        shared_circuit_components: Components that appear in all circuits
    """
    prompts: List[str]
    attention_similarity: np.ndarray
    hidden_state_divergence: np.ndarray
    circuit_overlap: float
    shared_circuit_components: List[Component]


@dataclass
class HookConfig:
    """
    Configuration for registering a hook.

    Attributes:
        module_name: Name of module to hook
        hook_type: Type of hook ('forward' or 'backward')
        component_type: Type of component being hooked
    """
    module_name: str
    hook_type: str = "forward"
    component_type: Optional[ComponentType] = None


@dataclass
class CacheMetadata:
    """
    Metadata for cached activations.

    Attributes:
        prompt: Original prompt
        model_name: Model used
        timestamp: When activations were cached
        shape_info: Information about tensor shapes
    """
    prompt: str
    model_name: str
    timestamp: str
    shape_info: Dict[str, Tuple[int, ...]]
