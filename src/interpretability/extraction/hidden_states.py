"""
Hidden state extraction and analysis.

This module provides functions for extracting and analyzing hidden states
(residual stream) from transformer models.
"""

from typing import Optional, List, Tuple
import torch
import numpy as np
from loguru import logger

from ..core.types import HiddenStates, ResidualComponents
from ..core.model_wrapper import ModelWrapper


def extract_hidden_states(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layers: Optional[List[int]] = None,
    include_tokens: bool = True
) -> HiddenStates:
    """
    Extract hidden states from model layers.

    Args:
        model: ModelWrapper instance
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Optional attention mask
        layers: Specific layers to extract (None for all)
        include_tokens: Whether to include token strings

    Returns:
        HiddenStates object
    """
    logger.debug(f"Extracting hidden states from {input_ids.shape[0]} sequences")

    # Run forward pass with hidden states
    with torch.no_grad():
        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

    if output.hidden_states is None:
        raise ValueError("Model did not return hidden states")

    # Stack hidden states: [num_layers, batch_size, seq_len, hidden_dim]
    all_hidden_states = torch.stack(output.hidden_states, dim=0)

    # Filter specific layers if requested
    if layers is not None:
        all_hidden_states = all_hidden_states[layers]
        layer_names = [model.get_layer_names()[i] for i in layers]
    else:
        # Include embedding layer (first hidden state)
        layer_names = ["embedding"] + model.get_layer_names()

    # Get tokens if requested
    tokens = None
    if include_tokens:
        tokens = model.tokenizer.convert_ids_to_tokens(input_ids[0])

    hidden_states = HiddenStates(
        states=all_hidden_states,
        layer_names=layer_names,
        tokens=tokens
    )

    logger.info(
        f"Extracted hidden states: "
        f"{hidden_states.num_layers} layers, "
        f"{hidden_states.seq_len} tokens, "
        f"{hidden_states.hidden_dim} dims"
    )

    return hidden_states


def decompose_residual_stream(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    layer: int,
    attention_mask: Optional[torch.Tensor] = None
) -> ResidualComponents:
    """
    Decompose residual stream into attention and MLP contributions.

    Args:
        model: ModelWrapper instance
        input_ids: Input token IDs
        layer: Layer index to decompose
        attention_mask: Optional attention mask

    Returns:
        ResidualComponents with decomposed activations
    """
    logger.debug(f"Decomposing residual stream at layer {layer}")

    # Get layer module
    layer_module = model.get_layer_module(layer)

    # Storage for components
    attn_output = None
    mlp_output = None
    residual = None
    layer_norm_scale = None

    # Hook to capture attention output
    def attn_hook(module, input, output):
        nonlocal attn_output
        if isinstance(output, tuple):
            attn_output = output[0].detach().clone()
        else:
            attn_output = output.detach().clone()

    # Hook to capture MLP output
    def mlp_hook(module, input, output):
        nonlocal mlp_output
        mlp_output = output.detach().clone()

    # Hook to capture full layer output
    def layer_hook(module, input, output):
        nonlocal residual
        if isinstance(output, tuple):
            residual = output[0].detach().clone()
        else:
            residual = output.detach().clone()

    # Register hooks
    handles = []

    # Find attention and MLP submodules
    for name, module in layer_module.named_modules():
        if "attn" in name.lower() and len(name.split('.')) == 1:
            handles.append(module.register_forward_hook(attn_hook))
        elif "mlp" in name.lower() and len(name.split('.')) == 1:
            handles.append(module.register_forward_hook(mlp_hook))

    # Hook on full layer
    handles.append(layer_module.register_forward_hook(layer_hook))

    try:
        # Run forward pass
        with torch.no_grad():
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Compute layer norm scale if available
        if hasattr(layer_module, 'input_layernorm'):
            with torch.no_grad():
                # Get input to layer
                prev_output = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                ).hidden_states[layer]

                # Compute norm scale
                layer_norm_scale = layer_module.input_layernorm(prev_output)

    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()

    # Create components object
    components = ResidualComponents(
        attention_contribution=attn_output if attn_output is not None else torch.zeros_like(residual),
        mlp_contribution=mlp_output if mlp_output is not None else torch.zeros_like(residual),
        layer_norm_scale=layer_norm_scale if layer_norm_scale is not None else torch.ones_like(residual),
        residual=residual if residual is not None else torch.zeros(1)
    )

    logger.debug(f"Decomposed residual stream at layer {layer}")
    return components


def compute_hidden_state_norms(
    hidden_states: HiddenStates
) -> torch.Tensor:
    """
    Compute L2 norms of hidden states.

    Args:
        hidden_states: HiddenStates object

    Returns:
        Norms [num_layers, batch_size, seq_len]
    """
    norms = torch.norm(hidden_states.states, p=2, dim=-1)
    logger.debug(f"Computed hidden state norms: {norms.shape}")
    return norms


def compute_cosine_similarity(
    hidden_states: HiddenStates,
    token_idx1: int,
    token_idx2: int
) -> torch.Tensor:
    """
    Compute cosine similarity between two token representations across layers.

    Args:
        hidden_states: HiddenStates object
        token_idx1: First token index
        token_idx2: Second token index

    Returns:
        Similarity scores [num_layers, batch_size]
    """
    states1 = hidden_states.states[:, :, token_idx1, :]  # [layers, batch, hidden]
    states2 = hidden_states.states[:, :, token_idx2, :]  # [layers, batch, hidden]

    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(states1, states2, dim=-1)

    logger.debug(f"Computed cosine similarity between tokens {token_idx1} and {token_idx2}")
    return cos_sim


def get_token_evolution(
    hidden_states: HiddenStates,
    token_idx: int
) -> torch.Tensor:
    """
    Get evolution of a token's representation across layers.

    Args:
        hidden_states: HiddenStates object
        token_idx: Token index

    Returns:
        Token representations [num_layers, batch_size, hidden_dim]
    """
    evolution = hidden_states.states[:, :, token_idx, :]
    return evolution


def compute_layer_similarity(
    hidden_states: HiddenStates,
    layer_idx1: int,
    layer_idx2: int,
    method: str = "cka"
) -> float:
    """
    Compute similarity between two layers' representations.

    Args:
        hidden_states: HiddenStates object
        layer_idx1: First layer index
        layer_idx2: Second layer index
        method: Similarity method ('cka', 'cosine', 'correlation')

    Returns:
        Similarity score
    """
    states1 = hidden_states.states[layer_idx1, 0]  # [seq_len, hidden_dim]
    states2 = hidden_states.states[layer_idx2, 0]  # [seq_len, hidden_dim]

    if method == "cka":
        # Centered Kernel Alignment
        score = _compute_cka(states1, states2)
    elif method == "cosine":
        # Average cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            states1.unsqueeze(1),
            states2.unsqueeze(0),
            dim=-1
        )
        score = cos_sim.mean().item()
    elif method == "correlation":
        # Pearson correlation
        states1_flat = states1.flatten()
        states2_flat = states2.flatten()
        score = torch.corrcoef(torch.stack([states1_flat, states2_flat]))[0, 1].item()
    else:
        raise ValueError(f"Unknown similarity method: {method}")

    logger.debug(f"Layer similarity ({method}): {score:.4f}")
    return score


def _compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two matrices.

    Args:
        X: First matrix [n, d1]
        Y: Second matrix [n, d2]

    Returns:
        CKA score
    """
    # Linear CKA
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices
    XX = X @ X.T
    YY = Y @ Y.T

    # CKA formula
    numerator = (XX * YY).sum()
    denominator = torch.sqrt((XX * XX).sum() * (YY * YY).sum())

    cka = (numerator / denominator).item() if denominator > 0 else 0.0
    return cka


def compute_principal_components(
    hidden_states: HiddenStates,
    layer_idx: int,
    n_components: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute principal components of hidden states at a layer.

    Args:
        hidden_states: HiddenStates object
        layer_idx: Layer index
        n_components: Number of components to compute

    Returns:
        Tuple of (components, explained_variance)
    """
    states = hidden_states.states[layer_idx, 0]  # [seq_len, hidden_dim]

    # Center the data
    states_centered = states - states.mean(dim=0, keepdim=True)

    # Compute SVD
    U, S, Vt = torch.svd(states_centered)

    # Get top components
    components = Vt[:n_components]  # [n_components, hidden_dim]

    # Explained variance
    explained_variance = (S[:n_components] ** 2) / (S ** 2).sum()

    logger.debug(
        f"Computed {n_components} principal components, "
        f"explained variance: {explained_variance.sum().item():.2%}"
    )

    return components, explained_variance


def detect_outlier_tokens(
    hidden_states: HiddenStates,
    layer_idx: int,
    threshold: float = 3.0
) -> List[int]:
    """
    Detect outlier tokens based on hidden state norms.

    Args:
        hidden_states: HiddenStates object
        layer_idx: Layer index to analyze
        threshold: Z-score threshold for outliers

    Returns:
        List of outlier token indices
    """
    states = hidden_states.states[layer_idx, 0]  # [seq_len, hidden_dim]

    # Compute norms
    norms = torch.norm(states, p=2, dim=-1)

    # Compute z-scores
    mean = norms.mean()
    std = norms.std()
    z_scores = (norms - mean) / (std + 1e-10)

    # Find outliers
    outliers = torch.where(torch.abs(z_scores) > threshold)[0].tolist()

    logger.debug(f"Found {len(outliers)} outlier tokens at layer {layer_idx}")
    return outliers


def compute_hidden_state_stats(
    hidden_states: HiddenStates
) -> dict:
    """
    Compute statistics about hidden states.

    Args:
        hidden_states: HiddenStates object

    Returns:
        Dictionary of statistics
    """
    states = hidden_states.states

    stats = {
        "mean": states.mean().item(),
        "std": states.std().item(),
        "min": states.min().item(),
        "max": states.max().item(),
        "norm_mean": torch.norm(states, dim=-1).mean().item(),
        "norm_std": torch.norm(states, dim=-1).std().item(),
    }

    # Per-layer statistics
    layer_norms = torch.norm(states, dim=-1).mean(dim=(1, 2))  # [layers]
    stats["layer_norms"] = layer_norms.tolist()

    logger.debug(f"Computed hidden state statistics")
    return stats
