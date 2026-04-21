"""
Attention pattern extraction and analysis.

This module provides functions for extracting, processing, and analyzing
attention patterns from transformer models.
"""

from typing import Optional, List, Tuple
import torch
import pandas as pd
import numpy as np
from loguru import logger

from ..core.types import AttentionPatterns
from ..core.model_wrapper import ModelWrapper


def extract_attention_patterns(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layers: Optional[List[int]] = None,
    include_tokens: bool = True
) -> AttentionPatterns:
    """
    Extract attention patterns from model.

    Args:
        model: ModelWrapper instance
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Optional attention mask
        layers: Specific layers to extract (None for all)
        include_tokens: Whether to include token strings

    Returns:
        AttentionPatterns object with attention weights
    """
    logger.debug(f"Extracting attention patterns from {input_ids.shape[0]} sequences")

    # For some models, we need to call the model directly rather than through the wrapper
    # to ensure output_attentions is properly handled
    with torch.no_grad():
        # Try using the model directly first
        raw_outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

    # Check what we got back
    if not hasattr(raw_outputs, 'attentions') or raw_outputs.attentions is None:
        raise ValueError(
            "Model did not return attention weights. "
            "The model may not support attention output extraction. "
            f"Model type: {type(model.model).__name__}"
        )

    # Check if attentions is empty
    if len(raw_outputs.attentions) == 0:
        raise ValueError(
            f"Model returned empty attention tuple. "
            f"Model config output_attentions: {getattr(model.config, 'output_attentions', 'unknown')}. "
            f"Try setting model.config.output_attentions = True before calling this function."
        )

    # Convert tuple of tensors to stacked tensor
    # Each attention tensor: [batch_size, num_heads, seq_len, seq_len]
    try:
        all_attentions = torch.stack(raw_outputs.attentions, dim=0)  # [num_layers, batch, heads, seq, seq]
    except RuntimeError as e:
        raise ValueError(
            f"Failed to stack attention tensors: {e}. "
            f"Received {len(raw_outputs.attentions)} attention tensors. "
            f"First tensor shape: {raw_outputs.attentions[0].shape if raw_outputs.attentions else 'N/A'}"
        ) from e

    # Filter specific layers if requested
    if layers is not None:
        all_attentions = all_attentions[layers]
        layer_names = [model.get_layer_names()[i] for i in layers]
    else:
        layer_names = model.get_layer_names()

    # Get number of heads per layer
    num_heads_per_layer = [model.get_attention_heads(i) for i in range(len(layer_names))]

    # Get tokens if requested
    tokens = None
    if include_tokens:
        tokens = model.tokenizer.convert_ids_to_tokens(input_ids[0])

    patterns = AttentionPatterns(
        patterns=all_attentions,
        layer_names=layer_names,
        num_heads_per_layer=num_heads_per_layer,
        tokens=tokens
    )

    logger.info(
        f"Extracted attention patterns: "
        f"{patterns.num_layers} layers, "
        f"{patterns.seq_len} tokens"
    )

    return patterns


def compute_attention_rollout(
    attention_patterns: AttentionPatterns,
    discard_ratio: float = 0.1,
    head_fusion: str = "mean"
) -> torch.Tensor:
    """
    Compute attention rollout (cumulative attention flow).

    Attention rollout tracks how information flows through layers by
    multiplying attention matrices across layers.

    Args:
        attention_patterns: Attention patterns to process
        discard_ratio: Ratio of attention to discard at each layer
        head_fusion: How to combine heads ('mean', 'max', 'min')

    Returns:
        Rollout attention [batch_size, seq_len, seq_len]
    """
    patterns = attention_patterns.patterns  # [layers, batch, heads, seq, seq]
    num_layers, batch_size, num_heads, seq_len, _ = patterns.shape

    # Fuse attention heads
    if head_fusion == "mean":
        fused = patterns.mean(dim=2)  # [layers, batch, seq, seq]
    elif head_fusion == "max":
        fused = patterns.max(dim=2)[0]
    elif head_fusion == "min":
        fused = patterns.min(dim=2)[0]
    else:
        raise ValueError(f"Unknown head_fusion: {head_fusion}")

    # Add residual connections (identity matrix)
    identity = torch.eye(seq_len, device=fused.device).unsqueeze(0).unsqueeze(0)
    fused = fused + identity
    fused = fused / fused.sum(dim=-1, keepdim=True)  # Renormalize

    # Apply discard ratio (keep top-k attention)
    if discard_ratio > 0:
        flat_fused = fused.view(num_layers, batch_size, -1)
        threshold_idx = int(flat_fused.shape[-1] * discard_ratio)
        threshold = torch.kthvalue(flat_fused, threshold_idx, dim=-1, keepdim=True)[0]
        threshold = threshold.view(num_layers, batch_size, 1, 1)
        fused = torch.where(fused < threshold, torch.zeros_like(fused), fused)
        fused = fused / fused.sum(dim=-1, keepdim=True)

    # Rollout: multiply attention matrices across layers
    rollout = torch.eye(seq_len, device=fused.device).unsqueeze(0)  # [1, seq, seq]
    rollout = rollout.expand(batch_size, -1, -1)  # [batch, seq, seq]

    for layer_idx in range(num_layers):
        rollout = torch.bmm(fused[layer_idx], rollout)

    logger.debug(f"Computed attention rollout: {rollout.shape}")
    return rollout


def get_head_importance(
    attention_patterns: AttentionPatterns,
    target_tokens: List[int],
    method: str = "mean"
) -> pd.DataFrame:
    """
    Rank attention heads by importance to target tokens.

    Args:
        attention_patterns: Attention patterns
        target_tokens: Indices of target tokens to analyze
        method: Aggregation method ('mean', 'max', 'sum')

    Returns:
        DataFrame with head importance scores
    """
    patterns = attention_patterns.patterns  # [layers, batch, heads, seq, seq]
    num_layers, batch_size, num_heads, seq_len, _ = patterns.shape

    importance_scores = []

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            # Get attention weights for this head
            head_attn = patterns[layer_idx, :, head_idx, :, :]  # [batch, seq, seq]

            # Compute importance to target tokens
            scores = []
            for target_idx in target_tokens:
                if target_idx < seq_len:
                    # Attention TO target token (all tokens attending to target)
                    attn_to_target = head_attn[:, :, target_idx]  # [batch, seq]

                    if method == "mean":
                        score = attn_to_target.mean().item()
                    elif method == "max":
                        score = attn_to_target.max().item()
                    elif method == "sum":
                        score = attn_to_target.sum().item()
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    scores.append(score)

            avg_score = np.mean(scores) if scores else 0.0

            importance_scores.append({
                "layer": layer_idx,
                "head": head_idx,
                "importance": avg_score,
                "layer_name": attention_patterns.layer_names[layer_idx]
            })

    df = pd.DataFrame(importance_scores)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    logger.debug(f"Computed head importance for {len(target_tokens)} target tokens")
    return df


def compute_attention_entropy(
    attention_patterns: AttentionPatterns
) -> torch.Tensor:
    """
    Compute entropy of attention distributions.

    High entropy indicates diffuse attention, low entropy indicates focused attention.

    Args:
        attention_patterns: Attention patterns

    Returns:
        Entropy values [num_layers, batch_size, num_heads, seq_len]
    """
    patterns = attention_patterns.patterns  # [layers, batch, heads, seq, seq]

    # Avoid log(0) by adding small epsilon
    eps = 1e-10
    patterns_safe = patterns + eps

    # Compute entropy: -sum(p * log(p))
    entropy = -(patterns_safe * torch.log2(patterns_safe)).sum(dim=-1)

    logger.debug(f"Computed attention entropy: {entropy.shape}")
    return entropy


def get_attention_to_token(
    attention_patterns: AttentionPatterns,
    token_idx: int,
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Get attention weights directed TO a specific token.

    Args:
        attention_patterns: Attention patterns
        token_idx: Index of target token
        layer_idx: Specific layer (None for all)
        head_idx: Specific head (None for all)

    Returns:
        Attention weights [layers, batch, heads, seq] or subset
    """
    patterns = attention_patterns.patterns

    # Extract attention TO token (last dimension)
    attn_to_token = patterns[:, :, :, :, token_idx]  # [layers, batch, heads, seq]

    # Filter by layer
    if layer_idx is not None:
        attn_to_token = attn_to_token[layer_idx:layer_idx+1]

    # Filter by head
    if head_idx is not None:
        attn_to_token = attn_to_token[:, :, head_idx:head_idx+1]

    return attn_to_token


def get_attention_from_token(
    attention_patterns: AttentionPatterns,
    token_idx: int,
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Get attention weights directed FROM a specific token.

    Args:
        attention_patterns: Attention patterns
        token_idx: Index of source token
        layer_idx: Specific layer (None for all)
        head_idx: Specific head (None for all)

    Returns:
        Attention weights [layers, batch, heads, seq] or subset
    """
    patterns = attention_patterns.patterns

    # Extract attention FROM token (second-to-last dimension)
    attn_from_token = patterns[:, :, :, token_idx, :]  # [layers, batch, heads, seq]

    # Filter by layer
    if layer_idx is not None:
        attn_from_token = attn_from_token[layer_idx:layer_idx+1]

    # Filter by head
    if head_idx is not None:
        attn_from_token = attn_from_token[:, :, head_idx:head_idx+1]

    return attn_from_token


def aggregate_attention_heads(
    attention_patterns: AttentionPatterns,
    method: str = "mean"
) -> torch.Tensor:
    """
    Aggregate attention across heads.

    Args:
        attention_patterns: Attention patterns
        method: Aggregation method ('mean', 'max', 'sum')

    Returns:
        Aggregated attention [num_layers, batch_size, seq_len, seq_len]
    """
    patterns = attention_patterns.patterns

    if method == "mean":
        aggregated = patterns.mean(dim=2)
    elif method == "max":
        aggregated = patterns.max(dim=2)[0]
    elif method == "sum":
        aggregated = patterns.sum(dim=2)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    logger.debug(f"Aggregated attention heads using {method}: {aggregated.shape}")
    return aggregated


def find_induction_heads(
    attention_patterns: AttentionPatterns,
    min_attention: float = 0.3
) -> List[Tuple[int, int]]:
    """
    Identify potential induction heads.

    Induction heads show characteristic patterns where tokens attend
    strongly to previous tokens with similar context.

    Args:
        attention_patterns: Attention patterns
        min_attention: Minimum attention weight to consider

    Returns:
        List of (layer_idx, head_idx) tuples
    """
    patterns = attention_patterns.patterns
    num_layers, batch_size, num_heads, seq_len, _ = patterns.shape

    induction_heads = []

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            head_attn = patterns[layer_idx, 0, head_idx]  # [seq, seq]

            # Check for diagonal offset pattern (token attending to token at offset)
            # Induction heads often show attention to positions offset by a fixed amount
            for offset in range(1, min(5, seq_len)):
                diagonal_sum = torch.diagonal(head_attn, offset=offset).sum().item()
                if diagonal_sum / seq_len > min_attention:
                    induction_heads.append((layer_idx, head_idx))
                    break

    logger.debug(f"Found {len(induction_heads)} potential induction heads")
    return induction_heads


def compute_attention_distance(
    attention_patterns: AttentionPatterns
) -> torch.Tensor:
    """
    Compute average attention distance (how far tokens attend).

    Args:
        attention_patterns: Attention patterns

    Returns:
        Average attention distance [num_layers, batch_size, num_heads, seq_len]
    """
    patterns = attention_patterns.patterns
    num_layers, batch_size, num_heads, seq_len, _ = patterns.shape

    # Create position matrix
    positions = torch.arange(seq_len, device=patterns.device).float()
    position_matrix = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq, seq]
    position_matrix = position_matrix.abs()

    # Weight positions by attention
    # [layers, batch, heads, seq, seq] * [seq, seq] -> [layers, batch, heads, seq, seq]
    weighted_distances = patterns * position_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Sum over attended positions
    avg_distance = weighted_distances.sum(dim=-1)  # [layers, batch, heads, seq]

    logger.debug(f"Computed attention distance: {avg_distance.shape}")
    return avg_distance
