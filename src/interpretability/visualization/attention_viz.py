"""
Attention pattern visualization.

This module provides interactive visualizations of attention patterns
using Plotly.
"""

from typing import Optional, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from loguru import logger

from ..core.types import AttentionPatterns


def plot_attention_heatmap(
    attention_patterns: AttentionPatterns,
    tokens: List[str],
    layer: int,
    head: Optional[int] = None,
    batch_idx: int = 0,
    title: Optional[str] = None,
    colorscale: str = "Viridis"
) -> go.Figure:
    """
    Create interactive attention heatmap.

    Args:
        attention_patterns: Attention patterns
        tokens: Token strings for labels
        layer: Layer index
        head: Head index (None for aggregated across heads)
        batch_idx: Batch index
        title: Optional plot title
        colorscale: Plotly colorscale name

    Returns:
        Plotly Figure
    """
    # Get attention weights
    if head is None:
        # Aggregate across heads
        attn = attention_patterns.patterns[layer, batch_idx].mean(dim=0).cpu().numpy()
        head_label = "all heads (mean)"
    else:
        attn = attention_patterns.patterns[layer, batch_idx, head].cpu().numpy()
        head_label = f"head {head}"

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attn,
        x=tokens,
        y=tokens,
        colorscale=colorscale,
        hovertemplate="From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Attention<br>Weight")
    ))

    # Update layout
    plot_title = title or f"Attention Pattern - Layer {layer}, {head_label}"
    fig.update_layout(
        title=plot_title,
        xaxis_title="To Token",
        yaxis_title="From Token",
        width=800,
        height=700,
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed")  # Start from top
    )

    logger.debug(f"Created attention heatmap for layer {layer}")
    return fig


def plot_attention_heads_grid(
    attention_patterns: AttentionPatterns,
    tokens: List[str],
    layer: int,
    batch_idx: int = 0,
    max_heads: int = 16,
    colorscale: str = "Viridis"
) -> go.Figure:
    """
    Create grid of attention heatmaps for multiple heads.

    Args:
        attention_patterns: Attention patterns
        tokens: Token strings
        layer: Layer index
        batch_idx: Batch index
        max_heads: Maximum number of heads to display
        colorscale: Colorscale name

    Returns:
        Plotly Figure
    """
    num_heads = attention_patterns.patterns.shape[2]
    num_heads = min(num_heads, max_heads)

    # Calculate grid dimensions
    cols = 4
    rows = (num_heads + cols - 1) // cols

    # Create subplots
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Head {i}" for i in range(num_heads)],
        horizontal_spacing=0.05,
        vertical_spacing=0.08
    )

    for head_idx in range(num_heads):
        row = head_idx // cols + 1
        col = head_idx % cols + 1

        attn = attention_patterns.patterns[layer, batch_idx, head_idx].cpu().numpy()

        fig.add_trace(
            go.Heatmap(
                z=attn,
                colorscale=colorscale,
                showscale=(head_idx == 0),
                hovertemplate=f"Head {head_idx}<br>From: %{{y}}<br>To: %{{x}}<br>Attn: %{{z:.3f}}<extra></extra>"
            ),
            row=row,
            col=col
        )

    # Update layout
    fig.update_layout(
        title=f"Attention Heads - Layer {layer}",
        height=250 * rows,
        width=1000,
        showlegend=False
    )

    # Update axes
    for i in range(1, num_heads + 1):
        fig.update_xaxes(showticklabels=False, row=(i-1)//cols + 1, col=(i-1)%cols + 1)
        fig.update_yaxes(showticklabels=False, row=(i-1)//cols + 1, col=(i-1)%cols + 1)

    logger.debug(f"Created attention heads grid for layer {layer} ({num_heads} heads)")
    return fig


def plot_attention_rollout(
    rollout: np.ndarray,
    tokens: List[str],
    title: str = "Attention Rollout",
    colorscale: str = "Blues"
) -> go.Figure:
    """
    Plot attention rollout (cumulative attention).

    Args:
        rollout: Rollout attention matrix [seq_len, seq_len]
        tokens: Token strings
        title: Plot title
        colorscale: Colorscale name

    Returns:
        Plotly Figure
    """
    if isinstance(rollout, np.ndarray):
        rollout_np = rollout
    else:
        rollout_np = rollout.cpu().numpy()

    fig = go.Figure(data=go.Heatmap(
        z=rollout_np,
        x=tokens,
        y=tokens,
        colorscale=colorscale,
        hovertemplate="From: %{y}<br>To: %{x}<br>Cumulative Attention: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="To Token",
        yaxis_title="From Token",
        width=800,
        height=700,
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed")
    )

    logger.debug("Created attention rollout plot")
    return fig


def plot_head_importance(
    importance_df,
    top_k: int = 20,
    title: str = "Attention Head Importance"
) -> go.Figure:
    """
    Plot attention head importance ranking.

    Args:
        importance_df: DataFrame from get_head_importance()
        top_k: Number of top heads to display
        title: Plot title

    Returns:
        Plotly Figure
    """
    df_top = importance_df.head(top_k)

    # Create labels
    labels = [f"L{row['layer']}H{row['head']}" for _, row in df_top.iterrows()]

    fig = go.Figure(data=go.Bar(
        x=df_top["importance"],
        y=labels,
        orientation='h',
        marker=dict(
            color=df_top["importance"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Layer-Head",
        height=max(400, top_k * 20),
        width=800,
        yaxis=dict(autorange="reversed")
    )

    logger.debug(f"Created head importance plot (top {top_k})")
    return fig


def plot_attention_entropy(
    entropy: np.ndarray,
    tokens: List[str],
    layer: int,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot attention entropy across tokens and heads.

    Args:
        entropy: Entropy values [num_heads, seq_len]
        tokens: Token strings
        layer: Layer index
        title: Optional plot title

    Returns:
        Plotly Figure
    """
    if not isinstance(entropy, np.ndarray):
        entropy = entropy.cpu().numpy()

    num_heads = entropy.shape[0]
    head_labels = [f"H{i}" for i in range(num_heads)]

    fig = go.Figure(data=go.Heatmap(
        z=entropy,
        x=tokens,
        y=head_labels,
        colorscale="RdYlGn_r",  # Red = high entropy, Green = low
        hovertemplate="Head: %{y}<br>Token: %{x}<br>Entropy: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Entropy<br>(bits)")
    ))

    plot_title = title or f"Attention Entropy - Layer {layer}"
    fig.update_layout(
        title=plot_title,
        xaxis_title="Token",
        yaxis_title="Head",
        width=800,
        height=max(400, num_heads * 20),
        xaxis=dict(tickangle=-45)
    )

    logger.debug(f"Created attention entropy plot for layer {layer}")
    return fig


def plot_attention_distance(
    distance: np.ndarray,
    tokens: List[str],
    layer: int,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot average attention distance (how far tokens attend).

    Args:
        distance: Distance values [num_heads, seq_len]
        tokens: Token strings
        layer: Layer index
        title: Optional plot title

    Returns:
        Plotly Figure
    """
    if not isinstance(distance, np.ndarray):
        distance = distance.cpu().numpy()

    num_heads = distance.shape[0]
    head_labels = [f"H{i}" for i in range(num_heads)]

    fig = go.Figure(data=go.Heatmap(
        z=distance,
        x=tokens,
        y=head_labels,
        colorscale="Plasma",
        hovertemplate="Head: %{y}<br>Token: %{x}<br>Avg Distance: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Distance<br>(tokens)")
    ))

    plot_title = title or f"Attention Distance - Layer {layer}"
    fig.update_layout(
        title=plot_title,
        xaxis_title="Token",
        yaxis_title="Head",
        width=800,
        height=max(400, num_heads * 20),
        xaxis=dict(tickangle=-45)
    )

    logger.debug(f"Created attention distance plot for layer {layer}")
    return fig


def plot_token_attention_evolution(
    attention_patterns: AttentionPatterns,
    token_idx: int,
    direction: str = "to",
    batch_idx: int = 0
) -> go.Figure:
    """
    Plot how attention to/from a token evolves across layers.

    Args:
        attention_patterns: Attention patterns
        token_idx: Token index to analyze
        direction: 'to' or 'from'
        batch_idx: Batch index

    Returns:
        Plotly Figure
    """
    num_layers = attention_patterns.num_layers
    seq_len = attention_patterns.seq_len

    # Extract attention evolution
    if direction == "to":
        # Attention TO token: [..., :, token_idx]
        attn_evolution = attention_patterns.patterns[:, batch_idx, :, :, token_idx]
    else:  # from
        # Attention FROM token: [..., token_idx, :]
        attn_evolution = attention_patterns.patterns[:, batch_idx, :, token_idx, :]

    # Average across heads
    attn_evolution = attn_evolution.mean(dim=1).cpu().numpy()  # [layers, seq_len]

    # Get token string
    token_str = attention_patterns.tokens[token_idx] if attention_patterns.tokens else f"Token {token_idx}"
    other_tokens = attention_patterns.tokens if attention_patterns.tokens else [f"T{i}" for i in range(seq_len)]

    fig = go.Figure(data=go.Heatmap(
        z=attn_evolution,
        x=other_tokens,
        y=[f"L{i}" for i in range(num_layers)],
        colorscale="Viridis",
        hovertemplate="Layer: %{y}<br>Token: %{x}<br>Attention: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Attention")
    ))

    title = f"Attention {direction.upper()} '{token_str}' Across Layers"
    fig.update_layout(
        title=title,
        xaxis_title="Token",
        yaxis_title="Layer",
        width=800,
        height=600,
        xaxis=dict(tickangle=-45)
    )

    logger.debug(f"Created attention evolution plot for token {token_idx}")
    return fig


def export_attention_html(
    attention_patterns: AttentionPatterns,
    tokens: List[str],
    output_path: str,
    layer: int,
    head: Optional[int] = None
):
    """
    Export attention heatmap to HTML file.

    Args:
        attention_patterns: Attention patterns
        tokens: Token strings
        output_path: Output HTML file path
        layer: Layer index
        head: Head index (None for aggregated)
    """
    fig = plot_attention_heatmap(attention_patterns, tokens, layer, head)
    fig.write_html(output_path)
    logger.info(f"Exported attention visualization to {output_path}")
