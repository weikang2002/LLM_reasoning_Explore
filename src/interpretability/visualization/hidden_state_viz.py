"""
Hidden state visualization.

This module provides visualizations for hidden states including
dimensionality reduction, evolution tracking, and clustering.
"""

from typing import Optional, List, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from loguru import logger

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from ..core.types import HiddenStates


def project_hidden_states(
    hidden_states: HiddenStates,
    method: str = "umap",
    dimensions: int = 2,
    layer_idx: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Project hidden states to lower dimensions.

    Args:
        hidden_states: HiddenStates object
        method: Projection method ('umap', 'tsne', 'pca')
        dimensions: Number of dimensions (2 or 3)
        layer_idx: Specific layer (None to project all layers)
        **kwargs: Additional arguments for projection method

    Returns:
        Projected coordinates [n_samples, dimensions]
    """
    logger.debug(f"Projecting hidden states using {method} to {dimensions}D")

    # Get states to project
    if layer_idx is not None:
        states = hidden_states.states[layer_idx, 0].cpu().numpy()  # [seq_len, hidden_dim]
    else:
        # Flatten all layers and tokens
        states = hidden_states.states[:, 0].reshape(-1, hidden_states.hidden_dim).cpu().numpy()

    # Apply dimensionality reduction
    if method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        reducer = UMAP(n_components=dimensions, **kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=dimensions, **kwargs)
    elif method == "pca":
        reducer = PCA(n_components=dimensions, **kwargs)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    projected = reducer.fit_transform(states)
    logger.debug(f"Projected {states.shape[0]} samples to {dimensions}D")

    return projected


def plot_layer_evolution(
    hidden_states: HiddenStates,
    token_idx: int,
    method: str = "umap",
    dimensions: int = 2,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot how a token's representation evolves across layers.

    Args:
        hidden_states: HiddenStates object
        token_idx: Token index to track
        method: Projection method
        dimensions: 2 or 3
        title: Optional plot title

    Returns:
        Plotly Figure
    """
    # Get token evolution across layers
    evolution = hidden_states.states[:, 0, token_idx, :].cpu().numpy()  # [num_layers, hidden_dim]

    # Project to lower dimensions
    if method == "umap":
        if not UMAP_AVAILABLE:
            method = "pca"
            logger.warning("UMAP not available, falling back to PCA")
        else:
            reducer = UMAP(n_components=dimensions)
            projected = reducer.fit_transform(evolution)
    elif method == "tsne":
        reducer = TSNE(n_components=dimensions)
        projected = reducer.fit_transform(evolution)
    elif method == "pca":
        reducer = PCA(n_components=dimensions)
        projected = reducer.fit_transform(evolution)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create figure
    num_layers = hidden_states.num_layers
    layer_indices = np.arange(num_layers)
    token_str = hidden_states.tokens[token_idx] if hidden_states.tokens else f"Token {token_idx}"

    if dimensions == 2:
        fig = go.Figure()

        # Add scatter trace
        fig.add_trace(go.Scatter(
            x=projected[:, 0],
            y=projected[:, 1],
            mode='markers+lines+text',
            marker=dict(
                size=10,
                color=layer_indices,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Layer")
            ),
            line=dict(color='gray', width=1),
            text=[f"L{i}" for i in layer_indices],
            textposition="top center",
            hovertemplate="Layer %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
        ))

        # Highlight start and end
        fig.add_trace(go.Scatter(
            x=[projected[0, 0], projected[-1, 0]],
            y=[projected[0, 1], projected[-1, 1]],
            mode='markers',
            marker=dict(size=15, color=['green', 'red'], symbol=['circle', 'star']),
            showlegend=True,
            name='Start/End',
            hoverinfo='skip'
        ))

        fig.update_layout(
            xaxis_title=f"{method.upper()} 1",
            yaxis_title=f"{method.upper()} 2",
            width=800,
            height=600
        )

    else:  # 3D
        fig = go.Figure(data=[go.Scatter3d(
            x=projected[:, 0],
            y=projected[:, 1],
            z=projected[:, 2],
            mode='markers+lines+text',
            marker=dict(
                size=6,
                color=layer_indices,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Layer")
            ),
            line=dict(color='gray', width=3),
            text=[f"L{i}" for i in layer_indices],
            textposition="top center",
            hovertemplate="Layer %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title=f"{method.upper()} 1",
                yaxis_title=f"{method.upper()} 2",
                zaxis_title=f"{method.upper()} 3"
            ),
            width=900,
            height=700
        )

    plot_title = title or f"Token '{token_str}' Evolution Across Layers ({method.upper()})"
    fig.update_layout(title=plot_title)

    logger.debug(f"Created layer evolution plot for token {token_idx}")
    return fig


def plot_token_clusters(
    hidden_states: HiddenStates,
    layer_idx: int,
    method: str = "umap",
    dimensions: int = 2,
    color_by: str = "position",
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot token representations with clustering.

    Args:
        hidden_states: HiddenStates object
        layer_idx: Layer index
        method: Projection method
        dimensions: 2 or 3
        color_by: Color scheme ('position', 'norm', 'cluster')
        title: Optional plot title

    Returns:
        Plotly Figure
    """
    # Get hidden states for layer
    states = hidden_states.states[layer_idx, 0].cpu().numpy()  # [seq_len, hidden_dim]
    tokens = hidden_states.tokens if hidden_states.tokens else [f"T{i}" for i in range(states.shape[0])]

    # Project
    projected = project_hidden_states(
        hidden_states,
        method=method,
        dimensions=dimensions,
        layer_idx=layer_idx
    )

    # Determine colors
    if color_by == "position":
        colors = np.arange(len(tokens))
        colorbar_title = "Position"
        colorscale = "Viridis"
    elif color_by == "norm":
        colors = np.linalg.norm(states, axis=1)
        colorbar_title = "L2 Norm"
        colorscale = "Plasma"
    elif color_by == "cluster":
        from sklearn.cluster import KMeans
        n_clusters = min(5, len(tokens) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        colors = kmeans.fit_predict(states)
        colorbar_title = "Cluster"
        colorscale = "Viridis"
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    # Create figure
    if dimensions == 2:
        fig = go.Figure(data=go.Scatter(
            x=projected[:, 0],
            y=projected[:, 1],
            mode='markers+text',
            marker=dict(
                size=10,
                color=colors,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=colorbar_title)
            ),
            text=tokens,
            textposition="top center",
            hovertemplate="Token: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
        ))

        fig.update_layout(
            xaxis_title=f"{method.upper()} 1",
            yaxis_title=f"{method.upper()} 2",
            width=900,
            height=700
        )

    else:  # 3D
        fig = go.Figure(data=go.Scatter3d(
            x=projected[:, 0],
            y=projected[:, 1],
            z=projected[:, 2],
            mode='markers+text',
            marker=dict(
                size=6,
                color=colors,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=colorbar_title)
            ),
            text=tokens,
            textposition="top center",
            hovertemplate="Token: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title=f"{method.upper()} 1",
                yaxis_title=f"{method.upper()} 2",
                zaxis_title=f"{method.upper()} 3"
            ),
            width=1000,
            height=800
        )

    layer_name = hidden_states.layer_names[layer_idx]
    plot_title = title or f"Token Clusters - {layer_name} ({method.upper()})"
    fig.update_layout(title=plot_title)

    logger.debug(f"Created token cluster plot for layer {layer_idx}")
    return fig


def plot_hidden_state_norms(
    hidden_states: HiddenStates,
    title: str = "Hidden State Norms Across Layers"
) -> go.Figure:
    """
    Plot L2 norms of hidden states across layers.

    Args:
        hidden_states: HiddenStates object
        title: Plot title

    Returns:
        Plotly Figure
    """
    # Compute norms
    norms = np.linalg.norm(hidden_states.states[0].cpu().numpy(), axis=-1)  # [num_layers, seq_len]

    # For all layers
    all_norms = []
    for layer_idx in range(hidden_states.num_layers):
        layer_norms = np.linalg.norm(
            hidden_states.states[layer_idx, 0].cpu().numpy(),
            axis=-1
        )
        all_norms.append(layer_norms)

    all_norms = np.array(all_norms)  # [num_layers, seq_len]
    tokens = hidden_states.tokens if hidden_states.tokens else [f"T{i}" for i in range(all_norms.shape[1])]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=all_norms,
        x=tokens,
        y=[f"L{i}" for i in range(hidden_states.num_layers)],
        colorscale="Viridis",
        hovertemplate="Layer: %{y}<br>Token: %{x}<br>Norm: %{z:.3f}<extra></extra>",
        colorbar=dict(title="L2 Norm")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Token",
        yaxis_title="Layer",
        width=900,
        height=600,
        xaxis=dict(tickangle=-45)
    )

    logger.debug("Created hidden state norms plot")
    return fig


def plot_cosine_similarity_matrix(
    hidden_states: HiddenStates,
    layer_idx: int,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot cosine similarity matrix between all token pairs at a layer.

    Args:
        hidden_states: HiddenStates object
        layer_idx: Layer index
        title: Optional plot title

    Returns:
        Plotly Figure
    """
    states = hidden_states.states[layer_idx, 0].cpu().numpy()  # [seq_len, hidden_dim]
    tokens = hidden_states.tokens if hidden_states.tokens else [f"T{i}" for i in range(states.shape[0])]

    # Compute pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(states)

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=tokens,
        y=tokens,
        colorscale="RdBu",
        zmid=0,
        hovertemplate="Token 1: %{y}<br>Token 2: %{x}<br>Similarity: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Cosine<br>Similarity")
    ))

    layer_name = hidden_states.layer_names[layer_idx]
    plot_title = title or f"Token Similarity Matrix - {layer_name}"
    fig.update_layout(
        title=plot_title,
        xaxis_title="Token",
        yaxis_title="Token",
        width=800,
        height=750,
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed")
    )

    logger.debug(f"Created similarity matrix for layer {layer_idx}")
    return fig


def plot_layer_similarity_matrix(
    hidden_states: HiddenStates,
    method: str = "cosine",
    title: str = "Layer Similarity Matrix"
) -> go.Figure:
    """
    Plot similarity between all layer pairs.

    Args:
        hidden_states: HiddenStates object
        method: Similarity method ('cosine', 'cka')
        title: Plot title

    Returns:
        Plotly Figure
    """
    from ..extraction.hidden_states import compute_layer_similarity

    num_layers = hidden_states.num_layers
    sim_matrix = np.zeros((num_layers, num_layers))

    for i in range(num_layers):
        for j in range(num_layers):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif i < j:
                sim = compute_layer_similarity(hidden_states, i, j, method=method)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim  # Symmetric

    layer_labels = [f"L{i}" for i in range(num_layers)]

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=layer_labels,
        y=layer_labels,
        colorscale="Viridis",
        hovertemplate="Layer 1: %{y}<br>Layer 2: %{x}<br>Similarity: %{z:.3f}<extra></extra>",
        colorbar=dict(title=f"{method.upper()}<br>Similarity")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Layer",
        width=800,
        height=750,
        yaxis=dict(autorange="reversed")
    )

    logger.debug(f"Created layer similarity matrix using {method}")
    return fig


def plot_pca_variance(
    hidden_states: HiddenStates,
    layer_idx: int,
    n_components: int = 20,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot explained variance of principal components.

    Args:
        hidden_states: HiddenStates object
        layer_idx: Layer index
        n_components: Number of components to analyze
        title: Optional plot title

    Returns:
        Plotly Figure
    """
    from ..extraction.hidden_states import compute_principal_components

    components, explained_var = compute_principal_components(
        hidden_states,
        layer_idx,
        n_components
    )

    explained_var_np = explained_var.cpu().numpy()
    cumulative_var = np.cumsum(explained_var_np)

    fig = go.Figure()

    # Individual variance
    fig.add_trace(go.Bar(
        x=np.arange(n_components),
        y=explained_var_np,
        name="Individual",
        marker_color="lightblue",
        hovertemplate="PC %{x}<br>Variance: %{y:.2%}<extra></extra>"
    ))

    # Cumulative variance
    fig.add_trace(go.Scatter(
        x=np.arange(n_components),
        y=cumulative_var,
        name="Cumulative",
        mode='lines+markers',
        line=dict(color='red', width=2),
        yaxis="y2",
        hovertemplate="PC %{x}<br>Cumulative: %{y:.2%}<extra></extra>"
    ))

    layer_name = hidden_states.layer_names[layer_idx]
    plot_title = title or f"PCA Explained Variance - {layer_name}"

    fig.update_layout(
        title=plot_title,
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance",
        yaxis2=dict(
            title="Cumulative Variance",
            overlaying="y",
            side="right"
        ),
        width=900,
        height=500,
        showlegend=True
    )

    logger.debug(f"Created PCA variance plot for layer {layer_idx}")
    return fig


def create_animation_layer_evolution(
    hidden_states: HiddenStates,
    method: str = "umap",
    title: str = "Token Evolution Animation"
) -> go.Figure:
    """
    Create animated visualization of tokens evolving across layers.

    Args:
        hidden_states: HiddenStates object
        method: Projection method
        title: Plot title

    Returns:
        Plotly Figure with animation
    """
    num_layers = hidden_states.num_layers
    seq_len = hidden_states.seq_len
    tokens = hidden_states.tokens if hidden_states.tokens else [f"T{i}" for i in range(seq_len)]

    # Project each layer separately
    all_projections = []
    for layer_idx in range(num_layers):
        projected = project_hidden_states(
            hidden_states,
            method=method,
            dimensions=2,
            layer_idx=layer_idx
        )
        all_projections.append(projected)

    # Create frames
    frames = []
    for layer_idx, projected in enumerate(all_projections):
        frame = go.Frame(
            data=[go.Scatter(
                x=projected[:, 0],
                y=projected[:, 1],
                mode='markers+text',
                marker=dict(size=10, color=np.arange(seq_len), colorscale='Viridis'),
                text=tokens,
                textposition="top center"
            )],
            name=f"L{layer_idx}"
        )
        frames.append(frame)

    # Initial frame
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    # Add animation controls
    fig.update_layout(
        title=title,
        xaxis_title=f"{method.upper()} 1",
        yaxis_title=f"{method.upper()} 2",
        width=900,
        height=700,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "steps": [
                {
                    "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": f"L{i}",
                    "method": "animate"
                }
                for i, f in enumerate(frames)
            ],
            "x": 0.1,
            "len": 0.9,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top"
        }]
    )

    logger.debug("Created layer evolution animation")
    return fig
