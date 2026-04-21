"""
Computational graph visualization for circuit analysis.

Visualize information flow through the model as an interactive graph.
"""

from typing import List, Optional, Dict, Tuple, Set
import numpy as np
import networkx as nx
import plotly.graph_objects as go

from ..core.types import Circuit, Component
from ..analysis.information_flow import FlowGraph


def build_computation_graph(
    flow_graph: FlowGraph,
    circuit: Optional[Circuit] = None,
    layers_to_show: Optional[List[int]] = None,
    positions_to_show: Optional[List[int]] = None,
    min_weight: float = 0.1,
) -> nx.DiGraph:
    """
    Build computation graph for visualization.

    Args:
        flow_graph: Information flow graph
        circuit: Optional circuit to highlight
        layers_to_show: Subset of layers to visualize (None = all)
        positions_to_show: Subset of token positions to show (None = all)
        min_weight: Minimum edge weight to include

    Returns:
        NetworkX DiGraph ready for visualization

    Example:
        >>> graph = build_computation_graph(
        ...     flow_graph,
        ...     circuit=discovered_circuit,
        ...     layers_to_show=[10, 15, 20],
        ...     min_weight=0.2
        ... )
        >>> fig = render_graph_interactive(graph)
    """
    # Filter flow graph
    if layers_to_show is None:
        layers_to_show = list(range(flow_graph.num_layers))
    if positions_to_show is None:
        positions_to_show = list(range(len(flow_graph.tokens)))

    # Create subgraph
    nodes_to_keep = set()
    for layer in layers_to_show:
        for pos in positions_to_show:
            nodes_to_keep.add((layer, pos))

    subgraph = flow_graph.graph.subgraph(nodes_to_keep).copy()

    # Filter edges by weight
    edges_to_remove = [
        (u, v) for u, v, data in subgraph.edges(data=True)
        if data.get("weight", 0) < min_weight
    ]
    subgraph.remove_edges_from(edges_to_remove)

    # Mark circuit components if provided
    if circuit is not None:
        circuit_nodes = set()
        for comp in circuit.components:
            if comp.type == "attention" or comp.type == "residual":
                # Mark all positions at this layer
                for pos in positions_to_show:
                    if (comp.layer, pos) in subgraph:
                        circuit_nodes.add((comp.layer, pos))

        nx.set_node_attributes(subgraph, False, "in_circuit")
        for node in circuit_nodes:
            subgraph.nodes[node]["in_circuit"] = True

    return subgraph


def render_graph_interactive(
    graph: nx.DiGraph,
    layout: str = "hierarchical",
    title: str = "Computation Graph",
    width: int = 1000,
    height: int = 800,
) -> go.Figure:
    """
    Render computation graph as interactive Plotly figure.

    Args:
        graph: NetworkX graph from build_computation_graph
        layout: Layout algorithm ("hierarchical", "spring", "circular")
        title: Plot title
        width: Figure width
        height: Figure height

    Returns:
        Plotly figure

    Example:
        >>> fig = render_graph_interactive(graph, layout="hierarchical")
        >>> fig.show()
        >>> fig.write_html("computation_graph.html")
    """
    if len(graph.nodes) == 0:
        # Empty graph
        fig = go.Figure()
        fig.add_annotation(
            text="No nodes to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig

    # Compute layout
    if layout == "hierarchical":
        pos = _hierarchical_layout(graph)
    elif layout == "spring":
        pos = nx.spring_layout(graph, k=1.0, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Extract edge coordinates
    edge_traces = []

    # Group edges by weight for different styling
    edges_by_weight = {}
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        if weight not in edges_by_weight:
            edges_by_weight[weight] = []
        edges_by_weight[weight].append((u, v, data))

    # Create trace for each weight group
    for weight, edges in edges_by_weight.items():
        edge_x = []
        edge_y = []
        for u, v, data in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Color and width based on weight
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(
                width=max(0.5, weight * 3),
                color=f"rgba(150, 150, 150, {weight})",
            ),
            hoverinfo="none",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    # Extract node coordinates
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Get node info
        layer, position = node
        token = graph.nodes[node].get("token", f"T{position}")
        in_circuit = graph.nodes[node].get("in_circuit", False)

        node_text.append(f"Layer {layer}<br>Token: {token}<br>Position: {position}")

        # Color: red if in circuit, blue otherwise
        node_colors.append("red" if in_circuit else "lightblue")

        # Size: larger if in circuit
        node_sizes.append(15 if in_circuit else 10)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="white"),
        ),
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=title,
        titlefont_size=16,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=width,
        height=height,
    )

    return fig


def export_graphviz(
    graph: nx.DiGraph,
    output_path: str,
    format: str = "svg",
) -> None:
    """
    Export graph using Graphviz for publication-quality rendering.

    Requires: pygraphviz or pydot

    Args:
        graph: NetworkX graph
        output_path: Output file path
        format: Output format ("svg", "png", "pdf")

    Example:
        >>> export_graphviz(graph, "circuit.svg", format="svg")
    """
    try:
        from networkx.drawing.nx_agraph import to_agraph
    except ImportError:
        try:
            from networkx.drawing.nx_pydot import to_pydot
            pydot_graph = to_pydot(graph)
            pydot_graph.write(output_path, format=format)
            return
        except ImportError:
            raise ImportError(
                "Graphviz export requires pygraphviz or pydot.\n"
                "Install with: pip install pygraphviz  OR  pip install pydot"
            )

    # Use pygraphviz
    agraph = to_agraph(graph)
    agraph.layout(prog="dot")  # Hierarchical layout
    agraph.draw(output_path, format=format)


def plot_circuit_overview(
    circuit: Circuit,
    num_layers: int,
    num_heads: int,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create overview visualization of circuit components.

    Shows which attention heads and layers are in the circuit.

    Args:
        circuit: Discovered circuit
        num_layers: Total number of layers in model
        num_heads: Number of attention heads per layer
        title: Plot title

    Returns:
        Plotly heatmap figure

    Example:
        >>> fig = plot_circuit_overview(circuit, num_layers=28, num_heads=16)
        >>> fig.show()
    """
    # Create matrix: layers × heads
    matrix = np.zeros((num_layers, num_heads))

    for comp, importance in zip(circuit.components, circuit.importance_scores):
        if comp.type == "attention" and comp.head is not None:
            matrix[comp.layer, comp.head] = importance

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"H{i}" for i in range(num_heads)],
        y=[f"L{i}" for i in range(num_layers)],
        colorscale="Reds",
        hovertemplate="Layer %{y}<br>Head %{x}<br>Importance: %{z:.3f}<extra></extra>",
    ))

    if title is None:
        title = f"Circuit Overview ({len(circuit.components)} components)"

    fig.update_layout(
        title=title,
        xaxis_title="Attention Head",
        yaxis_title="Layer",
        yaxis=dict(autorange="reversed"),
    )

    return fig


def plot_circuit_comparison(
    circuits: List[Circuit],
    circuit_names: List[str],
    num_layers: int,
    num_heads: int,
) -> go.Figure:
    """
    Compare multiple circuits side-by-side.

    Args:
        circuits: List of circuits to compare
        circuit_names: Names for each circuit
        num_layers: Total layers
        num_heads: Heads per layer

    Returns:
        Plotly figure with subplots

    Example:
        >>> circuits = [easy_circuit, medium_circuit, hard_circuit]
        >>> names = ["Easy Math", "Medium Math", "Hard Math"]
        >>> fig = plot_circuit_comparison(circuits, names, num_layers=28, num_heads=16)
    """
    from plotly.subplots import make_subplots

    n_circuits = len(circuits)
    fig = make_subplots(
        rows=1,
        cols=n_circuits,
        subplot_titles=circuit_names,
    )

    for i, (circuit, name) in enumerate(zip(circuits, circuit_names)):
        # Create matrix
        matrix = np.zeros((num_layers, num_heads))
        for comp, importance in zip(circuit.components, circuit.importance_scores):
            if comp.type == "attention" and comp.head is not None:
                matrix[comp.layer, comp.head] = importance

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=[f"H{j}" for j in range(num_heads)],
                y=[f"L{j}" for j in range(num_layers)],
                colorscale="Reds",
                showscale=(i == n_circuits - 1),  # Only show scale on last subplot
                hovertemplate=f"{name}<br>Layer %{{y}}<br>Head %{{x}}<br>Importance: %{{z:.3f}}<extra></extra>",
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title="Circuit Comparison",
        height=400,
    )

    # Update y-axes to reverse
    for i in range(1, n_circuits + 1):
        fig.update_yaxes(autorange="reversed", row=1, col=i)

    return fig


# Helper functions

def _hierarchical_layout(graph: nx.DiGraph) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Create hierarchical layout based on layer structure.

    Layers are arranged vertically, positions horizontally.
    """
    pos = {}

    # Group nodes by layer
    layers = {}
    for node in graph.nodes():
        layer, position = node
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(position)

    # Arrange positions
    max_layer = max(layers.keys())
    max_positions = max(len(positions) for positions in layers.values())

    for layer, positions in layers.items():
        y = 1.0 - (layer / max(max_layer, 1))  # Top to bottom
        for i, position in enumerate(sorted(positions)):
            x = (i + 0.5) / max(len(positions), 1)  # Left to right
            pos[(layer, position)] = (x, y)

    return pos
