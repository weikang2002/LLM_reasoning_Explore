"""
Information flow analysis: Track how information propagates through the model.

Key concepts:
- **Flow graph**: Token-to-token information flow weighted by attention
- **Reasoning path**: Sequence of tokens/layers showing how premises lead to conclusions
- **Bottleneck layers**: Layers where information compresses (low entropy, high focus)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
import torch
import numpy as np
import networkx as nx
from ..core.types import AttentionPatterns, HiddenStates


@dataclass
class FlowGraph:
    """
    Graph representing information flow through the model.

    Nodes: (layer, token_position) tuples
    Edges: Attention weights (source→target)

    Attributes:
        graph: NetworkX directed graph
        tokens: Token strings for each position
        num_layers: Number of layers
        attention_patterns: Original attention patterns (for reference)
    """
    graph: nx.DiGraph
    tokens: List[str]
    num_layers: int
    attention_patterns: AttentionPatterns

    def get_flow_to_token(
        self,
        target_position: int,
        layer: int,
        threshold: float = 0.1
    ) -> List[Tuple[int, int, float]]:
        """
        Get all tokens flowing into target at given layer.

        Args:
            target_position: Target token position
            layer: Layer index
            threshold: Minimum attention weight to include

        Returns:
            List of (source_layer, source_position, weight) tuples
        """
        target_node = (layer, target_position)
        if target_node not in self.graph:
            return []

        flows = []
        for source_node in self.graph.predecessors(target_node):
            weight = self.graph[source_node][target_node]["weight"]
            if weight >= threshold:
                flows.append((source_node[0], source_node[1], weight))

        return sorted(flows, key=lambda x: x[2], reverse=True)

    def get_flow_from_token(
        self,
        source_position: int,
        layer: int,
        threshold: float = 0.1
    ) -> List[Tuple[int, int, float]]:
        """Get all tokens receiving flow from source at given layer."""
        source_node = (layer, source_position)
        if source_node not in self.graph:
            return []

        flows = []
        for target_node in self.graph.successors(source_node):
            weight = self.graph[source_node][target_node]["weight"]
            if weight >= threshold:
                flows.append((target_node[0], target_node[1], weight))

        return sorted(flows, key=lambda x: x[2], reverse=True)


@dataclass
class ReasoningPath:
    """
    A reasoning path from premise tokens to conclusion token.

    Attributes:
        nodes: List of (layer, position) nodes in the path
        weights: Edge weights along the path
        tokens: Token strings at each position
        total_flow: Product of all edge weights
        path_length: Number of hops in path
    """
    nodes: List[Tuple[int, int]]
    weights: List[float]
    tokens: List[str]
    total_flow: float
    path_length: int

    def __str__(self) -> str:
        path_str = " → ".join([
            f"L{layer}:{token}" for (layer, pos), token in zip(self.nodes, self.tokens)
        ])
        return f"Path (flow={self.total_flow:.3f}): {path_str}"


def compute_information_flow(
    attention_patterns: AttentionPatterns,
    aggregation: str = "max",
    threshold: float = 0.01,
) -> FlowGraph:
    """
    Build information flow graph from attention patterns.

    The graph has nodes (layer, token_position) and edges weighted by attention.
    Attention is aggregated across heads using max or mean.

    Args:
        attention_patterns: Attention patterns from model
        aggregation: How to aggregate across heads ("max", "mean", "sum")
        threshold: Minimum attention weight for edge creation

    Returns:
        FlowGraph with nodes and edges

    Example:
        >>> from interpretability.extraction import extract_attention_patterns
        >>> attention = extract_attention_patterns(model, input_ids)
        >>> flow_graph = compute_information_flow(attention)
        >>> paths = trace_reasoning_path(
        ...     flow_graph,
        ...     source_tokens=[0, 1],  # Premise tokens
        ...     target_token=10  # Conclusion token
        ... )
    """
    # Aggregate attention across heads
    # attention_patterns.patterns: [num_layers, batch, num_heads, seq_len, seq_len]
    attn = attention_patterns.patterns[:, 0, :, :, :]  # [layers, heads, seq, seq]

    if aggregation == "max":
        attn_agg = attn.max(dim=1)[0]  # [layers, seq, seq]
    elif aggregation == "mean":
        attn_agg = attn.mean(dim=1)  # [layers, seq, seq]
    elif aggregation == "sum":
        attn_agg = attn.sum(dim=1)  # [layers, seq, seq]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    num_layers, seq_len, _ = attn_agg.shape
    tokens = attention_patterns.tokens or [f"T{i}" for i in range(seq_len)]

    # Build graph
    G = nx.DiGraph()

    # Add nodes for each (layer, position)
    for layer in range(num_layers):
        for pos in range(seq_len):
            G.add_node((layer, pos), token=tokens[pos], layer=layer, position=pos)

    # Add edges for attention weights above threshold
    for layer in range(num_layers):
        for target_pos in range(seq_len):
            for source_pos in range(seq_len):
                weight = attn_agg[layer, target_pos, source_pos].item()
                if weight >= threshold:
                    # Edge from (layer, source) to (layer, target)
                    G.add_edge(
                        (layer, source_pos),
                        (layer, target_pos),
                        weight=weight,
                        layer=layer,
                    )

    # Add inter-layer edges (information flows forward through residual stream)
    # Model as full connectivity between layers with weight 1.0
    for layer in range(num_layers - 1):
        for pos in range(seq_len):
            G.add_edge(
                (layer, pos),
                (layer + 1, pos),
                weight=1.0,
                layer=layer,
                is_residual=True,
            )

    return FlowGraph(
        graph=G,
        tokens=tokens,
        num_layers=num_layers,
        attention_patterns=attention_patterns,
    )


def trace_reasoning_path(
    flow_graph: FlowGraph,
    source_tokens: List[int],
    target_token: int,
    target_layer: Optional[int] = None,
    max_paths: int = 5,
    min_flow: float = 1e-4,
) -> List[ReasoningPath]:
    """
    Find reasoning paths from source tokens to target token.

    Uses shortest path with highest total flow (product of edge weights).

    Args:
        flow_graph: Information flow graph
        source_tokens: List of source token positions (e.g., premise tokens)
        target_token: Target token position (e.g., answer token)
        target_layer: Layer to end at (None = last layer)
        max_paths: Maximum number of paths to return
        min_flow: Minimum total flow for a path to be included

    Returns:
        List of ReasoningPath objects, sorted by total flow

    Example:
        >>> # Find paths from tokens 0,1 (premises) to token 10 (answer)
        >>> paths = trace_reasoning_path(
        ...     flow_graph,
        ...     source_tokens=[0, 1],
        ...     target_token=10,
        ...     max_paths=3
        ... )
        >>> for path in paths:
        ...     print(path)
    """
    if target_layer is None:
        target_layer = flow_graph.num_layers - 1

    target_node = (target_layer, target_token)
    if target_node not in flow_graph.graph:
        return []

    # Find paths from each source
    all_paths = []

    for source_pos in source_tokens:
        source_node = (0, source_pos)  # Start at layer 0
        if source_node not in flow_graph.graph:
            continue

        # Find all simple paths
        try:
            paths = nx.all_simple_paths(
                flow_graph.graph,
                source_node,
                target_node,
                cutoff=flow_graph.num_layers * 2  # Reasonable cutoff
            )

            for path in paths:
                # Compute total flow (product of edge weights)
                weights = []
                for i in range(len(path) - 1):
                    weight = flow_graph.graph[path[i]][path[i + 1]]["weight"]
                    weights.append(weight)

                total_flow = float(np.prod(weights))

                if total_flow >= min_flow:
                    path_tokens = [flow_graph.tokens[pos] for layer, pos in path]
                    all_paths.append(
                        ReasoningPath(
                            nodes=path,
                            weights=weights,
                            tokens=path_tokens,
                            total_flow=total_flow,
                            path_length=len(path) - 1,
                        )
                    )
        except nx.NetworkXNoPath:
            continue

    # Sort by total flow and return top-k
    all_paths.sort(key=lambda p: p.total_flow, reverse=True)
    return all_paths[:max_paths]


def detect_bottleneck_layers(
    flow_graph: FlowGraph,
    attention_patterns: Optional[AttentionPatterns] = None,
    method: str = "entropy",
) -> List[Tuple[int, float]]:
    """
    Identify layers where information compresses (bottlenecks).

    Bottleneck indicators:
    - Low entropy: Attention is focused on few tokens
    - High centrality: Many paths go through specific nodes
    - Low rank: Hidden states are low-dimensional

    Args:
        flow_graph: Information flow graph
        attention_patterns: Attention patterns (for entropy calculation)
        method: Detection method ("entropy", "centrality", "both")

    Returns:
        List of (layer, bottleneck_score) tuples, sorted by score

    Example:
        >>> bottlenecks = detect_bottleneck_layers(flow_graph)
        >>> print(f"Bottleneck layers: {[layer for layer, score in bottlenecks[:3]]}")
    """
    if attention_patterns is None:
        attention_patterns = flow_graph.attention_patterns

    bottleneck_scores = []

    for layer in range(flow_graph.num_layers):
        score = 0.0

        if method in ["entropy", "both"]:
            # Compute average attention entropy at this layer
            # Low entropy = focused attention = potential bottleneck
            attn = attention_patterns.patterns[layer, 0]  # [heads, seq, seq]
            entropy = _compute_attention_entropy(attn)
            # Invert so low entropy = high bottleneck score
            max_entropy = np.log(attn.shape[-1])  # Maximum possible entropy
            entropy_score = 1.0 - (entropy / max_entropy)
            score += entropy_score

        if method in ["centrality", "both"]:
            # Compute betweenness centrality for nodes at this layer
            # High centrality = many paths go through this layer
            layer_nodes = [(layer, pos) for pos in range(len(flow_graph.tokens))]
            centrality = nx.betweenness_centrality(
                flow_graph.graph,
                normalized=True,
                weight="weight"
            )
            avg_centrality = np.mean([centrality.get(node, 0.0) for node in layer_nodes])
            score += avg_centrality

        bottleneck_scores.append((layer, score))

    # Sort by score (higher = more bottleneck-like)
    bottleneck_scores.sort(key=lambda x: x[1], reverse=True)
    return bottleneck_scores


def compute_path_diversity(
    flow_graph: FlowGraph,
    source_tokens: List[int],
    target_token: int,
    target_layer: Optional[int] = None,
) -> float:
    """
    Measure diversity of reasoning paths from sources to target.

    High diversity = many independent paths (robust reasoning)
    Low diversity = few paths (brittle reasoning, potential bottleneck)

    Args:
        flow_graph: Information flow graph
        source_tokens: Source token positions
        target_token: Target token position
        target_layer: Target layer (None = last)

    Returns:
        Diversity score (0-1, higher = more diverse)
    """
    paths = trace_reasoning_path(
        flow_graph,
        source_tokens,
        target_token,
        target_layer,
        max_paths=100,  # Find many paths
    )

    if len(paths) == 0:
        return 0.0

    # Measure path diversity: count unique intermediate nodes
    all_intermediate_nodes = set()
    for path in paths:
        # Exclude source and target nodes
        intermediate = path.nodes[1:-1]
        all_intermediate_nodes.update(intermediate)

    # Normalize by maximum possible unique nodes
    max_nodes = (flow_graph.num_layers - 2) * len(flow_graph.tokens)
    diversity = len(all_intermediate_nodes) / max(max_nodes, 1)

    return diversity


def identify_critical_tokens(
    flow_graph: FlowGraph,
    target_token: int,
    target_layer: Optional[int] = None,
    top_k: int = 10,
) -> List[Tuple[int, int, float]]:
    """
    Identify tokens that are critical for target token.

    Critical = high total flow to target across all paths.

    Args:
        flow_graph: Information flow graph
        target_token: Target token position
        target_layer: Target layer (None = last)
        top_k: Number of critical tokens to return

    Returns:
        List of (layer, position, total_flow) tuples

    Example:
        >>> critical = identify_critical_tokens(flow_graph, target_token=10)
        >>> for layer, pos, flow in critical[:5]:
        ...     print(f"Token '{tokens[pos]}' at layer {layer}: {flow:.3f}")
    """
    if target_layer is None:
        target_layer = flow_graph.num_layers - 1

    target_node = (target_layer, target_token)

    # Compute total flow from each (layer, pos) to target
    token_flows = {}

    for layer in range(target_layer + 1):
        for pos in range(len(flow_graph.tokens)):
            source_node = (layer, pos)
            if source_node == target_node:
                continue

            # Sum flow over all paths from source to target
            try:
                # Use shortest path with weights
                path = nx.shortest_path(
                    flow_graph.graph,
                    source_node,
                    target_node,
                    weight=lambda u, v, d: -np.log(d.get("weight", 1e-10)),  # -log for shortest
                )

                # Compute flow along this path
                flow = 1.0
                for i in range(len(path) - 1):
                    flow *= flow_graph.graph[path[i]][path[i + 1]]["weight"]

                token_flows[(layer, pos)] = token_flows.get((layer, pos), 0.0) + flow

            except nx.NetworkXNoPath:
                continue

    # Sort by total flow
    critical_tokens = sorted(
        [(layer, pos, flow) for (layer, pos), flow in token_flows.items()],
        key=lambda x: x[2],
        reverse=True
    )

    return critical_tokens[:top_k]


# Helper functions

def _compute_attention_entropy(attention: torch.Tensor) -> float:
    """
    Compute average entropy of attention distribution.

    Args:
        attention: Attention tensor [heads, seq_len, seq_len]

    Returns:
        Average entropy across all heads and target positions
    """
    # Ensure probabilities sum to 1 (they should, but numerical errors)
    attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-10)

    # Compute entropy: -sum(p * log(p))
    log_attn = torch.log(attention + 1e-10)
    entropy = -(attention * log_attn).sum(dim=-1)  # [heads, seq_len]

    # Average across heads and positions
    return entropy.mean().item()
