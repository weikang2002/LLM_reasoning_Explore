"""Visualization modules for attention, hidden states, and circuits."""

from .attention_viz import (
    plot_attention_heatmap,
    plot_attention_heads_grid,
    plot_attention_rollout,
    plot_head_importance,
    plot_attention_entropy,
    plot_attention_distance,
    plot_token_attention_evolution,
    export_attention_html,
)

from .hidden_state_viz import (
    project_hidden_states,
    plot_layer_evolution,
    plot_token_clusters,
    plot_hidden_state_norms,
    plot_cosine_similarity_matrix,
    plot_layer_similarity_matrix,
    plot_pca_variance,
    create_animation_layer_evolution,
)

from .graph_viz import (
    build_computation_graph,
    render_graph_interactive,
    export_graphviz,
    plot_circuit_overview,
    plot_circuit_comparison,
)

__all__ = [
    # Attention visualization
    "plot_attention_heatmap",
    "plot_attention_heads_grid",
    "plot_attention_rollout",
    "plot_head_importance",
    "plot_attention_entropy",
    "plot_attention_distance",
    "plot_token_attention_evolution",
    "export_attention_html",

    # Hidden state visualization
    "project_hidden_states",
    "plot_layer_evolution",
    "plot_token_clusters",
    "plot_hidden_state_norms",
    "plot_cosine_similarity_matrix",
    "plot_layer_similarity_matrix",
    "plot_pca_variance",
    "create_animation_layer_evolution",

    # Graph visualization
    "build_computation_graph",
    "render_graph_interactive",
    "export_graphviz",
    "plot_circuit_overview",
    "plot_circuit_comparison",
]
