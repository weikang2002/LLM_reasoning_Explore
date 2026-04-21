"""Extraction modules for attention, hidden states, and logit lens."""

from .attention import (
    extract_attention_patterns,
    compute_attention_rollout,
    get_head_importance,
    compute_attention_entropy,
    get_attention_to_token,
    get_attention_from_token,
    aggregate_attention_heads,
    find_induction_heads,
    compute_attention_distance,
)

from .hidden_states import (
    extract_hidden_states,
    decompose_residual_stream,
    compute_hidden_state_norms,
    compute_cosine_similarity,
    get_token_evolution,
    compute_layer_similarity,
    compute_principal_components,
    detect_outlier_tokens,
    compute_hidden_state_stats,
)

from .logit_lens import (
    apply_logit_lens,
    find_answer_emergence_layer,
    track_token_rank,
    get_prediction_entropy,
    compare_predictions_across_layers,
    get_prediction_stability,
    find_convergence_layer,
    get_top_prediction_at_layer,
)

__all__ = [
    # Attention
    "extract_attention_patterns",
    "compute_attention_rollout",
    "get_head_importance",
    "compute_attention_entropy",
    "get_attention_to_token",
    "get_attention_from_token",
    "aggregate_attention_heads",
    "find_induction_heads",
    "compute_attention_distance",

    # Hidden States
    "extract_hidden_states",
    "decompose_residual_stream",
    "compute_hidden_state_norms",
    "compute_cosine_similarity",
    "get_token_evolution",
    "compute_layer_similarity",
    "compute_principal_components",
    "detect_outlier_tokens",
    "compute_hidden_state_stats",

    # Logit Lens
    "apply_logit_lens",
    "find_answer_emergence_layer",
    "track_token_rank",
    "get_prediction_entropy",
    "compare_predictions_across_layers",
    "get_prediction_stability",
    "find_convergence_layer",
    "get_top_prediction_at_layer",
]
