"""
Logit lens for analyzing intermediate predictions.

The logit lens applies the final layer norm and unembedding matrix to
intermediate hidden states to see what the model "thinks" at each layer.
"""

from typing import Optional, List, Tuple
import torch
from loguru import logger

from ..core.types import HiddenStates, LogitLensResults
from ..core.model_wrapper import ModelWrapper


def apply_logit_lens(
    model: ModelWrapper,
    hidden_states: HiddenStates,
    top_k: int = 10,
    normalize: bool = True
) -> LogitLensResults:
    """
    Apply logit lens to hidden states.

    Args:
        model: ModelWrapper instance
        hidden_states: Hidden states to analyze
        top_k: Number of top predictions to return per layer
        normalize: Whether to apply final layer norm

    Returns:
        LogitLensResults with predictions at each layer
    """
    logger.debug(f"Applying logit lens with top_k={top_k}")

    # Get model components
    lm_head = model.model.lm_head  # Unembedding matrix
    final_norm = None

    # Find final layer norm
    if hasattr(model.model, 'model'):
        if hasattr(model.model.model, 'norm'):
            final_norm = model.model.model.norm
        elif hasattr(model.model.model, 'final_layernorm'):
            final_norm = model.model.model.final_layernorm

    num_layers = hidden_states.num_layers
    batch_size = hidden_states.batch_size
    seq_len = hidden_states.seq_len

    # Storage for predictions
    all_predictions = []

    with torch.no_grad():
        for layer_idx in range(num_layers):
            layer_states = hidden_states.states[layer_idx]  # [batch, seq, hidden]

            # Apply final layer norm if requested
            if normalize and final_norm is not None:
                layer_states = final_norm(layer_states)

            # Apply unembedding
            logits = lm_head(layer_states)  # [batch, seq, vocab_size]

            # Get top-k predictions for this layer
            layer_predictions = []
            for batch_idx in range(batch_size):
                batch_predictions = []
                for pos_idx in range(seq_len):
                    position_logits = logits[batch_idx, pos_idx]

                    # Get top-k tokens
                    top_k_values, top_k_indices = torch.topk(position_logits, top_k)

                    # Convert to probabilities
                    probs = torch.softmax(position_logits, dim=-1)
                    top_k_probs = probs[top_k_indices]

                    # Decode tokens
                    top_k_tokens = model.tokenizer.convert_ids_to_tokens(
                        top_k_indices.tolist()
                    )

                    # Store as list of (token, prob, id) tuples
                    position_preds = [
                        (token, prob.item(), idx.item())
                        for token, prob, idx in zip(top_k_tokens, top_k_probs, top_k_indices)
                    ]
                    batch_predictions.append(position_preds)

                layer_predictions.append(batch_predictions)

            all_predictions.append(layer_predictions)

    results = LogitLensResults(
        predictions=all_predictions,
        layer_names=hidden_states.layer_names
    )

    logger.info(f"Applied logit lens across {num_layers} layers")
    return results


def find_answer_emergence_layer(
    logit_lens_results: LogitLensResults,
    target_token: int,
    threshold_rank: int = 5,
    position_idx: int = -1
) -> int:
    """
    Find the layer where target token first appears in top-k predictions.

    Args:
        logit_lens_results: Logit lens results
        target_token: Target token ID to find
        threshold_rank: Consider token as "emerged" if in top-k
        position_idx: Token position to analyze (-1 for last)

    Returns:
        Layer index where token first emerges, or -1 if never emerges
    """
    num_layers = logit_lens_results.num_layers

    for layer_idx in range(num_layers):
        # Get predictions for this layer at position
        layer_preds = logit_lens_results.get_layer_predictions(layer_idx, batch_idx=0)
        position_preds = layer_preds[position_idx]

        # Check if target token is in top-k
        for rank, (token, prob, token_id) in enumerate(position_preds):
            if token_id == target_token and rank < threshold_rank:
                logger.debug(
                    f"Target token {target_token} emerged at layer {layer_idx} "
                    f"(rank {rank}, prob {prob:.4f})"
                )
                return layer_idx

    logger.debug(f"Target token {target_token} never emerged in top-{threshold_rank}")
    return -1


def track_token_rank(
    logit_lens_results: LogitLensResults,
    target_token: int,
    position_idx: int = -1
) -> List[Optional[Tuple[int, float]]]:
    """
    Track the rank and probability of a target token across layers.

    Args:
        logit_lens_results: Logit lens results
        target_token: Target token ID
        position_idx: Token position to analyze

    Returns:
        List of (rank, probability) per layer, None if token not in top-k
    """
    num_layers = logit_lens_results.num_layers
    tracking = []

    for layer_idx in range(num_layers):
        layer_preds = logit_lens_results.get_layer_predictions(layer_idx, batch_idx=0)
        position_preds = layer_preds[position_idx]

        # Find target token
        found = False
        for rank, (token, prob, token_id) in enumerate(position_preds):
            if token_id == target_token:
                tracking.append((rank, prob))
                found = True
                break

        if not found:
            tracking.append(None)

    logger.debug(f"Tracked token {target_token} across {num_layers} layers")
    return tracking


def get_prediction_entropy(
    logit_lens_results: LogitLensResults,
    position_idx: int = -1
) -> List[float]:
    """
    Compute prediction entropy at each layer.

    High entropy indicates uncertain/diffuse predictions.

    Args:
        logit_lens_results: Logit lens results
        position_idx: Token position to analyze

    Returns:
        Entropy values per layer
    """
    num_layers = logit_lens_results.num_layers
    entropies = []

    for layer_idx in range(num_layers):
        layer_preds = logit_lens_results.get_layer_predictions(layer_idx, batch_idx=0)
        position_preds = layer_preds[position_idx]

        # Get probabilities
        probs = torch.tensor([prob for _, prob, _ in position_preds])

        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log2(probs + 1e-10)).sum().item()
        entropies.append(entropy)

    logger.debug(f"Computed prediction entropy across {num_layers} layers")
    return entropies


def compare_predictions_across_layers(
    logit_lens_results: LogitLensResults,
    layer_idx1: int,
    layer_idx2: int,
    position_idx: int = -1,
    top_k: int = 5
) -> float:
    """
    Compute overlap in top-k predictions between two layers.

    Args:
        logit_lens_results: Logit lens results
        layer_idx1: First layer
        layer_idx2: Second layer
        position_idx: Token position to analyze
        top_k: Number of top predictions to compare

    Returns:
        Jaccard similarity (overlap / union)
    """
    layer1_preds = logit_lens_results.get_layer_predictions(layer_idx1, batch_idx=0)
    layer2_preds = logit_lens_results.get_layer_predictions(layer_idx2, batch_idx=0)

    pos1_preds = layer1_preds[position_idx][:top_k]
    pos2_preds = layer2_preds[position_idx][:top_k]

    # Get token IDs
    tokens1 = set(token_id for _, _, token_id in pos1_preds)
    tokens2 = set(token_id for _, _, token_id in pos2_preds)

    # Jaccard similarity
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    similarity = intersection / union if union > 0 else 0.0

    logger.debug(
        f"Prediction overlap between layers {layer_idx1} and {layer_idx2}: "
        f"{similarity:.2%}"
    )
    return similarity


def get_prediction_stability(
    logit_lens_results: LogitLensResults,
    position_idx: int = -1,
    window_size: int = 3
) -> List[float]:
    """
    Measure prediction stability across consecutive layers.

    Args:
        logit_lens_results: Logit lens results
        position_idx: Token position to analyze
        window_size: Number of layers to compare

    Returns:
        Stability scores per layer (higher = more stable)
    """
    num_layers = logit_lens_results.num_layers
    stability_scores = []

    for layer_idx in range(num_layers):
        if layer_idx < window_size:
            stability_scores.append(0.0)
            continue

        # Compare with previous layers in window
        similarities = []
        for prev_idx in range(layer_idx - window_size, layer_idx):
            sim = compare_predictions_across_layers(
                logit_lens_results,
                prev_idx,
                layer_idx,
                position_idx
            )
            similarities.append(sim)

        # Average similarity = stability
        stability = sum(similarities) / len(similarities)
        stability_scores.append(stability)

    logger.debug(f"Computed prediction stability across {num_layers} layers")
    return stability_scores


def find_convergence_layer(
    logit_lens_results: LogitLensResults,
    position_idx: int = -1,
    stability_threshold: float = 0.8,
    window_size: int = 5
) -> int:
    """
    Find the layer where predictions converge (become stable).

    Args:
        logit_lens_results: Logit lens results
        position_idx: Token position to analyze
        stability_threshold: Minimum stability to consider converged
        window_size: Window for stability calculation

    Returns:
        Layer index where convergence occurs, or -1 if never converges
    """
    stability = get_prediction_stability(
        logit_lens_results,
        position_idx,
        window_size
    )

    for layer_idx, score in enumerate(stability):
        if score >= stability_threshold:
            logger.debug(f"Predictions converged at layer {layer_idx} (stability {score:.2%})")
            return layer_idx

    logger.debug("Predictions never fully converged")
    return -1


def get_top_prediction_at_layer(
    logit_lens_results: LogitLensResults,
    layer_idx: int,
    position_idx: int = -1
) -> Tuple[str, float, int]:
    """
    Get the top prediction at a specific layer and position.

    Args:
        logit_lens_results: Logit lens results
        layer_idx: Layer index
        position_idx: Token position

    Returns:
        Tuple of (token_string, probability, token_id)
    """
    layer_preds = logit_lens_results.get_layer_predictions(layer_idx, batch_idx=0)
    position_preds = layer_preds[position_idx]

    top_pred = position_preds[0]  # (token, prob, id)
    return top_pred
