"""
Gradient-based attribution: Identify which inputs/layers matter most.

Key methods:
- **Token attribution**: Which input tokens are important for output
- **Layer contribution**: How much each layer contributes to final prediction
- **Gradient attention**: Attention weighted by gradients (GradCAM-style)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import torch
import numpy as np
import pandas as pd

from ..core import ModelWrapper
from ..core.types import AttentionPatterns, HiddenStates


@dataclass
class AttributionResults:
    """
    Results from attribution analysis.

    Attributes:
        token_attributions: Importance score for each input token [seq_len]
        layer_contributions: Contribution of each layer [num_layers]
        tokens: Token strings
        target_token: Token ID being attributed
        method: Attribution method used
    """
    token_attributions: np.ndarray  # [seq_len]
    layer_contributions: Optional[np.ndarray] = None  # [num_layers]
    tokens: Optional[List[str]] = None
    target_token: Optional[int] = None
    method: str = "gradient"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert token attributions to DataFrame."""
        if self.tokens is None:
            tokens = [f"T{i}" for i in range(len(self.token_attributions))]
        else:
            tokens = self.tokens

        return pd.DataFrame({
            "token": tokens,
            "attribution": self.token_attributions,
        })

    def get_top_tokens(self, k: int = 10) -> pd.DataFrame:
        """Get top-k most important tokens."""
        df = self.to_dataframe()
        return df.nlargest(k, "attribution")


def compute_token_attribution(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    target_token: int,
    position: int = -1,
    method: str = "gradient",
) -> AttributionResults:
    """
    Compute importance of each input token for target prediction.

    Methods:
    - "gradient": Gradient of target logit w.r.t. input embeddings
    - "integrated_gradient": Integrated gradients (more accurate, slower)
    - "attention": Sum of attention to each token (gradient-free)

    Args:
        model: Model wrapper
        input_ids: Input tensor [1, seq_len]
        target_token: Token ID to attribute
        position: Output position to measure (-1 = last)
        method: Attribution method

    Returns:
        AttributionResults with token importance scores

    Example:
        >>> prompt = "What is 17 * 23? The answer is 391."
        >>> input_ids = model.tokenize(prompt)['input_ids']
        >>> target_token = model.tokenizer.encode("391")[0]
        >>> results = compute_token_attribution(model, input_ids, target_token)
        >>> print(results.get_top_tokens(5))
    """
    input_ids = input_ids.to(model.device)

    if method == "gradient":
        attributions = _gradient_attribution(model, input_ids, target_token, position)
    elif method == "integrated_gradient":
        attributions = _integrated_gradient_attribution(model, input_ids, target_token, position)
    elif method == "attention":
        attributions = _attention_attribution(model, input_ids, position)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get tokens
    tokens = model.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    return AttributionResults(
        token_attributions=attributions,
        tokens=tokens,
        target_token=target_token,
        method=method,
    )


def _gradient_attribution(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    target_token: int,
    position: int,
) -> np.ndarray:
    """Gradient-based attribution: d(logit)/d(embedding)."""
    # Get embeddings with gradients
    embeddings = model.model.get_input_embeddings()(input_ids)
    embeddings.requires_grad_(True)

    # Forward pass
    output = model.model(
        inputs_embeds=embeddings,
        output_hidden_states=False,
        output_attentions=False,
    )

    # Compute gradient of target logit
    target_logit = output.logits[0, position, target_token]
    target_logit.backward()

    # Attribution = gradient * embedding (gradient × input method)
    attributions = (embeddings.grad[0] * embeddings[0]).sum(dim=-1)
    attributions = attributions.abs().cpu().detach().numpy()

    return attributions


def _integrated_gradient_attribution(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    target_token: int,
    position: int,
    steps: int = 50,
) -> np.ndarray:
    """
    Integrated gradients: More accurate attribution.

    Integrates gradients along path from baseline (zero) to input.
    """
    # Get embeddings
    embeddings = model.model.get_input_embeddings()(input_ids).detach()

    # Baseline: zero embeddings
    baseline = torch.zeros_like(embeddings)

    # Interpolate between baseline and input
    accumulated_grads = torch.zeros_like(embeddings[0])

    for alpha in np.linspace(0, 1, steps):
        # Interpolated input
        interpolated = baseline + alpha * (embeddings - baseline)
        interpolated.requires_grad_(True)

        # Forward pass
        output = model.model(
            inputs_embeds=interpolated,
            output_hidden_states=False,
            output_attentions=False,
        )

        # Gradient
        target_logit = output.logits[0, position, target_token]
        target_logit.backward()

        # Accumulate
        accumulated_grads += interpolated.grad[0]

        # Clean up
        if interpolated.grad is not None:
            interpolated.grad.zero_()

    # Average gradients
    avg_grads = accumulated_grads / steps

    # Attribution = avg_gradient * (input - baseline)
    attributions = (avg_grads * (embeddings[0] - baseline[0])).sum(dim=-1)
    attributions = attributions.abs().cpu().detach().numpy()

    return attributions


def _attention_attribution(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    position: int,
) -> np.ndarray:
    """
    Attention-based attribution: Sum attention to each token.

    Gradient-free alternative.
    """
    with torch.no_grad():
        output = model.forward(
            input_ids,
            output_attentions=True,
        )

    # Sum attention across all layers and heads
    # attention: [num_layers, batch, num_heads, seq_len, seq_len]
    attention = torch.stack([attn[0] for attn in output.attentions])  # [layers, heads, seq, seq]

    # Sum attention TO each source token FROM target position
    target_position = position if position >= 0 else input_ids.shape[1] + position
    attention_to_sources = attention[:, :, target_position, :].sum(dim=(0, 1))  # [seq_len]

    attributions = attention_to_sources.cpu().numpy()
    return attributions


def compute_layer_contribution(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    target_token: int,
    position: int = -1,
    method: str = "gradient",
) -> AttributionResults:
    """
    Compute contribution of each layer to final prediction.

    Methods:
    - "gradient": Gradient norm at each layer
    - "logit_diff": Change in target logit when ablating layer
    - "norm": Hidden state norm at each layer

    Args:
        model: Model wrapper
        input_ids: Input tensor [1, seq_len]
        target_token: Token ID to attribute
        position: Output position (-1 = last)
        method: Contribution method

    Returns:
        AttributionResults with layer contributions

    Example:
        >>> results = compute_layer_contribution(model, input_ids, target_token)
        >>> plt.plot(results.layer_contributions)
        >>> plt.xlabel("Layer")
        >>> plt.ylabel("Contribution")
    """
    input_ids = input_ids.to(model.device)

    if method == "gradient":
        contributions = _gradient_layer_contribution(model, input_ids, target_token, position)
    elif method == "logit_diff":
        contributions = _ablation_layer_contribution(model, input_ids, target_token, position)
    elif method == "norm":
        contributions = _norm_layer_contribution(model, input_ids, position)
    else:
        raise ValueError(f"Unknown method: {method}")

    tokens = model.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    return AttributionResults(
        token_attributions=np.zeros(len(tokens)),  # Not computed
        layer_contributions=contributions,
        tokens=tokens,
        target_token=target_token,
        method=method,
    )


def _gradient_layer_contribution(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    target_token: int,
    position: int,
) -> np.ndarray:
    """Compute layer contribution via gradient norm."""
    # Forward pass with hidden states
    output = model.forward(
        input_ids,
        output_hidden_states=True,
    )

    # Get target logit
    target_logit = output.logits[0, position, target_token]

    # Compute gradient w.r.t. each layer's hidden state
    contributions = []

    for layer_hidden in output.hidden_states[1:]:  # Skip embedding layer
        # Compute gradient
        if layer_hidden.requires_grad:
            grad = torch.autograd.grad(
                target_logit,
                layer_hidden,
                retain_graph=True,
                create_graph=False,
            )[0]
            # Use gradient norm as contribution
            contribution = grad.norm(dim=-1).mean().item()
        else:
            # Enable gradients
            layer_hidden.requires_grad_(True)
            target_logit = output.logits[0, position, target_token]
            grad = torch.autograd.grad(
                target_logit,
                layer_hidden,
                retain_graph=True,
            )[0]
            contribution = grad.norm(dim=-1).mean().item()

        contributions.append(contribution)

    return np.array(contributions)


def _ablation_layer_contribution(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    target_token: int,
    position: int,
) -> np.ndarray:
    """Compute layer contribution via ablation."""
    # Baseline logit
    with torch.no_grad():
        baseline_output = model.forward(input_ids)
        baseline_logit = baseline_output.logits[0, position, target_token].item()

    contributions = []

    # Ablate each layer and measure logit change
    for layer_idx in range(model.num_layers):
        # TODO: Implement layer ablation via hooks
        # For now, use placeholder
        contribution = 0.0
        contributions.append(contribution)

    return np.array(contributions)


def _norm_layer_contribution(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    position: int,
) -> np.ndarray:
    """Layer contribution = hidden state norm (proxy for information content)."""
    with torch.no_grad():
        output = model.forward(
            input_ids,
            output_hidden_states=True,
        )

    contributions = []
    for hidden_state in output.hidden_states[1:]:  # Skip embedding
        # Use norm at target position as contribution
        norm = hidden_state[0, position].norm().item()
        contributions.append(norm)

    return np.array(contributions)


def compute_gradient_attention(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    target_token: int,
    position: int = -1,
) -> AttentionPatterns:
    """
    Compute gradient-weighted attention (GradCAM for attention).

    Weights attention by gradient to identify which attention heads
    are most important for target prediction.

    Args:
        model: Model wrapper
        input_ids: Input tensor [1, seq_len]
        target_token: Token ID to attribute
        position: Output position (-1 = last)

    Returns:
        AttentionPatterns with gradient weights applied

    Example:
        >>> grad_attention = compute_gradient_attention(model, input_ids, target_token)
        >>> # Visualize gradient-weighted attention
        >>> from interpretability.visualization import plot_attention_heatmap
        >>> fig = plot_attention_heatmap(grad_attention, tokens, layer=15)
    """
    input_ids = input_ids.to(model.device)

    # Forward pass with attention
    output = model.forward(
        input_ids,
        output_attentions=True,
    )

    # Get target logit
    target_logit = output.logits[0, position, target_token]

    # Compute gradients for each attention layer
    grad_attentions = []

    for attn in output.attentions:
        if not attn.requires_grad:
            attn.requires_grad_(True)

        # Gradient of target w.r.t. attention
        grad = torch.autograd.grad(
            target_logit,
            attn,
            retain_graph=True,
            create_graph=False,
        )[0]

        # Weight attention by gradient
        grad_weighted = attn * grad
        grad_attentions.append(grad_weighted.detach())

    # Stack into attention patterns
    patterns_tensor = torch.stack(grad_attentions)  # [layers, batch, heads, seq, seq]

    tokens = model.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    num_heads = patterns_tensor.shape[2]

    return AttentionPatterns(
        patterns=patterns_tensor,
        layer_names=[f"layer_{i}" for i in range(len(grad_attentions))],
        num_heads_per_layer=[num_heads] * len(grad_attentions),
        tokens=tokens,
    )


def visualize_token_attribution(
    attribution_results: AttributionResults,
    output_format: str = "text",
) -> str:
    """
    Create text visualization of token attributions.

    Args:
        attribution_results: Attribution results
        output_format: "text" or "html"

    Returns:
        Formatted string

    Example:
        >>> results = compute_token_attribution(model, input_ids, target_token)
        >>> print(visualize_token_attribution(results))
    """
    if attribution_results.tokens is None:
        return "No tokens available"

    # Normalize attributions to 0-1
    attrs = attribution_results.token_attributions
    attrs_norm = (attrs - attrs.min()) / (attrs.max() - attrs.min() + 1e-10)

    if output_format == "text":
        lines = []
        for token, attr in zip(attribution_results.tokens, attrs_norm):
            # Create bar visualization
            bar_length = int(attr * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"{token:15s} {bar} {attr:.3f}")
        return "\n".join(lines)

    elif output_format == "html":
        # Create HTML with color-coded tokens
        html_parts = []
        for token, attr in zip(attribution_results.tokens, attrs_norm):
            # Color from white (0) to red (1)
            red = int(255 * attr)
            color = f"rgb({red}, {255 - red}, {255 - red})"
            html_parts.append(
                f'<span style="background-color: {color}; padding: 2px 4px; margin: 1px;">'
                f'{token}</span>'
            )
        return " ".join(html_parts)

    else:
        raise ValueError(f"Unknown format: {output_format}")
