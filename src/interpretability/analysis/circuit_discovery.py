"""
Circuit discovery via activation patching and ablation.

Key concepts:
- **Activation patching**: Replace activations from a clean run with those from a
  corrupted run to measure causal effect on output
- **Ablation**: Zero out specific components to measure their importance
- **Circuit**: Minimal set of components that preserve model performance
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable, Union
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..core import ModelWrapper
from ..core.types import Component, Circuit


@dataclass
class PatchingResults:
    """Results from activation patching experiment.

    Attributes:
        components: List of components tested
        logit_diffs: Difference in target logit when patching each component
        patching_effects: Normalized effect (0 = no effect, 1 = full restoration)
        clean_logit: Baseline logit on clean input
        corrupted_logit: Logit on corrupted input (no patching)
        target_token: Token ID being predicted
    """
    components: List[Component]
    logit_diffs: np.ndarray  # [num_components]
    patching_effects: np.ndarray  # [num_components], normalized
    clean_logit: float
    corrupted_logit: float
    target_token: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        records = []
        for comp, logit_diff, effect in zip(
            self.components, self.logit_diffs, self.patching_effects
        ):
            records.append({
                "component_type": comp.type,
                "layer": comp.layer,
                "head": comp.head,
                "component_name": comp.name,
                "logit_diff": logit_diff,
                "patching_effect": effect,
            })
        return pd.DataFrame(records)

    def get_top_components(self, k: int = 10, by: str = "patching_effect") -> pd.DataFrame:
        """Get top-k most important components."""
        df = self.to_dataframe()
        return df.nlargest(k, by)


@dataclass
class AblationResults:
    """Results from component ablation experiment.

    Attributes:
        components: Components that were ablated
        ablated_logits: Logits after ablating each component
        importance_scores: Importance score (clean_logit - ablated_logit)
        baseline_logit: Logit without ablation
        target_token: Token ID being predicted
    """
    components: List[Component]
    ablated_logits: np.ndarray  # [num_components]
    importance_scores: np.ndarray  # [num_components]
    baseline_logit: float
    target_token: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        records = []
        for comp, logit, score in zip(
            self.components, self.ablated_logits, self.importance_scores
        ):
            records.append({
                "component_type": comp.type,
                "layer": comp.layer,
                "head": comp.head,
                "component_name": comp.name,
                "ablated_logit": logit,
                "importance_score": score,
            })
        return pd.DataFrame(records)

    def get_top_components(self, k: int = 10) -> pd.DataFrame:
        """Get top-k most important components."""
        df = self.to_dataframe()
        return df.nlargest(k, "importance_score")


def activation_patching_experiment(
    model: ModelWrapper,
    clean_input_ids: torch.Tensor,
    corrupted_input_ids: torch.Tensor,
    target_token: int,
    components: Optional[List[Component]] = None,
    layers: Optional[List[int]] = None,
    component_types: List[str] = ["attention", "mlp"],
    position: int = -1,
    show_progress: bool = True,
) -> PatchingResults:
    """
    Run activation patching experiment to identify critical components.

    **How it works**:
    1. Run clean input through model, cache activations (A_clean)
    2. Run corrupted input through model, cache activations (A_corrupted)
    3. For each component:
       - Run clean input but patch in corrupted activation for that component
       - Measure change in target token logit
       - Large change = component is important for correct reasoning

    Args:
        model: Model wrapper
        clean_input_ids: Input with correct reasoning [1, seq_len]
        corrupted_input_ids: Input with incorrect reasoning [1, seq_len]
        target_token: Token ID to measure (typically the answer)
        components: Specific components to test (if None, test all in layers)
        layers: Layer indices to test (if None, test all layers)
        component_types: Types of components to test ("attention", "mlp", "residual")
        position: Token position to measure (-1 = last token)
        show_progress: Show progress bar

    Returns:
        PatchingResults with logit differences and importance scores

    Example:
        >>> clean = "What is 2+2? The answer is 4."
        >>> corrupted = "What is 2+2? The answer is 5."
        >>> results = activation_patching_experiment(
        ...     model,
        ...     model.tokenize(clean)['input_ids'],
        ...     model.tokenize(corrupted)['input_ids'],
        ...     target_token=model.tokenizer.encode("4")[0]
        ... )
        >>> print(results.get_top_components(5))
    """
    device = model.device
    clean_input_ids = clean_input_ids.to(device)
    corrupted_input_ids = corrupted_input_ids.to(device)

    # Determine components to test
    if components is None:
        if layers is None:
            layers = list(range(model.num_layers))
        components = _enumerate_components(model, layers, component_types)

    # Run clean forward pass and get baseline
    with torch.no_grad():
        clean_output = model.forward(
            clean_input_ids,
            output_hidden_states=True,
            output_attentions=True
        )
        clean_logit = clean_output.logits[0, position, target_token].item()

    # Run corrupted forward pass and get baseline
    with torch.no_grad():
        corrupted_output = model.forward(
            corrupted_input_ids,
            output_hidden_states=True,
            output_attentions=True
        )
        corrupted_logit = corrupted_output.logits[0, position, target_token].item()

    # Cache clean activations
    clean_cache = {}
    for i, hidden_state in enumerate(clean_output.hidden_states[1:]):  # Skip embedding
        clean_cache[f"hidden_{i}"] = hidden_state.clone()
    if clean_output.attentions:
        for i, attn in enumerate(clean_output.attentions):
            clean_cache[f"attn_{i}"] = attn.clone()

    # Cache corrupted activations
    corrupted_cache = {}
    for i, hidden_state in enumerate(corrupted_output.hidden_states[1:]):
        corrupted_cache[f"hidden_{i}"] = hidden_state.clone()
    if corrupted_output.attentions:
        for i, attn in enumerate(corrupted_output.attentions):
            corrupted_cache[f"attn_{i}"] = attn.clone()

    # Test each component
    logit_diffs = []
    iterator = tqdm(components, desc="Patching components") if show_progress else components

    for component in iterator:
        # Patch activation for this component
        patched_logit = _patch_component(
            model,
            clean_input_ids,
            component,
            clean_cache,
            corrupted_cache,
            target_token,
            position,
        )
        logit_diffs.append(patched_logit - clean_logit)

    logit_diffs = np.array(logit_diffs)

    # Normalize patching effects: 0 = no effect, 1 = full restoration to corrupted
    baseline_diff = corrupted_logit - clean_logit
    if abs(baseline_diff) < 1e-8:
        patching_effects = np.zeros_like(logit_diffs)
    else:
        patching_effects = logit_diffs / baseline_diff

    return PatchingResults(
        components=components,
        logit_diffs=logit_diffs,
        patching_effects=patching_effects,
        clean_logit=clean_logit,
        corrupted_logit=corrupted_logit,
        target_token=target_token,
    )


def _patch_component(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    component: Component,
    clean_cache: Dict[str, torch.Tensor],
    corrupted_cache: Dict[str, torch.Tensor],
    target_token: int,
    position: int = -1,
) -> float:
    """Run forward pass with one component patched to corrupted activation."""

    # Create hook function to patch this component
    patched = False

    def patch_hook(module, input, output):
        nonlocal patched
        if patched:
            return output

        # Get corrupted activation for this component
        if component.type == "attention":
            key = f"attn_{component.layer}"
            if key in corrupted_cache:
                patched = True
                # Replace attention output
                if isinstance(output, tuple):
                    return (corrupted_cache[key],) + output[1:]
                return corrupted_cache[key]
        elif component.type in ["mlp", "residual"]:
            key = f"hidden_{component.layer}"
            if key in corrupted_cache:
                patched = True
                # Replace hidden state
                if isinstance(output, tuple):
                    return (corrupted_cache[key],) + output[1:]
                return corrupted_cache[key]

        return output

    # Register hook on the component's module
    module_name = _get_module_name(model, component)
    try:
        module = dict(model.model.named_modules())[module_name]
    except KeyError:
        # Component not found, return clean logit
        with torch.no_grad():
            output = model.forward(input_ids)
            return output.logits[0, position, target_token].item()

    handle = module.register_forward_hook(patch_hook)

    try:
        with torch.no_grad():
            output = model.forward(input_ids)
            logit = output.logits[0, position, target_token].item()
    finally:
        handle.remove()

    return logit


def ablate_components(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    target_token: int,
    components: Optional[List[Component]] = None,
    layers: Optional[List[int]] = None,
    component_types: List[str] = ["attention", "mlp"],
    position: int = -1,
    ablation_method: str = "zero",
    show_progress: bool = True,
) -> AblationResults:
    """
    Ablate (zero out) components to measure their importance.

    Args:
        model: Model wrapper
        input_ids: Input tensor [1, seq_len]
        target_token: Token ID to measure
        components: Specific components to ablate
        layers: Layer indices to test (if components is None)
        component_types: Types to test
        position: Token position to measure (-1 = last)
        ablation_method: "zero" or "mean" (replace with mean activation)
        show_progress: Show progress bar

    Returns:
        AblationResults with importance scores

    Example:
        >>> results = ablate_components(
        ...     model,
        ...     input_ids,
        ...     target_token=answer_token_id,
        ...     layers=[10, 15, 20]
        ... )
        >>> print(results.get_top_components(10))
    """
    device = model.device
    input_ids = input_ids.to(device)

    # Baseline: no ablation
    with torch.no_grad():
        baseline_output = model.forward(input_ids)
        baseline_logit = baseline_output.logits[0, position, target_token].item()

    # Determine components to test
    if components is None:
        if layers is None:
            layers = list(range(model.num_layers))
        components = _enumerate_components(model, layers, component_types)

    # Ablate each component
    ablated_logits = []
    iterator = tqdm(components, desc="Ablating components") if show_progress else components

    for component in iterator:
        ablated_logit = _ablate_component(
            model,
            input_ids,
            component,
            target_token,
            position,
            method=ablation_method,
        )
        ablated_logits.append(ablated_logit)

    ablated_logits = np.array(ablated_logits)
    importance_scores = baseline_logit - ablated_logits  # Positive = component increases logit

    return AblationResults(
        components=components,
        ablated_logits=ablated_logits,
        importance_scores=importance_scores,
        baseline_logit=baseline_logit,
        target_token=target_token,
    )


def _ablate_component(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    component: Component,
    target_token: int,
    position: int = -1,
    method: str = "zero",
) -> float:
    """Ablate a single component and return logit."""

    def ablation_hook(module, input, output):
        if method == "zero":
            # Zero out the output
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        elif method == "mean":
            # Replace with mean activation across sequence
            if isinstance(output, tuple):
                mean_activation = output[0].mean(dim=1, keepdim=True).expand_as(output[0])
                return (mean_activation,) + output[1:]
            return output.mean(dim=1, keepdim=True).expand_as(output)
        return output

    # Get module for this component
    module_name = _get_module_name(model, component)
    try:
        module = dict(model.model.named_modules())[module_name]
    except KeyError:
        # Component not found, return baseline
        with torch.no_grad():
            output = model.forward(input_ids)
            return output.logits[0, position, target_token].item()

    handle = module.register_forward_hook(ablation_hook)

    try:
        with torch.no_grad():
            output = model.forward(input_ids)
            logit = output.logits[0, position, target_token].item()
    finally:
        handle.remove()

    return logit


def discover_reasoning_circuit(
    model: ModelWrapper,
    clean_prompt: str,
    corrupted_prompt: str,
    target_token: int,
    threshold: float = 0.05,
    method: str = "patching",
    layers: Optional[List[int]] = None,
    **kwargs,
) -> Circuit:
    """
    Discover minimal circuit for reasoning task via iterative pruning.

    Algorithm:
    1. Test all components with activation patching or ablation
    2. Keep components above importance threshold
    3. Optionally verify circuit preserves performance

    Args:
        model: Model wrapper
        clean_prompt: Correct reasoning example
        corrupted_prompt: Incorrect reasoning example
        target_token: Token ID for answer
        threshold: Minimum importance to include (0-1 for patching, absolute for ablation)
        method: "patching" or "ablation"
        layers: Layer indices to search (None = all)
        **kwargs: Additional arguments for patching/ablation

    Returns:
        Circuit object with critical components

    Example:
        >>> circuit = discover_reasoning_circuit(
        ...     model,
        ...     clean_prompt="What is 2+2? The answer is 4.",
        ...     corrupted_prompt="What is 2+2? The answer is 5.",
        ...     target_token=model.tokenizer.encode("4")[0],
        ...     threshold=0.1  # Keep components with >10% effect
        ... )
        >>> print(f"Found {len(circuit.components)} critical components")
    """
    # Tokenize inputs
    clean_ids = model.tokenize(clean_prompt)['input_ids']
    corrupted_ids = model.tokenize(corrupted_prompt)['input_ids']

    if method == "patching":
        results = activation_patching_experiment(
            model,
            clean_ids,
            corrupted_ids,
            target_token,
            layers=layers,
            **kwargs,
        )
        # Keep components with patching effect > threshold
        important_mask = np.abs(results.patching_effects) > threshold
        important_components = [
            comp for comp, mask in zip(results.components, important_mask) if mask
        ]
        importance_scores = results.patching_effects[important_mask]

    elif method == "ablation":
        results = ablate_components(
            model,
            clean_ids,
            target_token,
            layers=layers,
            **kwargs,
        )
        # Keep components with importance > threshold
        important_mask = results.importance_scores > threshold
        important_components = [
            comp for comp, mask in zip(results.components, important_mask) if mask
        ]
        importance_scores = results.importance_scores[important_mask]

    else:
        raise ValueError(f"Unknown method: {method}. Use 'patching' or 'ablation'.")

    # Create circuit
    circuit = Circuit(
        components=important_components,
        importance_scores=importance_scores.tolist(),
        task_description=f"Reasoning: {clean_prompt[:50]}...",
        discovery_method=method,
        threshold=threshold,
    )

    return circuit


def evaluate_circuit(
    model: ModelWrapper,
    circuit: Circuit,
    test_prompts: List[str],
    target_tokens: List[int],
    baseline_accuracy: Optional[float] = None,
) -> Dict[str, float]:
    """
    Evaluate circuit performance on held-out examples.

    Args:
        model: Model wrapper
        circuit: Discovered circuit
        test_prompts: Test examples
        target_tokens: Expected answer tokens for each prompt
        baseline_accuracy: Baseline accuracy (if None, compute from full model)

    Returns:
        Dict with accuracy metrics:
            - baseline_accuracy: Full model accuracy
            - circuit_accuracy: Accuracy with only circuit components
            - accuracy_preservation: Ratio of circuit/baseline accuracy
    """
    if len(test_prompts) != len(target_tokens):
        raise ValueError("test_prompts and target_tokens must have same length")

    # Compute baseline accuracy if not provided
    if baseline_accuracy is None:
        correct = 0
        for prompt, target_token in zip(test_prompts, target_tokens):
            input_ids = model.tokenize(prompt)['input_ids'].to(model.device)
            with torch.no_grad():
                output = model.forward(input_ids)
                pred_token = output.logits[0, -1].argmax().item()
                if pred_token == target_token:
                    correct += 1
        baseline_accuracy = correct / len(test_prompts)

    # TODO: Evaluate with only circuit components active (requires intervention system)
    # For now, return baseline as placeholder
    circuit_accuracy = baseline_accuracy  # Placeholder

    return {
        "baseline_accuracy": baseline_accuracy,
        "circuit_accuracy": circuit_accuracy,
        "accuracy_preservation": circuit_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0.0,
        "num_components": len(circuit.components),
    }


# Helper functions

def _enumerate_components(
    model: ModelWrapper,
    layers: List[int],
    component_types: List[str],
) -> List[Component]:
    """Enumerate all components to test."""
    components = []

    for layer in layers:
        if "attention" in component_types:
            # Add each attention head
            for head in range(model.num_attention_heads):
                components.append(
                    Component(
                        type="attention",
                        layer=layer,
                        head=head,
                        name=f"L{layer}H{head}",
                    )
                )

        if "mlp" in component_types:
            components.append(
                Component(
                    type="mlp",
                    layer=layer,
                    head=None,
                    name=f"L{layer}_mlp",
                )
            )

        if "residual" in component_types:
            components.append(
                Component(
                    type="residual",
                    layer=layer,
                    head=None,
                    name=f"L{layer}_residual",
                )
            )

    return components


def _get_module_name(model: ModelWrapper, component: Component) -> str:
    """Get module name for a component."""
    # This is model-architecture specific
    # For DeepSeek/Qwen/Llama: model.layers.{i}.<component>

    if component.type == "attention":
        return f"model.layers.{component.layer}.self_attn"
    elif component.type == "mlp":
        return f"model.layers.{component.layer}.mlp"
    elif component.type == "residual":
        return f"model.layers.{component.layer}"
    else:
        raise ValueError(f"Unknown component type: {component.type}")
