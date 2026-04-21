"""
Analysis modules for circuit discovery and information flow.

This subpackage provides tools for:
- Activation patching experiments
- Circuit discovery via ablation
- Information flow analysis
- Gradient-based attribution
"""

from .circuit_discovery import (
    activation_patching_experiment,
    ablate_components,
    discover_reasoning_circuit,
    evaluate_circuit,
    PatchingResults,
    AblationResults,
)

from .information_flow import (
    compute_information_flow,
    trace_reasoning_path,
    detect_bottleneck_layers,
    compute_path_diversity,
    identify_critical_tokens,
    FlowGraph,
    ReasoningPath,
)

from .attribution import (
    compute_token_attribution,
    compute_layer_contribution,
    compute_gradient_attention,
    visualize_token_attribution,
    AttributionResults,
)

__all__ = [
    # Circuit discovery
    "activation_patching_experiment",
    "ablate_components",
    "discover_reasoning_circuit",
    "evaluate_circuit",
    "PatchingResults",
    "AblationResults",
    # Information flow
    "compute_information_flow",
    "trace_reasoning_path",
    "detect_bottleneck_layers",
    "compute_path_diversity",
    "identify_critical_tokens",
    "FlowGraph",
    "ReasoningPath",
    # Attribution
    "compute_token_attribution",
    "compute_layer_contribution",
    "compute_gradient_attention",
    "visualize_token_attribution",
    "AttributionResults",
]
