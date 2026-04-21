# Reasoning Interpretability: Exploring LLM Internal Reasoning

A mechanistic interpretability framework for exploring and understanding how reasoning emerges inside transformer-based language models.

## 🎯 Overview

This project provides tools to peek inside large language models and understand the internal computational processes that enable multi-step reasoning. By analyzing attention patterns, hidden states, and information flow, we can uncover the "circuits" and mechanisms that constitute thinking inside neural networks.

### Key Features

- **Model Loading**: Easy loading of reasoning-capable models with 4-bit/8-bit quantization for CPU inference
- **Activation Extraction**: Hook-based extraction of attention patterns, hidden states, and intermediate activations
- **Circuit Discovery**: Activation patching and ablation studies to identify critical reasoning components
- **Logit Lens**: Track when and where answers emerge across model layers
- **Visualization**: Interactive Plotly visualizations of attention patterns, hidden state evolution, and computational graphs
- **Experiment Management**: Reproducible experiments with caching and batch processing

### Supported Models

- **DeepSeek-R1-Distill** (1.5B, 7B, 8B): Models with explicit reasoning tokens
- *More models coming in future phases*

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd reasoning_explore

# Install dependencies
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### Basic Usage

```python
from interpretability.models import load_model
from interpretability.extraction import extract_attention_patterns
from interpretability.visualization import plot_attention_heatmap

# Load model with 4-bit quantization (CPU-friendly)
model = load_model(
    "deepseek-1.5b",
    device="cpu",
    quantization="4bit"
)

# Tokenize a reasoning prompt
prompt = "What is 17 * 23? Let me calculate step by step."
inputs = model.tokenize(prompt)

# Run forward pass with attention extraction
output = model.forward(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    output_attentions=True
)

# Extract and visualize attention patterns
attention = extract_attention_patterns(model, inputs["input_ids"])
tokens = model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Visualize attention at layer 16
fig = plot_attention_heatmap(attention, tokens, layer=16)
fig.show()
```

## 📁 Project Structure

```
reasoning_explore/
├── src/interpretability/          # Main package
│   ├── core/                      # Core abstractions
│   │   ├── types.py               # Type definitions
│   │   ├── model_wrapper.py       # Unified model interface
│   │   ├── hook_manager.py        # PyTorch hook management
│   │   └── activation_cache.py    # Activation storage
│   ├── models/                    # Model loaders
│   │   ├── base.py                # Base loader interface
│   │   ├── deepseek.py            # DeepSeek-specific loader
│   │   └── registry.py            # Model registry
│   ├── extraction/                # Activation extraction (Phase 2)
│   ├── analysis/                  # Circuit discovery (Phase 3)
│   ├── visualization/             # Visualization tools (Phase 2)
│   └── experiments/               # Experiment management (Phase 4)
├── notebooks/                     # Jupyter notebooks
├── config/                        # Configuration files
│   └── models.yaml                # Model configurations
├── tests/                         # Test suite
└── data/                          # Data storage
    ├── models/                    # Downloaded models
    ├── cache/                     # Activation cache
    └── experiments/               # Experiment results
```

## 🔧 Core Components

### ModelWrapper

Unified interface for loading and working with language models:

```python
from interpretability.core import ModelWrapper

# Load with custom settings
model = ModelWrapper(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device="cpu",
    quantization="4bit",
    torch_dtype=torch.float16
)

# Access model internals
print(f"Layers: {model.get_num_layers()}")
print(f"Hidden dim: {model.get_hidden_dim()}")
print(f"Attention heads: {model.get_attention_heads(0)}")

# Memory statistics
print(model.memory_stats())
```

### HookManager

Manage PyTorch hooks for activation extraction:

```python
# Register hooks on specific layers
handles = model.hook_manager.register_layer_hooks(
    layer_pattern="model.layers.{i}",
    layer_indices=[10, 15, 20]
)

# Run inference - activations are automatically captured
output = model.forward(input_ids)

# Retrieve activations
activations = model.hook_manager.get_all_activations()

# Temporary hooks with context manager
with model.hook_manager.temporary_hooks(['layer10', 'layer15']) as acts:
    output = model.forward(input_ids)
    layer10_acts = acts['layer10']
```

### ActivationCache

Efficient storage and retrieval of activations:

```python
from interpretability.core import ActivationCache

# Create cache with 1GB memory limit
cache = ActivationCache(max_size_mb=1000, disk_path="data/cache")

# Store activations
cache.store("prompt1_layer10", hidden_states, metadata={...})

# Retrieve
cached_acts = cache.retrieve("prompt1_layer10")

# Save to disk (Zarr format)
cache.save_to_disk()

# Cache statistics
print(cache.stats())
```

## 📊 Current Implementation Status

### ✅ Phase 1: Foundation (Weeks 1-2) - **COMPLETED**

- [x] Project structure and dependencies
- [x] Core type definitions
- [x] ModelWrapper with 4-bit quantization support
- [x] HookManager for activation extraction
- [x] ActivationCache with Zarr backing
- [x] DeepSeek model loader
- [x] Model registry
- [ ] Basic quickstart notebook *(in progress)*

### 🔄 Phase 2: Extraction & Visualization (Weeks 3-4) - **NEXT**

- [ ] Attention pattern extraction
- [ ] Hidden state extraction
- [ ] Logit lens implementation
- [ ] Interactive Plotly visualizations
- [ ] Exploration notebooks

### ⏳ Phase 3: Circuit Discovery (Weeks 5-6)

- [ ] Activation patching
- [ ] Circuit discovery algorithms
- [ ] Information flow analysis
- [ ] Computational graph visualization

### ⏳ Phase 4: Experimentation & Dashboard (Weeks 7-8)

- [ ] Experiment runner
- [ ] Comparative analysis tools
- [ ] Streamlit dashboard
- [ ] CLI scripts

## 🧪 Running Examples

### Example 1: Load and Inspect a Model

```python
from interpretability.models import load_model

# Load DeepSeek-R1-Distill-1.5B
model = load_model("deepseek-1.5b", device="cpu", quantization="4bit")

# Model info
print(model)
# Output:
# ModelWrapper(
#   model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,
#   layers=28,
#   params=1.5M,
#   size=2.1MB,
#   device=cpu,
#   quantization=4bit
# )
```

### Example 2: Extract Activations

```python
# Register hooks on attention layers
attention_pattern = "model.layers.{i}.self_attn"
handles = model.hook_manager.register_layer_hooks(
    attention_pattern,
    layer_indices=[0, 10, 20, 27]
)

# Run inference
prompt = "What is the capital of France? Let me think."
inputs = model.tokenize(prompt)
output = model.forward(**inputs)

# Get activations
activations = model.hook_manager.get_all_activations()
print(f"Captured {len(activations)} activation sets")

# Clean up
model.hook_manager.remove_all_hooks()
```

### Example 3: Cache Management

```python
# Create cache with disk backing
cache = model.cache
cache.disk_path = "data/cache/my_experiment"

# Store activations from multiple prompts
prompts = ["prompt1", "prompt2", "prompt3"]
for i, prompt in enumerate(prompts):
    inputs = model.tokenize(prompt)
    output = model.forward(**inputs, output_hidden_states=True)
    
    # Cache hidden states
    for layer_idx, hidden_state in enumerate(output.hidden_states):
        cache.store(
            f"prompt{i}_layer{layer_idx}",
            hidden_state,
            metadata={...}
        )

# Save to disk
cache.save_to_disk()
print(f"Cache stats: {cache.stats()}")
```

## 🔬 Research Applications

This framework enables various mechanistic interpretability research:

1. **Attention Analysis**: Understand which tokens attend to each other during reasoning
2. **Circuit Discovery**: Identify minimal sets of attention heads and neurons for reasoning
3. **Logit Lens Studies**: Track when correct answers emerge across layers
4. **Cross-Model Comparison**: Compare reasoning strategies across architectures
5. **Feature Attribution**: Determine which input tokens influence reasoning steps

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=interpretability --cov-report=html

# Run specific test file
pytest tests/test_core/test_model_wrapper.py
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

## 📝 Configuration

Model configurations are defined in [`config/models.yaml`](config/models.yaml):

```yaml
models:
  deepseek-r1-distill-1.5b:
    hf_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    loader: "DeepSeekLoader"
    size: "1.5B"
    recommended_quantization: "4bit"
    reasoning_tokens:
      start: ["<think>"]
      end: ["</think>"]
```

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

- Additional model loaders (Qwen, Phi, Llama)
- New analysis methods
- Visualization improvements
- Documentation and tutorials
- Bug fixes and optimizations

## 📚 References

This project builds on research in mechanistic interpretability:

- [Transformer Circuits Thread](https://transformer-circuits.pub/) (Anthropic)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Interpretability in the Wild](https://arxiv.org/abs/2211.00593)

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- HuggingFace Transformers for model infrastructure
- TransformerLens for interpretability utilities
- DeepSeek team for reasoning-capable models

---

**Status**: Phase 1 (Foundation) complete. Phase 2 (Extraction & Visualization) in progress.

**Hardware Requirements**: 
- Minimum: 4GB RAM (with 4-bit quantization)
- Recommended: 8GB+ RAM, GPU optional

**Time to First Analysis**: ~30 minutes from installation
