# TemporalSAEs

Codebase for the paper "Priors in Time: Missing for Language Model Interpretability"

This repository implements both standard Sparse Autoencoders (SAEs) and Temporal SAEs for language model interpretability. Temporal SAEs decompose language model activations into predictable (context-dependent) and novel (per-token) components using attention mechanisms.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training SAEs](#training-saes)
- [Architecture Overview](#architecture-overview)
- [Key Hyperparameters](#key-hyperparameters)
- [Codebase Structure](#codebase-structure)
- [Pretrained Models](#pretrained-models)
- [Configuration Files](#configuration-files)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd temporalSAEs

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets
pip install hydra-core omegaconf
pip install wandb tqdm
pip install sparsemax
```

## Quick Start

### 1. Precompute Activations

First, precompute activations from your target language model:

```bash
# For Gemma-2-2B
python precompute_llm_activations/run_precompute.py --config-name=precompute_gemma

# For Llama-3.1-8B  
python precompute_llm_activations/run_precompute.py --config-name=precompute_llama
```

### 2. Train Standard SAEs

```bash
# Train standard SAE on Gemma-2-2B
python train_standard_saes.py --config-name=standard_conf

# Train standard SAE on Llama-3.1-8B
python train_standard_saes.py --config-name=standard_conf_llama
```

### 3. Train Temporal SAEs

```bash
# Train temporal SAE on Gemma-2-2B
python train_temporal_saes.py --config-name=temporal_conf

# Train temporal SAE on Llama-3.1-8B
python train_temporal_saes.py --config-name=temporal_conf_llama
```

## Training SAEs

### Standard SAEs

Standard SAEs learn to reconstruct language model activations using various sparsity-inducing mechanisms:

**Supported SAE Types:**
- `relu`: ReLU activation with L1 regularization
- `topk`: Top-K sparsification (no regularization)
- `jumprelu`: JumpReLU with learnable thresholds
- `batchtopk`: Batch-level Top-K sparsification
- `SpaDE`: Sparsemax with distance-based penalty

**Training Command:**
```bash
python train_standard_saes.py --config-name=standard_conf
```

### Temporal SAEs

Temporal SAEs decompose activations into:
- **Predictable component**: Context-dependent, learned via attention
- **Novel component**: Per-token differences, learned via standard SAE

**Key Features:**
- Multi-head attention for context modeling
- Configurable number of attention layers
- Tied or separate encoder/decoder weights
- Multiple sparsity types for novel component

**Training Command:**
```bash
python train_temporal_saes.py --config-name=temporal_conf
```

## Architecture Overview

### Standard SAE Architecture

```
Input Activations → Encoder → Sparse Codes → Decoder → Reconstructed Activations
```

- **Encoder**: Linear projection + sparsity function
- **Decoder**: Linear projection (typically transpose of encoder)
- **Sparsity**: ReLU, TopK, JumpReLU, etc.

### Temporal SAE Architecture

```
Input Activations → [Predictable Path] → [Novel Path] → Combined Reconstruction
                    ↓                    ↓
                Attention Layers      SAE (TopK/ReLU)
                    ↓                    ↓
              Context Codes         Novel Codes
```

- **Predictable Path**: Multi-head attention over previous tokens
- **Novel Path**: Standard SAE on residual activations
- **Combined**: Sum of both reconstructions

## Key Hyperparameters

### Data Configuration
- `num_total_steps`: Total training steps (default: 200,000)
- `context_length`: Sequence length (default: 500)
- `batch_size`: Batch size (default: 100)
- `dtype`: Data type (default: "bfloat16")

### SAE Configuration

#### Standard SAEs
- `sae_type`: Type of SAE (`relu`, `topk`, `jumprelu`, `MP`, `batchtopk`, `SpaDE`)
- `exp_factor`: Expansion factor (width = dimin × exp_factor, default: 4)
- `kval_topk`: K for TopK sparsification (default: 192 for Gemma, 256 for Llama)
- `gamma_reg`: Regularization strength (default: 8 for ReLU, 10 for SpaDE)
- `scaling_factor`: Input scaling (Gemma: 0.0067, Llama: 0.085)

#### Temporal SAEs
- `sae_diff_type`: Type for novel component (`relu`, `topk`, `nullify`)
- `n_heads`: Number of attention heads (default: 4)
- `n_attn_layers`: Number of attention layers (default: 1)
- `bottleneck_factor`: Attention bottleneck factor (default: 1)
- `tied_weights`: Whether to tie encoder/decoder weights (default: True)

### Optimizer Configuration
- `learning_rate`: Learning rate (default: 1e-3)
- `weight_decay`: Weight decay (default: 1e-4)
- `beta1`, `beta2`: Adam betas (default: 0.9, 0.95)
- `grad_clip`: Gradient clipping (default: 1.0)
- `warmup_iters`: Learning rate warmup steps (default: 200)
- `min_lr`: Minimum learning rate (default: 9e-4)

### Logging Configuration
- `log_interval`: Logging frequency (default: 10)
- `save_interval`: Model saving frequency (default: 20,000)
- `wandb_project_name`: Weights & Biases project name

## Codebase Structure

```
temporalSAEs/
├── config/                          # Configuration files
│   ├── precompute/                  # Precomputation configs
│   │   ├── precompute_gemma.yaml
│   │   └── precompute_llama.yaml
│   └── sae_train/                   # Training configs
│       ├── standard_conf.yaml       # Standard SAE (Gemma)
│       ├── standard_conf_llama.yaml # Standard SAE (Llama)
│       ├── temporal_conf.yaml       # Temporal SAE (Gemma)
│       ├── temporal_conf_llama.yaml # Temporal SAE (Llama)
│       └── multirun.yaml           # Multi-run configuration
├── precompute_llm_activations/      # Activation precomputation
│   ├── run_precompute.py           # Main precomputation script
│   ├── cache_utils.py              # Caching utilities
│   └── compute_scaling_factor.py   # Scaling factor computation
├── sae/                            # SAE implementations
│   ├── saeStandard.py              # Standard SAE architectures
│   ├── saeTemporal.py              # Temporal SAE architecture
│   └── utils.py                    # SAE utilities
├── utils/                          # General utilities
│   ├── analysis.py                 # Analysis tools
│   ├── logging.py                  # Logging utilities
│   ├── obj.py                      # Object utilities
│   └── optimizer.py                # Optimizer utilities
├── train_standard_saes.py          # Standard SAE training
├── train_temporal_saes.py          # Temporal SAE training
└── upload_to_hf.py                 # HuggingFace upload utility
```

### Key Files

- **`train_standard_saes.py`**: Main training script for standard SAEs
- **`train_temporal_saes.py`**: Main training script for temporal SAEs
- **`sae/saeStandard.py`**: Standard SAE implementations (ReLU, TopK, JumpReLU, etc.)
- **`sae/saeTemporal.py`**: Temporal SAE with attention-based predictable component
- **`precompute_llm_activations/run_precompute.py`**: Precompute activations from language models

## Pretrained Models

Pretrained SAEs can be found at the following links. Note that contextual information that can enable predicted code to function well seems to be available primarily in middle / early layers, and hence we found later layer Temporal SAEs to not outperform standard SAEs. Nevertheless, to enable experimentation, we have released all SAEs trained in this work.

| Model | Layer | Type | Link |
|-------|-------|------|------|
| Gemma-2-2B | 12 | Temporal | [Download](https://huggingface.co/ekdeepslubana/temporalSAEs_gemma/tree/main) |
| Llama-3.1-8B | 15 | Temporal | [Download](https://huggingface.co/ekdeepslubana/temporalSAEs_llama/tree/main/layer_16) |
| Llama-3.1-8B | 26 | Temporal | [Download](https://huggingface.co/ekdeepslubana/temporalSAEs_llama/tree/main/layer_26) |

## Configuration Files

### Model-Specific Configurations

**Gemma-2-2B:**
- Hidden dimension: 2304
- Scaling factor: 0.0067
- TopK value: 192

**Llama-3.1-8B:**
- Hidden dimension: 4096  
- Scaling factor: 0.085
- TopK value: 256

### Custom Configurations

Create custom configs by modifying the YAML files in `config/sae_train/`. Key parameters to adjust:

- **Model size**: Change `llm.dimin` and `sae.exp_factor`
- **Sparsity**: Adjust `sae.kval_topk` and `sae.gamma_reg`
- **Training**: Modify `data.num_total_steps` and `optimizer.learning_rate`
- **Architecture**: Change `sae.n_heads` and `sae.n_attn_layers` for temporal SAEs