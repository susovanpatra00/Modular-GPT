# GPT Model Architecture

This document describes the modular GPT model architecture implemented in this project. The model is designed with configurable components that can be swapped out based on configuration settings.

## Overview

The GPT model follows a standard transformer architecture with the following key components:

- **Token Embedding**: Converts input tokens to dense vectors
- **Positional Encoding**: Adds positional information (configurable: learned embeddings or sinusoidal)
- **Transformer Blocks**: Stack of N transformer layers with attention and feedforward components
- **Layer Normalization**: Final normalization before output
- **Language Modeling Head**: Projects hidden states to vocabulary logits


## Model Components

### 1. Token and Positional Embeddings

- **Token Embedding**: `nn.Embedding(vocab_size, n_embd)`
- **Positional Encoding**: Configurable via `pos_type` parameter
  - `learned`: Standard learned position embeddings
  - `sinusoidal`: Sinusoidal positional encoding

### 2. Transformer Blocks

Each transformer block (`TransformerBlock`) contains:

- **Pre-normalization**: Applied before attention and feedforward
- **Multi-head Attention**: Configurable attention mechanism
- **Feedforward Network**: Configurable feedforward implementation
- **Residual Connections**: Skip connections around attention and FFN
- **Dropout**: Applied after attention and FFN outputs

### 3. Configurable Components

#### Attention Mechanisms
- **Flash Attention**: Memory-efficient attention implementation
- **Linear Attention**: Linear complexity attention
- **Local Window Attention**: Attention with limited window size

#### Feedforward Networks
- **Standard FFN**: Traditional two-layer MLP
- **SwiGLU FFN**: Gated feedforward with SwiGLU activation
- **Mixture of Experts (MoE)**: Sparse expert routing

#### Normalization Layers
- **LayerNorm**: Standard layer normalization
- **RMSNorm**: Root mean square normalization
- **ScaleNorm**: Simple scaling normalization

## Configuration

The model architecture is controlled through a YAML configuration file:

```yaml
# Model dimensions
n_embd: 256          # Embedding dimension
num_heads: 8         # Number of attention heads
n_layers: 6          # Number of transformer blocks
vocab_size: 64       # Vocabulary size (updated dynamically)
block_size: 256      # Maximum sequence length
dropout: 0.1         # Dropout rate

# Component selection
attention_type: flash_attention
attention_class: FlashAttention

ffn_type: ff_swiglu
ffn_class: GatedFeedForward

norm_type: rmsnorm
norm_class: RMSNorm

pos_type: sinusoidal
```

## Model Flow

1. **Input Processing**:
   - Input token IDs → Token embeddings
   - Add positional encoding
   - Apply dropout

2. **Transformer Processing**:
   - For each transformer block:
     - Apply pre-normalization
     - Multi-head attention with residual connection
     - Apply pre-normalization
     - Feedforward network with residual connection

3. **Output Generation**:
   - Final layer normalization
   - Project to vocabulary size via language modeling head
   - Optionally compute cross-entropy loss if targets provided

## Key Features

### Modular Design
- Components are dynamically imported based on configuration
- Easy to add new attention mechanisms, feedforward networks, or normalization layers
- Configuration-driven architecture selection

### Weight Initialization
- GPT-2 style weight initialization
- Normal distribution for linear layers and embeddings (std=0.02)
- Proper initialization for normalization layers

### Generation Support
- Built-in text generation with temperature and top-k sampling
- Handles variable sequence lengths up to `block_size`
- Efficient autoregressive generation

### Training Optimizations
- Gradient clipping for stability
- Optional weight tying between embeddings and output projection
- Support for mixed precision training

## Parameter Count

The model provides a method to count parameters:
- Total parameters including all components
- Non-embedding parameters (excluding position embeddings)
- Useful for model size analysis and memory estimation

## Usage Example

```python
from src.model.gpt import GPTModel
from types import SimpleNamespace

# Load configuration
config = SimpleNamespace(
    n_embd=256,
    num_heads=8,
    n_layers=6,
    vocab_size=1000,
    block_size=256,
    dropout=0.1,
    attention_type='flash_attention',
    attention_class='FlashAttention',
    ffn_type='ff_swiglu',
    ffn_class='GatedFeedForward',
    norm_type='rmsnorm',
    norm_class='RMSNorm'
)

# Create model
model = GPTModel(config)
print(f"Model parameters: {model.get_num_params():,}")

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 10))
logits = model(input_ids)
print(f"Output shape: {logits.shape}")  # [1, 10, vocab_size]
```

This modular architecture allows for easy experimentation with different components while maintaining a clean and extensible codebase.