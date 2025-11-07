# Feedforward Networks in Transformers

## What is a Feedforward Network?

A feedforward network (FFN) is a crucial component in transformer architectures that applies position-wise transformations to each token independently. Unlike attention mechanisms that model relationships between tokens, feedforward networks process each position separately, providing computational depth and non-linear transformations to the model.

In transformer blocks, the feedforward network typically follows the multi-head attention layer and consists of two linear transformations with a non-linear activation function in between:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

## Why Do We Need Feedforward Networks After Attention?

While attention mechanisms are powerful for modeling relationships between tokens, they have several limitations that feedforward networks address:

### 1. **Linear Transformations Only**
Attention mechanisms primarily perform linear combinations of input representations. The query-key-value computations and weighted aggregations are fundamentally linear operations. Without feedforward networks, transformers would be limited to linear transformations, severely restricting their expressiveness.

### 2. **Limited Computational Depth**
Attention provides a way to aggregate information across positions but doesn't add computational depth at each position. Feedforward networks provide this depth by applying multiple layers of transformations to each token's representation.

### 3. **Position-wise Processing**
While attention excels at cross-token interactions, feedforward networks complement this by providing sophisticated position-wise transformations. This allows the model to:
- Transform representations based on local context
- Apply complex non-linear functions to individual token embeddings
- Refine the representations after attention-based mixing

### 4. **Memory and Storage**
Feedforward networks can be viewed as providing a form of "memory" or "storage" for the model. The large intermediate dimensions (typically 4x the embedding size) allow the model to store and process complex patterns and features.

## How Feedforward Networks Bring Non-linearity

The key to feedforward networks' power lies in their non-linear activation functions. The mathematical formulation shows how non-linearity is introduced:

### Standard FFN Equation:
```
h = σ(xW₁ + b₁)
output = hW₂ + b₂
```

Where:
- `x` is the input (shape: [batch, seq_len, d_model])
- `W₁` expands to hidden dimension (shape: [d_model, d_hidden])
- `σ` is the non-linear activation function (GELU, ReLU, SiLU, etc.)
- `W₂` projects back to model dimension (shape: [d_hidden, d_model])

### Why Non-linearity Matters:
1. **Universal Approximation**: Non-linear activations enable the network to approximate any continuous function
2. **Feature Extraction**: Non-linearities allow the model to learn complex, non-linear patterns in the data
3. **Representational Power**: Without non-linearity, stacking multiple linear layers would be equivalent to a single linear transformation

The expansion to a larger hidden dimension (typically 4x) followed by non-linear activation and projection back creates a powerful transformation that can:
- Learn complex feature interactions
- Perform sophisticated pattern matching
- Store and retrieve learned representations

## Three Types of Feedforward Networks

This codebase implements three different feedforward architectures, each with unique characteristics and advantages:

### 1. Standard FFN ([`ff_standard.py`](src/feedforward/ff_standard.py))

**Architecture:**
```python
x → Linear(d_model → 4*d_model) → GELU → Dropout → Linear(4*d_model → d_model) → Dropout
```

**Key Features:**
- **Expansion Factor**: 4x the embedding dimension (configurable via `ffn_mult`)
- **Activation**: GELU (Gaussian Error Linear Unit)
- **Structure**: Simple two-layer MLP with intermediate expansion

**Significance:**
- **Simplicity**: Straightforward architecture that's easy to understand and implement
- **Proven Performance**: Used in original GPT and BERT models with excellent results
- **Computational Efficiency**: Balanced trade-off between performance and computational cost
- **Wide Adoption**: Standard choice in many transformer implementations

**Mathematical Form:**
```
FFN(x) = Dropout(Linear(Dropout(GELU(Linear(x)))))
```

### 2. SwiGLU FFN ([`ff_swiglu.py`](src/feedforward/ff_swiglu.py))

**Architecture:**
```python
x → [Linear(d_model → hidden), Linear(d_model → hidden)] → SiLU(w1) * w2 → Linear(hidden → d_model)
```

**Key Features:**
- **Gated Linear Unit**: Uses gating mechanism with SiLU activation
- **Reduced Parameters**: Hidden dimension is `2/3 * 4 * d_model` instead of `4 * d_model`
- **Three Projections**: w1, w2 (for gating), and w3 (output projection)

**Significance:**
- **Parameter Efficiency**: Achieves better performance with fewer parameters than standard FFN
- **Gating Mechanism**: The multiplication `SiLU(xW₁) * xW₂` provides sophisticated gating
- **Modern Architecture**: Used in LLaMA, Mistral, and other state-of-the-art models
- **Better Gradient Flow**: SiLU activation provides smoother gradients than GELU

**Mathematical Form:**
```
SwiGLU(x) = (SiLU(xW₁) ⊙ xW₂)W₃
```

Where `⊙` denotes element-wise multiplication.

### 3. Mixture of Experts (MoE) FFN ([`ff_moe.py`](src/feedforward/ff_moe.py))

**Architecture:**
```python
x → Router(top-2 selection) → [Expert₁, Expert₂, ..., ExpertN] → Weighted Combination
```

**Key Features:**
- **Multiple Experts**: Each expert is a standard FFN
- **Dynamic Routing**: Top-2 router selects which experts to use for each token
- **Sparse Activation**: Only 2 out of N experts are active per token
- **Learned Gating**: Router learns to assign tokens to appropriate experts

**Significance:**
- **Scalability**: Can increase model capacity without proportional increase in computation
- **Specialization**: Different experts can specialize in different types of patterns or domains
- **Efficiency**: Sparse activation means constant computational cost regardless of number of experts
- **Conditional Computation**: Tokens are processed differently based on their content

**Mathematical Form:**
```
MoE(x) = Σᵢ Gᵢ(x) * Expertᵢ(x)
```

Where:
- `Gᵢ(x)` is the gating weight for expert i (from top-2 selection)
- Only top-2 experts have non-zero gates

**Router Mechanism:**
```python
logits = xW_gate
top2_indices, top2_values = TopK(logits, k=2)
gates = Softmax(top2_values)
```

## Comparative Analysis

| Aspect | Standard FFN | SwiGLU FFN | MoE FFN |
|--------|-------------|------------|---------|
| **Parameters** | 8 * d_model² | ~5.3 * d_model² | N_experts * 8 * d_model² |
| **Computation** | 8 * d_model² | ~5.3 * d_model² | 2 * 8 * d_model² |
| **Activation** | GELU | SiLU + Gating | GELU (per expert) |
| **Specialization** | General | General | Expert-specific |
| **Memory** | Low | Low | High |
| **Complexity** | Simple | Medium | High |

## Implementation Details

### Standard FFN
- Uses [`nn.GELU()`](src/feedforward/ff_standard.py:18) activation
- Configurable expansion multiplier via `config.ffn_mult`
- Dropout applied after both linear layers

### SwiGLU FFN
- Implements the SwiGLU variant: `SiLU(xW₁) * xW₂`
- Hidden dimension calculation: `int(mult * n_embd * 2 / 3)`
- Single dropout applied to final output

### MoE FFN
- Top-2 routing with learned gating
- Each expert is a standard FFN
- Efficient implementation with expert masking
- Load balancing considerations for training stability

## When to Use Each Type

**Standard FFN**: 
- Default choice for most applications
- Good balance of performance and simplicity
- Suitable for smaller to medium-sized models

**SwiGLU FFN**:
- When parameter efficiency is important
- For larger models where every parameter counts
- Modern architectures requiring state-of-the-art performance

**MoE FFN**:
- When you need to scale model capacity
- For domain-specific applications requiring specialization
- When computational budget allows for the routing overhead

## Conclusion

Feedforward networks are essential components that complement attention mechanisms by providing:
1. **Non-linear transformations** through activation functions
2. **Position-wise processing** for local feature extraction
3. **Computational depth** for complex pattern learning
4. **Memory and storage** capabilities for the model

The three implementations showcase different approaches to achieving these goals, from the simplicity of standard FFN to the efficiency of SwiGLU and the scalability of MoE architectures.