# Normalization Techniques in Neural Networks

## What is Normalization?

Normalization is a technique used in neural networks to stabilize and accelerate training by controlling the distribution of inputs to each layer. It involves rescaling the activations to have desired statistical properties, typically zero mean and unit variance, or unit norm.

## Why Do We Need Normalization?

### 1. **Internal Covariate Shift**
As the network trains, the distribution of inputs to each layer changes due to parameter updates in previous layers. This phenomenon, known as internal covariate shift, can slow down training and make it harder for the network to converge.

### 2. **Gradient Flow**
Normalization helps maintain healthy gradient flow throughout the network, preventing vanishing or exploding gradients that can occur in deep networks.

### 3. **Training Stability**
By keeping activations in a reasonable range, normalization makes training more stable and less sensitive to initialization and learning rate choices.

### 4. **Faster Convergence**
Normalized networks typically converge faster because the optimization landscape becomes smoother and more predictable.

### 5. **Regularization Effect**
Some normalization techniques provide implicit regularization, helping to prevent overfitting.

## Three Normalization Techniques

### 1. LayerNorm (Layer Normalization)

**Implementation**: [`LayerNorm`](../src/normalization/layernorm.py:4)

**How it works**:
- Normalizes across the feature dimension for each token
- Computes mean and variance across the hidden dimension
- Applies learned scale (γ) and bias (β) parameters

**Formula**:
```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```

**When to use**:
- **Transformer models** (GPT, BERT, T5)
- **RNNs and sequence models** where batch normalization is problematic
- **Small batch sizes** where batch statistics are unreliable
- **Online/streaming inference** where you process one sample at a time

**Advantages**:
- Independent of batch size
- Works well with variable sequence lengths
- Stable across different batch sizes
- Good for transfer learning

**Disadvantages**:
- Slightly more computationally expensive than RMSNorm
- May not be as effective as batch normalization for CNNs

### 2. RMSNorm (Root Mean Square Normalization)

**Implementation**: [`RMSNorm`](../src/normalization/rmsnorm.py:4)

**How it works**:
- Normalizes by the root mean square (RMS) of activations
- Does NOT subtract the mean (unlike LayerNorm)
- Only applies learned scale parameter (γ), no bias

**Formula**:
```
RMSNorm(x) = γ * x / √(mean(x²) + ε)
```

**When to use**:
- **Large language models** (LLaMA, Mistral, PaLM)
- **Memory-constrained environments** where you want to reduce parameters
- **High-performance inference** where computational efficiency matters
- **Models that benefit from preserving activation magnitudes**

**Advantages**:
- **Faster computation** (no mean calculation)
- **Fewer parameters** (no bias term)
- **Better memory efficiency**
- **Empirically works well** for large language models
- **More stable gradients** in some cases

**Disadvantages**:
- May not center activations around zero
- Less studied than LayerNorm
- Might not work as well for all architectures

### 3. ScaleNorm

**Implementation**: [`ScaleNorm`](../src/normalization/scalenorm.py:4)

**How it works**:
- Normalizes each token vector by its L2 norm
- Applies a single global scale parameter (g)
- Simplest form of normalization

**Formula**:
```
ScaleNorm(x) = g * x / ||x||₂
```

**When to use**:
- **Lightweight transformer architectures**
- **Resource-constrained environments** (mobile, edge devices)
- **Research experiments** exploring minimal normalization
- **When you want maximum simplicity**

**Advantages**:
- **Minimal parameters** (single scalar)
- **Very fast computation**
- **Minimal memory overhead**
- **Simple to implement and debug**

**Disadvantages**:
- **Limited expressiveness** compared to LayerNorm/RMSNorm
- **May not provide sufficient normalization** for complex models
- **Less proven** in large-scale applications
- **May struggle with diverse activation patterns**

## Comparison Summary

| Technique | Parameters | Computation | Memory | Use Case |
|-----------|------------|-------------|---------|----------|
| **LayerNorm** | 2 × dim | High | High | General transformers, proven stability |
| **RMSNorm** | 1 × dim | Medium | Medium | Large LLMs, efficiency-focused |
| **ScaleNorm** | 1 scalar | Low | Low | Lightweight models, research |

## Choosing the Right Normalization

### Use **LayerNorm** when:
- Building standard transformer models
- You need proven stability and performance
- Working with diverse architectures
- Batch size varies significantly

### Use **RMSNorm** when:
- Building large language models
- Computational efficiency is critical
- You want to reduce parameter count
- Following modern LLM architectures (LLaMA-style)

### Use **ScaleNorm** when:
- Building lightweight/mobile models
- Experimenting with minimal architectures
- Resource constraints are extreme
- You need maximum simplicity

## Implementation Notes

All three implementations in this codebase:
- Support arbitrary hidden dimensions
- Include epsilon for numerical stability
- Use PyTorch's efficient tensor operations
- Are compatible with gradient computation
- Can be easily integrated into transformer blocks
