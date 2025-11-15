# 🧠 Modular GPT - A Learning Journey

A **modular transformer implementation** built for learning and experimentation with different GPT components. This project demonstrates how to build a configurable GPT model with swappable attention mechanisms, feedforward networks, and normalization layers.

## 🎯 Learning Objectives

This project was created to understand and experiment with:
- **Transformer Architecture**: Core components and their interactions
- **Attention Mechanisms**: Flash, Linear, and Local Window attention
- **Feedforward Networks**: Standard, SwiGLU, and Mixture of Experts (MoE)
- **Normalization Techniques**: LayerNorm, RMSNorm, and ScaleNorm
- **Positional Encodings**: Sinusoidal, RoPE, and ALiBi approaches
- **Training Dynamics**: Character-level language modeling on Shakespeare

## 🏗️ Modular Architecture

The model is **completely configurable** through YAML - swap components without changing code:

```yaml
# Choose your components
attention_type: flash_attention    # flash_attention | linear_attention | local_window_attention
ffn_type: ff_swiglu               # ff_standard | ff_swiglu | ff_moe
norm_type: rmsnorm                # layernorm | rmsnorm | scalenorm
pos_type: sinusoidal              # sinusoidal | learned
```

### 🧩 Available Components

| Component | Options | Purpose |
|-----------|---------|---------|
| **Attention** | Flash, Linear, Local Window | Memory efficiency vs. context length trade-offs |
| **Feedforward** | Standard, SwiGLU, MoE | Parameter efficiency and specialization |
| **Normalization** | LayerNorm, RMSNorm, ScaleNorm | Training stability and computational efficiency |
| **Positional** | Sinusoidal, Learned | Position encoding strategies |

## 🚀 Quick Start

### 1. Setup
```bash
git clone <repo-url>
cd Modular-GPT
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```

### 3. Test Generation
```bash
python inference.py
```

## 📊 Sample Output

```
🧠 GPT Model Inference Test
========================================
📂 Loading model from: experiments/checkpoints/gpt_model.pt
✅ Model loaded successfully!
   - Parameters: 3,175,164
   - Vocabulary size: 62
   - Architecture: 6 layers, 256 embedding dim

🎯 Testing with sample prompts:
========================================

1. Testing prompt: 'The quick brown fox'
🎯 Generating text...
   - Prompt: 'The quick brown fox'
   - Max new tokens: 80
   - Temperature: 0.8
   - Top-k: 40
   - Device: cuda
--------------------------------------------------
Generated: 'The quick brown foxen vouch, and so hand me
Your loving eyes so this condition.

AUFIDIUS:
So, you '
------------------------------

2. Testing prompt: 'Once upon a time'
🎯 Generating text...
   - Prompt: 'Once upon a time'
   - Max new tokens: 80
   - Temperature: 0.8
   - Top-k: 40
   - Device: cuda
--------------------------------------------------
Generated: 'Once upon a time.

MENENIUS:
Noble Menenius,
Be you then a present volour?

SICINIUS:
Sir, how c'
------------------------------

🤖 Interactive Mode - Enter prompts to generate text
Commands: 'quit' to exit, 'clear' to clear screen
============================================================
```

## 🔬 What I Learned

### **Attention Mechanisms**
- **Flash Attention**: 2-4x faster with exact same results through memory optimization
- **Linear Attention**: O(T) complexity but approximates full attention
- **Local Window**: Perfect for tasks with local dependencies

### **Feedforward Networks**
- **Standard FFN**: Simple 2-layer MLP with GELU activation
- **SwiGLU**: Gated mechanism with fewer parameters but better performance
- **MoE**: Sparse expert routing for scaling model capacity

### **Normalization Impact**
- **LayerNorm**: Standard choice, works everywhere
- **RMSNorm**: Faster computation, used in modern LLMs (LLaMA, Mistral)
- **ScaleNorm**: Minimal parameters for resource-constrained environments

### **Training Insights**
- **Character-level tokenization** works surprisingly well for Shakespeare
- **Gradient clipping** at 1.0 prevents training instability
- **Small models** (3M params) can learn meaningful patterns quickly
- **Loss progression**: 4-5 → 1-2 indicates successful pattern learning

## 📁 Project Structure

```
Modular-GPT/
├── src/
│   ├── attention/          # Flash, Linear, Local Window attention
│   ├── feedforward/        # Standard, SwiGLU, MoE networks
│   ├── normalization/      # LayerNorm, RMSNorm, ScaleNorm
│   ├── positional/         # Sinusoidal, ALiBi encodings
│   ├── model/             # GPT model and transformer blocks
│   └── train/             # Training utilities and dataset handling
├── docs/                  # Detailed documentation for each component
├── config/               # YAML configuration files
├── data/                 # Training data (Shakespeare text)
└── experiments/          # Model checkpoints and outputs
```

## 🎛️ Configuration

The model is controlled through [`config/config.yaml`](config/config.yaml):

```yaml
# Model Architecture
n_embd: 256          # Embedding dimension
num_heads: 8         # Attention heads
n_layers: 6          # Transformer layers
block_size: 256      # Context window

# Component Selection
attention_type: flash_attention
ffn_type: ff_swiglu
norm_type: rmsnorm
pos_type: sinusoidal

# Training Parameters
batch_size: 128
learning_rate: 1e-3
weight_decay: 0.01
```

## 📚 Documentation

Detailed explanations available in [`docs/`](docs/):
- [`attention.md`](docs/attention.md) - Deep dive into attention mechanisms
- [`feedforward.md`](docs/feedforward.md) - Feedforward network comparisons
- [`normalization.md`](docs/normalization.md) - Normalization techniques
- [`positional_encoding.md`](docs/positional_encoding.md) - Position encoding methods
- [`model_architecture.md`](docs/model_architecture.md) - Overall architecture
- [`training_loop.md`](docs/training_loop.md) - Training process details

## 🔧 Key Features

- **🔄 Modular Design**: Swap components via configuration
- **📈 Educational**: Extensive documentation and comments
- **⚡ Efficient**: Modern techniques (Flash Attention, RMSNorm, SwiGLU)
- **🎯 Practical**: Trains quickly on consumer hardware
- **🧪 Experimental**: Easy to test different architectural choices

## 🎓 Learning Outcomes

This project taught me:
1. **How transformers really work** - not just theory, but implementation details
2. **Trade-offs between different components** - memory vs. speed vs. accuracy
3. **Modern optimization techniques** - why Flash Attention and RMSNorm matter
4. **Training dynamics** - gradient clipping, learning rates, loss curves
5. **Modular software design** - building extensible, configurable systems

## 🚀 Future Experiments

- [ ] Implement more attention variants (Sparse, BigBird)
- [ ] Add different tokenization strategies (BPE, SentencePiece)
- [ ] Experiment with different datasets and tasks
- [ ] Add model parallelism for larger models
- [ ] Implement more advanced training techniques

---

*Built for learning and experimentation with transformer architectures. Each component is documented and designed to be educational.*