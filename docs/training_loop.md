# Training Loop Documentation

## Overview

So I've been working on training this GPT model, and I thought I'd document how the whole training process works. It's been quite a journey getting everything set up and running smoothly!

## What I'm Training On

I'm using Shakespeare's text as my training data - specifically what looks like Coriolanus and other plays. The text file (`data/input.txt`) contains the full text (first 10000 lines) with character names, dialogue, and stage directions. It's actually pretty cool to see the model learn from such rich, structured text.

## My Training Setup

### Model Configuration

I kept the model relatively small for quick experimentation:
- **Embedding dimension**: 256 (keeps things manageable)
- **Number of heads**: 8 (good balance for attention)
- **Layers**: 6 (enough depth without being too heavy)
- **Block size**: 256 tokens (decent context window)
- **Batch size**: 128 (fits well in memory)

I'm using some modern components that I've been experimenting with:
- **Flash Attention** for efficient attention computation
- **SwiGLU feedforward** networks (they seem to work better than standard FFN)
- **RMSNorm** instead of LayerNorm (slightly more stable in my experience)
- **Sinusoidal positional encoding** (classic but reliable)

### Training Parameters

I set up the training with these parameters:
- **Learning rate**: 1e-3 (a bit higher than usual, but works well for this size)
- **Weight decay**: 0.01 (helps with regularization)
- **Gradient clipping**: 1.0 (prevents those nasty gradient explosions)

## How the Training Actually Works

### Data Processing

The [`TextDataset`](src/train/dataset.py:5) class handles all the text processing:

1. **Text Loading**: Reads the entire Shakespeare text file
2. **Character-level tokenization**: Creates a vocabulary from all unique characters
3. **Encoding**: Converts text to integer sequences using a simple character-to-index mapping
4. **Chunking**: Creates training examples by sliding a window over the text

The dataset is pretty smart - it handles edge cases like when the text is shorter than expected and pads sequences when needed.

### The Training Loop

The main training happens in [`train_model()`](src/train/trainer.py:17). Here's how I structured it:

#### 1. Setup Phase
```python
# Load configuration and set device
config = load_config(config_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

I always check for CUDA availability because training on GPU is so much faster.

#### 2. Data Preparation
```python
dataset = TextDataset(data_path, config.block_size)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
```

The dataloader shuffles the data each epoch, which helps with training stability.

#### 3. Model Initialization
```python
model = GPTModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

I'm using AdamW because it handles weight decay better than regular Adam.

#### 4. The Actual Training Loop

This is where the magic happens:

```python
for epoch in range(epochs):
    for batch_idx, batch in enumerate(dataloader):
        input_ids, targets = [x.to(device) for x in batch]
        
        # Forward pass
        loss, _ = model(input_ids, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (super important!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
```

### Progress Tracking

I added some nice progress tracking that prints every 50 batches:
```
Epoch 1/10 | Batch 50/156 (32.1%) | Loss: 2.3456
```

This helps me keep track of how training is going without overwhelming the console.

### Timing (GPU Only)

When training on GPU, I use CUDA events to time each epoch:
```python
epoch_start_time = torch.cuda.Event(enable_timing=True)
epoch_end_time = torch.cuda.Event(enable_timing=True)
```

This gives me accurate timing information that's really helpful for planning longer training runs.

## What Gets Saved

After training completes, I save everything I need for inference:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'vocab_size': dataset.vocab_size,
    'stoi': dataset.stoi,  # string to index mapping
    'itos': dataset.itos   # index to string mapping
}, save_path)
```

The vocabulary mappings are crucial - without them, I can't decode the model's outputs back to text!

## Training Experience

### What Works Well

1. **Character-level tokenization** is surprisingly effective for Shakespeare's text
2. **Gradient clipping** at 1.0 prevents training instability
3. **Progress reporting every 50 batches** gives good feedback without spam
4. **Small model size** allows for quick experimentation

### Challenges I Faced

1. **Memory management** - had to be careful with batch sizes
2. **Vocabulary size** - Shakespeare uses a lot of unique characters including punctuation
3. **Sequence length** - balancing context window with memory usage

### Performance Notes

On my setup, training typically takes:
- **CPU**: Pretty slow, mainly for testing
- **GPU**: Much faster, can complete several epochs in reasonable time

The loss usually starts around 4-5 (random character prediction) and drops to around 1-2 after a few epochs, which indicates the model is learning the patterns in Shakespeare's text.

## Future Improvements

Some things I'm thinking about for next iterations:

1. **Learning rate scheduling** - could help with convergence
2. **Validation set** - currently just training, no validation tracking
3. **Better tokenization** - maybe subword tokenization for better efficiency
4. **Longer training** - more epochs might improve quality
5. **Model size experiments** - try different architectures

## Usage

To train the model, I just run:
```python
from src.train.trainer import train_model
model, dataset = train_model(
    config_path="config/config.yaml",
    data_path="data/input.txt",
    epochs=10,
    save_path="experiments/checkpoints/gpt_model.pt"
)
```

The function returns both the trained model and the dataset, which is handy for immediate testing or further experimentation.

That's pretty much how my training setup works! It's been a fun project getting this all working smoothly.