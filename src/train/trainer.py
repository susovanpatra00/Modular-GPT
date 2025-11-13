# src/train/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from types import SimpleNamespace
import os
from src.model.gpt import GPTModel
from src.train.dataset import TextDataset

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)

def train_model(config_path="configs/base_config.yaml", data_path="data/input.txt",
                epochs=10, save_path="model.pt"):
    """
    Train the GPT model.
    
    Args:
        config_path: Path to configuration file
        data_path: Path to training data
        epochs: Number of training epochs
        save_path: Path to save the trained model
    """
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    dataset = TextDataset(data_path, config.block_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print(f"Dataset loaded: {len(dataset)} samples, vocab_size: {dataset.vocab_size}")
    
    # Update config with actual vocab size
    config.vocab_size = dataset.vocab_size
    
    # Model
    model = GPTModel(config).to(device)
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Optimizer
    lr = float(getattr(config, 'learning_rate', 3e-4))
    weight_decay = float(getattr(config, 'weight_decay', 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids, targets = [x.to(device) for x in batch]
            loss, _ = model(input_ids, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Avg Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed | Avg Loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos
    }, save_path)
    print(f"✅ Training complete. Model saved to {save_path}")
    
    return model, dataset

# Alias for backward compatibility
train = train_model
