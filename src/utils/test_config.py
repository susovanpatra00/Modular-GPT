#!/usr/bin/env python3
"""
Test script to verify model configuration and estimate training time.
"""

import sys
import os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import yaml
from types import SimpleNamespace
from src.model.gpt import GPTModel
from src.train.dataset import TextDataset

def test_configuration():
    """Test the model configuration and estimate parameters."""
    
    # Load config - handle different working directories
    config_paths = [
        "config/config.yaml",           # From project root
        "../../config/config.yaml"     # From src/utils/
    ]
    
    config_dict = None
    for config_path in config_paths:
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            print(f"✅ Loaded config from: {config_path}")
            break
        except FileNotFoundError:
            continue
    
    if config_dict is None:
        print("❌ Could not find config file. Tried:")
        for path in config_paths:
            print(f"   - {path}")
        return False
    
    config = SimpleNamespace(**config_dict)
    
    print("🔧 Testing Model Configuration")
    print("=" * 50)
    
    # Test dataset - handle different working directories
    data_paths = [
        "data/input.txt",           # From project root
        "../../data/input.txt"     # From src/utils/
    ]
    
    dataset = None
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            dataset = TextDataset(path, config.block_size)
            break
    
    if dataset:
        actual_vocab_size = dataset.vocab_size
        print(f"📊 Dataset Info:")
        print(f"   - Data file: {data_path}")
        print(f"   - Actual vocab size: {actual_vocab_size}")
        print(f"   - Config vocab size: {config.vocab_size}")
        print(f"   - Dataset length: {len(dataset):,} samples")
        print(f"   - Block size: {config.block_size}")
        
        # Update config with actual vocab size
        config.vocab_size = actual_vocab_size
    else:
        print("⚠️  Data file not found, using config vocab size")
        print("   Tried paths:")
        for path in data_paths:
            print(f"   - {path}")
    
    # Test model creation
    try:
        model = GPTModel(config)
        total_params = model.get_num_params()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n🤖 Model Info:")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
        print(f"   - Layers: {config.n_layers}")
        print(f"   - Embedding dim: {config.n_embd}")
        print(f"   - Attention heads: {config.num_heads}")
        
        # Estimate memory usage
        batch_size = config.batch_size
        seq_len = config.block_size
        
        # Rough memory estimation (very approximate)
        activation_memory = batch_size * seq_len * config.n_embd * config.n_layers * 4 / 1024 / 1024  # MB
        total_memory = (total_params * 4 / 1024 / 1024 + activation_memory * 2) / 1024  # GB (params + activations + gradients)
        
        print(f"\n💾 Memory Estimation (approximate):")
        print(f"   - Model parameters: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        print(f"   - Activations (batch): ~{activation_memory:.1f} MB")
        print(f"   - Total GPU memory: ~{total_memory:.1f} GB")
        
        # Training time estimation
        if dataset:
            batches_per_epoch = len(dataset) // batch_size
            total_batches = batches_per_epoch * 3  # 3 epochs
            
            # Very rough estimate: small model on A100 can do ~1000-2000 tokens/sec
            tokens_per_batch = batch_size * seq_len
            estimated_time_minutes = (total_batches * tokens_per_batch) / 1500 / 60  # Conservative estimate
            
            print(f"\n⏱️  Training Time Estimation:")
            print(f"   - Batches per epoch: {batches_per_epoch}")
            print(f"   - Total batches (3 epochs): {total_batches}")
            print(f"   - Estimated time: ~{estimated_time_minutes:.1f} minutes")
            
            if estimated_time_minutes > 20:
                print("   ⚠️  Warning: Estimated time > 20 minutes")
                print("   💡 Consider reducing batch_size or block_size")
            else:
                print("   ✅ Estimated time looks good for quick training!")
        else:
            print(f"\n⏱️  Training Time Estimation: Skipped (no dataset found)")
        
        print(f"\n✅ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_configuration()