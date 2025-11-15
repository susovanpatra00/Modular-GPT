#!/usr/bin/env python3
"""
Simple training script for the Modular GPT model.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train.trainer import train_model

def validate_data_file(data_path, min_chars=10000):
    """Validate the data file for training."""
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        if len(data) < min_chars:
            print(f"❌ Data file too small: {len(data)} characters (minimum: {min_chars})")
            return False
        
        # Check vocabulary size
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        
        print(f"✅ Data validation passed:")
        print(f"   - File size: {len(data):,} characters")
        print(f"   - Vocabulary size: {vocab_size} unique characters")
        print(f"   - Sample characters: {repr(''.join(chars[:20]))}" + ("..." if len(chars) > 20 else ""))
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading data file: {e}")
        return False

def main():
    """Main training function."""
    print("🚀 Starting GPT training...")
    
    # Validate data file
    data_path = "data/input.txt"
    if not validate_data_file(data_path):
        print("Please ensure you have a valid training data file.")
        return
    
    try:
        # Start training with more epochs for larger model
        model, dataset = train_model(
            config_path="config/config.yaml",
            data_path=data_path,
            epochs=10,  # More epochs for better training with larger model
            save_path="experiments/checkpoints/gpt_model.pt"
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"Model saved with {model.get_num_params():,} parameters")
        print(f"Vocabulary size: {dataset.vocab_size}")
        print(f"Model architecture: {model.n_layers} layers, {model.n_embd} embedding dim")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()