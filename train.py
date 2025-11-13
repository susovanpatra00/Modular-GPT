#!/usr/bin/env python3
"""
Simple training script for the Modular GPT model.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train.trainer import train_model

def main():
    """Main training function."""
    print("🚀 Starting GPT training...")
    
    # Check if data file exists
    data_path = "data/input.txt"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure you have the training data file.")
        return
    
    try:
        # Start training
        model, dataset = train_model(
            config_path="configs/base_config.yaml",
            data_path=data_path,
            epochs=5,  # Start with fewer epochs for testing
            save_path="experiments/checkpoints/gpt_model.pt"
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"Model saved with {model.get_num_params():,} parameters")
        print(f"Vocabulary size: {dataset.vocab_size}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()