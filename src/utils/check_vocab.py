#!/usr/bin/env python3
"""
Script to check the actual vocabulary size of the dataset.
"""

def check_vocab_size(data_path):
    """Check the actual vocabulary size of the dataset."""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        
        print(f"Dataset: {data_path}")
        print(f"Total characters in dataset: {len(data):,}")
        print(f"Unique characters (vocab_size): {vocab_size}")
        print(f"Characters: {repr(''.join(chars[:50]))}" + ("..." if len(chars) > 50 else ""))
        
        return vocab_size, len(data)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return None, None

if __name__ == "__main__":
    # Adjust path since we're in src/utils/
    data_path = "../../data/input.txt"
    if not os.path.exists(data_path):
        data_path = "data/input.txt"  # Fallback for running from project root
    
    vocab_size, total_chars = check_vocab_size(data_path)
    if vocab_size:
        print(f"\nRecommended config update:")
        print(f"vocab_size: {vocab_size}")