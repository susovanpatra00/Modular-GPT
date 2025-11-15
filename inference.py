#!/usr/bin/env python3
"""
Simple inference script to test the trained GPT model.
"""

import sys
import os
import torch

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model.gpt import GPTModel

def load_trained_model(checkpoint_path="experiments/checkpoints/gpt_model.pt"):
    """Load the trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, None, None, None
    
    print(f"📂 Loading model from: {checkpoint_path}")
    # Fix for PyTorch 2.6+ security feature
    import types
    torch.serialization.add_safe_globals([types.SimpleNamespace])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint['config']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = checkpoint['vocab_size']
    
    # Create model
    model = GPTModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Parameters: {model.get_num_params():,}")
    print(f"   - Vocabulary size: {vocab_size}")
    print(f"   - Architecture: {config.n_layers} layers, {config.n_embd} embedding dim")
    
    return model, stoi, itos, config

def generate_text(model, stoi, itos, config, prompt="", max_new_tokens=100, temperature=0.8, top_k=40):
    """Generate text using the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Encode the prompt
    if prompt:
        encoded_prompt = [stoi.get(c, 0) for c in prompt]  # Use 0 for unknown chars
    else:
        encoded_prompt = [0]  # Start with first character
    
    # Convert to tensor
    input_ids = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    
    print(f"🎯 Generating text...")
    print(f"   - Prompt: '{prompt}'")
    print(f"   - Max new tokens: {max_new_tokens}")
    print(f"   - Temperature: {temperature}")
    print(f"   - Top-k: {top_k}")
    print(f"   - Device: {device}")
    print("-" * 50)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode the generated text
    generated_tokens = generated[0].cpu().tolist()
    generated_text = ''.join([itos.get(token, '?') for token in generated_tokens])
    
    return generated_text

def interactive_mode(model, stoi, itos, config):
    """Interactive text generation mode."""
    print("\n🤖 Interactive Mode - Enter prompts to generate text")
    print("Commands: 'quit' to exit, 'clear' to clear screen")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\n📝 Enter prompt (or 'quit'): ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif prompt.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            # Generate text
            generated = generate_text(
                model, stoi, itos, config, 
                prompt=prompt, 
                max_new_tokens=150,
                temperature=0.8,
                top_k=40
            )
            
            print(f"\n🎨 Generated text:")
            print(f"'{generated}'")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main inference function."""
    print("🧠 GPT Model Inference Test")
    print("=" * 40)
    
    # Load model
    model, stoi, itos, config = load_trained_model()
    if model is None:
        return
    
    # Test with some sample prompts
    sample_prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
        "Hello world",
        ""  # Empty prompt
    ]
    
    print(f"\n🎯 Testing with sample prompts:")
    print("=" * 40)
    
    for i, prompt in enumerate(sample_prompts, 1):
        print(f"\n{i}. Testing prompt: '{prompt}'")
        generated = generate_text(
            model, stoi, itos, config,
            prompt=prompt,
            max_new_tokens=80,
            temperature=0.8,
            top_k=40
        )
        print(f"Generated: '{generated}'")
        print("-" * 30)
    
    # Interactive mode
    try:
        interactive_mode(model, stoi, itos, config)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()