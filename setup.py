#!/usr/bin/env python3
"""
Setup script for Modular GPT project.
This script handles project setup, dependency installation, and data preparation.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return None

def download_file(url, destination, description=""):
    """Download a file from URL."""
    print(f"🔄 {description}")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✅ {description} - Success")
        return True
    except Exception as e:
        print(f"❌ {description} - Failed: {e}")
        return False

def setup_project():
    """Main setup function."""
    print("🚀 Setting up Modular GPT project...")
    
    # Create necessary directories
    directories = [
        "data",
        "experiments/checkpoints",
        "experiments/logs", 
        "experiments/results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install Python dependencies
    print("\n📦 Installing Python dependencies...")
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        cmd = f"{sys.executable} -m pip install -r {requirements_file}"
        run_command(cmd, "Installing requirements")
    else:
        # Install basic requirements
        basic_deps = ["torch>=2.0.0", "numpy", "pyyaml"]
        for dep in basic_deps:
            cmd = f"{sys.executable} -m pip install {dep}"
            run_command(cmd, f"Installing {dep}")
    
    # Download training data
    print("\n📊 Setting up training data...")
    data_file = "data/input.txt"
    
    if not os.path.exists(data_file):
        # Try to download Shakespeare dataset
        shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        success = download_file(shakespeare_url, data_file, "Downloading Shakespeare dataset")
        
        if not success:
            # Fallback: create a small sample dataset
            print("🔄 Creating sample dataset...")
            sample_text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd."""
            
            with open(data_file, 'w', encoding='utf-8') as f:
                # Repeat the text to make it longer
                f.write((sample_text + "\n") * 100)
            print("✅ Created sample dataset")
    else:
        print(f"✅ Training data already exists: {data_file}")
    
    # Verify setup
    print("\n🔍 Verifying setup...")
    
    # Check if we can import the main modules
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from src.model.gpt import GPTModel
        from src.train.trainer import load_config
        from src.train.dataset import TextDataset
        print("✅ All Python modules can be imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check data file
    if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
        print(f"✅ Training data ready: {data_file}")
    else:
        print(f"❌ Training data missing or empty: {data_file}")
        return False
    
    # Check config file
    config_file = "configs/base_config.yaml"
    if os.path.exists(config_file):
        print(f"✅ Configuration ready: {config_file}")
    else:
        print(f"❌ Configuration missing: {config_file}")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run training: python3 train.py")
    print("2. Or test with smaller model: python3 -c \"from src.train.trainer import train_model; train_model(config_path='configs/test_config.yaml', epochs=1)\"")
    
    return True

if __name__ == "__main__":
    success = setup_project()
    sys.exit(0 if success else 1)
