# src/train/dataset.py
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text_path, block_size):
        with open(text_path, 'r', encoding='utf-8') as f:
            self.data = f.read()
        self.chars = sorted(list(set(self.data)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.block_size = block_size
        self.vocab_size = len(self.chars)
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])
        self.encoded = torch.tensor(self.encode(self.data), dtype=torch.long)
        
    def __len__(self):
        # Ensure we return at least 0, even for small datasets
        return max(0, len(self.encoded) - self.block_size)
    
    def __getitem__(self, idx):
        # Handle edge case where dataset is smaller than expected
        if idx >= len(self):
            idx = len(self) - 1 if len(self) > 0 else 0
        
        chunk = self.encoded[idx:idx+self.block_size+1]
        
        # Pad if chunk is too small
        if len(chunk) < self.block_size + 1:
            # Repeat the data to fill the chunk
            while len(chunk) < self.block_size + 1:
                chunk = torch.cat([chunk, self.encoded[:min(len(self.encoded), self.block_size + 1 - len(chunk))]])
        
        x = chunk[:self.block_size]
        y = chunk[1:self.block_size+1]
        return x, y
