import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardFFN(nn.Module):
    """
    Standard 2-layer feedforward network used in GPT/Transformer blocks.
    Expands embedding dimension by 4x (config.ffn_mult default = 4).
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.mult = getattr(config, "ffn_mult", 4)
        self.dropout = config.dropout

        self.fc1 = nn.Linear(self.n_embd, self.mult * self.n_embd)
        self.fc2 = nn.Linear(self.mult * self.n_embd, self.n_embd)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
