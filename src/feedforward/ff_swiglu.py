import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """
    SwiGLU FeedForward block.
    Used in LLaMA and Mistral models for improved efficiency and performance.
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.mult = getattr(config, "ffn_hidden_mult", getattr(config, "ffn_mult", 4))
        self.dropout = config.dropout

        hidden_dim = int(self.mult * self.n_embd * 2 / 3)  # smaller than standard FFN
        self.w1 = nn.Linear(self.n_embd, hidden_dim)
        self.w2 = nn.Linear(self.n_embd, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, self.n_embd)
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x):
        # SwiGLU: (xW1 * σ(xW2))W3
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

# Alias for config compatibility
GatedFeedForward = SwiGLUFFN
