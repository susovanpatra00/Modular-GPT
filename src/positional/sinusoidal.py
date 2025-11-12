import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from Vaswani et al. (2017).
    Adds position-dependent sine and cosine signals to token embeddings.
    """

    def __init__(self, dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, dim)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        returns: (batch, seq_len, dim)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
