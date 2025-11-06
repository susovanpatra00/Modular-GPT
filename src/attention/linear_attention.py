import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    """
    LinearAttention: low-memory approximation using kernel feature maps.
    Complexity O(n) instead of O(n^2).
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)  # Shape: (B, T, C) -> (B, T, C)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)  # Shape: (B, T, C) -> (B, T, C)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)  # Shape: (B, T, C) -> (B, T, C)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)  # Shape: (B, T, C) -> (B, T, C)

    def feature_map(self, x):
        return F.elu(x) + 1  # Shape: unchanged, applies element-wise

    def forward(self, x):
        B, T, C = x.shape  # Input shape: (B, T, C)
        q = self.feature_map(  # Shape: (B, n_head, T, head_dim)
            self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, T, C) -> (B, n_head, T, head_dim)
        )
        k = self.feature_map(  # Shape: (B, n_head, T, head_dim)
            self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, T, C) -> (B, n_head, T, head_dim)
        )
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # Shape: (B, T, C) -> (B, n_head, T, head_dim)

        # Cumulative key-value projections
        kv = torch.einsum('bhnd,bhne->bhde', k, v)     # Shape: (B, n_head, head_dim, head_dim)
        z = 1 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-8)  # Shape: (B, n_head, T)
        out = torch.einsum('bhnd,bhde->bhne', q, kv)   # Shape: (B, n_head, T, head_dim)
        out = out * z.unsqueeze(-1)                    # Shape: (B, n_head, T, head_dim) * (B, n_head, T, 1)

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # Shape: (B, n_head, T, head_dim) -> (B, T, C)
        return self.out_proj(out)  # Shape: (B, T, C) -> (B, T, C)
