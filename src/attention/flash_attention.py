import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttention(nn.Module):
    """
    FlashAttention: memory-efficient scaled dot-product attention.
    Uses PyTorch's native implementation which calls FlashAttention kernels on supported GPUs.
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

    def forward(self, x, mask=None):
        B, T, C = x.size()  # Input shape: (B, T, C)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # Shape: (B, T, C) -> (B, n_head, T, head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # Shape: (B, T, C) -> (B, n_head, T, head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # Shape: (B, T, C) -> (B, n_head, T, head_dim)

        # FlashAttention 
        out = F.scaled_dot_product_attention(  # Shape: (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # Shape: (B, n_head, T, head_dim) -> (B, T, C)
        return self.out_proj(out)  # Shape: (B, T, C) -> (B, T, C)
