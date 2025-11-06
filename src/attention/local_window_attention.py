import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalWindowAttention(nn.Module):
    """
    LocalWindowAttention: limits attention context to a local window for efficiency.
    Each token attends only to its W preceding tokens.
    """
    def __init__(self, config, window_size=128):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.window_size = getattr(config, "window_size", window_size)

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)  # Shape: (B, T, C) -> (B, T, C)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)  # Shape: (B, T, C) -> (B, T, C)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)  # Shape: (B, T, C) -> (B, T, C)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)  # Shape: (B, T, C) -> (B, T, C)

    def forward(self, x):
        B, T, C = x.shape  # Input shape: (B, T, C)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # Shape: (B, T, C) -> (B, n_head, T, head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # Shape: (B, T, C) -> (B, n_head, T, head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # Shape: (B, T, C) -> (B, n_head, T, head_dim)

        # Compute local attention
        attn_out = torch.zeros_like(q)  # Shape: (B, n_head, T, head_dim)
        for t in range(T):
            start = max(0, t - self.window_size)
            q_t = q[:, :, t:t+1, :]                          # Shape: (B, n_head, 1, head_dim)
            k_win = k[:, :, start:t+1, :]                    # Shape: (B, n_head, W, head_dim) where W = window_size
            v_win = v[:, :, start:t+1, :]                    # Shape: (B, n_head, W, head_dim)
            attn_scores = torch.matmul(q_t, k_win.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Shape: (B, n_head, 1, W)
            attn_weights = F.softmax(attn_scores, dim=-1)    # Shape: (B, n_head, 1, W)
            attn_values = torch.matmul(attn_weights, v_win)  # Shape: (B, n_head, 1, head_dim)
            attn_out[:, :, t:t+1, :] = attn_values           # Update position t in output

        out = attn_out.transpose(1, 2).contiguous().view(B, T, C)  # Shape: (B, n_head, T, head_dim) -> (B, T, C)
        return self.out_proj(out)  # Shape: (B, T, C) -> (B, T, C)
