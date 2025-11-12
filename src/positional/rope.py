import math
import torch

def compute_freq(dim: int, seq_len: int, base: int = 10000):
    """
    Compute complex frequency matrix for RoPE.

    Args:
        dim (int): Embedding dimension.
        seq_len (int): Maximum sequence length.
        base (int): Frequency base (usually 10000).

    Returns:
        torch.Tensor: Precomputed RoPE frequencies of shape (seq_len, dim // 2, 2)
    """
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    seq_idx = torch.arange(seq_len, dtype=torch.float)
    freqs = torch.outer(seq_idx, theta)
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    freqs = torch.stack((freqs_cos, freqs_sin), dim=-1)
    return freqs


def apply_rope(q, k, freqs):
    """
    Apply RoPE to Q and K tensors.
    Args:
        q, k: (batch, seq_len, dim)
        freqs: precomputed RoPE frequencies from compute_freq()
    Returns:
        q_rot, k_rot: rotated Q and K tensors
    """
    seq_len = q.size(1)
    freqs = freqs[:seq_len].to(q.device)
    cos, sin = freqs[..., 0], freqs[..., 1]

    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_rot, k_rot
