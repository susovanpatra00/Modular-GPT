import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    RMSNorm: Root Mean Square Normalization
    (Used in models like LLaMA, Mistral)
    Unlike LayerNorm, it does NOT subtract the mean — 
    only rescales based on the root-mean-square of activations.
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps  # small constant to prevent division by zero
        self.weight = nn.Parameter(torch.ones(dim))  # learnable scale parameter (γ)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)

        # 1️⃣ Compute L2 norm along the hidden dimension (dim = -1)
        #    x.norm(2, dim=-1) means L2 norm → sqrt(sum(x_i^2))
        #    `dim=-1` means compute this for each token’s feature vector.
        #
        # 2️⃣ Divide by sqrt(hidden_dim) to convert L2 norm to RMS (root mean square)
        norm = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)

        # 3️⃣ Normalize x by dividing by RMS value
        #    Add eps for numerical stability.
        x_norm = x / (norm + self.eps)

        # 4️⃣ Scale by learnable weight (γ)
        return self.weight * x_norm
