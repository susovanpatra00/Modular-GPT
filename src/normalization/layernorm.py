import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Standard LayerNorm used in models like GPT, BERT, etc.
    It normalizes each token's hidden activations across its feature dimension.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps  # Small constant to avoid division by zero during normalization

        # Learnable scale and bias parameters (γ and β)
        # Initialized to ones and zeros respectively
        self.weight = nn.Parameter(torch.ones(dim))   # γ: scales normalized output
        self.bias = nn.Parameter(torch.zeros(dim))    # β: shifts normalized output

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        
        # Compute the mean across the last dimension (hidden_dim)
        # keepdim=True keeps the shape (batch_size, seq_len, 1)
        # so it can be broadcasted back to x
        mean = x.mean(-1, keepdim=True)

        # Compute the variance across the last dimension (hidden_dim)
        # unbiased=False gives the population variance (more stable for small batches)
        var = x.var(-1, unbiased=False, keepdim=True)

        # Normalize: subtract mean and divide by std (√var)
        # Add eps for numerical stability (avoid division by zero)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learned scale (γ) and bias (β)
        # Broadcasting automatically matches shapes
        return self.weight * x_norm + self.bias
