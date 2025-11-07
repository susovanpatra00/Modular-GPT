import torch
import torch.nn as nn

class ScaleNorm(nn.Module):
    """
    ScaleNorm: normalizes each token vector by its L2 norm.
    Used in some efficient and lightweight transformer architectures.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = x.norm(p=2, dim=-1, keepdim=True)
        return self.g * x / (norm + self.eps)
