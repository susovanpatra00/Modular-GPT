import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A single expert MLP."""
    def __init__(self, n_embd, hidden_mult=4, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_mult * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Top2Router(nn.Module):
    """Top-2 gating mechanism for MoE."""
    def __init__(self, n_embd, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Linear(n_embd, n_experts, bias=False)

    def forward(self, x):
        # Gating scores
        logits = self.gate(x)
        topk_vals, topk_idx = torch.topk(logits, k=2, dim=-1)  # top-2 experts
        gates = F.softmax(topk_vals, dim=-1)
        return topk_idx, gates

class MoEFFN(nn.Module):
    """
    Mixture of Experts FeedForward network.
    Each token routed to top-2 experts based on learned gating.
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_experts = getattr(config, "n_experts", 4)
        self.dropout = config.dropout
        self.experts = nn.ModuleList(
            [Expert(self.n_embd, hidden_mult=4, dropout=self.dropout) for _ in range(self.n_experts)]
        )
        self.router = Top2Router(self.n_embd, self.n_experts)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(B * T, C)

        topk_idx, gates = self.router(x_flat)  # (B*T, 2), (B*T, 2)
        output = torch.zeros_like(x_flat)

        for i in range(2):  # top-2 routing
            expert_idx = topk_idx[:, i]
            gate = gates[:, i].unsqueeze(-1)
            for eid in range(self.n_experts):
                mask = (expert_idx == eid)
                if mask.sum() == 0:
                    continue
                expert_inp = x_flat[mask]
                expert_out = self.experts[eid](expert_inp)
                output[mask] += gate[mask] * expert_out

        return output.view(B, T, C)
