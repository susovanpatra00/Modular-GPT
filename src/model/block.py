import torch
import torch.nn as nn

import inspect

# dynamically import selected modules
from importlib import import_module


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.n_embd

        # --- Load Attention ---
        attn_module = import_module(f"src.attention.{config.attention_type}")
        self.attn = attn_module.__dict__[config.attention_class](config)

        # --- Load FeedForward ---
        ffn_module = import_module(f"src.feedforward.{config.ffn_type}")
        self.ffn = ffn_module.__dict__[config.ffn_class](config)

        # --- Load Normalization ---
        norm_module = import_module(f"src.normalization.{config.norm_type}")
        # Pass eps parameter if available in config, otherwise use default
        norm_eps = getattr(config, 'norm_eps', 1e-5)
        self.norm1 = norm_module.__dict__[config.norm_class](dim, eps=norm_eps)
        self.norm2 = norm_module.__dict__[config.norm_class](dim, eps=norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    # def forward(self, x, mask=None):
    #     # Attention + Residual
    #     # Check if attention module supports mask parameter
    #     if hasattr(self.attn.forward, '__code__') and 'mask' in self.attn.forward.__code__.co_varnames:
    #         attn_out = self.attn(self.norm1(x), mask=mask)
    #     else:
    #         attn_out = self.attn(self.norm1(x))
    #     x = x + self.dropout(attn_out)

    #     # FeedForward + Residual
    #     ffn_out = self.ffn(self.norm2(x))
    #     x = x + self.dropout(ffn_out)

    #     return x
    def forward(self, x, mask=None):
        # Attention + Residual
        sig = inspect.signature(self.attn.forward)
        if 'mask' in sig.parameters:
            attn_out = self.attn(self.norm1(x), mask=mask)
        else:
            attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # FeedForward + Residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x
