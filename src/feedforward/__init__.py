from .ff_standard import StandardFFN
from .ff_swiglu import SwiGLUFFN
from .ff_moe import MoEFFN

# Alias for config compatibility
GatedFeedForward = SwiGLUFFN

__all__ = [
    'StandardFFN',
    'SwiGLUFFN',
    'MoEFFN',
    'GatedFeedForward'
]