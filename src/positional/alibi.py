import torch
import torch.nn as nn
import math

class ALiBiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) positional encoding.
    Instead of adding positional embeddings, it adds a bias to attention scores
    that is proportional to the distance between tokens.
    """
    
    def __init__(self, num_heads, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Create slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # Pre-compute bias matrix
        bias = self._build_alibi_bias(max_seq_len, slopes)
        self.register_buffer('bias', bias)
    
    def _get_slopes(self, num_heads):
        """Generate slopes for ALiBi bias."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(num_heads))
        else:
            # If not power of 2, use closest power of 2 and interpolate
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = get_slopes_power_of_2(2*closest_power_of_2)
            slopes = slopes_a + slopes_b[0::2][:num_heads-closest_power_of_2]
            return torch.tensor(slopes)
    
    def _build_alibi_bias(self, seq_len, slopes):
        """Build the ALiBi bias matrix."""
        # Create position matrix
        pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        pos = pos.abs()  # Distance between positions
        
        # Apply slopes to create bias for each head
        bias = pos.unsqueeze(0) * slopes.unsqueeze(1).unsqueeze(2)
        return -bias  # Negative because we want closer tokens to have higher scores
    
    def forward(self, seq_len):
        """
        Return ALiBi bias for the given sequence length.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            bias: Tensor of shape (num_heads, seq_len, seq_len)
        """
        if seq_len <= self.max_seq_len:
            return self.bias[:, :seq_len, :seq_len]
        else:
            # If sequence is longer than pre-computed, compute on the fly
            bias = self._build_alibi_bias(seq_len, self.slopes)
            return bias