import torch
import torch.nn as nn
from .block import TransformerBlock


class GPTModel(nn.Module):
    """
    GPT Model with modular components.
    Supports different attention mechanisms, feedforward networks, and normalization layers.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.n_layers = config.n_layers
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional encoding (configurable)
        if hasattr(config, 'pos_type'):
            from importlib import import_module
            pos_module = import_module(f"src.positional.{config.pos_type}")
            if config.pos_type == 'sinusoidal':
                self.pos_encoding = pos_module.SinusoidalPositionalEncoding(
                    config.n_embd,
                    getattr(config, 'pos_max_len', config.block_size)
                )
            else:
                # Fallback to learned embeddings for other types
                self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        else:
            # Default to learned position embeddings
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        from importlib import import_module
        norm_module = import_module(f"src.normalization.{config.norm_type}")
        norm_eps = getattr(config, 'norm_eps', 1e-5)
        self.ln_f = norm_module.__dict__[config.norm_class](config.n_embd, eps=norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between token embedding and lm_head (optional)
        if getattr(config, 'tie_weights', True):
            self.lm_head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, targets=None):
        """
        Forward pass of the GPT model.
        
        Args:
            input_ids: Token indices of shape (batch_size, sequence_length)
            targets: Target token indices for training (batch_size, sequence_length)
            
        Returns:
            If targets is None: logits of shape (batch_size, sequence_length, vocab_size)
            If targets is provided: (loss, logits)
        """
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # (B, T, C)
        
        # Apply positional encoding
        if hasattr(self, 'pos_encoding'):
            # Use configurable positional encoding (e.g., sinusoidal)
            x = self.pos_encoding(token_emb)
        else:
            # Use learned position embeddings
            pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            pos_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.position_embedding(pos_ids)   # (B, T, C)
            x = token_emb + pos_emb
        
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten for cross entropy
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1
            )
        
        if targets is not None:
            return loss, logits
        else:
            return logits
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non_embedding=True, the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, 'position_embedding'):
            n_params -= self.position_embedding.weight.numel()
        return n_params
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens using the model.
        
        Args:
            input_ids: Starting token indices (batch_size, sequence_length)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated token indices including the input
        """
        for _ in range(max_new_tokens):
            # Crop input_ids to the last block_size tokens
            input_ids_cond = input_ids if input_ids.size(1) <= self.block_size else input_ids[:, -self.block_size:]
            
            # Forward pass
            logits = self(input_ids_cond)
            
            # Get logits for the last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax and sample
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input_ids
            input_ids = torch.cat((input_ids, next_token), dim=1)
        
        return input_ids