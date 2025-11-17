"""
Transformer model components.
"""
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .nn_utils import Linear, Embedding, RMSNorm, silu
from .attention import MultiHeadSelfAttention


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    SwiGLU(x) = (SiLU(xW1) ⊙ xW3)W2
    where ⊙ is element-wise multiplication.
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Apply SwiGLU transformation."""
        return self.w2(silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.
    
    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=eps)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=True,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.ln2 = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(
        self,
        x: Float[Tensor, "batch sequence_length d_model"],
        token_positions: Int[Tensor, "batch sequence_length"] | None = None,
    ) -> Float[Tensor, "batch sequence_length d_model"]:
        """Apply Transformer block."""
        # Self-attention with residual
        x = x + self.attn(self.ln1(x), token_positions)
        # Feed-forward with residual
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    Transformer Language Model.
    
    Full decoder-only transformer with token embeddings,
    multiple transformer blocks, and output projection.
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        
        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = RMSNorm(d_model, eps=eps)
        
        # Output projection to vocabulary
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(
        self,
        token_ids: Int[Tensor, "batch_size sequence_length"],
    ) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        """
        Forward pass of the language model.
        
        Args:
            token_ids: Input token IDs
        
        Returns:
            Unnormalized logits for next token prediction
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        x = self.token_embeddings(token_ids)
        
        # Generate position IDs
        token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, token_positions)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits

