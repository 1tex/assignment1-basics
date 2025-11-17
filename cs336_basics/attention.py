"""
Attention mechanisms for Transformer models.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool
from torch import Tensor

from .nn_utils import Linear


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Query tensor
        K: Key tensor
        V: Value tensor
        mask: Optional boolean mask (True = keep, False = mask out)
    
    Returns:
        Attention output
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # Where mask is False, set to large negative value
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, V)
    
    return output


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    RoPE applies rotations to query and key vectors based on their positions.
    """
    
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequency tensor
        # freq_i = 1 / (theta ^ (2i / d_k)) for i in [0, d_k/2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_k"],
        token_positions: Int[Tensor, "... sequence_length"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_k"]:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor (query or key)
            token_positions: Position indices for each token
        
        Returns:
            Rotated tensor
        """
        seq_len = x.shape[-2]
        
        # Generate positions if not provided
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        
        # Compute frequencies: position * inv_freq
        # Shape: [..., seq_len, d_k/2]
        freqs = torch.einsum('...i,j->...ij', token_positions.float(), self.inv_freq)
        
        # Compute cos and sin
        cos_emb = torch.cos(freqs)  # [..., seq_len, d_k/2]
        sin_emb = torch.sin(freqs)  # [..., seq_len, d_k/2]
        
        # Reshape x for rotation: [..., seq_len, d_k/2, 2]
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        
        # Apply rotation
        # For each pair (x0, x1), rotate by angle θ:
        # x_rot0 = x0 * cos(θ) - x1 * sin(θ)
        # x_rot1 = x0 * sin(θ) + x1 * cos(θ)
        x_rot = torch.stack([
            x_reshaped[..., 0] * cos_emb - x_reshaped[..., 1] * sin_emb,
            x_reshaped[..., 0] * sin_emb + x_reshaped[..., 1] * cos_emb,
        ], dim=-1)
        
        # Reshape back: [..., seq_len, d_k]
        x_rot = x_rot.reshape(*x.shape)
        
        return x_rot


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Implements batched multi-head attention with optional RoPE.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        
        # Projections for Q, K, V (batched for all heads)
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        
        # RoPE if needed
        if use_rope:
            self.rope = RoPE(self.d_k, max_seq_len, theta)
        else:
            self.rope = None
    
    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, "... sequence_length"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor
            token_positions: Optional position indices for RoPE
        
        Returns:
            Attention output
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # [..., seq_len, d_model]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head: [..., seq_len, num_heads, d_k]
        # Then transpose to: [..., num_heads, seq_len, d_k]
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        
        # Apply RoPE if enabled
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # Create causal mask
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        
        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        
        # Reshape back: [..., num_heads, seq_len, d_k] -> [..., seq_len, d_model]
        attn_output = attn_output.transpose(-3, -2).contiguous()
        attn_output = attn_output.view(*batch_shape, seq_len, self.d_model)
        
        # Final output projection
        output = self.output_proj(attn_output)
        
        return output

