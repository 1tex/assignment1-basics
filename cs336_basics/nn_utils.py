"""
Neural network utility functions and layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


class Linear(nn.Module):
    """Linear transformation layer."""
    
    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        if bias:
            self.bias = nn.Parameter(torch.empty(d_out))
        else:
            self.bias = None
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Apply linear transformation."""
        output = F.linear(x, self.weight, self.bias)
        return output


class Embedding(nn.Module):
    """Embedding layer for token IDs."""
    
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=1)
    
    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        """Get embeddings for token IDs."""
        return F.embedding(token_ids, self.weight)


def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    """
    Compute softmax with numerical stability.
    
    Args:
        x: Input tensor
        dim: Dimension to apply softmax
    
    Returns:
        Softmax normalized tensor
    """
    # Subtract max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum


def silu(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """
    SiLU (Swish) activation function.
    
    SiLU(x) = x * sigmoid(x)
    """
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm(x) = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Apply RMSNorm."""
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_normed = x / rms
        return self.weight * x_normed


def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """
    Compute cross-entropy loss.
    
    Args:
        inputs: Unnormalized logits of shape (batch_size, vocab_size)
        targets: Target class indices of shape (batch_size,)
    
    Returns:
        Average cross-entropy loss (scalar)
    """
    # Compute log softmax with numerical stability
    log_probs = F.log_softmax(inputs, dim=-1)
    
    # Gather log probabilities for target classes
    batch_size = inputs.shape[0]
    target_log_probs = log_probs[torch.arange(batch_size), targets]
    
    # Return negative mean
    return -target_log_probs.mean()


def clip_grad_norm_(parameters, max_norm: float) -> None:
    """
    Clip gradients by global L2 norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum L2 norm
    """
    # Filter parameters that have gradients
    params_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(params_with_grad) == 0:
        return
    
    # Compute total norm
    total_norm = torch.sqrt(
        sum(torch.sum(p.grad ** 2) for p in params_with_grad)
    )
    
    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    
    # Clip gradients if needed
    if clip_coef < 1:
        for p in params_with_grad:
            p.grad.mul_(clip_coef)

