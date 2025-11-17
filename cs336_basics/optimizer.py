"""
Optimizers and learning rate schedulers.
"""
import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay).
    
    Implements the AdamW algorithm from "Decoupled Weight Decay Regularization"
    (Loshchilov & Hutter, 2019).
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Bias-corrected estimates
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # Update parameters with AdamW
                # theta_t = theta_{t-1} - lr * (m_t / (sqrt(v_t) + eps) + lambda * theta_{t-1})
                p.data.add_(corrected_exp_avg / (corrected_exp_avg_sq.sqrt() + eps), alpha=-lr)
                
                # Apply weight decay (decoupled)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        
        return loss


def get_cosine_schedule_with_warmup(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Get learning rate for a given iteration using cosine schedule with linear warmup.
    
    Args:
        it: Current iteration
        max_learning_rate: Maximum learning rate (alpha_max)
        min_learning_rate: Minimum learning rate (alpha_min)
        warmup_iters: Number of warmup iterations
        cosine_cycle_iters: Number of cosine annealing iterations
    
    Returns:
        Learning rate for the current iteration
    """
    # Linear warmup
    if it < warmup_iters:
        return (max_learning_rate / warmup_iters) * it
    
    # Cosine annealing
    elif it < cosine_cycle_iters:
        # Progress through the cosine cycle (0 to 1)
        # cosine_cycle_iters is the total iterations including warmup
        # so the cosine portion runs from warmup_iters to cosine_cycle_iters
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        # Cosine schedule: starts at max, ends at min
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    
    # After cosine cycle, stay at minimum
    else:
        return min_learning_rate

