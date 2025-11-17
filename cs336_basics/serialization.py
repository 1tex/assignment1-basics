"""
Model and optimizer serialization utilities.
"""
import os
import torch
from typing import IO, BinaryIO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """
    Save model and optimizer state to a checkpoint file.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        iteration: Current training iteration
        out: Path or file-like object to save to
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load model and optimizer state from a checkpoint file.
    
    Args:
        src: Path or file-like object to load from
        model: Model to restore state to
        optimizer: Optimizer to restore state to
    
    Returns:
        The iteration number from the checkpoint
    """
    checkpoint = torch.load(src, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']

