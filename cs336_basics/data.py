"""
Data loading utilities.
"""
import torch
import numpy as np
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of language modeling examples from the dataset.
    
    Args:
        dataset: 1D numpy array of integer token IDs
        batch_size: Number of examples to sample
        context_length: Length of each sequence
        device: PyTorch device to place tensors on
    
    Returns:
        Tuple of (inputs, labels) where:
        - inputs: Token IDs of shape (batch_size, context_length)
        - labels: Next token IDs of shape (batch_size, context_length)
          labels[i, j] = inputs[i, j+1]
    """
    # Total number of tokens in dataset
    n = len(dataset)
    
    # Valid starting positions: we need context_length + 1 tokens
    # (context_length for input + 1 for label)
    max_start_idx = n - context_length
    
    # Randomly sample starting indices
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Gather sequences
    inputs = np.stack([
        dataset[idx:idx + context_length]
        for idx in start_indices
    ])
    
    labels = np.stack([
        dataset[idx + 1:idx + context_length + 1]
        for idx in start_indices
    ])
    
    # Convert to PyTorch tensors
    inputs_tensor = torch.from_numpy(inputs).long().to(device)
    labels_tensor = torch.from_numpy(labels).long().to(device)
    
    return inputs_tensor, labels_tensor

