import numpy as np
import torch


def get_batch(
    dataset: np.array, batch_size: int, context_length: int, device: str
) -> tuple[torch.tensor, torch.tensor]:
    """
    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    starts = np.random.randint(0, len(dataset) - context_length, batch_size)

    tokens = torch.from_numpy(
        np.stack([dataset[i : i + context_length + 1] for i in starts])
    )

    x = tokens[:, :-1].to(device=device, dtype=torch.long)
    y = tokens[:, 1:].to(device=device, dtype=torch.long)

    return x, y
