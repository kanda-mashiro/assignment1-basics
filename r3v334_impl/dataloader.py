import numpy.typing as npt
import torch
import numpy
import random

def r_load_data(
    dataset: npt.NDArray, batch_size: int, context_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    batches_start = [
        random.randint(0, dataset.size - context_len - 1)
        for _ in range(batch_size)
    ]
    data = [
        torch.from_numpy(dataset[start:start+context_len]).to(device=device)
        for start in batches_start
    ]
    labels = [
        torch.from_numpy(dataset[(start+1):(start+1+context_len)]).to(device=device)
        for start in batches_start
    ]

    return (torch.stack(data), torch.stack(labels))