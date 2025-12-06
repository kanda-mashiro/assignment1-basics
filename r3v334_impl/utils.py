import math
from collections.abc import Iterable
import typing
import torch
import os
from jaxtyping import Float, Int

def r_annealing_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it/warmup_iters) * max_learning_rate

    if warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * \
            (1 + math.cos(
                (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
            )) * \
            (max_learning_rate - min_learning_rate)
    
    return min_learning_rate


def r_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
):

    s = 0
    for p in parameters:
        if p.grad is None:
            continue
        s += p.grad.pow(2).sum()
    
    s = s.sqrt()
    if s < max_l2_norm:
        return

    for p in parameters:
        if p.grad is None:
            continue
        p.grad *= max_l2_norm / (s + 1e-6)
    

def r_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration
        },
        out
    )

def r_load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    obj = torch.load(src)
    model.load_state_dict(
        obj["model"]
    )
    optimizer.load_state_dict(
        obj["optimizer"]
    )

    return obj["iteration"]


def r_cross_entropy(inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]) -> Float[torch.Tensor, ""]:
    maxv = inputs.max(dim=-1, keepdim=True).values
    logsum = ((inputs - maxv).exp().sum(-1, keepdim=True)).log()
    p = logsum + maxv - inputs[torch.arange(inputs.shape[0]), targets]

    return p.mean()