from collections.abc import Callable
import math
import torch.nn as nn
import torch
from typing import cast, List, Optional, Tuple, Union


class RAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = {
            "alpha": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "epsilon": eps,
            "decay_lambda": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            epsilon = group["epsilon"]
            decay_lambda = group["decay_lambda"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)

                g = p.grad.data
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)

                alpha_t = alpha * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.data -= alpha_t * (m / (v.sqrt() + epsilon))
                p.data -= alpha * decay_lambda * p.data
                
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


if __name__ == "__main__":
    weights = nn.Parameter(5 * torch.randn(100, 100))
    opt = RAdamW([weights])
    
    for t in range(100000):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
