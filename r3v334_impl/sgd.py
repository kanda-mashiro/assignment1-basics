from collections.abc import Callable
import math
import torch.nn as nn
import torch


class RSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError("learning rate must greater than 0")
        
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


if __name__ == "__main__":
    weights = nn.Parameter(5 * torch.randn(100, 100))
    opt = RSGD([weights], lr=10)
    
    for t in range(1000000):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
