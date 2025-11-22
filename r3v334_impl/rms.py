from typing import Any, Mapping
import einops
import torch.nn as nn
import torch
from einops import einsum


class MetaRms(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weights = nn.Parameter(torch.empty(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 RMS
        in_type = x.dtype
        x = x.to(torch.float32)
    

        rms = (((x ** 2 / self.d_model).sum(-1) + self.eps) ** 0.5).unsqueeze(-1)

        result = (x / rms) * self.weights

        return result.to(in_type)


        

