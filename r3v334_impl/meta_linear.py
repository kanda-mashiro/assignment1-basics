from typing import Any, Mapping
import einops
import torch.nn as nn
import torch
from einops import einsum


class MetaLinear(nn.Module):    

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_paramters()

    def reset_paramters(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(
            x, self.weights,
            "x y in_d, out_d in_d -> x y out_d",
        )