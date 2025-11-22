from typing import Any, Mapping
import einops
import torch.nn as nn
import torch
from einops import einsum
import einx

from r3v334_impl.silu import MetaSilu

class MetaSwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_layer = MetaSilu()
        result = silu_layer.forward(x @ self.w1.T) * (x @ self.w3.T)
        return result @ self.w2.T
        


