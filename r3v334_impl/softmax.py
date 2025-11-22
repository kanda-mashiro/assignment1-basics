from typing import Any, Mapping
import einops
from jaxtyping import Float, Int
import torch.nn as nn
import torch
from einops import einsum

class MetaSoftmax(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()

    def forward(self, in_features: torch.Tensor, dim: int = -1) -> torch.Tensor:
        max_v = in_features.max(dim=dim, keepdim=True)
        safe_in = in_features.sub(max_v.values)
        s = safe_in.exp().sum(dim=dim, keepdim=True)
        
        return safe_in.exp() / s
        