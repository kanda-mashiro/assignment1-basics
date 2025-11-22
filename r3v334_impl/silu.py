from typing import Any, Mapping
import einops
import torch.nn as nn
import torch
from einops import einsum

class MetaSilu(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))
        