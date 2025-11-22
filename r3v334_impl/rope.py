import token
from typing import Any, Mapping, override
import einops
from jaxtyping import Float, Int
import torch.nn as nn
import torch
from einops import einsum

class MetaRoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len


        # 初始化旋转矩阵
        r = torch.zeros(max_seq_len, d_k, d_k)
        for i in range(max_seq_len):
            for k in range(self.d_k // 2):
                row_idx = 2 * k
                theta_ik = i / (torch.math.pow(self.theta, 2 * k / self.d_k))

                r[i][row_idx][row_idx] = torch.math.cos(theta_ik)
                r[i][row_idx][row_idx+1] = -torch.math.sin(theta_ik)
                r[i][row_idx+1][row_idx] = torch.math.sin(theta_ik)
                r[i][row_idx+1][row_idx+1] = torch.math.cos(theta_ik)
        
        self.register_buffer(
            "rotate_m", r,
            persistent=False
        )

    @override
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        r = self.get_buffer("rotate_m")
        # print(x.shape, token_positions.shape)
        # print(r[token_positions].shape)
        return einsum(
            x, r[token_positions],
            "... seq_len d_in, ... seq_len d_out d_in -> ... seq_len d_out"
        )