from typing import Any, Mapping, override
import einops
import torch.nn as nn
import torch
from einops import einsum

from r3v334_impl.rope import MetaRoPE
from r3v334_impl.softmax import MetaSoftmax

class MetaScaledDotProductAttetion(nn.Module):
    def __init__(self):
        super(MetaScaledDotProductAttetion, self).__init__()
        self.sftmax = MetaSoftmax()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask_b: torch.Tensor | None) -> torch.Tensor:
        # 初始化 mask 数组
        mask = torch.where(
            mask_b, 0.0, -torch.inf
        )

        preatt = einsum(
            Q, K,
            "... queries d_k, ... keys d_k -> ... queries keys"
        ) / torch.math.sqrt(K.shape[-1])
        preatt = preatt + mask
        result = self.sftmax.forward(preatt, -1) @ V

        return result


class MetaMultiHeadSelfAttention(nn.Module):
    def __init__(self, 
        d_model: int, num_heads: int,
        q_proj: torch.Tensor, k_proj: torch.Tensor, v_proj: torch.Tensor, o_proj: torch.Tensor):

        super(MetaMultiHeadSelfAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj

        self.scaled_dot_product_att = MetaScaledDotProductAttetion()


    @override
    def forward(self, x: torch.Tensor):
        q = einsum(
            x, self.q_proj,
            "... seq_len d_in, d_k d_in -> ... seq_len d_k"
        )
        k = einsum(
            x, self.k_proj,
            "... seq_len d_in, d_k d_in -> ... seq_len d_k"
        )
        v = einsum(
            x, self.v_proj,
            "... seq_len d_in, d_k d_in -> ... seq_len d_k"
        )

        # 拆分
        # num_heads 作为 batch_size
        # ... seq_len d_model -> ... seq_len num_heads d_k -> ... num_heads seq_len d_k

        # ... num_heads seq_len  d_k
        q = q.reshape(*q.shape[:-1], self.num_heads, -1).transpose(-3, -2)
        k = k.reshape(*k.shape[:-1], self.num_heads, -1).transpose(-3, -2)
        v = v.reshape(*v.shape[:-1], self.num_heads, -1).transpose(-3, -2)
        
        # attetion
        attn = self.scaled_dot_product_att.forward(
            q, k, v,
            ~torch.triu(
                torch.fill(torch.zeros(q.shape[-2], k.shape[-2], dtype=torch.bool), True), 1
            ),
        )

        # -> 转回去
        result = attn.transpose(-3, -2)
        result = result.reshape(*result.shape[:-2], self.d_model)

        # 输出
        result = einsum(
            result, self.o_proj,
            "... seq_len d_v, d_model d_v -> ... seq_len d_model"
        )

        return result



class MetaMultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(self, 
        d_model: int, num_heads: int, max_seq_len: int, theta: float,
        q_proj: torch.Tensor, k_proj: torch.Tensor, v_proj: torch.Tensor, o_proj: torch.Tensor):

        super(MetaMultiHeadSelfAttentionWithRope, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj

        self.scaled_dot_product_att = MetaScaledDotProductAttetion()

        self.rope = MetaRoPE(
            d_model // num_heads, theta, max_seq_len,
        )


    @override
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        q = einsum(
            x, self.q_proj,
            "... seq_len d_in, d_k d_in -> ... seq_len d_k"
        )
        k = einsum(
            x, self.k_proj,
            "... seq_len d_in, d_k d_in -> ... seq_len d_k"
        )
        v = einsum(
            x, self.v_proj,
            "... seq_len d_in, d_k d_in -> ... seq_len d_k"
        )

        # 拆分
        # num_heads 作为 batch_size
        # ... seq_len d_model -> ... seq_len num_heads d_k -> ... num_heads seq_len d_k

        # ... num_heads seq_len  d_k
        q = q.reshape(*q.shape[:-1], self.num_heads, -1).transpose(-3, -2)
        k = k.reshape(*k.shape[:-1], self.num_heads, -1).transpose(-3, -2)

        # ... num_heads seq_len d_k
        # rope
        q = self.rope.forward(q, token_positions)
        k = self.rope.forward(k, token_positions)

        v = v.reshape(*v.shape[:-1], self.num_heads, -1).transpose(-3, -2)
        
        # attetion
        attn = self.scaled_dot_product_att.forward(
            q, k, v,
            ~torch.triu(
                torch.fill(torch.zeros(q.shape[-2], k.shape[-2], dtype=torch.bool), True), 1
            ),
        )

        # -> 转回去
        result = attn.transpose(-3, -2)
        result = result.reshape(*result.shape[:-2], self.d_model)

        # 输出
        result = einsum(
            result, self.o_proj,
            "... seq_len d_v, d_model d_v -> ... seq_len d_model"
        )

        return result