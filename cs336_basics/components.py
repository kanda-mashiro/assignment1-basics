from einops import einsum, rearrange
from typing import override, Mapping, Any

import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:

        super().__init__()

        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features + out_features))
        w = torch.nn.init.trunc_normal_(w, 0, std, -3*std, 3*std)

        self.p_w = nn.Parameter(w)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.p_w, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:

        super().__init__()

        emb = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        emb = torch.nn.init.trunc_normal_(emb, 0, 1, -3, 3)

        self.p_emb = nn.Parameter(emb)

    @override
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.p_emb[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None) -> None:

        super().__init__()

        self.d_model = d_model
        self.eps = eps
        w = torch.ones((d_model), device=device, dtype=dtype)
        self.p_w = nn.Parameter(w)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model

        x_dtype = x.dtype
        x = x.to(torch.float32)

        # rms = torch.sqrt(torch.sum(x**2, dim=[-1]) / self.d_model + self.eps)
        rms = ((x**2).sum(-1) / self.d_model + self.eps).sqrt()
        out = x / rms.unsqueeze(-1) * self.p_w

        return out.to(x_dtype)


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x / (1 + torch.exp(-x))
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:

        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        w1 = torch.randn([self.d_ff, self.d_model])
        w2 = torch.randn([self.d_model, self.d_ff])
        w3 = torch.randn([self.d_ff, self.d_model])
        self.p_w1 = nn.Parameter(w1)
        self.p_w2 = nn.Parameter(w2)
        self.p_w3 = nn.Parameter(w3)


    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up1 = einsum(x, self.p_w1, "... d_model, d_ff d_model -> ... d_ff")
        up1 = up1 * torch.sigmoid(up1)
        up3 = einsum(x, self.p_w3, "... d_model, d_ff d_model -> ... d_ff")
        up = up1 * up3
        return einsum(up, self.p_w2, "... d_ff, d_model d_ff -> ... d_model")


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        r = torch.zeros([max_seq_len, d_k, d_k])
        for idx in range(max_seq_len):
            for k in range(self.d_k // 2):
                theta_i_k = idx / self.theta**((2*k)/self.d_k)
                row, col = k*2, k*2
                r[idx][row][col] = math.cos(theta_i_k)
                r[idx][row+1][col] = math.sin(theta_i_k)
                r[idx][row][col+1] = -math.sin(theta_i_k)
                r[idx][row+1][col+1] = math.cos(theta_i_k)
        self.r = r

    @override
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: [... sequence_length d_k]
        token_positions: [... sequence_length]
        """
        return einsum(x, self.r[token_positions], "... seq_len col, ... seq_len row col -> ... seq_len row")


# 3.5.4
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x -= torch.max(x, dim=dim).values.unsqueeze(-1)
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim).unsqueeze(-1)


# 3.5.4
class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        """
        q: [... queries d_k]
        k: [... keys d_k]
        v: [... values d_v]
        mask: [... queries keys]
        """
        qk = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")
        qk = torch.masked_fill(qk, ~mask, float('-inf'))
        qk = softmax(qk / math.sqrt(k.shape[-1]), -1)
        return einsum(qk, v, "... queries keys, ... keys d_v -> ... queries d_v")

# 3.5.5
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int, device: None = None, dtype: None = None) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_head = num_head
        self.d_head = self.d_model // self.num_head

        q = torch.randn([d_model, d_model])
        k = torch.randn([d_model, d_model])
        v = torch.randn([d_model, d_model])
        o = torch.randn([d_model, d_model])

        self.p_q = nn.Parameter(q)
        self.p_k = nn.Parameter(k)
        self.p_v = nn.Parameter(v)
        self.p_o = nn.Parameter(o)

    def forward_v1(self, x: torch.Tensor) -> torch.Tensor:
        q = self.p_q.view(self.num_head, self.d_head, self.d_model)
        k = self.p_k.view(self.num_head, self.d_head, self.d_model)
        v = self.p_v.view(self.num_head, self.d_head, self.d_model)

        q = einsum(x, q, "... seq_len d_model, num_head d_head d_model -> ... num_head seq_len d_head")
        k = einsum(x, k, "... seq_len d_model, num_head d_head d_model -> ... num_head seq_len d_head")
        v = einsum(x, v, "... seq_len d_model, num_head d_head d_model -> ... num_head seq_len d_head")

        qk = einsum(q, k, "... num_head seq_q_len d_head, ... num_head seq_kv_len d_head -> ... num_head seq_q_len seq_kv_len")

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones([seq_len, seq_len], dtype=torch.bool))
        qk = qk.masked_fill(~mask, float("-inf"))
        qk = softmax(qk / math.sqrt(self.d_head), -1)

        qkv = einsum(qk, v, "... seq_q_len seq_kv_len, ... seq_kv_len d_head -> ... seq_q_len d_head")
        qkv = rearrange(qkv, "... num_head seq_len d_head -> ... seq_len (num_head d_head)")
        return einsum(qkv, self.p_o, "... seq_len d_model1, d_model2 d_model1 -> ... seq_len d_model2")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.p_q.view(self.num_head, self.d_head, self.d_model)
        k = self.p_k.view(self.num_head, self.d_head, self.d_model)
        v = self.p_v.view(self.num_head, self.d_head, self.d_model)

        q = einsum(x, q, "... seq_len d_model, num_head d_head d_model -> ... num_head seq_len d_head")
        k = einsum(x, k, "... seq_len d_model, num_head d_head d_model -> ... num_head seq_len d_head")
        v = einsum(x, v, "... seq_len d_model, num_head d_head d_model -> ... num_head seq_len d_head")

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones([seq_len, seq_len], dtype=torch.bool))
        qkv = ScaledDotProductAttention().forward(q, k, v, mask)
        qkv = rearrange(qkv, "... num_head seq_len d_head -> ... seq_len (num_head d_head)")
        return einsum(qkv, self.p_o, "... seq_len d_model1, d_model2 d_model1 -> ... seq_len d_model2")


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_head: int, max_seq_len: int, theta: float, device: None = None, dtype: None = None) -> None:
        super().__init__()

        self.theta = theta
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_head = num_head
        self.d_head = self.d_model // self.num_head

        q = torch.randn([d_model, d_model])
        k = torch.randn([d_model, d_model])
        v = torch.randn([d_model, d_model])
        o = torch.randn([d_model, d_model])

        self.p_q = nn.Parameter(q)
        self.p_k = nn.Parameter(k)
        self.p_v = nn.Parameter(v)
        self.p_o = nn.Parameter(o)

    @override
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        q = self.p_q.view(self.num_head, self.d_head, self.d_model)
        k = self.p_k.view(self.num_head, self.d_head, self.d_model)
        v = self.p_v.view(self.num_head, self.d_head, self.d_model)

        q = einsum(x, q, "... seq_len d_model, num_head d_head d_model -> ... num_head seq_len d_head")
        q = RoPE(self.theta, self.d_head, self.max_seq_len).forward(q, token_positions)
        k = einsum(x, k, "... seq_len d_model, num_head d_head d_model -> ... num_head seq_len d_head")
        k = RoPE(self.theta, self.d_head, self.max_seq_len).forward(k, token_positions)
        v = einsum(x, v, "... seq_len d_model, num_head d_head d_model -> ... num_head seq_len d_head")
        qk = einsum(q, k, "... num_head seq_q_len d_head, ... num_head seq_kv_len d_head -> ... num_head seq_q_len seq_kv_len")

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones([seq_len, seq_len], dtype=torch.bool))
        qk = qk.masked_fill(~mask, float("-inf"))
        qk = softmax(qk / math.sqrt(self.d_head), -1)

        qkv = einsum(qk, v, "... seq_q_len seq_kv_len, ... seq_kv_len d_head -> ... seq_q_len d_head")
        qkv = rearrange(qkv, "... num_head seq_len d_head -> ... seq_len (num_head d_head)")
        return einsum(qkv, self.p_o, "... seq_len d_model1, d_model2 d_model1 -> ... seq_len d_model2")


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_head: int, d_ff: int, max_seq_len: int, theta: float, device: None = None, dtype: None = None) -> None:
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_head, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.attn.forward(self.ln1.forward(x), torch.arange(0, x.shape[-2], 1))
        x += self.ffn.forward(self.ln2.forward(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()

        self.token_emb = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb.forward(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x