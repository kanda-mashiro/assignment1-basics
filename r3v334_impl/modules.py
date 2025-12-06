import math
from jaxtyping import Bool, Float, Int
import torch
from torch import Tensor, nn
from typing import override
import einops
from torch.nn.modules import normalization


class REmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @override
    def forward(self, token_ids: Int[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ... d_model"]:
        return self.weight[token_ids]


class RLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @override
    def forward(self, x: Float[torch.Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einops.einsum(
            x, self.weight,
            "... in_d, out_d in_d -> ... out_d",
        )

class RRMSNormal(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
        in_type = x.dtype

        x = x.to(torch.float32)

        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        return (self.weight * x).to(in_type)

class RRoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()


        # 初始化旋转矩阵
        r = torch.zeros(max_seq_len, d_k, d_k, device=device, dtype=dtype)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                row_idx = 2 * k
                theta_ik = i / (torch.math.pow(theta, 2 * k / d_k))

                r[i][row_idx][row_idx] = torch.math.cos(theta_ik)
                r[i][row_idx][row_idx+1] = -torch.math.sin(theta_ik)
                r[i][row_idx+1][row_idx] = torch.math.sin(theta_ik)
                r[i][row_idx+1][row_idx+1] = torch.math.cos(theta_ik)
        
        self.register_buffer(
            "rotate_m", r,
            persistent=False
        )

    @override
    def forward(self, x: Float[torch.Tensor, " ... seq_len d_k"], token_positions: Int[torch.Tensor, " ... seq_len"]) -> Float[torch.Tensor, " ... seq_len d_k"]:
        r = self.get_buffer("rotate_m")
        return einops.einsum(
            x, r[token_positions],
            "... seq_len d_in, ... seq_len d_out d_in -> ... seq_len d_out"
        )


class RSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
        return x / (1 + torch.exp(-x))



class RSwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()

        self.silu = RSiLU()

        self.w1 = RLinear(d_model, d_ff, device, dtype)
        self.w2 = RLinear(d_ff, d_model, device, dtype)
        self.w3 = RLinear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
        gate = self.silu(self.w1(x))

        value = self.w3(x)

        combined = gate * value

        return self.w2(combined)


class RSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_features: Float[torch.Tensor, " ..."], dim: int = -1) -> Float[torch.Tensor, " ..."]:
        max_v = in_features.max(dim=dim, keepdim=True)
        safe_in = in_features.sub(max_v.values)
        s = safe_in.exp().sum(dim=dim, keepdim=True)

        return safe_in.exp() / s


class RScaledDotProductAttetion(nn.Module):
    def __init__(self):
        super().__init__()

        self.sftmax = RSoftmax()

    @override
    def forward(self, 
                Q: Float[torch.Tensor, " ... queries d_k"],
                K: Float[torch.Tensor, " ... keys d_k"],
                V: Float[torch.Tensor, " ... values d_v"],
                mask: Bool[torch.Tensor, " ... queries keys"] | None
    ) -> Float[torch.Tensor, " ... queries d_v"]:
        # 初始化 mask 数组
        mask = torch.where(
            mask, 0.0, -torch.inf
        )

        preatt = einops.einsum(
            Q, K,
            "... queries d_k, ... keys d_k -> ... queries keys"
        ) / torch.math.sqrt(K.shape[-1])
        preatt = preatt + mask
        result = self.sftmax(preatt, -1) @ V

        return result


class RMultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                d_model: int,
                num_heads: int,
                max_seq_len: int = 0,
                theta: float = 0.0,
                use_rope: bool = False,
                device=None, dtype=None):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.scaled_dot_product_att = RScaledDotProductAttetion()

        self.q_proj = RLinear(d_model, d_model, device, dtype)
        self.k_proj = RLinear(d_model, d_model, device, dtype)
        self.v_proj = RLinear(d_model, d_model, device, dtype)
        self.output_proj = RLinear(d_model, d_model, device, dtype)

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RRoPE(
                d_k=d_model // num_heads,
                theta=theta,
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype
            )

    @override
    def forward(self, 
                x: Float[torch.Tensor, " ... seq_len d_in"],
                token_positions: Int[torch.Tensor, " ... seq_len"] | None = None
    ) -> Float[torch.Tensor, " ... seq_len d_out"]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 拆分
        # num_heads 作为 batch_size
        # ... seq_len d_model -> ... seq_len num_heads d_k -> ... num_heads seq_len d_k
        # ... num_heads seq_len d_k
        q = q.reshape(*q.shape[:-1], self.num_heads, -1).transpose(-3, -2)
        k = k.reshape(*k.shape[:-1], self.num_heads, -1).transpose(-3, -2)
        v = v.reshape(*v.shape[:-1], self.num_heads, -1).transpose(-3, -2)

        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        # attetion
        current_seq_len = q.shape[-2]
        mask = torch.tril(
                    torch.fill(
                        torch.empty(current_seq_len, current_seq_len, device=q.device, dtype=torch.bool),
                        True,
                    )
                )
        attn = self.scaled_dot_product_att(q, k, v, mask)

        # -> 转回去
        result = attn.transpose(-3, -2)
        # batch_size, seq_len num_heads d_k
        result = result.reshape(*result.shape[:-2], self.d_model)

        # 输出
        return self.output_proj(result)


class RTransformerBlock(nn.Module):
    def __init__(self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: any = None,
    ):

        super().__init__()

        self.ln1 = RRMSNormal(d_model, device=device)
        self.attn = RMultiHeadSelfAttention(d_model, num_heads, 
                                                max_seq_len=max_seq_len,
                                                theta=theta,
                                                use_rope=True,
                                                device=device)
        self.ln2 = RRMSNormal(d_model, device=device)
        self.ffn = RSwiGLU(d_model, d_ff, device=device)


    def forward(self, x: Float[torch.Tensor, " batch seq_len d_model"], pos_ids: Int[torch.Tensor, " batch seq_len"]) -> Float[torch.Tensor, " batch seq_len d_model"]:
        y = x + self.attn(
            self.ln1(x),
            pos_ids,
        )
        z = y + self.ffn(self.ln2(x))

        return z


class RTransformerLM(nn.Module):
    def __init__(self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: any = None,
    ):
        super().__init__()

        self.token_embeddings = REmbedding(vocab_size, d_model, device)
        self.layers = nn.ModuleList([
            RTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RRMSNormal(d_model, device=device)
        self.lm_head = RLinear(d_model, vocab_size, device=device)

    def forward(self, in_features: Int[torch.Tensor, " batch_size seq_len"]) -> Float[torch.Tensor, " batch_size seq_len vocab_size"]:
        # embedding
        x = self.token_embeddings(in_features)

        # 多个 transformer block
        for layer in self.layers:
            x = layer(x, torch.arange(0, x.shape[-2], device=x.device))
        
        # 归一化
        x = self.ln_final(x)

        # lm_head -> logits
        x = self.lm_head(x)

        return x