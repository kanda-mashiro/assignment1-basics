from turtle import forward
from typing import Any, Mapping, override
import einops
import torch.nn as nn
import torch
from einops import einsum
import einx
from jaxtyping import Bool, Float, Int

from r3v334_impl.attention import MetaMultiHeadSelfAttentionWithRope
from r3v334_impl.embd import MetaEmbedding
from r3v334_impl.meta_linear import MetaLinear
from r3v334_impl.rms import MetaRms
from r3v334_impl.silu import MetaSilu
from r3v334_impl.softmax import MetaSoftmax
from r3v334_impl.swiglu import MetaSwiGLU

class MetaTransformerBlock(nn.Module):
    def __init__(self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, torch.Tensor]):

        super().__init__()

        rms1 = MetaRms(d_model)
        rms1.load_state_dict({
            "weights": weights["ln1.weight"]
        })
        self.rms1 = rms1

        attWithRope = MetaMultiHeadSelfAttentionWithRope(
            d_model, num_heads, max_seq_len, theta,
            weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], weights["attn.v_proj.weight"],
            weights["attn.output_proj.weight"]
        )
        self.attWithRope = attWithRope

        rms2 = MetaRms(d_model)
        rms2.load_state_dict({
            "weights": weights["ln2.weight"]
        })
        self.rms2 = rms2

        swiglu = MetaSwiGLU(d_model, d_ff)
        swiglu.load_state_dict({
            "w1": weights["ffn.w1.weight"],
            "w2": weights["ffn.w2.weight"],
            "w3": weights["ffn.w3.weight"],
        })
        self.swiglu = swiglu


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.attWithRope.forward(
            self.rms1.forward(x),
            torch.arange(0, x.shape[-2], 1),
        )

        x += self.swiglu.forward(self.rms2(x))

        return x

        
class MetaTransformerLM(nn.Module):
    def __init__(self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, torch.Tensor],
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        ebd = MetaEmbedding(
            vocab_size, d_model
        )
        ebd.load_state_dict({
            "weights": weights["token_embeddings.weight"]
        })
        self.ebd = ebd

        layers = []
        for num_layer in range(num_layers):
            keys = [
                "attn.q_proj.weight",
                "attn.k_proj.weight",
                "attn.v_proj.weight",
                "attn.output_proj.weight",
                "ln1.weight",
                "ffn.w1.weight",
                "ffn.w2.weight",
                "ffn.w3.weight",
                "ln2.weight",
            ]
            block_weights = {}
            for key in keys:
                block_weights[key] = weights[f"layers.{num_layer}.{key}"]
            # print(block_weights)
            block = MetaTransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, block_weights)
            layers.append(block)
        
        final_norm = MetaRms(d_model)
        final_norm.load_state_dict({
            "weights": weights["ln_final.weight"]
        })
        self.final_norm = final_norm

        lm_head = MetaLinear(
            d_model, vocab_size
        )
        lm_head.load_state_dict({
            "weights": weights["lm_head.weight"]
        })

        self.lm_head = lm_head
        

        self.lm = nn.Sequential(
            self.ebd,
            *layers,
            self.final_norm,
            self.lm_head
        )

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self.lm.forward(in_features)