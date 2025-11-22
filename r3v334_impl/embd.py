from typing import Any, Mapping
import einops
import torch.nn as nn
import torch
from einops import einsum

class MetaEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super(MetaEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]