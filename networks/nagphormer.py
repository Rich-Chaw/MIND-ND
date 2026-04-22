import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import degree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch, Graph
from .gnn_utils import _get_init_features, _read_out


class _TokenSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x):
        n, t, _ = x.shape
        q = self.q(x).view(n, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(n, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(n, t, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(n, t, self.dim)
        return self.o(out)


class _NAGBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = _TokenSelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class NAGphormerEncoder(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        self.blocks = nn.ModuleList([_NAGBlock(self.num_features, num_heads) for _ in range(num_mps)])
        self.token_score = nn.Linear(self.num_features, 1)
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)

    def _normalized_aggregate(self, x, edge_index):
        src, dst = edge_index
        deg = degree(dst, x.size(0), dtype=x.dtype).clamp(min=1.0)
        msg = x[src] / deg[dst].unsqueeze(-1)
        return scatter_add(msg, dst, dim=0, dim_size=x.size(0))

    def forward(self, g: Batch):
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)

        for k, block in enumerate(self.blocks):
            x_hop = self._normalized_aggregate(x, g.edge_index)
            tokens = torch.stack([x, x_hop], dim=1)
            tokens = block(tokens)
            weights = torch.softmax(self.token_score(tokens).squeeze(-1), dim=1)
            x = (tokens * weights.unsqueeze(-1)).sum(dim=1)
            x = F.normalize(F.relu(x), p=2, dim=-1)
            x_profile[:, k * self.num_features : (k + 1) * self.num_features] = x

        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)


if __name__ == "__main__":
    device = torch.device("cpu")
    edge_index1 = np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
    edge_index2 = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)
    g1, g2 = Graph(edge_index1, 3), Graph(edge_index2, 4)
    batch = Batch(device, [g1, g2])
    model = NAGphormerEncoder(num_features=16, num_heads=4, num_mps=3).to(device)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (7, 2 * 16 * 3), f"Unexpected output shape: {out.shape}"
    print("NAGphormerEncoder smoke test passed:", out.shape)
