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


class _SparseExphormerLayer(nn.Module):
    def __init__(self, dim, num_heads, dim_edge=5, dropout=0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.e_proj = nn.Linear(dim_edge, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        e = self.e_proj(edge_attr).view(-1, self.num_heads, self.head_dim)

        # Exphormer-style attention: elementwise QK modulated by edge features.
        score = (q[dst] * k[src]) / math.sqrt(self.head_dim)
        score = score * e
        score = torch.exp(score.sum(dim=-1).clamp(min=-5.0, max=5.0))

        msg = v[src] * score.unsqueeze(-1)
        w_v = scatter_add(msg, dst, dim=0, dim_size=x.size(0))
        z = scatter_add(score.unsqueeze(-1), dst, dim=0, dim_size=x.size(0))
        out = (w_v / (z + 1e-6)).reshape(-1, self.dim)
        out = self.o_proj(out)
        out = self.dropout(out)

        x = self.norm1(x + out)
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class ExphormerEncoder(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        self.layers = nn.ModuleList(
            [_SparseExphormerLayer(self.num_features, num_heads, dim_edge=5) for _ in range(num_mps)]
        )
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)

    def _get_expander_inputs(self, g: Batch):
        edge_index = getattr(g, "expander_edge_index", None)
        edge_attr = getattr(g, "expander_edge_attr", None)
        if edge_index is None:
            edge_index = g.edge_index
        if edge_attr is None:
            src, dst = edge_index[0], edge_index[1]
            deg_dst = degree(dst, g.total_nodes, dtype=torch.float32).clamp(min=1.0)
            inv_sqrt_deg_src = deg_dst[src].pow(-0.5)
            inv_sqrt_deg_dst = deg_dst[dst].pow(-0.5)
            is_src_omni = (~g.non_omni_mask[src]).float()
            is_dst_omni = (~g.non_omni_mask[dst]).float()
            ones = torch.ones_like(inv_sqrt_deg_src)
            edge_attr = torch.stack(
                [ones, inv_sqrt_deg_src, inv_sqrt_deg_dst, is_src_omni, is_dst_omni], dim=-1
            )
        return edge_index, edge_attr

    def forward(self, g: Batch):
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        edge_index, edge_attr = self._get_expander_inputs(g)
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        for k, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
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
    model = ExphormerEncoder(num_features=16, num_heads=4, num_mps=3).to(device)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (7, 2 * 16 * 3), f"Unexpected output shape: {out.shape}"
    print("ExphormerEncoder smoke test passed:", out.shape)
