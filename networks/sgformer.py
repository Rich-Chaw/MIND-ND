import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch, Graph
from .gnn_utils import _get_init_features, _read_out


class _TransConvLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_in = x.unsqueeze(0)
        out, _ = self.attn(x_in, x_in, x_in, need_weights=False)
        out = out.squeeze(0)
        return self.norm(x + out)


class SGFormerEncoder(nn.Module):
    def __init__(
        self,
        num_features,
        num_heads,
        num_mps,
        positional_encoding=None,
        handcrafted_features=False,
        graph_weight=0.8,
    ):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features
        self.graph_weight = graph_weight

        self.trans_layers = nn.ModuleList([_TransConvLayer(self.num_features, num_heads) for _ in range(num_mps)])
        self.gcn_layers = nn.ModuleList([GCNConv(self.num_features, self.num_features) for _ in range(num_mps)])
        self.fuse_norms = nn.ModuleList([nn.LayerNorm(self.num_features) for _ in range(num_mps)])
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)

    def forward(self, g: Batch):
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        for k in range(self.num_mps):
            x_global = self.trans_layers[k](x)
            x_local = self.gcn_layers[k](x, g.edge_index)
            x = self.graph_weight * x_local + (1.0 - self.graph_weight) * x_global
            x = self.fuse_norms[k](x)
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
    model = SGFormerEncoder(num_features=16, num_heads=4, num_mps=3).to(device)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (7, 2 * 16 * 3), f"Unexpected output shape: {out.shape}"
    print("SGFormerEncoder smoke test passed:", out.shape)
