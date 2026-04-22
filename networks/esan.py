import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GraphNorm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch, Graph
from .gnn_utils import _get_init_features, _read_out


class ESANEncoder(nn.Module):
    """
    A single-file ESAN-style encoder:
    - stacked GIN blocks (subgraph-aware spirit via strong local aggregation)
    - JK-concat over layers
    - graph-level context injected by _read_out
    """

    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_mps):
            mlp = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.ReLU(),
                nn.Linear(self.num_features * 2, self.num_features),
            )
            self.convs.append(GINConv(nn=mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(self.num_features))

        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)

    def forward(self, g: Batch):
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        for k, conv in enumerate(self.convs):
            x = conv(x, g.edge_index)
            x = self.bns[k](x)
            x = F.relu(x)
            x = F.normalize(x, p=2, dim=-1)
            x_profile[:, k * self.num_features : (k + 1) * self.num_features] = x
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)


if __name__ == "__main__":
    device = torch.device("cpu")
    edge_index1 = np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
    edge_index2 = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)
    g1, g2 = Graph(edge_index1, 3), Graph(edge_index2, 4)
    batch = Batch(device, [g1, g2])
    model = ESANEncoder(num_features=16, num_heads=4, num_mps=3).to(device)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (7, 2 * 16 * 3), f"Unexpected output shape: {out.shape}"
    print("ESANEncoder smoke test passed:", out.shape)
