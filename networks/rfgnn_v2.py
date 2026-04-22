import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch


class LeanHybridConv(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha=0.1, theta=0.5, layer=1):
        super(LeanHybridConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.theta = theta
        self.layer = layer

        self.lin_src = nn.Linear(out_channels, 1, bias=False)
        self.lin_dst = nn.Linear(out_channels, 1, bias=False)

        self.lin_neigh = nn.Linear(out_channels, out_channels, bias=False)
        self.lin_flow = nn.Linear(out_channels, out_channels, bias=False)
        self.lin_center = nn.Linear(out_channels, out_channels, bias=False)

        self.layer_norm = nn.LayerNorm(out_channels)
        self.beta = torch.log(torch.tensor(theta / layer + 1))

    def forward(self, x, x_0, edge_index):
        phi_src = self.lin_src(x)
        phi_dst = self.lin_dst(x)

        out = self.propagate(edge_index, x=x, phi_src=phi_src, phi_dst=phi_dst)
        out = (1 - self.alpha) * out + self.alpha * x_0
        out = (1 - self.beta) * out + self.beta * self.lin_center(out)
        out = self.layer_norm(F.relu(out))

        # Keep unit-length embeddings per node for better training stability.
        out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_i, x_j, phi_src_i, phi_dst_j):
        alpha_ij = torch.sigmoid(phi_src_i + phi_dst_j)
        msg = self.lin_neigh(x_j) + self.lin_flow(x_j - x_i)
        return alpha_ij * msg


class ResiflowGNN_V2(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, alpha=0.1, theta=0.5, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        self.layers = nn.ModuleList([
            LeanHybridConv(self.num_features, self.num_features, alpha, theta, layer=l + 1)
            for l in range(num_mps)
        ])
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)
        self.register_buffer("x_init", torch.ones(1, num_features))

    def forward(self, g: Batch):
        from .gnn_utils import _get_init_features
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_0 = x.clone()
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)

        for k, layer in enumerate(self.layers):
            x = layer(x, x_0, g.edge_index)
            x_profile[:, k * self.num_features : (k + 1) * self.num_features] = x

        x_profile = self.graph_norm(x_profile, g.batch)
        x_profile = torch.cat([
            x_profile[g.non_omni_mask],
            x_profile[g.omni_ids[g.batch_non_omni]]
        ], dim=1)
        return x_profile
