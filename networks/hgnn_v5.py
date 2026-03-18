"""
HGNN_V4: Based on HGNN_V3 and MIND.
- No initial residual (alpha) or alpha coefficient.
- Multi-head like MIND (F divisible by H, per-head dimension D = F//H).
- Attention inside message: attention-weighted lin_neigh(x_j) + attention-weighted lin_flow(x_j - x_i),
  with two MLPs like MIND (mlp_a_neigh / mlp_a_flow) giving separate per-head attention for each term.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm

from utils.graph_data import Batch
from .gnn_utils import _get_init_features, _read_out


class LeanHybridConvV5(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha=0.1, theta=0.5, layer=1):
        super(LeanHybridConvV5, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha  # Initial residual weight
        self.theta = theta  # Identity mapping decay
        self.layer = layer  # Current layer index for beta_l calculation

        # Attention projections (Additive Attention)
        self.lin_src = nn.Linear(out_channels, 1, bias=False)
        self.lin_dst = nn.Linear(out_channels, 1, bias=False)

        # Message & Transformation weights
        self.lin_neigh = nn.Linear(out_channels, out_channels, bias=False)
        self.lin_flow = nn.Linear(out_channels, out_channels, bias=False)
        self.lin_center = nn.Linear(out_channels, out_channels, bias=False)
        
        self.layer_norm = nn.LayerNorm(out_channels)
        self.beta = torch.log(torch.tensor(theta / layer + 1))

    def forward(self, x, edge_index):
        x_init = x
        # 1. Pre-compute Attention Projections (N x 1)
        phi_src = self.lin_src(x)
        phi_dst = self.lin_dst(x)

        # 2. Propagate Messages
        # We pass phi_src/dst to message() via the edge_index mapping
        out = self.propagate(edge_index, x=x, phi_src=phi_src, phi_dst=phi_dst)

        # 3. GCNII Initial Residual & Identity Mapping
        # h_tilde = (1-alpha) * sum(m_ji) + alpha * h_0
        out = (1 - self.alpha) * out + self.alpha * x_init
        
        # 4. h_l = (1-beta) * h_tilde + beta * W_center * h_tilde
        out = (1 - self.beta) * out + self.beta * self.lin_center(out)

        return self.layer_norm(F.relu(out))

    def message(self, x_i, x_j, phi_src_i, phi_dst_j):
        # 0.1.1 Additive Attention Coefficient
        alpha_ij = torch.sigmoid(phi_src_i + phi_dst_j)

        # 0.1.2 Difference-Based Flow Message
        # Using (x_j - x_i) for LGNN-style flow
        msg = self.lin_neigh(x_j) + self.lin_flow(x_j - x_i)
        
        return alpha_ij * msg

class HGNN_V5(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, alpha=0.1, theta=0.5, positional_encoding=None, handcrafted_features=False, **kwargs):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_heads = num_heads
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features
        self.layers = nn.ModuleList([
            LeanHybridConvV5(num_features, num_features, alpha=alpha, theta=theta, layer=l + 1)
            # LeanHybridConvV4(num_features, num_features, theta=theta, layer=l + 1)
            for l in range(num_mps)
        ])
        self.graph_norm = GraphNorm(num_features * num_mps, eps=1e-4)
        self.register_buffer("x_init", torch.ones(1, num_features))

    def forward(self, g: Batch):
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        for k, layer in enumerate(self.layers):
            x = layer(x, g.edge_index)
            x_profile[:, k * self.num_features : (k + 1) * self.num_features] = x
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)