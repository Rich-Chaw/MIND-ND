
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.nn import GraphNorm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch
from .gnn_utils import _get_init_features, _read_out



class MINDConv(nn.Module):
    def __init__(self, F, H):
        super().__init__()
        assert F % H == 0, f'num_features {F} not divisible by num_heads {H}'
        self.F, self.H = F, H
        self.W_src = nn.Linear(F, F)
        self.W_dst = nn.Linear(F, F)
        self.mlp_a_src = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(F, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, H)
        )
        self.mlp_a_dst = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(F, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, H)
        )

    def forward(self, h, edge_index):
        '''
        edge_index=(src,dst) message passing direction: src -> dst
            src [E] dst [E]
        h [N,F]

        return h_next [N,F]
        '''
        src, dst = edge_index  # message passing direction: src -> dst
        N, F, H = h.shape[0], self.F, self.H
        D, E = F//H, len(src)

        g_src = self.W_src(h) # -> (N, F)
        g_dst = self.W_dst(h) # -> (N, F)

        msg_src, msg_dst = g_src[src], g_dst[dst] # -> (E, F), (E, F)
        
        a_src = torch.sigmoid(self.mlp_a_src(msg_src+msg_dst)).unsqueeze(-1) # -> (E, H, 1)
        a_dst = torch.sigmoid(self.mlp_a_dst(g_dst)).unsqueeze(-1)           # -> (N, H, 1)
        
        # Get message from source nodes -> (N, H, D)
        h_next = a_dst * g_dst.view(N, H, D) \
                 + scatter_add(a_src * msg_src.view(E, H, D), dst, 0, dim_size=N)
        return h_next.view(N, -1)


class MIND(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_heads = num_heads
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features
        self.convs = nn.ModuleList([MINDConv(self.num_features, num_heads) for _ in range(num_mps)])
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)

    def forward(self, g: Batch):
        x_k = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        for k, conv in enumerate(self.convs):
            x_k = conv(x_k, g.edge_index)
            x_profile[:, k * self.num_features : (k + 1) * self.num_features] = x_k
            x_k = torch.relu(x_k)
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)