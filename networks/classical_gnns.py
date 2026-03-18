
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphNorm, MessagePassing
from torch_geometric.nn.inits import glorot, reset, zeros
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops, softmax

from utils.graph_data import Batch
from .gnn_utils import _get_init_features, _read_out

class GCN(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        self.convs = nn.ModuleList()
        for _ in range(num_mps):
            # GCNConv doesn't typically use heads, so we ignore num_heads here
            self.convs.append(GCNConv(num_features, num_features))
            
        self.graph_norm = GraphNorm(num_features * num_mps, eps=1e-4)

    def forward(self, g):
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        
        for k, conv in enumerate(self.convs):
            x = conv(x, g.edge_index)
            x_profile[:, k*self.num_features : (k+1)*self.num_features] = x
            x = F.relu(x)
            # L2 normalize after each layer (same as FINDER) for stable RL training
            x = F.normalize(x, p=2, dim=-1)
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)

class GraphSAGE(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        self.convs = nn.ModuleList()
        for _ in range(num_mps):
            self.convs.append(SAGEConv(num_features, num_features, aggr='add'))
            
        self.graph_norm = GraphNorm(num_features * num_mps, eps=1e-4)

    def forward(self, g):
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        for k, conv in enumerate(self.convs):
            x = conv(x, g.edge_index)
            x_profile[:, k*self.num_features : (k+1)*self.num_features] = x
            x = F.relu(x)
            # L2 normalize after each layer (same as FINDER GraphDQN_modules) for stable RL training
            x = F.normalize(x, p=2, dim=-1)
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)

class GAT(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        assert self.num_features % num_heads == 0, f'num_features {self.num_features} not divisible by num_heads {num_heads}'
        head_dim = self.num_features // num_heads
        
        self.convs = nn.ModuleList()
        for _ in range(num_mps):
            self.convs.append(GATConv(self.num_features, head_dim, heads=num_heads, concat=True))
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)

    def forward(self, g):
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        for k, conv in enumerate(self.convs):
            x = conv(x, g.edge_index)
            x_profile[:, k*self.num_features : (k+1)*self.num_features] = x
            x = F.relu(x)
            # L2 normalize after each layer (same as FINDER) for stable RL training
            x = F.normalize(x, p=2, dim=-1)
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)