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

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        self.in_features = 2 * in_features if variant else in_features
        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False, **kwargs):
        super().__init__()
        nlayers = num_mps
        self.num_features = 5 if handcrafted_features else num_features
        self.nlayers = nlayers
        self.dropout = kwargs.get('dropout', 0.6)
        self.alpha = kwargs.get('alpha', 0.1)
        self.lamda = kwargs.get('lamda', 0.5)
        variant = kwargs.get('variant', False)
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(self.num_features, self.num_features, variant=variant))
        self.fc_in = nn.Linear(self.num_features, self.num_features)
        self.act_fn = nn.ReLU()
        self.graph_norm = GraphNorm(self.num_features * nlayers, eps=1e-4)

    def forward(self, g: Batch):
        adj = torch.sparse_coo_tensor(
            g.edge_index,
            torch.ones(g.edge_index.shape[1], device=g.device),
            (g.total_nodes, g.total_nodes)
        ).coalesce()
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fc_in(x))
        h0 = layer_inner
        x_profile = torch.empty(g.total_nodes, self.num_features * self.nlayers, device=g.device)
        x_profile[:, 0:self.num_features] = layer_inner
        
        # Apply GCNII layers
        for i, conv in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(conv(layer_inner, adj, h0, self.lamda, self.alpha, i + 1))
            if i + 1 < self.nlayers:
                x_profile[:, (i + 1) * self.num_features:(i + 2) * self.num_features] = layer_inner
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)
