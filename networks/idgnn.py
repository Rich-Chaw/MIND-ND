import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn import GraphNorm, MessagePassing
from torch_geometric.nn.inits import glorot, reset, zeros
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops, softmax

from utils.graph_data import Batch
from .gnn_utils import _get_init_features, _read_out



# ---------- IDGNN (identity-aware conv layers + base + variants) ----------
class GCNIDConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=True, normalize=True, **kwargs):
        super(GCNIDConvLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_id = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight_id)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, id, edge_weight=None):
        x_id = torch.index_select(x, dim=0, index=id)
        x_id = torch.matmul(x_id, self.weight_id)
        x = torch.matmul(x, self.weight)
        x.index_add_(0, id, x_id)
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(self.node_dim), edge_weight, self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class SAGEIDConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=False, concat=False, bias=True, **kwargs):
        super(SAGEIDConvLayer, self).__init__(aggr='mean', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.concat = concat
        in_channels = 2 * in_channels if concat else in_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_id = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight_id)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, edge_index, id, edge_weight=None, size=None, res_n_id=None):
        if not self.concat and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1, x.size(self.node_dim))
        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight, res_n_id=res_n_id, id=id)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out, x, res_n_id, id):
        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)
        elif self.concat and (isinstance(x, tuple) or isinstance(x, list)):
            assert res_n_id is not None
            aggr_out = torch.cat([x[0][res_n_id], aggr_out], dim=-1)
        aggr_out_id = torch.index_select(aggr_out, dim=0, index=id)
        aggr_out_id = torch.matmul(aggr_out_id, self.weight_id)
        aggr_out = torch.matmul(aggr_out, self.weight)
        aggr_out.index_add_(0, id, aggr_out_id)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out


class GATIDConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATIDConvLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.weight_id = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight_id)
        glorot(self.att)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, edge_index, id, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))
        if torch.is_tensor(x):
            x_id = torch.index_select(x, dim=0, index=id)
            x_id = torch.matmul(x_id, self.weight_id)
            x = torch.matmul(x, self.weight)
            x.index_add_(0, id, x_id)
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels) if x_i is not None else None
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) if x_i is not None else (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.out_channels) if self.concat else aggr_out.mean(dim=1)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GINIDConvLayer(MessagePassing):
    def __init__(self, nn_module, nn_id, eps=0, train_eps=False, **kwargs):
        super(GINIDConvLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.nn = nn_module
        self.nn_id = nn_id
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        reset(self.nn_id)
        if isinstance(self.eps, nn.Parameter):
            self.eps.data.fill_(self.initial_eps)
        else:
            self.eps.fill_(self.initial_eps)

    def forward(self, x, edge_index, id):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        x = (1 + self.eps) * x + self.propagate(edge_index, x=x)
        x_id = torch.index_select(x, dim=0, index=id)
        x_id = self.nn_id(x_id)
        x = self.nn(x)
        x.index_add_(0, id, x_id)
        return x

    def message(self, x_j):
        return x_j


class IDGNN(nn.Module):
    """Unified init: num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False."""
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False, **kwargs):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_heads = num_heads
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features
        self.convs = nn.ModuleList()
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)

    def forward(self, g: Batch):
        x_k = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        for k, conv in enumerate(self.convs):
            x_k = conv(x_k, g.edge_index, g.omni_ids)
            x_profile[:, k * self.num_features : (k + 1) * self.num_features] = x_k
            x_k = torch.relu(x_k)
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)


class IDGCN(IDGNN):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False, **kwargs):
        super().__init__(num_features, num_heads, num_mps, positional_encoding, handcrafted_features, **kwargs)
        for _ in range(num_mps):
            self.convs.append(GCNIDConvLayer(self.num_features, self.num_features))


class IDSAGE(IDGNN):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False, **kwargs):
        super().__init__(num_features, num_heads, num_mps, positional_encoding, handcrafted_features, **kwargs)
        for _ in range(num_mps):
            self.convs.append(SAGEIDConvLayer(self.num_features, self.num_features, concat=True))


class IDGAT(IDGNN):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False, **kwargs):
        super().__init__(num_features, num_heads, num_mps, positional_encoding, handcrafted_features, **kwargs)
        for _ in range(num_mps):
            self.convs.append(GATIDConvLayer(self.num_features, self.num_features // num_heads, heads=num_heads, concat=True))


class IDGIN(IDGNN):
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False, **kwargs):
        super().__init__(num_features, num_heads, num_mps, positional_encoding, handcrafted_features, **kwargs)
        for _ in range(num_mps):
            gin_nn = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(), nn.Linear(self.num_features, self.num_features))
            gin_nn_id = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(), nn.Linear(self.num_features, self.num_features))
            self.convs.append(GINIDConvLayer(gin_nn, gin_nn_id))

def test_idgnn():
    """Test all IDGNN variants (run from repo root)."""
    import torch
    import numpy as np
    from utils.graph_data import Graph, Batch

    device = torch.device('cpu')
    edge_index1 = np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
    graph1 = Graph(edge_index1, 3)
    edge_index2 = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    graph2 = Graph(edge_index2, 3)
    batch = Batch(device, [graph1, graph2])
    num_features, num_heads, num_mps = 64, 4, 3
    variants = [('IDGCN', IDGCN), ('IDSAGE', IDSAGE), ('IDGAT', IDGAT), ('IDGIN', IDGIN)]
    for name, Cls in variants:
        model = Cls(num_features, num_heads, num_mps)
        model.eval()
        with torch.no_grad():
            output = model(batch)
        expected_nodes = batch.total_nodes - batch.batch_size
        expected_features = 2 * num_features * num_mps
        assert output.shape == (expected_nodes, expected_features), f"{name} shape mismatch"
        print(f"  {name} OK")
    print("All IDGNN variants passed!")


if __name__ == "__main__":
    test_idgnn()
