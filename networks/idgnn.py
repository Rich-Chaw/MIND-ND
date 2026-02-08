'''follow GraphGym/graphgym/contrib/layer/idconv.py'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset, zeros
from torch_geometric.utils import (add_remaining_self_loops, add_self_loops,
                                   remove_self_loops, softmax)
from torch_scatter import scatter_add
from torch_geometric.nn import GraphNorm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch


class GCNIDConvLayer(MessagePassing):
    """Identity-aware GCN layer"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 normalize=True,
                 **kwargs):
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
    def norm(edge_index,
             num_nodes,
             edge_weight=None,
             improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, id, edge_weight=None):
        """Forward pass with identity-aware mechanism"""
        # Apply identity-aware transformation
        x_id = torch.index_select(x, dim=0, index=id)
        x_id = torch.matmul(x_id, self.weight_id)
        x = torch.matmul(x, self.weight)
        x.index_add_(0, id, x_id)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
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
    """Identity-aware GraphSAGE layer"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize=False,
                 concat=False,
                 bias=True,
                 **kwargs):
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
        zeros(self.bias)

    def forward(self,
                x,
                edge_index,
                id,
                edge_weight=None,
                size=None,
                res_n_id=None):
        if not self.concat and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(self.node_dim))

        return self.propagate(edge_index,
                              size=size,
                              x=x,
                              edge_weight=edge_weight,
                              res_n_id=res_n_id,
                              id=id)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out, x, res_n_id, id):
        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)
        elif self.concat and (isinstance(x, tuple) or isinstance(x, list)):
            assert res_n_id is not None
            aggr_out = torch.cat([x[0][res_n_id], aggr_out], dim=-1)

        # Apply identity-aware transformation
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
    """Identity-aware GAT layer adapted for MIND-ND"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 **kwargs):
        super(GATIDConvLayer, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.weight_id = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
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
        zeros(self.bias)

    def forward(self, x, edge_index, id, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            # Apply identity-aware transformation
            x_id = torch.index_select(x, dim=0, index=id)
            x_id = torch.matmul(x_id, self.weight_id)
            x = torch.matmul(x, self.weight)
            x.index_add_(0, id, x_id) 

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class GINIDConvLayer(MessagePassing):
    """Identity-aware GIN layer adapted for MIND-ND"""
    def __init__(self, nn, nn_id, eps=0, train_eps=False, **kwargs):
        super(GINIDConvLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.nn = nn
        self.nn_id = nn_id
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        reset(self.nn_id)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, id):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        x = (1 + self.eps) * x + self.propagate(edge_index, x=x)
        
        # Apply identity-aware transformation
        x_id = torch.index_select(x, dim=0, index=id)
        x_id = self.nn_id(x_id)
        x = self.nn(x)
        x.index_add_(0, id, x_id)
        return x

    def message(self, x_j):
        return x_j


class IDGNN(nn.Module):
    """Base Identity-aware GNN adapted for MIND-ND framework"""
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None):
        super().__init__()
        self.num_features = num_features  # F
        self.num_mps = num_mps  # K layers
        self.num_heads = num_heads
        self.positional_encoding = positional_encoding
        
        # Initialize node features
        self.register_buffer("x_init", torch.ones(1, num_features))
        self.convs = nn.ModuleList()
        # Graph normalization
        self.graph_norm = GraphNorm(num_features * num_mps, eps=1e-4)

    def forward(self, g: Batch):
        """
        Forward pass that outputs concatenation of node and graph embeddings
        return x_profile (N, 2KF)
        """
        # Initialize profile tensor (N+B, KF)
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, 
                               device=self.x_init.device)
        
        # Initialize node features (N+B, F)
        if self.positional_encoding == 'RW':
            from utils.graph_data import random_walk_positional_encoding
            x_k = random_walk_positional_encoding(g, self.num_features, self.x_init.device)
        else:
            x_k = self.x_init.expand(g.total_nodes, -1)
        
        # Apply conv layers
        for k, conv in enumerate(self.convs):
            x_k = conv(x_k, g.edge_index, g.omni_ids)
            # Store layer output in profile
            x_profile[:, k*self.num_features : (k+1)*self.num_features] = x_k
            x_k = torch.relu(x_k)
        
        # Apply graph normalization
        x_profile = self.graph_norm(x_profile, g.batch)  # (N+B, KF)
        x_profile = torch.cat([
            x_profile[g.non_omni_mask],  # Node embeddings
            x_profile[g.omni_ids[g.batch_non_omni]]  # Graph embeddings
        ], dim=1)  # (N, 2KF)
        
        return x_profile


class IDGCN(IDGNN):
    """Identity-aware GCN variant"""
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None):
        super().__init__(num_features, num_heads, num_mps, positional_encoding)
        for _ in range(num_mps):
            self.convs.append(GCNIDConvLayer(num_features, num_features))


class IDSAGE(IDGNN):
    """Identity-aware GraphSAGE variant"""
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None):
        super().__init__(num_features, num_heads, num_mps, positional_encoding)
        for _ in range(num_mps):
            self.convs.append(SAGEIDConvLayer(num_features, num_features, concat=True))


class IDGAT(IDGNN):
    """Identity-aware GAT variant"""
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None):
        super().__init__(num_features, num_heads, num_mps, positional_encoding)
        for _ in range(num_mps):
            self.convs.append(GATIDConvLayer(num_features, num_features // num_heads, 
                                           heads=num_heads, concat=True))


class IDGIN(IDGNN):
    """Identity-aware GIN variant"""
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None):
        super().__init__(num_features, num_heads, num_mps, positional_encoding)
        for _ in range(num_mps):
            gin_nn = nn.Sequential(
                nn.Linear(num_features, num_features), 
                nn.ReLU(),
                nn.Linear(num_features, num_features)
            )
            gin_nn_id = nn.Sequential(
                nn.Linear(num_features, num_features), 
                nn.ReLU(),
                nn.Linear(num_features, num_features)
            )
            self.convs.append(GINIDConvLayer(gin_nn, gin_nn_id))


def test_idgnn():
    """Test all IDGNN variants"""
    print("Testing IDGNN variants...")
    
    # Create dummy graph data
    import numpy as np
    from utils.graph_data import Graph, Batch
    
    device = torch.device('cpu')
    
    # Create two simple graphs
    # Graph 1: triangle (3 nodes, 6 edges for undirected)
    edge_index1 = np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
    graph1 = Graph(edge_index1, 3)
    
    # Graph 2: line (3 nodes, 4 edges for undirected)  
    edge_index2 = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    graph2 = Graph(edge_index2, 3)
    
    # Create batch
    batch = Batch(device, [graph1, graph2])
    
    # Test parameters
    num_features = 64
    num_heads = 4
    num_mps = 3
    
    # Test all variants
    variants = [
        ('IDGCN', IDGCN),
        ('IDSAGE', IDSAGE), 
        ('IDGAT', IDGAT),
        ('IDGIN', IDGIN)
    ]
    
    for variant_name, variant_class in variants:
        print(f"\nTesting {variant_name}:")
        
        # Create model
        model = variant_class(num_features, num_heads, num_mps)
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            output = model(batch)
        
        print(f"  Input graphs: {batch.batch_size} graphs with {batch.total_nodes} total nodes")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: ({batch.total_nodes - batch.batch_size}, {2 * num_features * num_mps})")
        
        # Verify output shape
        expected_nodes = batch.total_nodes - batch.batch_size  # N (excluding omni nodes)
        expected_features = 2 * num_features * num_mps  # 2KF
        assert output.shape == (expected_nodes, expected_features), \
            f"Shape mismatch: got {output.shape}, expected ({expected_nodes}, {expected_features})"
        
        print(f"  ✓ Shape test passed!")
        print(f"  Output stats: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    
    print("\n✓ All IDGNN variants tested successfully!")


if __name__ == "__main__":
    test_idgnn()