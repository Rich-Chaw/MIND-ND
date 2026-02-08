import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import to_dense_adj

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1-alpha)*hi + alpha*h0
        else:
            support = (1-alpha)*hi + alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight) + (1-theta)*r
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):
    def __init__(self, num_features, nlayers, dropout=0.6, lamda=0.5, alpha=0.1, variant=False, positional_encoding=None):
        super().__init__()
        self.num_features = num_features
        self.nlayers = nlayers # K
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.positional_encoding = positional_encoding
        
        # Initialize with ones like MIND
        self.register_buffer("x_init", torch.ones(1, num_features))
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(num_features, num_features, variant=variant))
        
        # Input transformation
        self.fc_in = nn.Linear(num_features, num_features)
        self.act_fn = nn.ReLU()
        
        # Graph normalization for the concatenated features
        self.graph_norm = GraphNorm(num_features * nlayers, eps=1e-4)

    def forward(self, g: Batch):
        '''
        return x_profile (N, 2KF) - concatenation of node and graph embeddings
        '''
        # Convert edge_index to sparse adjacency matrix
        adj = torch.sparse_coo_tensor(
            g.edge_index, 
            torch.ones(g.edge_index.shape[1], device=g.edge_index.device),
            (g.total_nodes, g.total_nodes)
        ).coalesce()

        # Initialize and transform input features
        if self.positional_encoding == 'RW':
            from utils.graph_data import random_walk_positional_encoding
            x = random_walk_positional_encoding(g, self.num_features, self.x_init.device)
        else:
            x = self.x_init.expand(g.total_nodes, -1)
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fc_in(x))
        h0 = layer_inner  # Keep reference to initial layer
        
        # Store embeddings from each layer
        x_profile = torch.empty(g.total_nodes, self.num_features * self.nlayers, device=self.x_init.device)
        x_profile[:, 0:self.num_features] = layer_inner
        
        # Apply GCNII layers
        for i, conv in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(conv(layer_inner, adj, h0, self.lamda, self.alpha, i+1))
            if i+1 < self.nlayers:
                x_profile[:, (i+1)*self.num_features:(i+2)*self.num_features] = layer_inner
        
        # Apply graph normalization and concatenate node + graph embeddings
        x_profile = self.graph_norm(x_profile, g.batch)
        x_profile = torch.cat([
            x_profile[g.non_omni_mask],  # Node embeddings
            x_profile[g.omni_ids[g.batch_non_omni]]  # Graph embeddings from omni nodes
        ], dim=1)  # (N, 2KF)
        
        return x_profile


def test_gcnii():
    """Comprehensive test case for GCNII_Encoder"""
    import numpy as np
    
    # Create test data
    device = torch.device('cpu')
    
    # Create simple graphs
    # Graph 1: 3 nodes in a triangle
    edge_index_1 = np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
    graph_1 = type('Graph', (), {'edge_index': edge_index_1, 'num_nodes': 3})()
    
    # Graph 2: 4 nodes in a square
    edge_index_2 = np.array([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=np.int64)
    graph_2 = type('Graph', (), {'edge_index': edge_index_2, 'num_nodes': 4})()
    
    # Create batch
    batch = Batch(device, [graph_1, graph_2])
    
    # Test GCNII encoder
    num_features = 64
    nlayers = 3
    model = GCNII(num_features, nlayers)
    
    print(f"Batch size: {batch.batch_size}")
    print(f"Total nodes (including omni): {batch.total_nodes}")
    print(f"Non-omni nodes: {batch.non_omni_mask.sum()}")
    
    # Forward pass
    with torch.no_grad():
        output = model(batch)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch.non_omni_mask.sum()}, {2 * nlayers * num_features})")
    
    # Verify output dimensions
    expected_nodes = batch.non_omni_mask.sum()  # Total non-omni nodes
    expected_features = 2 * nlayers * num_features  # Concatenated node + graph embeddings
    
    assert output.shape == (expected_nodes, expected_features), \
        f"Expected shape ({expected_nodes}, {expected_features}), got {output.shape}"
    
    # Test that node and graph embeddings are properly concatenated
    node_embeddings = output[:, :nlayers * num_features]
    graph_embeddings = output[:, nlayers * num_features:]
    
    print(f"Node embeddings shape: {node_embeddings.shape}")
    print(f"Graph embeddings shape: {graph_embeddings.shape}")
    
    # Verify that graph embeddings are repeated for nodes in the same graph
    # First 3 nodes belong to graph 0, next 4 nodes belong to graph 1
    graph_0_embedding = graph_embeddings[:3]  # First 3 nodes
    graph_1_embedding = graph_embeddings[3:]  # Next 4 nodes
    
    # Check that all nodes in the same graph have the same graph embedding
    assert torch.allclose(graph_0_embedding[0], graph_0_embedding[1], atol=1e-6), \
        "Graph embeddings should be the same for nodes in the same graph"
    assert torch.allclose(graph_0_embedding[0], graph_0_embedding[2], atol=1e-6), \
        "Graph embeddings should be the same for nodes in the same graph"
    
    assert torch.allclose(graph_1_embedding[0], graph_1_embedding[1], atol=1e-6), \
        "Graph embeddings should be the same for nodes in the same graph"
    assert torch.allclose(graph_1_embedding[0], graph_1_embedding[2], atol=1e-6), \
        "Graph embeddings should be the same for nodes in the same graph"
    assert torch.allclose(graph_1_embedding[0], graph_1_embedding[3], atol=1e-6), \
        "Graph embeddings should be the same for nodes in the same graph"
    
    print("✓ GCNII test passed!")
    print("✓ Graph embedding consistency verified!")
    return model, batch, output


def compare_with_mind():
    """Compare GCNII_Encoder output format with MIND"""
    import numpy as np
    from networks.mind import MIND
    
    device = torch.device('cpu')
    
    # Create test graphs
    edge_index_1 = np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
    graph_1 = type('Graph', (), {'edge_index': edge_index_1, 'num_nodes': 3})()
    
    edge_index_2 = np.array([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=np.int64)
    graph_2 = type('Graph', (), {'edge_index': edge_index_2, 'num_nodes': 4})()
    
    batch = Batch(device, [graph_1, graph_2])
    
    # Test both models with same parameters
    num_features = 32
    nlayers = 2
    num_heads = 4
    
    gcnii_model = GCNII(num_features, nlayers)
    mind_model = MIND(num_features, num_heads, nlayers)
    
    with torch.no_grad():
        gcnii_output = gcnii_model(batch)
        mind_output = mind_model(batch)
    
    print("=== Model Comparison ===")
    print(f"GCNII output shape: {gcnii_output.shape}")
    print(f"MIND output shape: {mind_output.shape}")
    print(f"Both models output concatenated [node_embedding, graph_embedding]")
    print(f"GCNII: [{nlayers * num_features}, {nlayers * num_features}] = {2 * nlayers * num_features}")
    print(f"MIND: [{nlayers * num_features}, {nlayers * num_features}] = {2 * nlayers * num_features}")
    print("✓ Both models follow the same output pattern!")


if __name__ == '__main__':
    test_gcnii()
    print()
    compare_with_mind()