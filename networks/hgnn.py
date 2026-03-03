import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, EdgeConv, GraphNorm
from torch_geometric.utils import add_self_loops, degree

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch


class HybridLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha, beta):
        super(HybridLayer, self).__init__(aggr='add')
        self.alpha = alpha  # GCNII Initial residual weight
        self.beta = beta    # GCNII Identity mapping weight
        
        # ID-GNN: Separate weights for center node and neighbors
        self.W_center = Linear(in_channels, out_channels)
        self.W_neigh = Linear(in_channels, out_channels)
        
        # LGNN / Edge-Aware logic: Processing edge/line graph features
        # EdgeConv takes cat(x_i, x_j - x_i) or cat(x_i, x_j)
        self.edge_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
        self.edge_conv = EdgeConv(self.edge_mlp)

    def forward(self, x, x_0, edge_index, edge_attr=None):
        # 1. LGNN Component: Implicit Line Graph Update via EdgeConv
        # We update edge features based on the nodes they connect
        new_edge_attr = self.edge_conv(x, edge_index)
        
        # 2. GCNII + ID-GNN Component: Node Update
        # Initial Residual connection (from GCNII)
        x_initial = (1 - self.alpha) * x + self.alpha * x_0
        
        # Message Passing (ID-GNN style heterogeneous weighting)
        out = self.propagate(edge_index, x=x_initial, edge_attr=new_edge_attr)
        
        # Identity Mapping (from GCNII)
        out = (1 - self.beta) * out + self.beta * self.W_center(x)
        
        return F.relu(out), new_edge_attr

    def message(self, x_j, edge_attr):
        # Combine neighbor features with edge features
        # Handle potential dimension mismatch by using broadcasting or reshaping
        if edge_attr is not None and edge_attr.shape[0] == x_j.shape[0]:
            return F.relu(self.W_neigh(x_j) + edge_attr)
        else:
            return F.relu(self.W_neigh(x_j))


class HGNN(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, alpha=0.1, theta=0.5, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps  # K layers
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        # Create hybrid layers
        self.layers = nn.ModuleList([
            HybridLayer(self.num_features, self.num_features, alpha, theta/(l+1))
            for l in range(num_mps)
        ])
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)

    def forward(self, g: Batch):
        '''
        return x_profile (N, 2KF) - concatenation of node and graph embeddings
        '''
        from .gnn_utils import _get_init_features
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_0 = x.clone()  # Keep initial features for residual connections
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        
        edge_attr = None  # Will be initialized by first layer
        
        # Apply hybrid layers
        for k, layer in enumerate(self.layers):
            x, edge_attr = layer(x, x_0, g.edge_index, edge_attr)
            # Store layer output in profile
            x_profile[:, k*self.num_features : (k+1)*self.num_features] = x
        
        # Apply graph normalization
        x_profile = self.graph_norm(x_profile, g.batch)  # (N+B, KF)
        
        # Concatenate node embeddings with graph embeddings (omni node embeddings)
        # Following MIND's pattern: [node_embedding, graph_embedding]
        x_profile = torch.cat([
            x_profile[g.non_omni_mask],  # Node embeddings (N, KF)
            x_profile[g.omni_ids[g.batch_non_omni]]  # Graph embeddings (N, KF)
        ], dim=1)  # (N, 2KF)
        
        return x_profile


# Simple test case
def test_hgnn():
    """Test HGNN with a simple batch of graphs"""
    import numpy as np
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.graph_data import Graph, Batch
    
    device = torch.device('cpu')
    
    # Create simple test graphs
    # Graph 1: triangle (3 nodes)
    edge_index1 = np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
    graph1 = Graph(edge_index1, 3)
    
    # Graph 2: line (4 nodes)  
    edge_index2 = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)
    graph2 = Graph(edge_index2, 4)
    
    # Create batch
    batch = Batch(device, [graph1, graph2])
    
    # Initialize HGNN
    num_features = 16
    num_heads = 4  # Not used in HGNN but kept for consistency
    num_mps = 3
    
    model = HGNN(num_features, num_heads, num_mps)
    
    # Forward pass
    with torch.no_grad():
        output = model(batch)
    
    print(f"Input batch size: {batch.batch_size}")
    print(f"Total nodes (including omni): {batch.total_nodes}")
    print(f"Non-omni nodes: {batch.non_omni_mask.sum()}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch.non_omni_mask.sum()}, {2 * num_features * num_mps})")
    
    assert output.shape == (batch.non_omni_mask.sum(), 2 * num_features * num_mps), \
        f"Output shape mismatch: {output.shape} vs expected {(batch.non_omni_mask.sum(), 2 * num_features * num_mps)}"
    
    print("HGNN test passed!")
    return output


if __name__ == "__main__":
    test_hgnn()