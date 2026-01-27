import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, GraphNorm
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch


class OptimizedHybridLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha, beta):
        super(OptimizedHybridLayer, self).__init__(aggr='add')
        self.alpha = alpha
        # GCNII: Decaying identity mapping weight
        self.beta = beta
        
        self.W_center = Linear(in_channels, out_channels, bias=False)
        self.W_neigh = Linear(in_channels, out_channels, bias=False)
        
        # LGNN: Edge MLP for flow-based features
        self.edge_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )

    def forward(self, x, x_0, edge_index):
        # 1. ID-GNN-Fast / GCNII Residual
        # Mix current state with initial structural features (x_0)
        x_res = (1 - self.alpha) * x + self.alpha * x_0
        
        # 2. Edge-Aware Propagate
        # We pass edge_index to simulate line-graph flow
        out = self.propagate(edge_index, x=x_res)
        
        # 3. GCNII Identity Mapping
        # Regularize the weight matrix to prevent over-smoothing
        out = (1 - self.beta) * out + self.beta * self.W_center(out)
        
        return F.relu(out)

    def message(self, x_i, x_j):
        # Implicit LGNN: Compute edge features on the fly
        # This simulates the Line Graph without the O(E^2) storage
        edge_feature = self.edge_mlp(torch.cat([x_i, x_j], dim=-1))
        
        # ID-GNN: Heterogeneous message (distinguish neighbor j from self i)
        return self.W_neigh(x_j) + edge_feature


class HGNN_V2(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, alpha=0.1, theta=0.5):
        super().__init__()
        self.num_features = num_features  # F
        self.num_mps = num_mps  # K layers
        
        # Initialize with all-ones features like MIND
        self.register_buffer("x_init", torch.ones(1, num_features))
        
        # Create optimized hybrid layers
        self.layers = nn.ModuleList([
            OptimizedHybridLayer(num_features, num_features, alpha, math.log(theta / (l + 1) + 1)) 
            for l in range(num_mps)
        ])
        
        # Graph normalization for the profile
        self.graph_norm = GraphNorm(num_features * num_mps, eps=1e-4)

    def forward(self, g: Batch):
        '''
        return x_profile (N, 2KF) - concatenation of node and graph embeddings
        '''
        # Initialize features for all nodes (N+B, F)
        x = self.x_init.expand(g.total_nodes, -1)
        x_0 = x.clone()  # Keep initial features for residual connections
        
        # Store layer outputs for profile construction
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, 
                               device=self.x_init.device)
        
        # Apply optimized hybrid layers
        for k, layer in enumerate(self.layers):
            x = layer(x, x_0, g.edge_index)
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
def test_hgnn_v2():
    """Test HGNN_V2 with a simple batch of graphs"""
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
    
    # Initialize HGNN_V2
    num_features = 16
    num_heads = 4  # Not used in HGNN_V2 but kept for consistency
    num_mps = 3
    
    model = HGNN_V2(num_features, num_heads, num_mps)
    
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
    
    print("HGNN_V2 test passed!")
    
    # Test with different parameters
    print("\nTesting with different layer configurations:")
    for num_layers in [1, 2, 4, 8]:
        model_test = HGNN_V2(num_features, num_heads, num_layers)
        with torch.no_grad():
            output_test = model_test(batch)
        expected_features = 2 * num_features * num_layers
        print(f"  Layers: {num_layers}, Output shape: {output_test.shape}, Expected: ({batch.non_omni_mask.sum()}, {expected_features})")
        assert output_test.shape == (batch.non_omni_mask.sum(), expected_features)
    
    print("✓ All layer configuration tests passed!")
    return output


def compare_hgnn_versions():
    """Compare HGNN and HGNN_V2 performance"""
    import numpy as np
    import time
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.graph_data import Graph, Batch
    from hgnn import HGNN
    
    device = torch.device('cpu')
    
    # Create larger test graphs for performance comparison
    # Graph 1: larger graph (20 nodes)
    edges1 = []
    for i in range(20):
        for j in range(i+1, min(i+4, 20)):  # Connect to next 3 nodes
            edges1.extend([[i, j], [j, i]])
    edge_index1 = np.array(edges1, dtype=np.int64).T
    graph1 = Graph(edge_index1, 20)
    
    # Graph 2: another large graph (15 nodes)
    edges2 = []
    for i in range(15):
        for j in range(i+1, min(i+3, 15)):  # Connect to next 2 nodes
            edges2.extend([[i, j], [j, i]])
    edge_index2 = np.array(edges2, dtype=np.int64).T
    graph2 = Graph(edge_index2, 15)
    
    batch = Batch(device, [graph1, graph2])
    
    # Test parameters
    num_features = 32
    num_heads = 4
    num_mps = 4
    
    # Initialize both models
    hgnn_v1 = HGNN(num_features, num_heads, num_mps)
    hgnn_v2 = HGNN_V2(num_features, num_heads, num_mps)
    
    print("Performance Comparison:")
    print(f"Batch: {batch.batch_size} graphs, {batch.total_nodes} total nodes")
    
    # Test HGNN (v1)
    start_time = time.time()
    with torch.no_grad():
        output_v1 = hgnn_v1(batch)
    v1_time = time.time() - start_time
    
    # Test HGNN_V2
    start_time = time.time()
    with torch.no_grad():
        output_v2 = hgnn_v2(batch)
    v2_time = time.time() - start_time
    
    print(f"HGNN (v1):    {v1_time:.4f}s, Output shape: {output_v1.shape}")
    print(f"HGNN_V2:      {v2_time:.4f}s, Output shape: {output_v2.shape}")
    print(f"Speedup:      {v1_time/v2_time:.2f}x")
    
    # Verify outputs have same shape
    assert output_v1.shape == output_v2.shape, f"Shape mismatch: {output_v1.shape} vs {output_v2.shape}"
    print("✓ Both versions produce same output shape")


if __name__ == "__main__":
    test_hgnn_v2()
    print("\n" + "="*50)
    compare_hgnn_versions()