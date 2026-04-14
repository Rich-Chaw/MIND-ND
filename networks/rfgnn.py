import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch


class LeanHybridConv(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha=0.1, theta=0.5, layer=1):
        super(LeanHybridConv, self).__init__(aggr='add')
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

    def forward(self, x, x_0, edge_index):
        # 1. Pre-compute Attention Projections (N x 1)
        phi_src = self.lin_src(x)
        phi_dst = self.lin_dst(x)

        # 2. Propagate Messages
        # We pass phi_src/dst to message() via the edge_index mapping
        out = self.propagate(edge_index, x=x, phi_src=phi_src, phi_dst=phi_dst)

        # 3. GCNII Initial Residual & Identity Mapping
        # h_tilde = (1-alpha) * sum(m_ji) + alpha * h_0
        out = (1 - self.alpha) * out + self.alpha * x_0

        # h_l = (1-beta) * h_tilde + beta * W_center * h_tilde
        out = (1 - self.beta) * out + self.beta * self.lin_center(out)

        return self.layer_norm(F.relu(out))

    def message(self, x_i, x_j, phi_src_i, phi_dst_j):
        # 0.1.1 Additive Attention Coefficient
        alpha_ij = torch.sigmoid(phi_src_i + phi_dst_j)

        # 0.1.2 Difference-Based Flow Message
        # Using (x_j - x_i) for LGNN-style flow
        msg = self.lin_neigh(x_j) + self.lin_flow(x_j - x_i)
        
        return alpha_ij * msg


class ResiflowGNN(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, alpha=0.1, theta=0.5, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps  # K layers
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        # Create lean hybrid layers with attention mechanism
        self.layers = nn.ModuleList([
            LeanHybridConv(self.num_features, self.num_features, alpha, theta, layer=l+1)
            for l in range(num_mps)
        ])
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)
        self.register_buffer("x_init", torch.ones(1, num_features))

    def forward(self, g: Batch):
        '''
        return x_profile (N, 2KF) - concatenation of node and graph embeddings
        '''
        from .gnn_utils import _get_init_features
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        x_0 = x.clone()  # Keep initial features for residual connections
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        
        # Apply lean hybrid layers with attention
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
def test_rfgnn():
    """Test ResiflowGNN with a simple batch of graphs"""
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
    
    # Initialize ResiflowGNN
    num_features = 16
    num_heads = 4  # Not directly used in LeanHybridConv but kept for consistency
    num_mps = 3
    
    model = ResiflowGNN(num_features, num_heads, num_mps)
    
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
    
    print("ResiflowGNN test passed!")
    
    # Test with different parameters
    print("\nTesting with different layer configurations:")
    for num_layers in [1, 2, 4, 8]:
        model_test = ResiflowGNN(num_features, num_heads, num_layers)
        with torch.no_grad():
            output_test = model_test(batch)
        expected_features = 2 * num_features * num_layers
        print(f"  Layers: {num_layers}, Output shape: {output_test.shape}, Expected: ({batch.non_omni_mask.sum()}, {expected_features})")
        assert output_test.shape == (batch.non_omni_mask.sum(), expected_features)
    
    print("✓ All layer configuration tests passed!")
    
    # Test attention mechanism
    print("\nTesting attention mechanism properties:")
    model_attention = ResiflowGNN(num_features=8, num_heads=2, num_mps=2, alpha=0.2, theta=0.3)
    with torch.no_grad():
        output_attention = model_attention(batch)
    print(f"  Attention model output shape: {output_attention.shape}")
    print(f"  Expected: ({batch.non_omni_mask.sum()}, {2 * 8 * 2})")
    assert output_attention.shape == (batch.non_omni_mask.sum(), 2 * 8 * 2)
    print("✓ Attention mechanism test passed!")
    
    return output


def compare_hgnn_versions():
    """Compare HGNN, HGNN_V2, and ResiflowGNN performance"""
    import numpy as np
    import time
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.graph_data import Graph, Batch
    from hgnn import HGNN
    from hgnn_v2 import HGNN_V2
    
    device = torch.device('cpu')
    
    # Create larger test graphs for performance comparison
    # Graph 1: larger graph (20 nodes)
    edges1 = []
    for i in range(100):
        for j in range(i+1, min(i+4, 20)):  # Connect to next 3 nodes
            edges1.extend([[i, j], [j, i]])
    edge_index1 = np.array(edges1, dtype=np.int64).T
    graph1 = Graph(edge_index1, 20)
    
    # Graph 2: another large graph (15 nodes)
    edges2 = []
    for i in range(150):
        for j in range(i+1, min(i+3, 15)):  # Connect to next 2 nodes
            edges2.extend([[i, j], [j, i]])
    edge_index2 = np.array(edges2, dtype=np.int64).T
    graph2 = Graph(edge_index2, 15)
    
    batch = Batch(device, [graph1, graph2])
    
    # Test parameters
    num_features = 32
    num_heads = 4
    num_mps = 4
    
    # Initialize all models
    hgnn_v1 = HGNN(num_features, num_heads, num_mps)
    hgnn_v2 = HGNN_V2(num_features, num_heads, num_mps)
    rfgnn = ResiflowGNN(num_features, num_heads, num_mps)
    
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
    
    # Test ResiflowGNN
    start_time = time.time()
    with torch.no_grad():
        output_v3 = rfgnn(batch)
    v3_time = time.time() - start_time
    
    print(f"HGNN (v1):    {v1_time:.4f}s, Output shape: {output_v1.shape}")
    print(f"HGNN_V2:      {v2_time:.4f}s, Output shape: {output_v2.shape}")
    print(f"ResiflowGNN:      {v3_time:.4f}s, Output shape: {output_v3.shape}")
    print(f"V2 vs V1 Speedup: {v1_time/v2_time:.2f}x")
    print(f"V3 vs V1 Speedup: {v1_time/v3_time:.2f}x")
    print(f"V3 vs V2 Speedup: {v2_time/v3_time:.2f}x")
    
    # Verify outputs have same shape
    assert output_v1.shape == output_v2.shape == output_v3.shape, \
        f"Shape mismatch: {output_v1.shape} vs {output_v2.shape} vs {output_v3.shape}"
    print("✓ All versions produce same output shape")


if __name__ == "__main__":
    test_rfgnn()
    print("\n" + "="*50)
    compare_hgnn_versions()