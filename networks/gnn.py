# classical GNNS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphNorm

# classical GNNS
class GCN(nn.Module):
    def __init__(self, num_features, num_heads, num_mps):
        super().__init__()
        self.num_features = num_features
        self.num_mps = num_mps
        # Initialize with all-ones features like MIND/HGNN
        self.register_buffer("x_init", torch.ones(1, num_features))
        
        self.convs = nn.ModuleList()
        for _ in range(num_mps):
            # GCNConv doesn't typically use heads, so we ignore num_heads here
            self.convs.append(GCNConv(num_features, num_features))
            
        self.graph_norm = GraphNorm(num_features * num_mps, eps=1e-4)

    def forward(self, g):
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=self.x_init.device)
        x = self.x_init.expand(g.total_nodes, -1)
        
        for k, conv in enumerate(self.convs):
            x = conv(x, g.edge_index)
            x_profile[:, k*self.num_features : (k+1)*self.num_features] = x
            x = F.relu(x)
            
        x_profile = self.graph_norm(x_profile, g.batch)
        
        # Concatenate node embeddings and graph (omni-node) embeddings
        x_profile = torch.cat([
            x_profile[g.non_omni_mask],
            x_profile[g.omni_ids[g.batch_non_omni]]
        ], dim=1)
        return x_profile

class GraphSAGE(nn.Module):
    def __init__(self, num_features, num_heads, num_mps):
        super().__init__()
        self.num_features = num_features
        self.num_mps = num_mps
        self.register_buffer("x_init", torch.ones(1, num_features))
        
        self.convs = nn.ModuleList()
        for _ in range(num_mps):
            self.convs.append(SAGEConv(num_features, num_features))
            
        self.graph_norm = GraphNorm(num_features * num_mps, eps=1e-4)

    def forward(self, g):
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=self.x_init.device)
        x = self.x_init.expand(g.total_nodes, -1)
        
        for k, conv in enumerate(self.convs):
            x = conv(x, g.edge_index)
            x_profile[:, k*self.num_features : (k+1)*self.num_features] = x
            x = F.relu(x)
            
        x_profile = self.graph_norm(x_profile, g.batch)
        x_profile = torch.cat([
            x_profile[g.non_omni_mask],
            x_profile[g.omni_ids[g.batch_non_omni]]
        ], dim=1)
        return x_profile

class GAT(nn.Module):
    def __init__(self, num_features, num_heads, num_mps):
        super().__init__()
        self.num_features = num_features
        self.num_mps = num_mps
        self.register_buffer("x_init", torch.ones(1, num_features))
        
        assert num_features % num_heads == 0, f'num_features {num_features} not divisible by num_heads {num_heads}'
        head_dim = num_features // num_heads
        
        self.convs = nn.ModuleList()
        for _ in range(num_mps):
            self.convs.append(GATConv(num_features, head_dim, heads=num_heads, concat=True))
            
        self.graph_norm = GraphNorm(num_features * num_mps, eps=1e-4)

    def forward(self, g):
        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=self.x_init.device)
        x = self.x_init.expand(g.total_nodes, -1)
        
        for k, conv in enumerate(self.convs):
            x = conv(x, g.edge_index)
            x_profile[:, k*self.num_features : (k+1)*self.num_features] = x
            x = F.relu(x)
            
        x_profile = self.graph_norm(x_profile, g.batch)
        x_profile = torch.cat([
            x_profile[g.non_omni_mask],
            x_profile[g.omni_ids[g.batch_non_omni]]
        ], dim=1)
        return x_profile

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# advanced GNNS
from networks.mind import MIND
from networks.idgnn import IDGAT,IDGCN,IDSAGE,IDGIN
from networks.gcnii import GCNII
from networks.hgnn import HGNN
from networks.hgnn_v2 import HGNN_V2
from networks.hgnn_v3 import HGNN_V3
from networks.hgnn_v4 import HGNN_V4

GNN_ENCODER = {
    'gcn':GCN,
    'graphsage':GraphSAGE,
    # 'gin':GIN,
    'gat': GAT,
    'mind': MIND,
    'idgcn': IDGCN,
    'idgat': IDGAT,
    'idgin': IDGIN,
    'idsage': IDSAGE,
    'gcnii':GCNII,
    'hgnn': HGNN,
    'hgnn_v2': HGNN_V2,
    'hgnn_v3': HGNN_V3,
    'hgnn_v4': HGNN_V4,
}

if __name__ == '__main__':
    import numpy as np
    import sys
    import os
    import torch
    import time
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

    # Test parameters
    num_features = 16
    num_heads = 4
    num_mps = 6

    # model = MIND(num_features, num_heads, num_mps)
    # model = HGNN_V2(num_features, num_heads, num_mps)

    models_to_test = [GCN, GraphSAGE, GAT]
    for ModelClass in models_to_test:
        print(f"\nTesting {ModelClass.__name__}...")
        model = ModelClass(num_features, num_heads, num_mps)
        
        start_time = time.time()
        with torch.no_grad():
            output = model(batch)
        elapsed = time.time() - start_time
        
        print(f"Time: {elapsed:.4f}s, Output shape: {output.shape}")
        
        # Expected shape: (Total non-omni nodes, 2 * num_features * num_mps)
        expected_nodes = 3 + 4
        expected_dim = 2 * num_features * num_mps
        assert output.shape == (expected_nodes, expected_dim), \
            f"Shape mismatch! Expected ({expected_nodes}, {expected_dim}), got {output.shape}"