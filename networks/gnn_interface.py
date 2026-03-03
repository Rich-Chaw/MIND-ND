# Unified GNN module: classical + MIND, HGNN_V4, GCNII, IDGNN
from re import X
import torch
import sys
import os
# ---------- GNN registry and remaining imports ----------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.hgnn import HGNN
from networks.hgnn_v2 import HGNN_V2
from networks.hgnn_v3 import HGNN_V3
from networks.hgnn_v4 import HGNN_V4
from networks.classical_gnns import GCN, GraphSAGE, GAT
from networks.mind import MIND
from networks.idgnn import IDGCN, IDGAT, IDGIN, IDSAGE
from networks.gcnii import GCNII
from networks.hm_gnn import HM_GNN
from networks.hm_gnn_v2 import HM_GNN_V2

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
    'hm_gnn': HM_GNN,
    'hm_gnn_v2': HM_GNN_V2,
}

if __name__ == '__main__':
    import numpy as np
    import sys
    import os
    import torch
    import time
    from utils.graph_data import Graph, Batch, ig_to_data, handcrafted_node_features

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
    x = handcrafted_node_features(batch, batch.device)
    print(x)

    # Test parameters
    num_features = 16
    num_heads = 4
    num_mps = 6

    models_to_test = [GCN, GraphSAGE, GAT, HM_GNN, HM_GNN_V2, HGNN_V4, MIND]
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