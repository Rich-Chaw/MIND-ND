import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, EdgeConv
from torch_geometric.utils import add_self_loops, degree

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

    def forward(self, x, x_0, edge_index, edge_attr):
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
        return F.relu(self.W_neigh(x_j) + edge_attr)

class DeepHybridGNN(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, alpha=0.1, theta=0.5):
        super(DeepHybridGNN, self).__init__()
        self.node_lin = Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList([
            HybridLayer(hidden_channels, hidden_channels, alpha, (theta/(l+1))) 
            for l in range(num_layers)
        ])
        self.regressor = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Initial all-one features
        x = self.node_lin(x)
        x_0 = x.clone()
        edge_attr = None # Initialized internally by EdgeConv in first layer
        
        for layer in self.layers:
            x, edge_attr = layer(x, x_0, edge_index, edge_attr)
            
        # Output node-wise score (Betweenness Centrality)
        return self.regressor(x)

# --- Usage Example ---
# num_nodes = 100
# x = torch.ones((num_nodes, 1)) # All-one node features
# edge_index = torch.randint(0, num_nodes, (2, 500))
# model = DeepHybridGNN(num_layers=16, in_channels=1, hidden_channels=32, out_channels=1)
# output = model(x, edge_index)


import math

class OptimizedHybridLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha, theta, layer_idx):
        super(OptimizedHybridLayer, self).__init__(aggr='add')
        self.alpha = alpha
        # GCNII: Decaying identity mapping weight
        self.beta = math.log(theta / (layer_idx + 1) + 1)
        
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

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