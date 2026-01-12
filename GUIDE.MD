import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, node_dim, graph_dim, hidden_dim=256):
        super(Discriminator, self).__init__()
        # Input: Concat of selected node embedding and global graph embedding
        self.fc = nn.Sequential(
            nn.Linear(node_dim + graph_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Output probability of being 'Expert'
        )

    def forward(self, node_emb, graph_emb):
        # x shape: [batch_size, node_dim + graph_dim]
        x = torch.cat([node_emb, graph_emb], dim=-1)
        return self.fc(x)