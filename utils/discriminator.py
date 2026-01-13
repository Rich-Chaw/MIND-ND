import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from utils import Batch, ig_to_data

class Discriminator(nn.Module):
    def __init__(self, e_size):
        super().__init__()
        # Input: graph embedding from policy network
        self.net = nn.Sequential(
            nn.Linear(e_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Output logit for binary classification
        )

    def forward(self, e):
        # e: [batch, e_size] - graph embeddings
        return self.net(e).squeeze(-1)  # [batch]

class DiscriminatorDataset:
    def __init__(self, graphs, actions, labels, seed, device):
        """
        graphs: list of igraph graphs
        actions: list of integer actions
        labels: list of binary labels (0=student, 1=teacher)
        """

        self.device = device
        self.rng = np.random.default_rng(seed)

        # Shuffle the dataset
        indices = self.rng.permutation(len(graphs))
        self.graphs = [graphs[i] for i in indices]
        # keep actions/labels as tensors so we can index with torch tensors later
        self.actions = torch.as_tensor([actions[i] for i in indices], device=device, dtype=torch.long)
        self.labels = torch.as_tensor([labels[i] for i in indices], device=device, dtype=torch.float32)

        self.size = len(self.graphs)
        
    def sample(self, batch_size, encoder):
        """Sample a batch from the dataset"""
        if len(self.graphs) == 0:
            return None, None
            
        indices = torch.randint(0, self.size, (min(batch_size, self.size),), device=self.device)

        # graphs stay as a Python list, so index via Python integers
        batch_graphs = [self.graphs[i] for i in indices.tolist()]
        batch_actions = self.actions[indices]

        g = Batch(self.device, [ig_to_data(g) for g in batch_graphs])
        e = encoder(g) #[N,2KF]
        batch_x = e[g.act_offsets + batch_actions]
        batch_y = self.labels[indices]
        return batch_x, batch_y
    
    def __len__(self):
        return len(self.graphs)

def train_discriminator(dataset, discriminator, encoder, batch_size=64, num_epochs=10, lr=0.001):
    """Train discriminator to distinguish teacher vs student embeddings"""
    if dataset is None or len(dataset) == 0:
        return
        
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, eps=1e-4)
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(num_epochs):
        x, y = dataset.sample(batch_size,encoder)
        if x is None:
            continue
        # Forward pass
        logits = discriminator(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss
