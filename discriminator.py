import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch_scatter import scatter_mean
from utils.graph_data import Batch, ig_to_data
from utils import graph_models
from networks.gnn import GNN_ENCODER

class Discriminator(nn.Module):
    def __init__(self, e_size):
        super().__init__()
        # Input: graph embedding from GNN encoder
        # Binary classification: 0 = SBM-like (community), 1 = LPA-like (power-law)
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


class GraphDataset:
    """Dataset of (igraph graph, label) for graph-type classification.
    Label 0 = SBM-like (community structure), Label 1 = LPA-like (power-law).
    """
    def __init__(self, graphs, labels, seed, device):
        self.device = device
        self.rng = np.random.default_rng(seed)
        indices = self.rng.permutation(len(graphs))
        self.graphs = [graphs[i] for i in indices]
        self.labels = torch.as_tensor([labels[i] for i in indices], device=device, dtype=torch.float32)
        self.size = len(self.graphs)

    def sample(self, batch_size, encoder):
        """Sample a batch; return (graph_embeddings [B, E], labels [B])."""
        if self.size == 0:
            return None, None
        n = min(batch_size, self.size)
        indices = self.rng.choice(self.size, size=n, replace=False)
        batch_graphs = [self.graphs[i] for i in indices]
        batch_labels = self.labels[indices]
        g = Batch(self.device, [ig_to_data(gr) for gr in batch_graphs])
        with torch.no_grad() if not encoder.training else torch.enable_grad():
            out = encoder(g)  # [N_total, 2*KF]
        embed_dim = out.shape[1] // 2
        graph_emb = scatter_mean(
            out[:, embed_dim:], g.batch_non_omni, dim=0, dim_size=g.batch_size
        )  # [B, KF]
        return graph_emb, batch_labels

    def __len__(self):
        return self.size


class GraphActionDataset:
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

def train_discriminator(dataset, discriminator, encoder, batch_size=64, num_epochs=10, lr=0.001, train_encoder=True):
    """Train discriminator (and optionally encoder) for graph-type classification."""
    if dataset is None or len(dataset) == 0:
        return 0.0
    params = list(discriminator.parameters())
    if train_encoder and encoder is not None:
        params = list(encoder.parameters()) + params
    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-4)
    num_batches_per_epoch = max(1, len(dataset) // batch_size)
    total_loss = 0.0
    total_batches = 0
    for epoch in range(num_epochs):
        encoder.train()
        discriminator.train()
        for _ in range(num_batches_per_epoch):
            x, y = dataset.sample(batch_size, encoder)
            if x is None:
                continue
            logits = discriminator(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def evaluate(encoder, discriminator, dataset, device, batch_size=64):
    """Compute accuracy and average loss on dataset."""
    if dataset is None or len(dataset) == 0:
        return 0.0, 0.0
    encoder.eval()
    discriminator.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = np.arange(start, min(start + batch_size, len(dataset)))
            batch_graphs = [dataset.graphs[i] for i in indices]
            labels = dataset.labels[indices]
            g = Batch(device, [ig_to_data(gr) for gr in batch_graphs])
            out = encoder(g)
            embed_dim = out.shape[1] // 2
            graph_emb = scatter_mean(out[:, embed_dim:], g.batch_non_omni, dim=0, dim_size=g.batch_size)
            logits = discriminator(graph_emb)
            pred = (logits >= 0).long()
            correct += (pred == labels.long()).sum().item()
            total += labels.shape[0]
            total_loss += F.binary_cross_entropy_with_logits(logits, labels).item()
            n_batches += 1
    acc = correct / max(total, 1)
    avg_loss = total_loss / max(n_batches, 1)
    return acc, avg_loss


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Train discriminator: SBM-like (0) vs LPA-like (1)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_graphs", type=int, default=1000, help="Training graphs per class (SBM + LPA)")
    parser.add_argument("--num_test_graphs", type=int, default=200, help="Test graphs per class")
    parser.add_argument("--gnn", type=str, default="rfgnn", choices=list(GNN_ENCODER.keys()))
    parser.add_argument("--num_features", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_mps", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="saved/discriminator")
    parser.add_argument("--visualize", action="store_true", help="Save t-SNE of test embeddings")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    graph_embed_dim = args.num_features * args.num_mps  # KF

    # --- Generate graphs: SBM-like (0) vs LPA-like (1) ---
    print("Generating graphs...")
    graphs_sbm = []
    graphs_lpa = []
    rng = np.random.default_rng(args.seed)

    def gen_sbm():
        N = int(rng.integers(100, 201))
        p_in = float(rng.uniform(0.1, 0.2))
        p_out = float(rng.uniform(0.001, 0.01))
        num_blocks = int(rng.integers(2, 5))
        return graph_models.SBM(N, p_in, p_out, num_blocks=num_blocks)

    def gen_dcsbm():
        N = int(rng.integers(100, 201))
        p_in = float(rng.uniform(0.1, 0.2))
        p_out = float(rng.uniform(0.001, 0.01))
        num_blocks = int(rng.integers(2, 5))
        return graph_models.DCSBM(N, p_in, p_out, num_blocks)

    def gen_lpa():
        N = int(rng.integers(100, 201))
        m = int(rng.choice([2, 3, 4, 5, 6, 7]))
        gamma = float(2.5 + rng.random())
        return graph_models.LPA(N, m, gamma)

    def gen_copy():
        N = int(rng.integers(100, 201))
        m = int(rng.choice([2, 3, 4, 5]))
        gamma = float(2.1 + rng.uniform(0, 0.8))
        return graph_models.copying_model(N, m, gamma)

    n_per_class = args.num_train_graphs + args.num_test_graphs
    while len(graphs_sbm) < n_per_class:
        g = gen_sbm() if rng.random() < 0.5 else gen_dcsbm()
        if g.is_connected():
            graphs_sbm.append(g)
    while len(graphs_lpa) < n_per_class:
        g = gen_lpa() if rng.random() < 0.5 else gen_copy()
        if g.is_connected():
            graphs_lpa.append(g)
    graphs_sbm = graphs_sbm[:n_per_class]
    graphs_lpa = graphs_lpa[:n_per_class]
    train_sbm = graphs_sbm[: args.num_train_graphs]
    test_sbm = graphs_sbm[args.num_train_graphs:]
    train_lpa = graphs_lpa[: args.num_train_graphs]
    test_lpa = graphs_lpa[args.num_train_graphs:]

    train_graphs = train_sbm + train_lpa
    train_labels = [0] * len(train_sbm) + [1] * len(train_lpa)
    test_graphs = test_sbm + test_lpa
    test_labels = [0] * len(test_sbm) + [1] * len(test_lpa)

    train_dataset = GraphDataset(train_graphs, train_labels, args.seed, device)
    test_dataset = GraphDataset(test_graphs, test_labels, args.seed + 1, device)
    print(f"Train: {len(train_dataset)} graphs, Test: {len(test_dataset)} graphs")

    encoder = GNN_ENCODER[args.gnn](args.num_features, args.num_heads, args.num_mps).to(device)
    discriminator = Discriminator(graph_embed_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(discriminator.parameters()), lr=args.lr, eps=1e-4
    )
    num_batches_per_epoch = max(1, len(train_dataset) // args.batch_size)
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.num_epochs):
        encoder.train()
        discriminator.train()
        epoch_loss = 0.0
        for _ in range(num_batches_per_epoch):
            x, y = train_dataset.sample(args.batch_size, encoder)
            if x is None:
                continue
            logits = discriminator(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_acc, train_loss = evaluate(encoder, discriminator, train_dataset, device, args.batch_size)
        test_acc, test_loss = evaluate(encoder, discriminator, test_dataset, device, args.batch_size)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "num_features": args.num_features,
                    "num_heads": args.num_heads,
                    "num_mps": args.num_mps,
                    "gnn": args.gnn,
                    "epoch": epoch + 1,
                    "test_acc": test_acc,
                },
                os.path.join(args.save_dir, "discriminator_best.ckpt"),
            )
        print(
            f"Epoch {epoch+1}/{args.num_epochs} loss={epoch_loss/max(num_batches_per_epoch,1):.4f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} (best={best_acc:.4f} @ {best_epoch})"
        )

    print("Loading best checkpoint and final evaluation...")
    ckpt = torch.load(os.path.join(args.save_dir, "discriminator_best.ckpt"), map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    test_acc, test_loss = evaluate(encoder, discriminator, test_dataset, device, args.batch_size)
    print(f"Best model test accuracy: {test_acc:.4f}, test loss: {test_loss:.4f}")

    if args.visualize:
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            encoder.eval()
            discriminator.eval()
            all_emb = []
            all_labels = []
            with torch.no_grad():
                for start in range(0, len(test_dataset), args.batch_size):
                    indices = np.arange(start, min(start + args.batch_size, len(test_dataset)))
                    batch_graphs = [test_dataset.graphs[i] for i in indices]
                    g = Batch(device, [ig_to_data(gr) for gr in batch_graphs])
                    out = encoder(g)
                    embed_dim = out.shape[1] // 2
                    graph_emb = scatter_mean(out[:, embed_dim:], g.batch_non_omni, dim=0, dim_size=g.batch_size)
                    all_emb.append(graph_emb.cpu().numpy())
                    all_labels.append(test_dataset.labels[indices].cpu().numpy())
            X = np.vstack(all_emb)
            y = np.concatenate(all_labels)
            X_tsne = TSNE(n_components=2, random_state=args.seed).fit_transform(X)
            plt.figure(figsize=(8, 6))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis", alpha=0.7)
            plt.colorbar(label="0=SBM-like, 1=LPA-like")
            plt.title("Graph embeddings (t-SNE)")
            plt.savefig(os.path.join(args.save_dir, "discriminator_tsne.png"))
            plt.close()
            print(f"Saved t-SNE to {args.save_dir}/discriminator_tsne.png")
        except Exception as e:
            print(f"Visualization skipped: {e}")