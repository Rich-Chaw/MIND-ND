"""
Pre-train a graph-type encoder with InfoNCE: same dataset as sac_graph_task.
Groups: SBM+DCSBM (close), LPA+COPY (close), ER (single type).
Encoder = GNN + MLP (same architecture as TaskEncoder in sac_graph_task) for easy loading.
"""
import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timedelta
from torch_scatter import scatter_mean

from utils import load_g
from utils.graph_data import Batch, ig_to_data
from networks.gnn_interface import GNN_ENCODER


# --- Type -> group: SBM/DCSBM=0, LPA/COPY=1, ER=2 ---
TYPE_TO_GROUP = {"SBM": 0, "DCSBM": 0, "LPA": 1, "COPY": 1, "ER": 2}


def parse_type_from_name(name):
    """Graph name is e.g. '100_150_SBM_DCSBM_LPA_COPY_ER_6000_00001_SBM' -> type is last segment: SBM."""
    return name.split("_")[-1].upper()


def type_to_group(graph_type):
    """Map graph type string to group id. Unknown types -> None (filtered out in build)."""
    return TYPE_TO_GROUP.get(graph_type.upper())


def load_graphs_from_dir(data_dir, device=None):
    """Load graphs from directory; same convention as sac_graph_task / DismantleEnv."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    graph_data = []
    for p in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, p)
        if not os.path.isfile(path) or not p.endswith(".pkl"):
            continue
        name = f"{os.path.basename(data_dir)}_{os.path.splitext(os.path.basename(p))[0]}"
        g = load_g(path, name)
        graph_data.append(g)
    return graph_data


def build_graphs_and_groups(data_dir):
    """Load graphs and assign group id from name. Returns (graphs, group_ids), filtering unknown types."""
    graphs = load_graphs_from_dir(data_dir)
    groups = []
    kept = []
    for g in graphs:
        t = parse_type_from_name(g["name"])
        grp = type_to_group(t)
        if grp is None:
            continue
        kept.append(g)
        groups.append(grp)
    return kept, np.array(groups, dtype=np.int64)

class TaskEncoder(nn.Module):
    """
    Task encoder: encodes a batch of graphs (Batch) into task embeddings z of shape (B, latent_dim).
    Uses a GNN (same architecture as policy/Q) to get node embeddings, then mean-pools per graph,
    then a 2-layer MLP to produce z.
    """
    def __init__(self, num_features, num_heads, num_mps, gnn, hidden_dim=128, latent_dim=16):
        super().__init__()
        self.gnn = GNN_ENCODER[gnn](num_features, num_heads, num_mps)
        e_size = (num_features * num_mps) * 2  # same as policy/Q: node + omni concat
        self.mlp = nn.Sequential(
            nn.Linear(e_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, batch):
        """
        Args:
            batch: Batch of graphs (from utils or utils.graph_data)
        Returns:
            z: [batch_size, latent_dim] task embedding
        """
        node_emb = self.gnn(batch)  # [N, 2KF]
        graph_emb = scatter_mean(
            node_emb, batch.batch_non_omni, dim=0, dim_size=batch.batch_size,
        )  # [B, 2KF]
        z = self.mlp(graph_emb)  # [B, latent_dim]
        return z

    def get_z(self, batch):
        """Convenience method; same as forward(batch)."""
        return self.forward(batch)


# --- Dataset ---
class GraphGroupDataset:
    """(igraph graph, group_id). Groups: 0=SBM+DCSBM, 1=LPA+COPY, 2=ER."""

    def __init__(self, graphs, group_ids, seed, device):
        self.device = device
        self.rng = np.random.default_rng(seed)
        idx = self.rng.permutation(len(graphs))
        self.graphs = [graphs[i] for i in idx]
        self.group_ids = torch.as_tensor(group_ids[idx], device=device, dtype=torch.long)
        self.size = len(self.graphs)

    def sample(self, batch_size, encoder):
        if self.size == 0:
            return None, None
        n = min(batch_size, self.size)
        indices = self.rng.choice(self.size, size=n, replace=False)
        batch_graphs = [self.graphs[i] for i in indices]
        batch_groups = self.group_ids[indices]
        g = Batch(self.device, [ig_to_data(gr) for gr in batch_graphs])
        with torch.no_grad() if not encoder.training else torch.enable_grad():
            z = encoder(g)
        return z, batch_groups

    def __len__(self):
        return self.size


# --- InfoNCE: same group = positive, other groups = negative ---
def info_nce_loss_same_group(z, group_ids, temperature=0.1):
    """
    z: [B, D], group_ids: [B] in {0,1,2}.
    For anchor i, positives = {j != i : group_ids[j] == group_ids[i]}, negatives = rest.
    Loss_i = -log( sum_pos exp(sim_i_pos/tau) / sum_{j!=i} exp(sim_ij/tau) ).
    """
    B = z.shape[0]
    z_n = F.normalize(z, dim=1)
    logits = (z_n @ z_n.T) / max(temperature, 1e-8)
    # mask self-similarity out of denominator
    logits_no_self = logits.clone()
    logits_no_self.fill_diagonal_(-1e9)
    denom = torch.logsumexp(logits_no_self, dim=1)

    group_ids_np = group_ids.cpu().numpy()
    loss_list = []
    for i in range(B):
        gi = group_ids_np[i]
        pos_mask = (group_ids == gi) & (torch.arange(B, device=z.device) != i)
        if pos_mask.sum().item() == 0:
            continue
        num = torch.logsumexp(logits[i, pos_mask], dim=0)
        loss_list.append(denom[i] - num)
    if not loss_list:
        return torch.tensor(0.0, device=z.device)
    return torch.stack(loss_list).mean()


def evaluate_info_nce(encoder, dataset, device, batch_size=64, temperature=0.1):
    """Average InfoNCE loss over dataset."""
    if len(dataset) == 0:
        return 0.0
    encoder.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = np.arange(start, min(start + batch_size, len(dataset)))
            batch_graphs = [dataset.graphs[i] for i in indices]
            batch_groups = dataset.group_ids[indices]
            g = Batch(device, [ig_to_data(gr) for gr in batch_graphs])
            z = encoder(g)
            loss = info_nce_loss_same_group(z, batch_groups, temperature)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def get_embeddings(encoder, dataset, device, batch_size=64):
    """Collect embeddings and group_ids for a dataset. Returns (Z, group_ids) as numpy."""
    if len(dataset) == 0:
        return np.zeros((0, encoder.latent_dim)), np.array([], dtype=np.int64)
    encoder.eval()
    all_z, all_groups = [], []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = np.arange(start, min(start + batch_size, len(dataset)))
            batch_graphs = [dataset.graphs[i] for i in indices]
            batch_groups = dataset.group_ids[indices]
            g = Batch(device, [ig_to_data(gr) for gr in batch_graphs])
            z = encoder(g)
            all_z.append(z.cpu().numpy())
            all_groups.append(batch_groups.cpu().numpy())
    return np.vstack(all_z), np.concatenate(all_groups)


def visualize_embeddings(encoder, dataset, device, save_dir, batch_size=64, seed=42, dataset_label="valid"):
    """t-SNE of encoder embeddings on dataset, colored by graph type (SBM, DCSBM, LPA, COPY, ER). Saves plot to save_dir."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Visualization skipped (install sklearn, matplotlib): {e}")
        return
    Z, _ = get_embeddings(encoder, dataset, device, batch_size)
    n = Z.shape[0]
    if n == 0:
        print("No samples to visualize.")
        return
    if n < 5:
        print(f"Too few samples ({n}) for t-SNE, skipping.")
        return
    # Types in same order as Z (dataset order)
    types = np.array([parse_type_from_name(dataset.graphs[i]["name"]) for i in range(len(dataset))])
    perplexity = min(30, max(5, n - 1))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity)
    Z_2d = tsne.fit_transform(Z)
    type_order = ["SBM", "DCSBM", "LPA", "COPY", "ER"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in type_order:
        mask = types == t
        if mask.sum() == 0:
            continue
        ax.scatter(
            Z_2d[mask, 0], Z_2d[mask, 1],
            label=t, alpha=0.7, s=30
        )
    ax.legend()
    ax.set_title(f"Encoder embeddings (t-SNE) on {dataset_label} set by type (n={n})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    out_path = os.path.join(save_dir, "encoder_tsne_valid.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved t-SNE plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train graph-type encoder with InfoNCE (SBM+DCSBM close, LPA+COPY close, ER single)."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_dir",
        type=str,
        default="graphs/train/100_150_SBM_DCSBM_LPA_COPY_ER_6000_copy",
        help="Same as sac_graph_task train_dir",
    )
    parser.add_argument(
        "--valid_dir",
        type=str,
        default="graphs/valid/100_150_SBM_DCSBM_LPA_COPY_ER_100_copy",
        help="Optional; same as sac_graph_task valid_dir (set to '' to disable)",
    )
    parser.add_argument("--gnn", type=str, default="hgnn_v4", choices=list(GNN_ENCODER.keys()))
    parser.add_argument("--num_features", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_mps", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="saved/task_encoder")
    parser.add_argument("--visualize", action="store_true", help="After training, plot t-SNE of best ckpt on valid set")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, args.gnn,f"{time_string}")

    print("Loading graphs (same dataset as sac_graph_task)...")
    train_graphs, train_groups = build_graphs_and_groups(args.train_dir)
    print(f"Train: {len(train_graphs)} graphs, groups 0/1/2: {(train_groups==0).sum()}/{(train_groups==1).sum()}/{(train_groups==2).sum()}")

    train_dataset = GraphGroupDataset(train_graphs, train_groups, args.seed, device)
    valid_dataset = None
    if args.valid_dir and args.valid_dir.strip() and os.path.isdir(args.valid_dir):
        valid_graphs, valid_groups = build_graphs_and_groups(args.valid_dir)
        if len(valid_graphs) > 0:
            valid_dataset = GraphGroupDataset(valid_graphs, valid_groups, args.seed + 1, device)
            print(f"Valid: {len(valid_dataset)} graphs")

    encoder = TaskEncoder(
        args.num_features, args.num_heads, args.num_mps, args.gnn,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, eps=1e-4)
    num_batches = max(1, len(train_dataset) // args.batch_size)

    best_loss = float("inf")
    best_epoch = 0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        encoder.train()
        epoch_loss = 0.0
        for _ in range(num_batches):
            z, group_ids = train_dataset.sample(args.batch_size, encoder)
            if z is None:
                continue
            loss = info_nce_loss_same_group(z, group_ids, args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / max(num_batches, 1)
        val_loss = (
            evaluate_info_nce(encoder, valid_dataset, device, args.batch_size, args.temperature)
            if valid_dataset is not None
            else train_loss
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "num_features": args.num_features,
                    "num_heads": args.num_heads,
                    "num_mps": args.num_mps,
                    "gnn": args.gnn,
                    "hidden_dim": args.hidden_dim,
                    "latent_dim": args.latent_dim,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                },
                os.path.join(args.save_dir, "task_encoder_best.ckpt"),
            )
        print(
            f"Epoch {epoch+1}/{args.num_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"(best={best_loss:.4f} @ {best_epoch})"
        )

    print("Done. Load encoder into sac_graph_task TaskEncoder via encoder_state_dict (same architecture).")

    if args.visualize:
        ckpt_path = os.path.join(args.save_dir, "task_encoder_best.ckpt")
        if not os.path.isfile(ckpt_path):
            print("No best checkpoint found, skipping visualization.")
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            viz_dataset = valid_dataset if valid_dataset is not None else train_dataset
            viz_label = "valid" if valid_dataset is not None else "train"
            visualize_embeddings(
                encoder, viz_dataset, device, args.save_dir,
                args.batch_size, args.seed, dataset_label=viz_label
            )
