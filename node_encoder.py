"""
Supervised node encoder: predict degree and betweenness per node.
Loss = degree reconstruction loss + betweenness reconstruction loss.
Dataset is not divided into groups; val loss is printed per type for analysis.
Visualization: one random graph per type, highlight top-10 predicted degree and top-10 predicted betweenness.
"""
import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
from torch_scatter import scatter_mean

from utils import load_g
from utils.graph_data import Batch, ig_to_data
from networks.gnn import GNN_ENCODER


def parse_type_from_name(name):
    """Graph name e.g. '100_150_..._00001_SBM' -> type is last segment: SBM."""
    return name.split("_")[-1].upper()


def load_graphs_from_dir(data_dir):
    """Load graphs from directory (same convention as task_encoder)."""
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


def compute_degree_betweenness(g):
    """
    g: igraph (original, before omni). Returns (degree, betweenness) for each node.
    degree: list of int; betweenness: list of float.
    We normalize: target_degree = log(1 + degree), target_betweenness = betweenness / (max + 1e-8).
    """
    degree = np.array(g.degree(), dtype=np.float32)
    betweenness = np.array(g.betweenness(), dtype=np.float32)
    target_degree = np.log(1.0 + degree)
    max_bc = float(np.max(betweenness)) + 1e-8
    target_betweenness = betweenness / max_bc
    return target_degree, target_betweenness


class NodeEncoder(nn.Module):
    """
    Node encoder: GNN -> per-node embedding -> MLP -> (pred_degree, pred_betweenness).
    Same GNN as TaskEncoder; output is per-node (N, 2) for reconstruction.
    """

    def __init__(self, num_features, num_heads, num_mps, gnn, hidden_dim=128):
        super().__init__()
        self.gnn = GNN_ENCODER[gnn](num_features, num_heads, num_mps)
        e_size = (num_features * num_mps) * 2
        self.mlp = nn.Sequential(
            nn.Linear(e_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch):
        """
        batch: Batch
        Returns: pred [N, 2] (degree, betweenness) in normalized space.
        """
        node_emb = self.gnn(batch)  # [N, 2KF]
        pred = self.mlp(node_emb)   # [N, 2]
        return pred


# --- Dataset (no groups) ---
class NodeEncoderDataset:
    """List of graphs (igraph with name). No group split."""

    def __init__(self, graphs, seed, device):
        self.device = device
        self.rng = np.random.default_rng(seed)
        idx = self.rng.permutation(len(graphs))
        self.graphs = [graphs[i] for i in idx]
        self.size = len(self.graphs)

    def get_batch(self, indices, encoder):
        """Build Batch and targets (degree, betweenness) for given indices."""
        if not indices:
            return None, None, None
        batch_graphs = [self.graphs[i] for i in indices]
        graph_datas = [ig_to_data(g) for g in batch_graphs]
        batch = Batch(self.device, graph_datas)
        # Targets: concatenate degree and betweenness per graph (original nodes only)
        deg_list, bc_list = [], []
        for g in batch_graphs:
            td, tb = compute_degree_betweenness(g)
            deg_list.append(td)
            bc_list.append(tb)
        target_degree = torch.tensor(np.concatenate(deg_list), dtype=torch.float32, device=self.device)
        target_betweenness = torch.tensor(np.concatenate(bc_list), dtype=torch.float32, device=self.device)
        return batch, target_degree, target_betweenness

    def __len__(self):
        return self.size


def reconstruction_loss(pred, target_degree, target_betweenness):
    """pred [N, 2]: (degree, betweenness). MSE on both."""
    loss_degree = F.mse_loss(pred[:, 0], target_degree)
    loss_betweenness = F.mse_loss(pred[:, 1], target_betweenness)
    return loss_degree + loss_betweenness, loss_degree.item(), loss_betweenness.item()


def evaluate(encoder, dataset, device, batch_size=64):
    """Average reconstruction loss over dataset. Returns total, loss_degree, loss_betweenness."""
    if len(dataset) == 0:
        return 0.0, 0.0, 0.0
    encoder.eval()
    total_loss = 0.0
    total_deg = 0.0
    total_bc = 0.0
    n_batches = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            indices = list(range(start, end))
            batch, td, tb = dataset.get_batch(indices, encoder)
            if batch is None:
                continue
            pred = encoder(batch)
            loss, ld, lb = reconstruction_loss(pred, td, tb)
            total_loss += loss.item()
            total_deg += ld
            total_bc += lb
            n_batches += 1
    n = max(n_batches, 1)
    return total_loss / n, total_deg / n, total_bc / n


def evaluate_by_type(encoder, dataset, device, batch_size=64):
    """
    Compute val loss per graph type. Returns dict type -> (loss, loss_degree, loss_betweenness)
    and list of (type, indices in dataset).
    """
    if len(dataset) == 0:
        return {}, []
    types = np.array([parse_type_from_name(dataset.graphs[i]["name"]) for i in range(len(dataset))])
    unique_types = np.unique(types)
    encoder.eval()
    results = {}
    with torch.no_grad():
        for t in unique_types:
            mask = types == t
            indices = np.where(mask)[0].tolist()
            if not indices:
                continue
            # Process in mini-batches to avoid OOM
            total_loss = 0.0
            total_deg = 0.0
            total_bc = 0.0
            n_b = 0
            for start in range(0, len(indices), batch_size):
                sub = indices[start : start + batch_size]
                batch, td, tb = dataset.get_batch(sub, encoder)
                if batch is None:
                    continue
                pred = encoder(batch)
                loss, ld, lb = reconstruction_loss(pred, td, tb)
                total_loss += loss.item()
                total_deg += ld
                total_bc += lb
                n_b += 1
            if n_b > 0:
                results[t] = (total_loss / n_b, total_deg / n_b, total_bc / n_b)
    return results, list(results.keys())


def visualize_one_graph(encoder, g_igraph, device, save_path, top_k=10):
    """
    Draw one graph: highlight top-k highest predicted degree and top-k highest predicted betweenness.
    g_igraph: igraph (original, no omni) with g['name'].
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping visualization.")
        return
    encoder.eval()
    n = g_igraph.vcount()
    if n == 0:
        return
    # Single-graph batch
    graph_data = ig_to_data(g_igraph)
    batch = Batch(device, [graph_data])
    with torch.no_grad():
        pred = encoder(batch)  # [n, 2]
    pred_deg = pred[:, 0].cpu().numpy()
    pred_bc = pred[:, 1].cpu().numpy()
    top_degree_nodes = np.argsort(pred_deg)[::-1][:top_k].tolist()
    top_betweenness_nodes = np.argsort(pred_bc)[::-1][:top_k].tolist()
    # Layout and draw with igraph
    try:
        layout = g_igraph.layout_fruchterman_reingold()
        coords = np.array(layout.coords)
    except Exception:
        coords = np.random.randn(n, 2).astype(np.float32)
    # Build edge list for plotting (igraph may be directed after ig_to_data; use original edges)
    el = g_igraph.get_edgelist()
    if not el:
        edges = np.array([[], []]).T
    else:
        edges = np.array(el)
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot edges
    for e in edges:
        i, j = e[0], e[1]
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], "k-", lw=0.5, alpha=0.5, zorder=0)
    # Default nodes
    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=20, zorder=1, label="node")
    # Top degree
    td_x = coords[top_degree_nodes, 0]
    td_y = coords[top_degree_nodes, 1]
    ax.scatter(td_x, td_y, c="C0", s=120, marker="o", edgecolors="darkblue", linewidths=2, zorder=2, label=f"top-{top_k} pred degree")
    # Top betweenness
    tb_x = coords[top_betweenness_nodes, 0]
    tb_y = coords[top_betweenness_nodes, 1]
    ax.scatter(tb_x, tb_y, c="C1", s=120, marker="s", edgecolors="darkred", linewidths=2, zorder=2, label=f"top-{top_k} pred betweenness")
    ax.legend()
    ax.set_title(f"Type {parse_type_from_name(g_igraph['name'])} (n={n})")
    ax.axis("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


def visualize_per_type(encoder, dataset, device, save_dir, seed=42, top_k=10):
    """Random sample one graph per type; visualize top-k predicted degree and betweenness."""
    if len(dataset) == 0:
        return
    types = np.array([parse_type_from_name(dataset.graphs[i]["name"]) for i in range(len(dataset))])
    unique_types = np.unique(types).tolist()
    rng = np.random.default_rng(seed)
    os.makedirs(save_dir, exist_ok=True)
    for t in unique_types:
        mask = types == t
        indices = np.where(mask)[0]
        idx = int(rng.choice(indices))
        g = dataset.graphs[idx]
        out_path = os.path.join(save_dir, f"node_encoder_viz_{t}.png")
        visualize_one_graph(encoder, g, device, out_path, top_k=top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train node encoder with degree + betweenness reconstruction loss."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_dir",
        type=str,
        default="graphs/train/100_150_SBM_DCSBM_LPA_COPY_ER_6000_copy",
        help="Training graphs directory",
    )
    parser.add_argument(
        "--valid_dir",
        type=str,
        default="graphs/valid/100_150_SBM_DCSBM_LPA_COPY_ER_100_copy",
        help="Validation directory (empty to disable)",
    )
    parser.add_argument("--gnn", type=str, default="hgnn_v4", choices=list(GNN_ENCODER.keys()))
    parser.add_argument("--num_features", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_mps", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="saved/node_encoder")
    parser.add_argument("--visualize", action="store_true", help="After training, plot one graph per type with top-10 degree/betweenness")
    parser.add_argument("--viz_top_k", type=int, default=10, help="Top-k nodes to highlight in visualization")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, args.gnn,f"{time_string}")

    print("Loading graphs...")
    train_graphs = load_graphs_from_dir(args.train_dir)
    print(f"Train: {len(train_graphs)} graphs")

    train_dataset = NodeEncoderDataset(train_graphs, args.seed, device)
    valid_dataset = None
    if args.valid_dir and args.valid_dir.strip() and os.path.isdir(args.valid_dir):
        valid_graphs = load_graphs_from_dir(args.valid_dir)
        if len(valid_graphs) > 0:
            valid_dataset = NodeEncoderDataset(valid_graphs, args.seed + 1, device)
            print(f"Valid: {len(valid_dataset)} graphs")

    encoder = NodeEncoder(
        args.num_features, args.num_heads, args.num_mps, args.gnn,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, eps=1e-4)
    num_batches = max(1, len(train_dataset) // args.batch_size)

    best_loss = float("inf")
    best_epoch = 0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        encoder.train()
        epoch_loss = 0.0
        epoch_deg = 0.0
        epoch_bc = 0.0
        for _ in range(num_batches):
            indices = np.random.choice(len(train_dataset), size=min(args.batch_size, len(train_dataset)), replace=False)
            batch, target_degree, target_betweenness = train_dataset.get_batch(indices.tolist(), encoder)
            if batch is None:
                continue
            pred = encoder(batch)
            loss, ld, lb = reconstruction_loss(pred, target_degree, target_betweenness)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_deg += ld
            epoch_bc += lb

        train_loss = epoch_loss / max(num_batches, 1)
        train_deg = epoch_deg / max(num_batches, 1)
        train_bc = epoch_bc / max(num_batches, 1)

        if valid_dataset is not None:
            val_loss, val_deg, val_bc = evaluate(encoder, valid_dataset, device, args.batch_size)
            loss_by_type, type_list = evaluate_by_type(encoder, valid_dataset, device, args.batch_size)
        else:
            val_loss, val_deg, val_bc = train_loss, train_deg, train_bc
            loss_by_type, type_list = {}, []

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
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                },
                os.path.join(args.save_dir, "node_encoder_best.ckpt"),
            )

        print(
            f"Epoch {epoch+1}/{args.num_epochs} train_loss={train_loss:.4f} (deg={train_deg:.4f} bc={train_bc:.4f}) "
            f"val_loss={val_loss:.4f} (deg={val_deg:.4f} bc={val_bc:.4f}) best={best_loss:.4f} @ {best_epoch}"
        )
        # Print val loss by type for analysis
        if loss_by_type:
            parts = ["  val_by_type:"]
            for t in type_list:
                tl, td, tb = loss_by_type[t]
                parts.append(f" {t}={tl:.4f}(deg={td:.4f},bc={tb:.4f})")
            print("".join(parts))

    print("Done.")

    if args.visualize and valid_dataset is not None:
        ckpt_path = os.path.join(args.save_dir, "node_encoder_best.ckpt")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            viz_dir = os.path.join(args.save_dir, "viz_per_type")
            visualize_per_type(
                encoder, valid_dataset, device, viz_dir,
                seed=args.seed, top_k=args.viz_top_k,
            )
        else:
            print("No best checkpoint found, skipping visualization.")
