import argparse
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from networks.classical_gnns import GAT, GCN, GraphSAGE
from networks.gcnii import GCNII
from networks.idgnn import IDGCN
from networks.rfgnn import ResiflowGNN
from utils.graph_data import Batch, Graph

PLOT_CONFIG = {
    "font.family": ["Times New Roman", "SimHei"],
    "font.size": 20,
    "font.serif": ["SimHei"],
}
plt.rcParams.update(PLOT_CONFIG)
plt.rcParams["axes.unicode_minus"] = False


MODEL_REGISTRY = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
    "IDGNN": IDGCN,
    "GCNII": GCNII,
    "ResiFlow-GNN": ResiflowGNN,
}

MODEL_COLORS = {
    "GCN": "#1f77b4",
    "GraphSAGE": "#ff7f0e",
    "GAT": "#2ca02c",
    "IDGNN": "#9467bd",
    "GCNII": "#8c564b",
    "ResiFlow-GNN": "#d62728",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def nx_to_graph_data(g: nx.Graph) -> Graph:
    edges = np.array(g.edges(), dtype=np.int64)
    if edges.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        rev = edges[:, [1, 0]]
        both = np.concatenate([edges, rev], axis=0)
        edge_index = both.T
    return Graph(edge_index=edge_index, num_nodes=g.number_of_nodes())


def make_graph(graph_type: str, n: int, p: float, seed: int) -> Graph:
    if graph_type == "er":
        g_nx = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    elif graph_type == "ba":
        m = max(1, int(p * n))
        g_nx = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    elif graph_type == "ws":
        k = max(2, int(p * n))
        if k % 2 == 1:
            k += 1
        g_nx = nx.watts_strogatz_graph(n=n, k=min(k, n - 1), p=0.2, seed=seed)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")

    if g_nx.number_of_edges() == 0 and n > 1:
        g_nx.add_edge(0, 1)
    g_nx.remove_edges_from(nx.selfloop_edges(g_nx))
    return nx_to_graph_data(g_nx)


def build_batch(
    device: torch.device,
    graph_type: str,
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
    seed: int,
) -> Batch:
    graphs = []
    for i in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        graphs.append(make_graph(graph_type, n, edge_prob, seed + i))
    return Batch(device, graphs)


def make_input_features(
    batch: Batch,
    num_features: int,
    feature_init: str,
) -> torch.Tensor:
    if feature_init == "ones":
        return torch.ones(batch.total_nodes, num_features, device=batch.device, dtype=torch.float32)
    if feature_init == "random":
        return torch.randn(batch.total_nodes, num_features, device=batch.device, dtype=torch.float32)
    raise ValueError(f"Unsupported feature_init: {feature_init}")


def compute_collapse_distances(x: torch.Tensor, batch: Batch) -> np.ndarray:
    x_non = x[batch.non_omni_mask]
    if x_non.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    center = x_non.mean(dim=0, keepdim=True)
    d = torch.norm(x_non - center, p=2, dim=-1)
    return d.detach().cpu().numpy()


def compute_layer_metrics(x: torch.Tensor, batch: Batch) -> Dict[str, float]:
    x_non = x[batch.non_omni_mask]
    if x_non.shape[0] < 2:
        return {
            "dirichlet_energy": 0.0,
            "mean_pairwise_dist": 0.0,
            "mean_pairwise_cos": 1.0,
            "one_minus_pairwise_cos": 0.0,
        }

    src, dst = batch.edge_index.long()
    edge_diff = x[src] - x[dst]
    dirichlet = float(edge_diff.pow(2).sum(dim=-1).mean().item())

    pd = torch.pdist(x_non, p=2)
    mean_pairwise_dist = float(pd.mean().item()) if pd.numel() > 0 else 0.0

    xn = F.normalize(x_non, p=2, dim=-1, eps=1e-12)
    sim = xn @ xn.t()
    tri = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
    mean_pairwise_cos = float(sim[tri[0], tri[1]].mean().item())

    return {
        "dirichlet_energy": dirichlet,
        "mean_pairwise_dist": mean_pairwise_dist,
        "mean_pairwise_cos": mean_pairwise_cos,
        "one_minus_pairwise_cos": float(1.0 - mean_pairwise_cos),
    }


@torch.no_grad()
def extract_layer_embeddings_and_metrics(
    model_name: str,
    batch: Batch,
    input_x: torch.Tensor,
    num_features: int,
    num_heads: int,
    num_mps: int,
    use_l2_norm: bool,
) -> Dict[str, List]:
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        num_features=num_features,
        num_heads=num_heads,
        num_mps=num_mps,
        positional_encoding=None,
        handcrafted_features=False,
    ).to(batch.device)
    model.eval()

    x = input_x.clone()
    collapse_by_layer: List[np.ndarray] = []
    metric_list: List[Dict[str, float]] = []
    if model_name == "IDGNN":
        for conv in model.convs:
            x = conv(x, batch.edge_index, batch.omni_ids)
            x = F.relu(x)
            if use_l2_norm:
                x = F.normalize(x, p=2, dim=-1, eps=1e-12)
            collapse_by_layer.append(compute_collapse_distances(x, batch))
            metric_list.append(compute_layer_metrics(x, batch))
    elif model_name == "GCNII":
        adj = torch.sparse_coo_tensor(
            batch.edge_index,
            torch.ones(batch.edge_index.shape[1], device=batch.device),
            (batch.total_nodes, batch.total_nodes),
        ).coalesce()
        x = model.act_fn(model.fc_in(x))
        h0 = x.clone()
        for i, conv in enumerate(model.convs):
            x = conv(x, adj, h0, model.lamda, model.alpha, i + 1)
            x = F.relu(x)
            if use_l2_norm:
                x = F.normalize(x, p=2, dim=-1, eps=1e-12)
            collapse_by_layer.append(compute_collapse_distances(x, batch))
            metric_list.append(compute_layer_metrics(x, batch))
    elif hasattr(model, "convs"):
        for conv in model.convs:
            x = conv(x, batch.edge_index)
            x = F.relu(x)
            if use_l2_norm:
                x = F.normalize(x, p=2, dim=-1, eps=1e-12)
            collapse_by_layer.append(compute_collapse_distances(x, batch))
            metric_list.append(compute_layer_metrics(x, batch))
    elif hasattr(model, "layers"):
        x_0 = x.clone()
        for layer in model.layers:
            x = layer(x, x_0, batch.edge_index)
            if use_l2_norm:
                x = F.normalize(x, p=2, dim=-1, eps=1e-12)
            collapse_by_layer.append(compute_collapse_distances(x, batch))
            metric_list.append(compute_layer_metrics(x, batch))
    else:
        raise ValueError(f"Unsupported model architecture for {model_name}")
    return {"collapse_distances": collapse_by_layer, "metrics": metric_list}


def plot_oversmoothing_proof(results_by_model: Dict[str, Dict[str, List]], save_path: str) -> None:
    model_names = list(results_by_model.keys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_cos, ax_dir = axes[0], axes[1]

    for model_name in model_names:
        color = MODEL_COLORS[model_name]
        metrics: List[Dict[str, float]] = results_by_model[model_name]["metrics"]
        xs = np.arange(1, len(metrics) + 1)
        cos = np.array([m["mean_pairwise_cos"] for m in metrics], dtype=np.float64)
        dir_e = np.array([m["dirichlet_energy"] for m in metrics], dtype=np.float64)

        ax_cos.plot(xs, cos, color=color, lw=2.6, label=model_name)
        ax_dir.plot(xs, dir_e + 1e-16, color=color, lw=2.6, label=model_name)
        ax_cos.scatter([1], [cos[0]], color=color, s=70, zorder=3)
        ax_dir.scatter([1], [dir_e[0] + 1e-16], color=color, s=70, zorder=3)

    ax_cos.set_title("平均余弦相似度曲线", fontsize=20, fontweight="medium")
    ax_cos.set_xlabel("Layer")
    ax_cos.set_ylabel("Mean pairwise cosine")
    ax_cos.set_ylim(0.0, 1.02)
    ax_cos.grid(alpha=0.25)
    ax_cos.legend(fontsize=14)

    ax_dir.set_yscale("log")
    ax_dir.set_title("Dirichlet 能量曲线（对数坐标）", fontsize=20, fontweight="medium")
    ax_dir.set_xlabel("Layer")
    ax_dir.set_ylabel("Dirichlet energy")
    ax_dir.grid(alpha=0.25)
    ax_dir.legend(fontsize=14)

    # fig.suptitle("Oversmoothing diagnostics", fontsize=22, y=1.03)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    print(f"Saved to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot oversmoothing curves: mean cosine similarity and Dirichlet energy."
    )
    parser.add_argument("--graph_type", type=str, default="ba", choices=["er", "ba", "ws"])
    parser.add_argument("--num_graphs", type=int, default=1)
    parser.add_argument("--min_nodes", type=int, default=500)
    parser.add_argument("--max_nodes", type=int, default=500)
    parser.add_argument("--edge_prob", type=float, default=0.02)
    parser.add_argument("--num_features", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_mps", type=int, default=24, help="Maximum message passing layers (L_max).")
    parser.add_argument("--feature_init", type=str, default="ones", choices=["ones", "random"])
    parser.add_argument("--disable_l2_norm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default="oversmoothing_proof.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    batch = build_batch(
        device=device,
        graph_type=args.graph_type,
        num_graphs=args.num_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        edge_prob=args.edge_prob,
        seed=args.seed,
    )
    input_x = make_input_features(batch=batch, num_features=args.num_features, feature_init=args.feature_init)

    print(
        f"Setting: graph_type={args.graph_type}, graphs={args.num_graphs}, "
        f"nodes=[{args.min_nodes},{args.max_nodes}], F={args.num_features}, "
        f"L_max={args.num_mps}, init={args.feature_init}, "
        f"L2_norm={'off' if args.disable_l2_norm else 'on'}"
    )

    results_by_model: Dict[str, Dict[str, List]] = {}
    for model_name in ["GCN", "GraphSAGE", "GAT", "IDGNN", "GCNII", "ResiFlow-GNN"]:
    # for model_name in ["GCN", "GraphSAGE", "GAT"]:
        results_by_model[model_name] = extract_layer_embeddings_and_metrics(
            model_name=model_name,
            batch=batch,
            input_x=input_x,
            num_features=args.num_features,
            num_heads=args.num_heads,
            num_mps=args.num_mps,
            use_l2_norm=not args.disable_l2_norm,
        )
        print(f"Collected layer embeddings for {model_name.upper()}.")

    plot_oversmoothing_proof(results_by_model=results_by_model, save_path=args.save_path)
    print("Interpretation: 余弦相似度若快速接近 1 且 Dirichlet 能量长期处于极小量级，可判定模型已发生过平滑。")


if __name__ == "__main__":
    main()
