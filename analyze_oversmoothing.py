import argparse
import random
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from networks.classical_gnns import GAT, GCN, GraphSAGE
from utils.graph_data import Batch, Graph


MODEL_REGISTRY = {
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "gat": GAT,
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


def graphwise_metrics(x: torch.Tensor, batch: Batch) -> Dict[str, float]:
    x_non = x[batch.non_omni_mask]
    b_non = batch.batch_non_omni

    vars_per_graph: List[float] = []
    cos_per_graph: List[float] = []

    for gid in range(batch.batch_size):
        mask = b_non == gid
        xi = x_non[mask]
        if xi.shape[0] < 2:
            continue

        var_val = xi.var(dim=0, unbiased=False).mean().item()
        vars_per_graph.append(var_val)

        xi_norm = F.normalize(xi, p=2, dim=-1, eps=1e-12)
        sim = xi_norm @ xi_norm.t()
        tri = torch.triu_indices(sim.size(0), sim.size(1), offset=1, device=sim.device)
        cos_mean = sim[tri[0], tri[1]].mean().item()
        cos_per_graph.append(cos_mean)

    return {
        "mean_feature_var": float(np.mean(vars_per_graph)) if vars_per_graph else 0.0,
        "mean_pairwise_cos": float(np.mean(cos_per_graph)) if cos_per_graph else 1.0,
    }


@torch.no_grad()
def run_oversmoothing_probe(
    model_name: str,
    batch: Batch,
    num_features: int,
    num_heads: int,
    num_mps: int,
    use_l2_norm: bool,
) -> List[Tuple[int, Dict[str, float]]]:
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        num_features=num_features,
        num_heads=num_heads,
        num_mps=num_mps,
        positional_encoding=None,
        handcrafted_features=False,
    ).to(batch.device)
    model.eval()

    x = torch.ones(batch.total_nodes, model.num_features, device=batch.device, dtype=torch.float32)
    layer_stats: List[Tuple[int, Dict[str, float]]] = []

    init_metrics = graphwise_metrics(x, batch)
    layer_stats.append((0, init_metrics))

    for lid, conv in enumerate(model.convs, start=1):
        x = conv(x, batch.edge_index)
        x = F.relu(x)
        if use_l2_norm:
            x = F.normalize(x, p=2, dim=-1, eps=1e-12)
        layer_stats.append((lid, graphwise_metrics(x, batch)))

    return layer_stats


def print_table(model_name: str, stats: List[Tuple[int, Dict[str, float]]]) -> None:
    print(f"\n=== {model_name.upper()} ===")
    print("layer\tmean_feature_var\tmean_pairwise_cos")
    for layer_id, m in stats:
        print(f"{layer_id}\t{m['mean_feature_var']:.6f}\t{m['mean_pairwise_cos']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe oversmoothing under all-ones node features.")
    parser.add_argument("--gnn", type=str, default="all", choices=["all", "gcn", "graphsage", "gat"])
    parser.add_argument("--graph_type", type=str, default="er", choices=["er", "ba", "ws"])
    parser.add_argument("--num_graphs", type=int, default=16)
    parser.add_argument("--min_nodes", type=int, default=80)
    parser.add_argument("--max_nodes", type=int, default=140)
    parser.add_argument("--edge_prob", type=float, default=0.05)
    parser.add_argument("--num_features", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_mps", type=int, default=8)
    parser.add_argument("--disable_l2_norm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
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

    model_names = list(MODEL_REGISTRY.keys()) if args.gnn == "all" else [args.gnn]
    print(
        f"Probe setting: graphs={args.num_graphs}, nodes=[{args.min_nodes},{args.max_nodes}], "
        f"graph_type={args.graph_type}, F={args.num_features}, K={args.num_mps}, "
        f"L2_norm={'off' if args.disable_l2_norm else 'on'}"
    )

    for model_name in model_names:
        stats = run_oversmoothing_probe(
            model_name=model_name,
            batch=batch,
            num_features=args.num_features,
            num_heads=args.num_heads,
            num_mps=args.num_mps,
            use_l2_norm=not args.disable_l2_norm,
        )
        print_table(model_name, stats)

    print("\nInterpretation:")
    print("- mean_feature_var 趋近 0 代表节点表征越来越相同（过平滑增强）。")
    print("- mean_pairwise_cos 趋近 1 代表节点间方向越来越一致（过平滑增强）。")


if __name__ == "__main__":
    main()
