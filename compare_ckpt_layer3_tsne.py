import argparse
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from networks.dismantle import load_dismantler
from utils.common import load_g
from utils.graph_data import Batch, ig_to_data


DEFAULT_CKPTS = [
    "saved/rfgnn/sac_teacher_betweenness_20260310_083133/24999.ckpt",
    "saved/gin/sac_20260328_001514/19999.ckpt",
    "saved/gcn/sac_teacher_20260326_134020/9999.ckpt",
    "saved/graphsage/sac_teacher_20260324_224237/33999.ckpt",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_batch(graph_path: str, device: torch.device) -> Batch:
    name = os.path.splitext(os.path.basename(graph_path))[0]
    g = load_g(graph_path, name=name)
    data = ig_to_data(g)
    return Batch(device, [data])


def sample_rows(x: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if x.shape[0] <= max_points:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_points, replace=False)
    return x[idx]


def extract_layer_embedding(
    ckpt_path: str,
    batch: Batch,
    layer_idx_1based: int,
    max_points: int,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, int]]:
    policy = load_dismantler(ckpt_pth=ckpt_path, device=device)
    policy.eval()

    with torch.no_grad():
        emb = policy.graph_embedding(batch)  # [N_non_omni, 2*K*F]

    total_dim = emb.shape[1]
    node_dim = total_dim // 2
    k = policy.graph_embedding.num_mps
    if node_dim % k != 0:
        raise ValueError(
            f"维度无法按层切分: ckpt={ckpt_path}, total_dim={total_dim}, node_dim={node_dim}, K={k}"
        )

    per_layer_dim = node_dim // k
    if layer_idx_1based < 1 or layer_idx_1based > k:
        raise ValueError(f"layer_idx_1based={layer_idx_1based} 超出范围 1..{k}")

    s = (layer_idx_1based - 1) * per_layer_dim
    e = layer_idx_1based * per_layer_dim
    layer_emb = emb[:, s:e].detach().cpu().numpy()
    layer_emb = sample_rows(layer_emb, max_points=max_points, seed=seed)

    meta = {
        "num_nodes_used": int(layer_emb.shape[0]),
        "per_layer_dim": int(per_layer_dim),
        "num_mps": int(k),
    }
    return layer_emb, meta


def plot_joint_tsne(
    features_list: List[np.ndarray],
    labels: List[str],
    output_path: str,
    perplexity: float,
    random_state: int,
) -> None:
    x = np.concatenate(features_list, axis=0)
    y = np.concatenate(
        [np.full(feat.shape[0], i, dtype=np.int64) for i, feat in enumerate(features_list)],
        axis=0,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    z = tsne.fit_transform(x)

    plt.figure(figsize=(11, 8))
    cmap = plt.get_cmap("tab10")
    for i, label in enumerate(labels):
        mask = y == i
        plt.scatter(
            z[mask, 0],
            z[mask, 1],
            s=6,
            alpha=0.65,
            color=cmap(i % 10),
            label=label,
        )
    plt.title("t-SNE of Layer-3 Node Embeddings Across Checkpoints")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare layer-3 embeddings across ckpts via t-SNE.")
    parser.add_argument(
        "--graph_path",
        type=str,
        default="graphs/real/information/dblp-cite.pkl",
        help="Graph pkl path.",
    )
    parser.add_argument(
        "--ckpts",
        nargs="+",
        default=DEFAULT_CKPTS,
        help="Checkpoint paths to compare.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--layer", type=int, default=3, help="1-based layer index.")
    parser.add_argument("--max_points_per_ckpt", type=int, default=3000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="results/tsne_layer3_ckpt_compare_dblp_cite.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch = build_batch(args.graph_path, device=device)

    features = []
    labels = []
    print(f"Graph: {args.graph_path}")
    print(f"Total non-omni nodes: {int(batch.non_omni_mask.sum().item())}")
    print("")

    for ckpt in args.ckpts:
        emb, meta = extract_layer_embedding(
            ckpt_path=ckpt,
            batch=batch,
            layer_idx_1based=args.layer,
            max_points=args.max_points_per_ckpt,
            seed=args.seed,
            device=device,
        )
        features.append(emb)
        label = os.path.relpath(ckpt).replace("\\", "/")
        labels.append(label)
        print(
            f"[OK] {label} | K={meta['num_mps']} | layer_dim={meta['per_layer_dim']} | "
            f"used_nodes={meta['num_nodes_used']}"
        )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_joint_tsne(
        features_list=features,
        labels=labels,
        output_path=args.output,
        perplexity=args.perplexity,
        random_state=args.seed,
    )

    print("")
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
