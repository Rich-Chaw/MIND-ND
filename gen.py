"""
图生成脚本：多源训练池 + GNN-VAE（Fermi-Dirac 解码器）+ 去噪训练，生成 200 节点图。
参考 gen.py 中的实验步骤，代码风格参考 sac.py。
"""
import os
import json
import pickle
import random
import igraph as ig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from dataclasses import dataclass, field
from datetime import datetime
from utils import load_g
from networks.hgcn import HGCNEncoder, PoincareBall, HypLinear
from utils.graph_data import ig_to_data
from utils.graph_models import NPSO, BA, WS


def _perturb_edges(g: ig.Graph, add_ratio: float = 0.05, remove_ratio: float = 0.05, rng: np.random.Generator = None):
    """对图施加 5%~10% 的边扰动（随机加边/删边）。"""
    rng = rng or np.random.default_rng()
    n = g.vcount()
    el = g.get_edgelist()
    edge_set = set((min(u, v), max(u, v)) for u, v in el)
    m = len(edge_set)
    to_remove = int(m * remove_ratio)
    to_add = int(m * add_ratio)
    arr = np.array(list(edge_set))
    if to_remove and len(arr) > 0:
        ridx = rng.choice(len(arr), size=min(to_remove, len(arr)), replace=False)
        for i in ridx:
            edge_set.discard((int(arr[i, 0]), int(arr[i, 1])))
    for _ in range(to_add):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            edge_set.add((min(u, v), max(u, v)))
    g2 = ig.Graph(n=n)
    g2.add_edges(list(edge_set))
    return g2


def _ig_to_edge_index(g: ig.Graph):
    """用 ig_to_data 得到图数据，去掉 omni 边后得到 (edge_index, n) 供 VAE 使用。"""
    d = ig_to_data(g)
    mask = (d.edge_index[0] < d.num_nodes) & (d.edge_index[1] < d.num_nodes)
    return d.edge_index[:, mask], d.num_nodes


def build_pool(
    pool_size: int,
    n_min: int,
    n_max: int,
    real_dirs: list,
    edge_perturb: float = 0.07,
    rng: np.random.Generator = None,
):
    """构建结构异质训练池：真实图（若存在）+ 噪声 NPSO/BA/WS。"""
    rng = rng or np.random.default_rng()
    graphs = []
    # 真实网络
    for d in real_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d))[: pool_size // 4]:
            path = os.path.join(d, f)
            if not os.path.isfile(path) or not f.endswith(".pkl"):
                continue
            try:
                g = load_g(path, name=f)
                if g.vcount() < n_min or g.vcount() > n_max * 2:
                    continue
                if g.vcount() > n_max:
                    g = g.subgraph(rng.choice(g.vcount(), n_max, replace=False))
                g.simplify()
                graphs.append(g)
            except Exception:
                pass
    # 噪声规则图
    need = max(0, pool_size - len(graphs))
    for _ in range(need):
        n = int(rng.integers(n_min, n_max + 1))
        kind = rng.choice(["npso", "ba", "ws"])
        if kind == "npso":
            m = max(1, rng.integers(2, min(10, n // 20)))
            g = NPSO(n, m=m, beta=float(rng.uniform(0.5, 1.0)))
        elif kind == "ba":
            m = max(1, min(5, n // 50))
            g = BA(n, m)
        else:
            k = max(2, min(10, n // 20))
            p = float(rng.uniform(0.01, 0.3))
            g = WS(n, k=k, p=p)
        g = _perturb_edges(g, add_ratio=edge_perturb, remove_ratio=edge_perturb, rng=rng)
        graphs.append(g)
    return graphs


class GraphVAE(nn.Module):
    """GNN-VAE：3 层 HGCN 编码 -> 双曲 μ, σ；Fermi-Dirac 双曲距离解码 P(A_ij)=1/(exp((d_L(z_i,z_j)-r)/τ)+1)。"""

    def __init__(self, in_dim: int, hidden: int, latent: int, r: float = 1.0, tau: float = 0.1, c: float = 1.0):
        super().__init__()
        self.latent = latent
        self.r, self.tau, self.c = r, tau, c
        self.manifold = PoincareBall()
        self.encoder = HGCNEncoder(in_dim, hidden, c=c, dropout=0.0, num_layers=3)
        self.mu_hyp = HypLinear(self.manifold, hidden, latent, c, 0.0, True)
        self.logvar_lin = nn.Linear(hidden, latent)

    def encode(self, x, edge_index, n):
        h = self.encoder(x, edge_index, n)
        mu = self.manifold.proj(self.mu_hyp(h), self.c)
        h_tan = self.manifold.logmap0(h, self.c)
        logvar = self.logvar_lin(h_tan)
        return mu, logvar

    def reparam(self, mu, logvar):
        mu_tan = self.manifold.logmap0(mu, self.c)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_tan = mu_tan + eps * std
        z = self.manifold.proj(self.manifold.expmap0(z_tan, self.c), self.c)
        return z

    def decode(self, z, edge_index):
        """双曲距离 d_L(z_i,z_j)，Fermi-Dirac: P(A_ij=1) = 1/(exp((d_L - r)/τ) + 1)。"""
        i, j = edge_index[0], edge_index[1]
        d = self.manifold.dist(z[i], z[j], self.c)
        return torch.sigmoid(-(d - self.r) / self.tau)

    def forward(self, x, edge_index, n):
        mu, logvar = self.encode(x, edge_index, n)
        z = self.reparam(mu, logvar)
        return z, mu, logvar


def _train_step(model, optimizer, ei, n, device, edge_drop: float, beta: float, rng: np.random.Generator):
    """单步：随机丢弃部分边做去噪，重建 + KL。"""
    model.train()
    ei = torch.tensor(ei, dtype=torch.long, device=device)
    x = torch.ones(n, 1, device=device)
    # 去噪：随机丢弃部分边
    E = ei.shape[1]
    if edge_drop > 0 and E > 0:
        keep = rng.random(E) > edge_drop
        ei_c = ei[:, keep]
    else:
        ei_c = ei
    z, mu, logvar = model(x, ei_c, n)
    p_pos = model.decode(z, ei)
    n_neg = min(ei.shape[1] * 2, n * n // 2)
    neg_i = torch.tensor(rng.integers(0, n, n_neg), device=device, dtype=torch.long)
    neg_j = torch.tensor(rng.integers(0, n, n_neg), device=device, dtype=torch.long)
    neg_ei = torch.stack([neg_i, neg_j], dim=0)
    p_neg = model.decode(z, neg_ei)
    recon = -(
        torch.log(p_pos.clamp(1e-8, 1 - 1e-8)).mean()
        + torch.log((1 - p_neg).clamp(1e-8, 1 - 1e-8)).mean()
    ) / 2
    mu_tan = model.manifold.logmap0(mu, model.c)
    kl = -0.5 * (1 + logvar - mu_tan.pow(2) - logvar.exp()).sum(dim=1).mean()
    loss = recon + beta * kl
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(model, optimizer, pool, device, edge_drop: float, beta: float, rng: np.random.Generator):
    indices = rng.permutation(len(pool))
    total_loss = 0.0
    for idx in indices:
        g = pool[idx]
        ei, n = _ig_to_edge_index(g)
        if ei.shape[1] == 0:
            continue
        total_loss += _train_step(model, optimizer, ei, n, device, edge_drop, beta, rng)
    return total_loss / max(len(pool), 1)


def generate_graph(model, n_nodes: int, device, rng: np.random.Generator = None):
    """从双曲隐空间采样 n_nodes 个点，用 Fermi-Dirac 双曲距离解码得到邻接矩阵。"""
    rng = rng or np.random.default_rng()
    model.eval()
    m, c, r, tau = model.manifold, model.c, model.r, model.tau
    with torch.no_grad():
        z_tan = torch.tensor(rng.standard_normal((n_nodes, model.latent)), dtype=torch.float32, device=device) * 0.1
        z = m.proj(m.expmap0(z_tan, c), c)
        d = m.dist_matrix(z, c)
        p = torch.sigmoid(-(d - r) / tau)
        triu = torch.triu(p, diagonal=1)
        thresh = float(triu.mean().item())
        ei = (triu > thresh).nonzero(as_tuple=False).t()
        ei_np = ei.cpu().numpy()
        edge_list = list(zip(ei_np[0].tolist(), ei_np[1].tolist()))
    g = ig.Graph(n=n_nodes)
    if edge_list:
        g.add_edges(edge_list)
    g.simplify()
    return g


@dataclass
class Args:
    seed: int = 0
    device: str = "cuda:0"
    pool_size: int = 600
    n_min: int = 500
    n_max: int = 1000
    latent_dim: int = 16
    hidden_dim: int = 64
    r: float = 1.0
    tau: float = 0.1
    beta: float = 0.1
    edge_drop: float = 0.2
    edge_perturb: float = 0.07
    epochs: int = 50
    lr: float = 1e-3
    gen_nodes: int = 200
    real_dirs: list = field(default_factory=lambda: [
        "graphs/real/tech",
        "graphs/real/bio",
        "graphs/real/social",
        "graphs/real/information",
    ])
    save_dir: str = "saved/generation"


def main():
    args = tyro.cli(Args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Building training pool...")
    pool = build_pool(
        args.pool_size, args.n_min, args.n_max,
        args.real_dirs, args.edge_perturb, rng,
    )
    print(f"Pool size: {len(pool)}")

    model = GraphVAE(1, args.hidden_dim, args.latent_dim, r=args.r, tau=args.tau, c=1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        loss = train_epoch(model, optimizer, pool, device, args.edge_drop, args.beta, rng)
        print(f"Epoch {epoch + 1}/{args.epochs} loss={loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            ckpt_path = os.path.join(args.save_dir, f"best_model_{time_string}.ckpt")
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"  -> saved {ckpt_path}")

    # 生成 200 节点图
    ckpt_path = os.path.join(args.save_dir, f"best_model_{time_string}.ckpt")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
    g_gen = generate_graph(model, args.gen_nodes, device, rng=rng)
    out_path = os.path.join(args.save_dir, f"gen_200_{time_string}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(g_gen, f)
        
    from utils.structural_diversity_analysis import analyze_powerlaw,statistics
    print(statistics(g_gen))
    analyze_powerlaw(g_gen)

if __name__ == "__main__":
    main()
