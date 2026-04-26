import random
from typing import List, Tuple

import igraph as ig
import matplotlib.pyplot as plt

from utils.plot import plot_graph
from utils.graph_models import statistics
from utils.common import load_g


def random_walk_subgraph(
    g: ig.Graph,
    target_size: int,
    restart_prob: float = 0.15,
    max_steps: int = 20000,
) -> ig.Graph:
    if g.vcount() <= target_size:
        return g.copy()
    if g.vcount() == 0:
        return ig.Graph()

    start = random.randrange(g.vcount())
    cur = start
    visited = {cur}
    steps = 0

    while len(visited) < target_size and steps < max_steps:
        steps += 1
        if random.random() < restart_prob:
            cur = start
        nbrs = g.neighbors(cur)
        if len(nbrs) == 0:
            cur = random.randrange(g.vcount())
        else:
            cur = random.choice(nbrs)
        visited.add(cur)

    return g.subgraph(list(visited))

def mhrw_subgraph(
    g,
    target_size: int,
    max_steps: int = 200000,
    seed: int = None,
):
    """
    Metropolis-Hastings Random Walk 子图采样
    - 目标：尽量减小普通随机游走对高阶节点的偏置
    - 返回：包含 target_size 个节点的诱导子图（若步数不够则返回当前已采样部分）
    """
    import random
    import igraph as ig

    if seed is not None:
        random.seed(seed)

    if g.vcount() == 0:
        return ig.Graph()
    if g.vcount() <= target_size:
        return g.copy()

    cur = random.randrange(g.vcount())
    visited = {cur}
    steps = 0
    while len(visited) < target_size and steps < max_steps:
        steps += 1
        nbrs = g.neighbors(cur)
        if not nbrs:
            # 遇到孤立点则随机重启
            cur = random.randrange(g.vcount())
            visited.add(cur)
            continue
        nxt = random.choice(nbrs)
        deg_cur = max(g.degree(cur), 1)
        deg_nxt = max(g.degree(nxt), 1)
        # MH 接受率：alpha = min(1, deg(cur)/deg(nxt))
        alpha = min(1.0, deg_cur / deg_nxt)
        if random.random() < alpha:
            cur = nxt
        visited.add(cur)

    return g.subgraph(list(visited))

def forest_fire_subgraph(
    g: ig.Graph,
    target_size: int,
    burn_prob: float = 0.35,
    k: int = 10,
    max_steps: int = 20000,
) -> ig.Graph:
    if g.vcount() <= target_size:
        return g.copy()
    if g.vcount() == 0:
        return ig.Graph()

    seed = random.randrange(g.vcount())
    burned = {seed}
    frontier = [seed]
    steps = 0

    while len(burned) < target_size and frontier and steps < max_steps:
        steps += 1
        node = frontier.pop(0)
        nbrs = [n for n in g.neighbors(node) if n not in burned]
        random.shuffle(nbrs)
        nbrs = [n for n in nbrs if random.random() < burn_prob][:k]
        for n in nbrs:
            if random.random() < burn_prob:
                burned.add(n)
                frontier.append(n)
                if len(burned) >= target_size:
                    break
        if not frontier and len(burned) < target_size:
            remain = [i for i in range(g.vcount()) if i not in burned]
            if remain:
                jump = random.choice(remain)
                burned.add(jump)
                frontier.append(jump)

    sg = g.subgraph(list(burned))
    sg = sg.connected_components().giant()

    return sg


def analyze_graph(before_g: ig.Graph, after_g: ig.Graph, title: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_graph(before_g, ax=axes[0], node_size=4)
    axes[0].set_title(f"{title} before")
    plot_graph(after_g, ax=axes[1], node_size=4)
    axes[1].set_title(f"{title} after")
    plt.tight_layout()
    plt.show()

    print(f"[{title}] before stats:\n", statistics(before_g))
    print(f"[{title}] after stats:\n", statistics(after_g))


def sample_real_subgraphs(
    real_graphs: List[ig.Graph],
    num_subgraphs: int,
    min_nodes: int = 200,
    max_nodes: int = 500,
    ratio: List[float] = [0.4, 0.4, 0.2],  # [rw, mhrw, ff]
) -> List[ig.Graph]:

    if len(real_graphs) == 0:
        raise ValueError("real_graphs is empty")
    s = float(sum(ratio))
    p = [r / s for r in ratio]

    out = []
    methods = ["rw", "mhrw", "ff"]

    for i in range(num_subgraphs):
        src = random.choice(real_graphs)
        target_size = random.randint(min_nodes, max_nodes)
        if src.vcount() < target_size:
            continue
        method = random.choices(methods, weights=p, k=1)[0]

        if method == "rw":
            sg = random_walk_subgraph(src, target_size=target_size)
        elif method == "mhrw":
            sg = mhrw_subgraph(src, target_size=target_size)
        else:
            sg = forest_fire_subgraph(src, target_size=target_size)

        sg["name"] = f"{src['name']}_sub_{method}_{i}"
        sg["type"] = "real"
        out.append(sg)

    return out


def _test_case():
    # g = ig.Graph.Erdos_Renyi(n=600, p=0.01)
    # g["name"] = "toy_er"
    
    g_name = 'graphs/real/tech/internet-topology.pkl'

    # g_name = 'graphs/gen/citeseer.pkl'
    # g_name = 'graphs/gen/cora.pkl'
    # g_name = 'graphs/gen/ego-facebook.pkl'

    g = load_g(g_name,name=g_name)

    rw = random_walk_subgraph(g, 250)
    mhrw = mhrw_subgraph(g, 250)
    ff = forest_fire_subgraph(g, 250)
    analyze_graph(g, rw, title="rw_test")
    analyze_graph(g, mhrw, title="mhrw_test")
    analyze_graph(g, ff, title="ff_test")


if __name__ == "__main__":
    _test_case()
