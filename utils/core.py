"""
Core / core-CI reward-shaping helpers. Graph attributes core2_init and core_ci_init
are set on dataset copies by DismantleEnv.init_core (episode s0, fixed for the whole trajectory).
"""
from __future__ import annotations

from typing import List

import igraph as ig


def get_core_nodes(g: ig.Graph, mode: str = "k2") -> List[int]:
    """
    mode='k2'      -> nodes with coreness >= 2
    mode='maxcore' -> nodes in the highest-coreness shell
    """
    if g.vcount() == 0 or g.ecount() == 0:
        return []
    coreness = g.coreness()
    if mode == "k2":
        return [i for i, k in enumerate(coreness) if k >= 2]
    if mode == "maxcore":
        max_core = max(coreness) if len(coreness) > 0 else 0
        return [i for i, k in enumerate(coreness) if k == max_core]
    return []


def core2_size(g: ig.Graph) -> int:
    """Number of vertices with coreness >= 2 (2-core size)."""
    return len(get_core_nodes(g, mode="k2"))


def max_ci_in_max_core(g: ig.Graph) -> float:
    """
    Phi(s): max CI among nodes in the current max-coreness subgraph
    (same local CI as in baseline.adaptive_ci, radius-2 ball on induced subgraph).
    """
    core_nodes = get_core_nodes(g, mode="maxcore")
    if len(core_nodes) == 0:
        return 0.0
    g_core = g.induced_subgraph(core_nodes)
    if g_core.vcount() == 0 or g_core.ecount() == 0:
        return 0.0

    max_ci = 0.0
    for v in g_core.vs:
        neighbors_1 = set(g_core.neighbors(v.index))
        neighbors_2 = set()
        for n1 in neighbors_1:
            neighbors_2.update(g_core.neighbors(n1))
        ball_2 = neighbors_1.union(neighbors_2) - {v.index}
        k_v = len(neighbors_1)
        ci_v = (k_v - 1) * sum(len(g_core.neighbors(u)) - 1 for u in ball_2)
        if ci_v > max_ci:
            max_ci = float(ci_v)
    return max_ci


def set_core_inits(graphs: List[ig.Graph], shaping_method: str) -> None:
    """
    Write episode-initial normalization constants on each **dataset** graph (full graph at reset).
    igraph copies in GraphPool preserve these graph attributes across steps of the same episode.
    """
    if not graphs or shaping_method not in ("core", "core_ci"):
        return
    for g in graphs:
        if g is None or g.vcount() == 0:
            continue
        if shaping_method == "core":
            g["core2_init"] = core2_size(g)
        else:
            g["core_ci_init"] = max_ci_in_max_core(g)
