"""
Figure 3.1 illustration (core clique vs leaf-branch).

Outputs:
1) fig3_1a_core_leaf_schematic.png : network schematic with highlighted v_core (red) and v_leaf (blue)
2) fig3_1b_lcc_curve.png           : normalized LCC size decay curve across removal steps
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np

# Inline palette to avoid importing `utils` package (it may require torch in some envs).
PALETTE2 = ["#92A478", "#B3DE69", "#F7B6D2", "#ACE0CF", "#7EA1EF", "#F19F69", "#C195C4", "#FFD966", "#BAB4D8", "#C27BA0", "#76A5AF", "#7E6FB1", "#C9B458"]
MAIN_METHOD_COLOR = "#EA4252"


# Keep fonts consistent with visualize_powerlaw.py
plt.rcParams.update(
    {
        "font.family": ["Times New Roman", "SimHei"],
        "font.size": 20,
        "font.serif": ["SimHei"],  # 中文黑体
    }
)
plt.rcParams["axes.unicode_minus"] = False  # 用 ASCII 减号，避免负号也变成方块
# Ensure mathtext uses a fontset with good Latin glyph coverage.
plt.rcParams["mathtext.fontset"] = "stix"


CORE_COLOR = MAIN_METHOD_COLOR  # red
LEAF_COLOR = PALETTE2[4]  # blue-ish


@dataclass(frozen=True)
class CoreLeafGraphSpec:
    core_size: int
    n_branches: int
    branch_length: int  # number of edges from v_leaf endpoint to the far end


def build_core_leaf_graph(spec: CoreLeafGraphSpec, seed: int = 0):
    """
    Build an undirected graph:
    - Core: a clique (fully connected)
    - Periphery: multiple path branches

    Construction detail (to match the “instant LCC drop from removing v_leaf” idea):
    - Each branch endpoint v_leaf is directly connected to a clique node.
    - Removing v_leaf disconnects the whole branch from the clique immediately.

    We choose v_core from the clique such that it has no branch attachments,
    so removing v_core initially only decreases LCC slightly.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    core_nodes = list(range(spec.core_size))
    v_core = core_nodes[0]

    # Clique
    for i in range(spec.core_size):
        for j in range(i + 1, spec.core_size):
            G.add_edge(i, j)

    # Build branches; attach each branch to a clique node except v_core.
    # Important for the desired behavior:
    # Allow multiple branches to attach to the same clique node (with replacement),
    # so removing a core node can disconnect many branches at once.
    available_attach_nodes = [u for u in core_nodes if u != v_core]
    if spec.n_branches > len(available_attach_nodes):
        # With replacement, n_branches can be arbitrarily large; v_core is still un-attached.
        pass

    branch_length_nodes = spec.branch_length + 1  # number of nodes on each path
    first_branch_node_id = spec.core_size

    v_leaf_nodes: list[int] = []
    branch_nodes: set[int] = set()
    depth_from_leaf: dict[int, int] = {}

    # Each branch b chooses its attachment clique node u[b] (with replacement).
    attach_nodes = rng.choice(available_attach_nodes, size=spec.n_branches, replace=True).tolist()

    for b, attach_u in enumerate(attach_nodes):
        base = first_branch_node_id + b * branch_length_nodes
        # v_leaf endpoint is depth 0
        leaf_endpoint = base
        v_leaf_nodes.append(leaf_endpoint)

        # Create path leaf_endpoint (=depth0) - depth1 - ... - depthL (=far end)
        prev = None
        for d in range(branch_length_nodes):
            node = base + d
            branch_nodes.add(node)
            depth_from_leaf[node] = d
            if prev is not None:
                G.add_edge(prev, node)
            prev = node

        # Connect clique to the leaf endpoint
        G.add_edge(attach_u, leaf_endpoint)

    attach_count_by_core = {u: 0 for u in core_nodes}
    for attach_u in attach_nodes:
        attach_count_by_core[attach_u] += 1

    return {
        "G": G,
        "core_nodes": core_nodes,
        "v_core": v_core,
        "v_leaf_nodes": v_leaf_nodes,
        "branch_nodes": branch_nodes,
        "depth_from_leaf": depth_from_leaf,
        "attach_count_by_core": attach_count_by_core,
        "spec": spec,
    }


def lcc_size_after_removals(G: nx.Graph, removal_sequence: list[int]):
    """
    Returns:
      lcc_sizes: list of LCC sizes after each removal step (length = len(removal_sequence))
    """
    H = G.copy()
    lcc_sizes = []
    for node in removal_sequence:
        if node in H:
            H.remove_node(node)
        if H.number_of_nodes() == 0:
            lcc_sizes.append(0)
            continue
        # LCC size in undirected graph
        largest_cc = 0
        for comp in nx.connected_components(H):
            largest_cc = max(largest_cc, len(comp))
        lcc_sizes.append(largest_cc)
    return lcc_sizes


def choose_core_first_sequence(
    core_nodes: list[int],
    v_core: int,
    v_leaf_nodes: list[int],
    n_steps: int,
    attach_count_by_core: dict[int, int],
    depth_from_leaf: dict[int, int],
    branch_nodes_set: set[int],
):
    """
    Strategy B (long-term optimal but immediate reward not dominant):
    1) Remove v_core first (small immediate LCC drop).
    2) Do a few low-impact branch-tail removals (keeps early slope mild).
    3) Attack core hubs with many attached branches (strong late-stage fragmentation).
    """
    seq = [v_core]

    warmup_steps = min(3, max(0, n_steps - 1))
    branch_nodes_sorted = sorted(branch_nodes_set, key=lambda x: depth_from_leaf.get(x, 0), reverse=True)
    seq.extend(branch_nodes_sorted[:warmup_steps])

    core_rest = [u for u in core_nodes if u != v_core]
    core_rest_sorted = sorted(core_rest, key=lambda u: attach_count_by_core.get(u, 0), reverse=True)
    seq.extend(core_rest_sorted)

    if len(seq) < n_steps:
        seq.extend(v_leaf_nodes)

    # Deduplicate while preserving order
    seen = set()
    seq = [x for x in seq if not (x in seen or seen.add(x))]
    return seq[:n_steps]


def choose_leaf_first_sequence(v_leaf_nodes: list[int], all_branch_nodes: list[int], n_steps: int, depth_from_leaf: dict[int, int]):
    """
    Strategy A:
    - Remove all v_leaf endpoints first (highest “instant” impact)
    - If still need more steps, remove remaining branch nodes by decreasing depth_from_leaf
      (this keeps the clique intact for a long time, matching the “poor long-term” behavior)
    """
    seq = v_leaf_nodes.copy()
    if len(seq) >= n_steps:
        return seq[:n_steps]

    # Remove remaining branch nodes (excluding already selected v_leaf nodes)
    remaining = [u for u in all_branch_nodes if u not in set(seq)]
    remaining_sorted = sorted(remaining, key=lambda x: depth_from_leaf.get(x, 0), reverse=True)
    seq.extend(remaining_sorted)
    return seq[:n_steps]


def layout_core_leaf_schematic(G: nx.Graph, core_nodes: list[int], v_leaf_nodes: list[int], v_core: int, spec: CoreLeafGraphSpec):
    """
    Create a deterministic hand-tuned layout to look like “core clique in center, branches around”.
    """
    pos: dict[int, tuple[float, float]] = {}

    # Core on a circle
    R_core = 1.6
    core_angles = np.linspace(0, 2 * np.pi, len(core_nodes), endpoint=False)
    for idx, u in enumerate(core_nodes):
        theta = core_angles[idx]
        pos[u] = (R_core * float(np.cos(theta)), R_core * float(np.sin(theta)))

    # Place each branch outward from its attached clique node.
    # The branch endpoint v_leaf is adjacent to the clique in the construction,
    # so we locate it near its adjacent clique node, then extend the path outward.
    delta = 0.55
    branch_len_nodes = spec.branch_length + 1

    # Map branch endpoint -> its neighboring clique node in G (for placement direction)
    # (v_leaf node has exactly one neighbor in clique due to the construction).
    core_set = set(core_nodes)
    for leaf_endpoint in v_leaf_nodes:
        neighbors = [nbr for nbr in G.neighbors(leaf_endpoint) if nbr in core_set]
        if not neighbors:
            # Fallback: place using nearest core node
            attach_u = min(core_nodes, key=lambda u: (pos[u][0] - pos[v_core][0]) ** 2 + (pos[u][1] - pos[v_core][1]) ** 2)
        else:
            attach_u = neighbors[0]

        ax, ay = pos[attach_u]
        base_angle = float(np.arctan2(ay, ax))

        # Give each branch a slight angular jitter based on its index for aesthetics
        jitter = 0.14 * (v_leaf_nodes.index(leaf_endpoint) - (len(v_leaf_nodes) - 1) / 2)
        theta = base_angle + jitter

        # v_leaf endpoint is depth 0 in our construction; draw it just outside the core
        leaf_pos = (ax + delta * np.cos(theta), ay + delta * np.sin(theta))
        pos[leaf_endpoint] = leaf_pos

        # Extend along the path away from the core:
        # find the unique non-clique neighbor on the path and walk forward for branch_length steps
        path_dir_nodes = [nbr for nbr in G.neighbors(leaf_endpoint) if nbr not in core_set]
        if not path_dir_nodes:
            continue
        next_node = path_dir_nodes[0]

        prev = leaf_endpoint
        cur = next_node
        for step in range(1, branch_len_nodes):
            # step corresponds to depth=step in the branch path
            pos[cur] = (leaf_pos[0] + step * delta * np.cos(theta), leaf_pos[1] + step * delta * np.sin(theta))

            # walk to the next node (degree 1 or 2 in a path)
            nxt_candidates = [nbr for nbr in G.neighbors(cur) if nbr != prev]
            if not nxt_candidates:
                break
            prev, cur = cur, nxt_candidates[0]

    # For any remaining nodes (shouldn't happen), place via spring layout
    remaining = [u for u in G.nodes if u not in pos]
    if remaining:
        spring = nx.spring_layout(G.subgraph(remaining), seed=42)
        for u in remaining:
            pos[u] = tuple(float(x) for x in spring[u])

    return pos


def draw_fig3_1a_schematic(core_leaf_bundle, save_path: str):
    G: nx.Graph = core_leaf_bundle["G"]
    core_nodes: list[int] = core_leaf_bundle["core_nodes"]
    v_core: int = core_leaf_bundle["v_core"]
    v_leaf_nodes: list[int] = core_leaf_bundle["v_leaf_nodes"]
    spec: CoreLeafGraphSpec = core_leaf_bundle["spec"]

    # pick one leaf endpoint as the highlighted v_leaf
    v_leaf = v_leaf_nodes[0]

    # For readability, only draw a small subset of branches in the schematic.
    n_show_branches = min(4, len(v_leaf_nodes))
    show_leaf_endpoints = [v_leaf] + [x for x in v_leaf_nodes[1:] if x != v_leaf][: (n_show_branches - 1)]

    # Layout only those shown leaves (but still uses the full graph structure).
    pos = layout_core_leaf_schematic(G, core_nodes, v_leaf_nodes=show_leaf_endpoints, v_core=v_core, spec=spec)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_aspect("equal")
    ax.axis("off")

    core_set = set(core_nodes)
    show_branch_set = set()
    core_set_local = core_set
    branch_len_nodes = spec.branch_length + 1
    for leaf_endpoint in show_leaf_endpoints:
        # Walk along the unique path away from the core to include branch internal nodes.
        next_nodes = [nbr for nbr in G.neighbors(leaf_endpoint) if nbr not in core_set_local]
        if not next_nodes:
            show_branch_set.add(leaf_endpoint)
            continue
        cur = next_nodes[0]
        prev = leaf_endpoint
        show_branch_set.add(leaf_endpoint)
        for _ in range(1, branch_len_nodes):
            show_branch_set.add(cur)
            nxt_candidates = [nbr for nbr in G.neighbors(cur) if nbr != prev]
            if not nxt_candidates:
                break
            prev, cur = cur, nxt_candidates[0]

    nodes_to_keep = set(core_nodes) | show_branch_set
    H = G.subgraph(nodes_to_keep).copy()

    # Draw edges (thin + light)
    core_edges = [(u, v) for u, v in H.edges if u in core_set and v in core_set]
    tree_edges = [(u, v) for u, v in H.edges if not (u in core_set and v in core_set)]

    for u, v in core_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#7D848C", linewidth=1.25, alpha=0.62, zorder=1)

    for u, v in tree_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#9AA2A8", linewidth=1.05, alpha=0.65, zorder=1)

    # Draw nodes
    for u in H.nodes:
        x, y = pos[u]
        if u == v_core:
            color = CORE_COLOR
            size = 120
            marker = "o"
            z = 2
        elif u == v_leaf:
            color = LEAF_COLOR
            size = 120
            marker = "o"
            z = 2
        elif u in core_set:
            color = "#AAB2BA"
            size = 110
            marker = "o"
            z = 2
        else:
            color = "#C2C8CD"
            size = 70
            marker = "o"
            z = 2

        ax.scatter([x], [y], s=size, c=color, marker=marker, edgecolors="none", alpha=0.95, zorder=z)

    # Subtle labels
    ax.text(pos[v_core][0]+0.22, pos[v_core][1] + 0.18, r"$v_{\mathrm{core}}$", ha="center", va="bottom", fontsize=20)
    ax.text(pos[v_leaf][0], pos[v_leaf][1] + 0.18, r"$v_{\mathrm{leaf}}$", ha="center", va="bottom", fontsize=20)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=350, bbox_inches="tight")
    plt.show()


def draw_fig3_1b_lcc_curve(core_leaf_bundle, n_steps: int, save_path: str):
    G: nx.Graph = core_leaf_bundle["G"]
    core_nodes: list[int] = core_leaf_bundle["core_nodes"]
    v_core: int = core_leaf_bundle["v_core"]
    v_leaf_nodes: list[int] = core_leaf_bundle["v_leaf_nodes"]
    branch_nodes_set: set[int] = core_leaf_bundle["branch_nodes"]
    depth_from_leaf: dict[int, int] = core_leaf_bundle["depth_from_leaf"]
    attach_count_by_core: dict[int, int] = core_leaf_bundle["attach_count_by_core"]

    # Strategy A: remove v_leaf first (and keep removing branch-side “leaf endpoints”)
    branch_nodes_sorted = sorted(branch_nodes_set)
    leaf_seq = choose_leaf_first_sequence(
        v_leaf_nodes=v_leaf_nodes,
        all_branch_nodes=branch_nodes_sorted,
        n_steps=n_steps,
        depth_from_leaf=depth_from_leaf,
    )

    # Strategy B: remove v_core first (and keep removing clique nodes)
    core_seq = choose_core_first_sequence(
        core_nodes=core_nodes,
        v_core=v_core,
        v_leaf_nodes=v_leaf_nodes,
        n_steps=n_steps,
        attach_count_by_core=attach_count_by_core,
        depth_from_leaf=depth_from_leaf,
        branch_nodes_set=branch_nodes_set,
    )

    initial_lcc = max((len(c) for c in nx.connected_components(G)), default=0)
    lcc_leaf = lcc_size_after_removals(G, leaf_seq)
    lcc_core = lcc_size_after_removals(G, core_seq)

    # Include step 0
    steps = np.arange(0, n_steps + 1)
    y_leaf = np.array([initial_lcc] + lcc_leaf, dtype=float) / max(initial_lcc, 1)
    y_core = np.array([initial_lcc] + lcc_core, dtype=float) / max(initial_lcc, 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        steps,
        y_leaf,
        color=LEAF_COLOR,
        linewidth=2.2,
        linestyle="-",
        marker="o",
        markersize=4.2,
        # Avoid mixing Chinese with mathtext ($...$). Some Matplotlib backends
        # may render the whole label in math mode, causing missing glyphs.
        label="移除 v_leaf",
    )
    ax.plot(
        steps,
        y_core,
        color=CORE_COLOR,
        linewidth=2.2,
        linestyle="-",
        marker="s",
        markersize=4.2,
        label="移除 v_core",
    )

    ax.set_xlabel("步数", fontsize=20)
    ax.set_ylabel("归一化LCC规模", fontsize=20)
    # ax.set_title("LCC Size Decay (Core-first vs Leaf-first)", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 1.05)
    # Force integer step ticks on x-axis.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=20, frameon=False, loc="upper right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=350, bbox_inches="tight")
    plt.show()


def main():
    # Parameters tuned to make the “leaf-first” dismantling very poor in the long run:
    # - v_core has no branch attachments -> tiny initial LCC drop
    # - branches are attached to core nodes with replacement -> removing a core node disconnects many branches at once
    # - while leaf-first removes one branch per step, leaving the dense core + many remaining branches intact
    spec = CoreLeafGraphSpec(
        core_size=14,
        n_branches=30,
        branch_length=6,  # each branch has 7 nodes
    )

    core_leaf_bundle = build_core_leaf_graph(spec=spec, seed=7)

    out_dir = "visualizations"
    save_a = os.path.join(out_dir, "fig3_1a_core_leaf_schematic.png")
    save_b = os.path.join(out_dir, "fig3_1b_lcc_curve.png")

    draw_fig3_1a_schematic(core_leaf_bundle, save_path=save_a)
    draw_fig3_1b_lcc_curve(core_leaf_bundle, n_steps=20, save_path=save_b)


if __name__ == "__main__":
    main()

