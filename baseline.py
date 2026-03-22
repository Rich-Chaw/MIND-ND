import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
from scipy.sparse.linalg import eigsh
from copy import deepcopy
from typing import Callable, Dict, List, Tuple
import os
import time
import gc

def ensure_attribute(graph):
    """Ensure graph has static_id attribute and n_init"""
    if 'static_id' not in graph.vs.attributes():
        graph.vs['static_id'] = list(range(graph.vcount()))
    
    if 'n_init' not in graph.attributes():
        graph['n_init'] = graph.vcount()

def get_lcc_size(graph):
    """Get the size of the largest connected component"""
    if graph.vcount() == 0:
        return 0
    components = graph.connected_components()
    return max(components.sizes())

def is_terminal(graph,threshold):
    if threshold == None:
        target_size = 3
    else: target_size = int(graph['n_init']*threshold)

    if get_lcc_size(graph) < target_size or graph.ecount() == 0:
        return True
    else: return False

def spectral_dismantling(G, max_steps=None, threshold=None):
    temp_G = G.copy()
    ensure_attribute(temp_G)
    removals = []
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    for _ in range(max_steps):
        if is_terminal(temp_G,threshold): break
        
        # 1. Get Laplacian (as sparse matrix)
        L = np.array(temp_G.laplacian())
        
        # 2. Get the Fiedler Vector (2nd smallest eigenvalue)
        vals, vecs = eigsh(L.astype(float), k=2, which='SM')
        fiedler_vec = vecs[:, 1]
        
        # 3. Target nodes on the "cut" boundary (values near 0)
        idx_to_remove = np.argmin(np.abs(fiedler_vec))
        
        # Map back to original vertex name/index
        original_idx = temp_G.vs[idx_to_remove]['static_id']
        removals.append(original_idx)
        temp_G.delete_vertices(idx_to_remove)
        
        # Explicit memory cleanup
        del L, vals, vecs, fiedler_vec
        
        # Force garbage collection every 10 iterations to prevent accumulation
        if len(removals) % 10 == 0:
            gc.collect()
        
    return removals

def core_hd(G, max_steps=None, threshold=None):
    temp_G = G.copy()
    ensure_attribute(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    
    for _ in range(max_steps):
        if is_terminal(temp_G,threshold): break
        
        # 1. Calculate Coreness (k-shell decomposition)
        coreness = temp_G.coreness()
        
        # 2. Find nodes in the 2-core or higher
        core_2_indices = [i for i, k in enumerate(coreness) if k >= 2]
        
        if core_2_indices:
            # Pick node with highest degree within the 2-core
            subgraph_degrees = temp_G.degree(core_2_indices)
            max_idx_in_list = np.argmax(subgraph_degrees)
            idx_to_remove = core_2_indices[max_idx_in_list]
        else:
            # Fallback to standard high-degree if no 2-core remains
            idx_to_remove = np.argmax(temp_G.degree())
            
        removals.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)
        
    return removals

def adaptive_degree(G, max_steps=None, threshold=None):
    temp_G = G.copy()
    ensure_attribute(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    for _ in range(max_steps):
        if is_terminal(temp_G,threshold): break
        
        idx_to_remove = np.argmax(temp_G.degree())
        removals.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)
    return removals

def adaptive_k_shell(G, max_steps=None, threshold=None):
    """Adaptive k-shell (coreness) based dismantling.
    At each step, remove a node with the highest k-shell index; break ties by degree.
    """
    temp_G = G.copy()
    ensure_attribute(temp_G)
    removals = []

    if max_steps is None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)

    for _ in range(max_steps):
        if is_terminal(temp_G, threshold):
            break

        coreness = temp_G.coreness()
        max_core = max(coreness) if len(coreness) > 0 else 0
        candidate_indices = [i for i, k in enumerate(coreness) if k == max_core]

        if candidate_indices:
            degrees = temp_G.degree(candidate_indices)
            best_local = int(np.argmax(degrees))
            idx_to_remove = candidate_indices[best_local]
        else:
            idx_to_remove = int(np.argmax(temp_G.degree()))

        removals.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)

    return removals

def betweenness(G, max_steps=None, threshold=None):
    """betweenness centrality dismantling - compute all betweenness from beginning"""
    temp_G = G.copy()
    ensure_attribute(temp_G)
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    
    betweenness= temp_G.betweenness()
    sorted_indices = np.argsort(betweenness)[::-1]
    
    # Return the static_ids of top max_steps nodes
    removals = sorted_indices[:max_steps]

    return removals

def adaptive_betweenness(G, max_steps=None, threshold=None):
    """Adaptive betweenness centrality dismantling"""
    temp_G = G.copy()
    ensure_attribute(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    for _ in range(max_steps):
        if is_terminal(temp_G,threshold): break
        
        betweenness = temp_G.betweenness()
        idx_to_remove = np.argmax(betweenness)
        removals.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)
    return removals

def adaptive_pagerank(G, max_steps=None, threshold=None):
    """Adaptive PageRank dismantling"""
    temp_G = G.copy()
    ensure_attribute(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    for _ in range(max_steps):
        if is_terminal(temp_G,threshold): break
        
        pagerank = temp_G.pagerank()
        idx_to_remove = np.argmax(pagerank)
        removals.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)
    return removals

def adaptive_ci(G, max_steps=None, threshold=None):
    """Adaptive Collective Influence dismantling"""
    temp_G = G.copy()
    ensure_attribute(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    
    for _ in range(max_steps):
        if is_terminal(temp_G,threshold): break
        
        # Collective Influence with radius 2
        ci_scores = []
        for v in temp_G.vs:
            # CI(v) = (k_v - 1) * sum_{u in ball(v,2)} (k_u - 1)
            neighbors_1 = set(temp_G.neighbors(v.index))
            neighbors_2 = set()
            for n1 in neighbors_1:
                neighbors_2.update(temp_G.neighbors(n1))
            
            ball_2 = neighbors_1.union(neighbors_2) - {v.index}
            
            k_v = len(neighbors_1)
            ci_score = (k_v - 1) * sum(len(temp_G.neighbors(u)) - 1 for u in ball_2)
            ci_scores.append(ci_score)
        
        idx_to_remove = np.argmax(ci_scores)
        removals.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)
    return removals


def adaptive_greedy_lcc(G, max_steps=None, threshold=None):
    """Greedy: at each step remove the node whose removal minimizes LCC of the remaining graph."""
    temp_G = G.copy()
    ensure_attribute(temp_G)
    removals = []
    if max_steps is None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    for _ in range(max_steps):
        if is_terminal(temp_G, threshold):
            break
        best_lcc = float("inf")
        idx_to_remove = 0
        for v in temp_G.vs:
            g_test = temp_G.copy()
            g_test.delete_vertices(v.index)
            lcc = get_lcc_size(g_test)
            if lcc < best_lcc:
                best_lcc = lcc
                idx_to_remove = v.index
        removals.append(temp_G.vs[idx_to_remove]["static_id"])
        temp_G.delete_vertices(idx_to_remove)
    return removals


def random_dismantling(G, max_steps=None, threshold=None):
    """Random dismantling for comparison"""
    temp_G = G.copy()
    ensure_attribute(temp_G)
    
    # Get all node IDs and shuffle them
    node_ids = [v['static_id'] for v in temp_G.vs]
    np.random.shuffle(node_ids)
    
    # Return only max_steps nodes if specified
    if max_steps is not None:
        node_ids = node_ids[:max_steps]
    return node_ids

from scipy.integrate import simpson
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def _largest_eigenvalue(g: "ig.Graph") -> float:
    """计算无向图邻接矩阵的最大特征值（谱半径）。"""
    n = g.vcount()
    if n == 0 or g.ecount() == 0:
        return 0.0
    edgelist = g.get_edgelist()
    if not edgelist:
        return 0.0
    rows = [e[0] for e in edgelist]
    cols = [e[1] for e in edgelist]
    # 构造对称邻接矩阵
    rows_sym = rows + cols
    cols_sym = cols + rows
    data_sym = np.ones(len(rows_sym), dtype=float)
    A = csr_matrix((data_sym, (rows_sym, cols_sym)), shape=(n, n))
    try:
        vals = eigsh(A, k=1, which="LM", return_eigenvectors=False)
        return float(vals[0])
    except Exception:
        return 0.0

def evaluate_sol(graph, removals, threshold=None):
    '''
    Evaluate a dismantling solution by computing AUC and robustness

    Args:
        graph: igraph.Graph object
        removals: list of node indices in removal order
    Returns:
        auc: Area under the curve (using Simpson's rule) in MIND
        robustness: Robustness metric following FINDER C++ getRobustness implementation
    '''
    """
    使用结构性指标评估拆解序列。

    返回：
    - auc: 与原实现一致的 AUC（基于 LCC 比例的 Simpson 积分）
    - robustness: 与原实现一致的鲁棒性指标
    - lcc_sizes: 随拆除比例变化的 LCC 比例列表（长度为实际执行的删除步数）
    - lambda_max_list: 随拆除比例变化的最大邻接矩阵特征值列表
    - removed_sizes: 对应每一步的节点移除比例列表
    """
    if len(removals) == 0:
        print("empty removal when evaluate sol")
        return 0.0, 0.0, [], [], []

    temp_G = graph.copy()
    ensure_attribute(temp_G)
    n_init = temp_G['n_init']

    lcc_sizes = [n_init]
    # lambda_max_list = [_largest_eigenvalue(temp_G)]
    removed_sizes = [0]

    removed_count = 0

    for node_id in removals:
        if is_terminal(temp_G, threshold):
            break

        node_id = int(node_id)
        # 按 static_id 找到当前图中的索引
        vertex_idx = [i for i, v in enumerate(temp_G.vs) if v['static_id'] == node_id]
        if not vertex_idx:
            # 该点可能已被删除，跳过
            continue
        vertex_idx = vertex_idx[0]
        temp_G.delete_vertices(vertex_idx)
        removed_count += 1

        # 计算 LCC 比例
        if temp_G.vcount() > 0:
            lcc_size = get_lcc_size(temp_G)
        else:
            lcc_size = 0.0
        lcc_sizes.append(lcc_size / n_init)
        removed_sizes.append(removed_count / n_init)

        # 计算当前图的最大特征值
        # lambda_max = _largest_eigenvalue(temp_G)
        # lambda_max_list.append(lambda_max)

    auc = simpson(lcc_sizes[1:], dx=1)/ n_init if lcc_sizes else 0.0
    robustness = sum(lcc_sizes[::-1][:-2]) / n_init if lcc_sizes else 0.0
    return auc, robustness, lcc_sizes, removed_sizes

def _sir_simulate_with_seeds(adj_list, n_nodes, seeds, beta, gamma, n_simulations, max_steps, rng):
    """
    在给定图（通过邻接表）和给定初始感染集合 seeds 上运行多次 SIR，
    返回：平均最终感染比例、平均峰值感染比例。
    """
    if n_nodes == 0 or not seeds:
        return 0.0, 0.0

    seeds = list(set(seeds))
    outbreak_ratios = []
    peak_ratios = []

    for _ in range(n_simulations):
        state = np.zeros(n_nodes, dtype=np.int32)  # 0=S,1=I,2=R
        for s in seeds:
            if 0 <= s < n_nodes:
                state[s] = 1

        peak_infected = np.sum(state == 1)

        for _ in range(max_steps):
            new_infections = set()
            recoveries = []
            for v in range(n_nodes):
                if state[v] == 1:
                    for u in adj_list[v]:
                        if state[u] == 0 and u not in new_infections:
                            if rng.random() < beta:
                                new_infections.add(u)
                    if rng.random() < gamma:
                        recoveries.append(v)
            for u in new_infections:
                state[u] = 1
            for v in recoveries:
                state[v] = 2

            current_infected = np.sum(state == 1)
            if current_infected > peak_infected:
                peak_infected = current_infected

            if len(recoveries) == 0 and len(new_infections) == 0:
                break

        outbreak_size = np.sum(state >= 1)
        outbreak_ratios.append(outbreak_size / n_nodes)
        peak_ratios.append(peak_infected / n_nodes)

    return float(np.mean(outbreak_ratios)), float(np.mean(peak_ratios))


def evaluate_sir_seeds_as_sources(
    graph,
    removals,
    beta=0.2,
    gamma=0.1,
    n_simulations=50,
    max_steps=500,
    seed=None,
):
    """
    视角一：将拆解序列前 k 个节点作为 SIR 的感染源（Seeds）。

    含义：如果前 k 个节点作为种子在 SIR 中引发的最终感染规模/峰值越大，
    说明该拆解序列确实找到了具有强传播能力的“核心”节点。

    返回：
    - removed_sizes: 每个 k 对应的 k / N 初始节点比例
    - final_outbreak_list: 对应的平均最终感染比例列表
    - peak_outbreak_list: 对应的平均峰值感染比例列表
    """
    if len(removals) == 0:
        print("empty removal when evaluate_sir_seeds_as_sources")
        return [], [], []

    rng = np.random.default_rng(seed)
    temp_G = graph.copy()
    ensure_attribute(temp_G)
    n_init = temp_G.vcount()

    # static_id -> index 映射在整个过程中保持不变（不删点）
    id_to_idx = {v["static_id"]: i for i, v in enumerate(temp_G.vs)}
    adj = temp_G.get_adjlist()

    removed_sizes = [0]
    final_outbreak_list = [0]
    peak_outbreak_list = [0]

    # 逐步增加 seed 集合的大小：k = 1,2,...,len(removals)
    for k in range(1, len(removals) + 1):
        current_seeds_ids = removals[:k]
        current_seeds = [id_to_idx[sid] for sid in current_seeds_ids if sid in id_to_idx]

        avg_final, avg_peak = _sir_simulate_with_seeds(
            adj_list=adj,
            n_nodes=n_init,
            seeds=current_seeds,
            beta=beta,
            gamma=gamma,
            n_simulations=n_simulations,
            max_steps=max_steps,
            rng=rng,
        )

        removed_sizes.append(k / n_init)
        final_outbreak_list.append(avg_final)
        peak_outbreak_list.append(avg_peak)

    return removed_sizes, final_outbreak_list, peak_outbreak_list


def evaluate_sir_after_removals(
    graph,
    removals,
    beta=0.2,
    gamma=0.1,
    n_simulations=50,
    max_steps=500,
    seed=None,
):
    """
    视角二：先从网络中删除前 k 个拆解节点，然后在残余网络中随机选择感染源做 SIR。

    这对应“拆解之后的网络对流行病的抑制能力”。

    返回：
    - removed_sizes: 每个 k 对应的 k / N 初始节点比例
    - final_outbreak_list: 对应的平均最终感染比例列表
    - peak_outbreak_list: 对应的平均峰值感染比例列表
    """
    if len(removals) == 0:
        print("empty removal when evaluate_sir_after_removals")
        return [], [], []

    rng = np.random.default_rng(seed)
    temp_G = graph.copy()
    ensure_attribute(temp_G)
    n_init = temp_G.vcount()

    removed_sizes = []
    final_outbreak_list = []
    peak_outbreak_list = []

    # 逐步累积删除：k = 1,2,...,len(removals)
    removed = 0
    for k in range(1, len(removals) + 1):
        node_id = int(removals[k - 1])
        if temp_G.vcount() > 0:
            vertex_idx = [i for i, v in enumerate(temp_G.vs) if v["static_id"] == node_id]
            if vertex_idx:
                temp_G.delete_vertices(vertex_idx[0])
                removed += 1

        n_remaining = temp_G.vcount()
        if n_remaining == 0:
            removed_sizes.append(removed / n_init)
            final_outbreak_list.append(0.0)
            peak_outbreak_list.append(0.0)
            # 之后所有 k，网络都为空，直接填充 0
            for kk in range(k + 1, len(removals) + 1):
                removed_sizes.append(kk / n_init)
                final_outbreak_list.append(0.0)
                peak_outbreak_list.append(0.0)
            break

        adj = temp_G.get_adjlist()

        outbreak_ratios = []
        peak_ratios = []

        for _ in range(n_simulations):
            state = np.zeros(n_remaining, dtype=np.int32)
            initial_infected = rng.integers(0, n_remaining)
            state[initial_infected] = 1

            peak_infected = 1

            for _ in range(max_steps):
                new_infections = set()
                recoveries = []
                for v in range(n_remaining):
                    if state[v] == 1:
                        for u in adj[v]:
                            if state[u] == 0 and u not in new_infections:
                                if rng.random() < beta:
                                    new_infections.add(u)
                        if rng.random() < gamma:
                            recoveries.append(v)
                for u in new_infections:
                    state[u] = 1
                for v in recoveries:
                    state[v] = 2

                current_infected = np.sum(state == 1)
                if current_infected > peak_infected:
                    peak_infected = current_infected

                if len(recoveries) == 0 and len(new_infections) == 0:
                    break

            outbreak_size = np.sum(state >= 1)
            outbreak_ratios.append(outbreak_size / n_remaining)
            peak_ratios.append(peak_infected / n_remaining)

        removed_sizes.append(removed / n_init)
        final_outbreak_list.append(float(np.mean(outbreak_ratios)))
        peak_outbreak_list.append(float(np.mean(peak_ratios)))

    return removed_sizes, final_outbreak_list, peak_outbreak_list


def igraph_to_networkx(graph):
    edgelist = graph.get_edgelist()
    graph = nx.Graph()
    graph.add_edges_from(edgelist)
    return graph

def evaluate_sol_networkx(graph, removals):
    graph = igraph_to_networkx(graph)
    # Calculate robustness in forward order (same as EvaluateSol)
    # This removes nodes one by one and tracks the largest connected component
    graph_test = graph.copy()
    num_nodes = graph.number_of_nodes()
    total_max_num = 0.0
    max_wcc_sz_list_forward = []
    # Remove nodes in forward order (same as dismantling process)
    for node in removals:
        # Remove the node from the graph
        if node in graph_test:
            graph_test.remove_node(node)
        
        # Find the largest connected component after removal
        if graph_test.number_of_nodes() > 0:
            max_cc_size = max(len(c) for c in nx.connected_components(graph_test))
        else:
            max_cc_size = 0
        
        total_max_num += max_cc_size
        max_wcc_sz_list_forward.append(max_cc_size / num_nodes)
    robustness = total_max_num / (num_nodes * num_nodes)
    
    return robustness
    

# Base methods that return only removals
BASE_METHODS = {
    "Random": random_dismantling,
    "CoreHD": core_hd,
    "Spectral": spectral_dismantling,
    "Degree": adaptive_degree,
    "BetweennessNA": betweenness,
    "Betweenness": adaptive_betweenness,
    "PageRank": adaptive_pagerank,
    "CI": adaptive_ci,
}


def _wrap_with_runtime(func):
    def wrapped(graph, max_steps=None, threshold=None):
        start = time.time()
        removals = func(graph, max_steps=max_steps, threshold=threshold)
        runtime = time.time() - start
        return removals, runtime

    return wrapped


# Public METHODS dict: functions must return (removals, runtime) for baseline_dismantling
METHODS = {name: _wrap_with_runtime(f) for name, f in BASE_METHODS.items()}

def baseline_dismantling(graph, methods, max_steps=None, threshold=0.1,visualize=False):
    ensure_attribute(graph)
    methods_results = {}
    for name, func in methods.items():
        # Get the sequence of nodes to remove
        removals, runtime = func(graph,max_steps=max_steps, threshold=threshold)

        auc, r, lcc_sizes, removed_sizes = evaluate_sol(graph,removals,threshold=threshold)
        print(f"method {name}: AUC={auc:.6f}, Robustness={r:.6f}, runtime={runtime:.6f}")

        methods_results[name] = {
            'removals': removals,
            'auc': auc,
            'robustness': r,
            'lcc_sizes': lcc_sizes,
            'removed_sizes': removed_sizes,
            # 'lambda_list': lambda_list,
        }

    if visualize:
        from visualize_dismantling import visualize_multiple_curve
        visualize_multiple_curve(graph,methods_results)

    return methods_results


def baseline_dismantling_batch(
    graph_list: List[ig.Graph],
    batch_size: int=None,
    method: str = "Degree",
    max_steps=None,
    threshold=0.1,
) -> Tuple[List[float], List[float], List[List[int]]]:
    """
    对多张图批量运行同一基线拆解方法，接口与 test.mind_dismantling_batch 的返回值形式一致：
    返回每张图对应的 auc、鲁棒性、移除序列列表。

    batch_size 仅用于按块遍历 graph_list（与 DismantleEnv 的 batch 语义对齐）；基线启发式本身仍逐图计算。
    """
    if method not in METHODS:
        raise KeyError(f"Unknown baseline method {method!r}; valid keys: {sorted(METHODS)}")
    func = METHODS[method]
    auc_list: List[float] = []
    robustness_list: List[float] = []
    removals_list: List[List[int]] = []

    for g in graph_list:
        g = deepcopy(g)
        ensure_attribute(g)
        removals, _ = func(g, max_steps=max_steps, threshold=threshold)
        auc, r, _, _ = evaluate_sol(g, removals, threshold=threshold)
        auc_list.append(auc)
        robustness_list.append(r)
        removals_list.append(removals)

    print("Avg AUC:", np.mean(auc_list))
    print("Avg Robustness:", np.mean(robustness_list))
    return auc_list, robustness_list, removals_list


#-----------------------------------------------------------------
# Import FINDER methods
def FINDER_dismantling(graph, max_steps=None, threshold=None):
    from baseline_rl.FINDER import FINDER_wrapper
    removals, score, MaxCCList, runtime = FINDER_wrapper(graph)
    return removals, runtime

def NIRM_dismantling(graph, max_steps=None, threshold=None):
    from baseline_rl.NIRM import NIRM_wrapper
    removals, runtime = NIRM_wrapper(graph)
    return removals, runtime

def GDM_dismantling(graph, max_steps=None, threshold=0.1):
    from baseline_rl.GDM import GDM_wrapper
    removals, score, MaxCCList, runtime = GDM_wrapper(graph,heuristic="GDM",threshold=threshold)
    return removals, runtime

def CoreGDM_dismantling(graph, max_steps=None, threshold=0.1):
    from baseline_rl.GDM import GDM_wrapper
    removals, score, MaxCCList, runtime = GDM_wrapper(graph,heuristic="CoreGDM",threshold=threshold)
    return removals, runtime

def GND_dismantling(graph, max_steps=None, threshold=0.1):
    from baseline_rl.GND import GND_wrapper
    removals, score, MaxCCList, runtime = GND_wrapper(graph,threshold=threshold)
    return removals, runtime

def DomiRank_dismantling(graph, max_steps=None, threshold=None):
    from baseline_rl.DomiRank import DomiRank_wrapper
    removals, score, MaxCCList, runtime = DomiRank_wrapper(graph)
    return removals, runtime

def TSAM_dismantling(graph, max_steps=None, threshold=0.1):
    from baseline_rl.TSAM import TSAM_wrapper
    removals, score, MaxCCList, runtime = TSAM_wrapper(graph, threshold=threshold)
    return removals, runtime

METHODS.update({
    "FINDER": FINDER_dismantling,
    "NIRM": NIRM_dismantling,
    "GND": GND_dismantling,
    "GDM": GDM_dismantling,
    "CoreGDM": CoreGDM_dismantling,
    "DomiRank": DomiRank_dismantling,
    "TSAM": TSAM_dismantling,
})


if __name__ == "__main__":
    print("Starting dismantling visualization demo...")
    np.random.seed(42)
    
    # Define methods
    methods = {
        "Spectral": spectral_dismantling,
        "CoreHD": core_hd,
        "Adaptive Degree": adaptive_degree,
        "Random": random_dismantling
    }
    
    # Test different graph types
    graphs = {
        "Small World": ig.Graph.Watts_Strogatz(1, 50, 4, 0.1),
        "Random Geometric": ig.Graph.GRG(n=60, radius=0.15),
        "Grid": ig.Graph.Lattice([8, 8], circular=False),
        "Scale Free": ig.Graph.Barabasi(n=60, m=2)
    }
    for graph_name, graph in graphs.items():
        if not graph.is_connected():
            print(f"{graph_name} is unconnected, extract largest connected component... ")
            graphs[graph_name] = graph.connected_components().giant()
    
    for graph_name, graph in graphs.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {graph_name} graph...")
        print(f"Nodes: {graph.vcount()}, Edges: {graph.ecount()}")
        
        output_dir = os.path.join("baseline_dismantling_analysis",f"{graph_name.lower().replace(' ', '_')}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # Ensure graph has static_id
        ensure_attribute(graph)
        
        # Run all methods and collect results
        methods_results = {}
        n_init = graph.vcount()
        
        print("Running dismantling methods...")
        for name, func in methods.items():
            print(f"  Running {name}...")
            removals = func(graph)
            methods_results[name] = removals
        
        # Create static comparison plot
        from visualize_dismantling import visualize_multiple_curve,visualize_multiple_dynamic
        save_path = visualize_multiple_curve(graph,methods_results,
                                        save_path=os.path.join(output_dir, 'dismantling_comparison.png'))
        

