import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
import networkx as nx
from scipy.sparse.linalg import eigsh
from copy import deepcopy
from typing import Callable, Dict
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
    if len(removals) == 0:
        print("empty removal when evaluate sol")
        return 0.0, 0.0

    from scipy.integrate import simpson

    temp_G = graph.copy()
    ensure_attribute(temp_G)
    n_init = temp_G['n_init']
    gcc_eps = []

    for node_id in removals:
        if is_terminal(temp_G,threshold):
            break
        node_id = int(node_id)
        vertex_idx = [i for i, v in enumerate(temp_G.vs) if v['static_id'] == node_id][0]
        temp_G.delete_vertices(vertex_idx)
        
        # Calculate normalized LCC size after removal
        if temp_G.vcount() > 0:
            lcc_size = get_lcc_size(temp_G)
        else:
            lcc_size = 0.0
        gcc_eps.append(lcc_size/n_init)


    auc = simpson(gcc_eps, dx=1) if gcc_eps else 0.0
    # robustness = sum(gcc_eps[::-1][:-1]) / n_init
    robustness = sum(gcc_eps) / n_init if gcc_eps else 0.0
    return auc, robustness


def evaluate_sol_sir(graph, removals, n_removals=None, beta=0.2, gamma=0.1, n_simulations=50, max_steps=500, seed=None):
    '''
    Evaluate a dismantling solution using the Susceptible-Infected-Recovered (SIR) epidemic model.
    Simulates epidemic spread on the graph *after* applying the removals; lower outbreak size
    indicates better containment (removals are more effective from this perspective).

    Args:
        graph: igraph.Graph object
        removals: list of node indices (static_id) in removal order
        n_removals: number of removals to apply before SIR (default: all)
        beta: per-edge infection probability per step
        gamma: recovery probability per step (infected -> recovered)
        n_simulations: number of SIR runs to average over (each with random initial infected)
        max_steps: maximum SIR time steps per run
        seed: random seed for reproducibility
    Returns:
        avg_outbreak_ratio: average (over runs) of (infected+recovered) / N_remaining
        std_outbreak_ratio: standard deviation of outbreak ratio (optional stability measure)
    '''
    if len(removals) == 0:
        print("empty removal when evaluate_sol_sir")
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    temp_G = graph.copy()
    ensure_attribute(temp_G)
    n_apply = len(removals) if n_removals is None else min(n_removals, len(removals))

    for node_id in removals[:n_apply]:
        if temp_G.vcount() == 0:
            break
        node_id = int(node_id)
        try:
            vertex_idx = next(i for i, v in enumerate(temp_G.vs) if v['static_id'] == node_id)
        except StopIteration:
            continue
        temp_G.delete_vertices(vertex_idx)

    n_remaining = temp_G.vcount()
    if n_remaining == 0:
        return 0.0, 0.0

    # state: 0 = S, 1 = I, 2 = R
    outbreak_ratios = []
    adj = temp_G.get_adjlist()

    for _ in range(n_simulations):
        state = np.zeros(n_remaining, dtype=np.int32)  # all S
        initial_infected = rng.integers(0, n_remaining)
        state[initial_infected] = 1

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
            if len(recoveries) == 0 and len(new_infections) == 0:
                break

        outbreak_size = np.sum(state >= 1)
        outbreak_ratios.append(outbreak_size / n_remaining)

    avg_outbreak_ratio = float(np.mean(outbreak_ratios))
    std_outbreak_ratio = float(np.std(outbreak_ratios))
    return avg_outbreak_ratio, std_outbreak_ratio


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


# Usage
METHODS = {
    "Random": random_dismantling,
    "CoreHD": core_hd,
    "Spectral": spectral_dismantling,
    "Degree": adaptive_degree,
    "BetweennessNA": betweenness,
    "Betweenness": adaptive_betweenness,
    "PageRank": adaptive_pagerank,
    "CI": adaptive_ci,
}

def baseline_dismantling(graph, methods, max_steps=None, threshold=None,visualize=False):
    ensure_attribute(graph)
    methods_results = {}
    for name, func in methods.items():
        # Get the sequence of nodes to remove
        removals = func(graph,max_steps=max_steps, threshold=threshold)

        auc, r = evaluate_sol(graph,removals)
        print(f"method {name}: AUC={auc:.6f}, Robustness={r:.6f}")

        methods_results[name] = removals

    if visualize:
        from visualize_dismantling import visualize_multiple_curve
        visualize_multiple_curve(graph,methods_results)

    return methods_results

#-----------------------------------------------------------------
# Import FINDER methods
def FINDER_dismantling(graph, max_steps=None):
    from baseline_rl.FINDER import FINDER_wrapper
    removals, score, MaxCCList = FINDER_wrapper(graph)
    return removals

def NIRM_dismantling(graph, max_steps=None):
    from baseline_rl.NIRM import NIRM_wrapper
    removals, score, MaxCCList = NIRM_wrapper(graph)
    return removals

METHODS.update({
    "FINDER": FINDER_dismantling,
    "NIRM": NIRM_dismantling
})


def gnd_wrapper(graph, max_steps=None):
    """Wrapper for GND weighted dismantling"""
    pass

METHODS.update({
    "GND": gnd_wrapper
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
        

