import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
from scipy.sparse.linalg import eigsh
from copy import deepcopy
from typing import Callable, Dict
import os
import time
import gc

def ensure_static_id(graph):
    """Ensure graph has static_id attribute"""
    if 'static_id' not in graph.vs.attributes():
        graph.vs['static_id'] = list(range(graph.vcount()))

def get_lcc_size(graph):
    """Get the size of the largest connected component"""
    if graph.vcount() == 0:
        return 0
    components = graph.connected_components()
    return max(components.sizes())

def spectral_dismantling(G, max_steps=None):
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removals = []
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    for _ in range(max_steps):
        if temp_G.vcount() <= 2 or temp_G.ecount() == 0: 
            break
        
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


def spectral_dismantling_advance(G, max_steps=None):
    """
    Improved spectral dismantling following GND approach
    Uses power iteration and spectral clustering
    much slower but more stable and performs better than spectral_dismantling
    """
    import gc
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removals = []
    
    if max_steps is not None:
        max_steps = min(temp_G.vcount(), max_steps)
    
    while True:
        if temp_G.vcount() <= 2 or temp_G.ecount() == 0: 
            break
        
        # Check max_steps limit
        if max_steps is not None and len(removals) >= max_steps:
            break
        
        # Convert to adjacency matrix
        adj_matrix = np.array(temp_G.get_adjacency().data)
        n = adj_matrix.shape[0]
        
        if n <= 2:
            break
        
        # Power iteration for Fiedler vector (more stable than eigsh for small matrices)
        x = np.random.uniform(-1, 1, n)
        
        # Create Laplacian matrix L = D - A
        degrees = np.sum(adj_matrix, axis=1)
        L = np.diag(degrees) - adj_matrix
        
        # Power iteration to find smallest non-zero eigenvalue's eigenvector
        for _ in range(50):  # More iterations for stability
            # Shift to avoid zero eigenvalue: (L + I)^-1
            try:
                L_shifted = L + np.eye(n)
                x_new = np.linalg.solve(L_shifted, x)
                # Orthogonalize against constant vector
                x_new = x_new - np.mean(x_new)
                norm = np.linalg.norm(x_new)
                if norm > 1e-10:
                    x = x_new / norm
                else:
                    break
            except np.linalg.LinAlgError:
                # Fallback to simple degree-based removal
                idx_to_remove = np.argmax(temp_G.degree())
                removals.append(temp_G.vs[idx_to_remove]['static_id'])
                temp_G.delete_vertices(idx_to_remove)
                break
        else:
            # Use Fiedler vector to partition and find cut
            # Find edges crossing the partition (positive vs negative values)
            crossing_edges = []
            for edge in temp_G.es:
                u, v = edge.tuple
                if x[u] * x[v] < 0:  # Different signs
                    crossing_edges.append((u, v))
            
            if len(crossing_edges) == 0:
                # No clear partition, remove node closest to zero
                idx_to_remove = np.argmin(np.abs(x))
            else:
                # Remove node that appears in most crossing edges (simple heuristic)
                node_counts = {}
                for u, v in crossing_edges:
                    node_counts[u] = node_counts.get(u, 0) + 1
                    node_counts[v] = node_counts.get(v, 0) + 1
                
                if node_counts:
                    idx_to_remove = max(node_counts.keys(), key=lambda k: node_counts[k])
                else:
                    idx_to_remove = np.argmax(temp_G.degree())
            
            removals.append(temp_G.vs[idx_to_remove]['static_id'])
            temp_G.delete_vertices(idx_to_remove)
        
        # Explicit memory cleanup for large matrices
        del adj_matrix, degrees, L, x
        if 'L_shifted' in locals():
            del L_shifted
        if 'x_new' in locals():
            del x_new
        
        # Force garbage collection every 5 iterations (more frequent due to larger matrices)
        if len(removals) % 5 == 0:
            gc.collect()
        
    return removals

def core_hd(G, max_steps=None):
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    
    for _ in range(max_steps):
        if temp_G.vcount() <= 2 or temp_G.ecount() == 0: 
            break
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

def adaptive_degree(G, max_steps=None):
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    for _ in range(max_steps):
        if temp_G.vcount() <= 2 or temp_G.ecount() == 0: 
            break
        
        idx_to_remove = np.argmax(temp_G.degree())
        removals.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)
    return removals

def adaptive_betweenness(G, max_steps=None):
    """Adaptive betweenness centrality dismantling"""
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    for _ in range(max_steps):
        if temp_G.vcount() <= 2 or temp_G.ecount() == 0: 
            break
        
        betweenness = temp_G.betweenness()
        idx_to_remove = np.argmax(betweenness)
        removals.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)
    return removals

def adaptive_pagerank(G, max_steps=None):
    """Adaptive PageRank dismantling"""
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    for _ in range(max_steps):
        if temp_G.vcount() <= 2 or temp_G.ecount() == 0: 
            break
        
        pagerank = temp_G.pagerank()
        idx_to_remove = np.argmax(pagerank)
        removals.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)
    return removals

def adaptive_ci(G, max_steps=None):
    """Adaptive Collective Influence dismantling"""
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    
    for _ in range(max_steps):
        if temp_G.vcount() <= 2 or temp_G.ecount() == 0: 
            break
        
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


def random_dismantling(G, max_steps=None):
    """Random dismantling for comparison"""
    temp_G = G.copy()
    ensure_static_id(temp_G)
    
    # Get all node IDs and shuffle them
    node_ids = [v['static_id'] for v in temp_G.vs]
    np.random.shuffle(node_ids)
    
    # Return only max_steps nodes if specified
    if max_steps is not None:
        return node_ids[:max_steps]
    return node_ids

def bpd_dismantling(G, max_steps=None):
    """
    Belief Propagation Decimation (Min-Sum inspired) for Network Dismantling.
    Targets the Feedback Vertex Set (nodes that break cycles).
    """
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removals = []
    
    if max_steps == None:
        max_steps = temp_G.vcount()
    else:
        max_steps = min(temp_G.vcount(), max_steps)
    
    # Parameters for the BP algorithm
    x = 10.0  # Sensitivity parameter (resembles inverse temperature)
    
    for _ in range(max_steps):
        if temp_G.vcount() <= 2 or temp_G.ecount() == 0: 
            break
        
        N = temp_G.vcount()
        edges = temp_G.get_edgelist()
        adj = temp_G.get_adjlist()
        
        # Initialize messages for each directed edge (2 * E)
        messages = {(u, v): 1.0/3.0 for u, v in edges}
        messages.update({(v, u): 1.0/3.0 for u, v in edges})
        
        # Simple BP iteration (fixed number of steps for stability)
        for _ in range(5):
            new_messages = {}
            for u, v in messages:
                # Calculate the product of incoming messages from neighbors other than v
                neighbors_of_u = adj[u]
                prod_val = 1.0
                for w in neighbors_of_u:
                    if w != v:
                        prod_val *= messages.get((w, u), 0.5)
                
                # Simplified update rule for the 'cavity' probability
                new_messages[(u, v)] = np.tanh(x * prod_val)
            messages.update(new_messages)
            
        # Compute marginals: How likely is this node to be part of a cycle?
        marginals = np.zeros(N)
        for i in range(N):
            prod_all = 1.0
            for neighbor in adj[i]:
                prod_all *= messages.get((neighbor, i), 0.5)
            marginals[i] = 1.0 - prod_all
            
        # Remove the node with the highest marginal probability
        node_to_del_idx = np.argmax(marginals)
        removals.append(temp_G.vs[node_to_del_idx]['static_id'])
        temp_G.delete_vertices(node_to_del_idx)
        
    return removals

def evaluate_sol(graph, removals):
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
        return 0.0 ,0.0

    from scipy.integrate import simpson
    
    temp_G = graph.copy()
    ensure_static_id(temp_G)
    n_init = temp_G.vcount()
    
    # Track LCC size at each step (normalized)
    gcc_eps = []
    
    # Remove nodes one by one and track LCC after each removal
    for node_id in removals:
        if temp_G.vcount() == 0:
            break
        
        # Find the vertex with matching static_id
        try:
            vertex_idx = [i for i, v in enumerate(temp_G.vs) if v['static_id'] == node_id][0]
            temp_G.delete_vertices(vertex_idx)
            
            # Calculate normalized LCC size after removal
            if temp_G.vcount() > 0:
                lcc_size = get_lcc_size(temp_G) / n_init
            else:
                lcc_size = 0.0
            gcc_eps.append(lcc_size)
        except (IndexError, KeyError):
            # Node already removed or doesn't exist
            continue
    
    auc = simpson(gcc_eps, dx=1)
    robustness = sum(gcc_eps[::-1][:-1]) / n_init
    
    # Calculate robustness using the same method as FINDER C++ getRobustness
    # Start with empty graph and add nodes back in reverse order
    # active_nodes = set()
    # total_max_num = 0.0
    
    # # Process removals in reverse order (last removed first)
    # for node_id in reversed(removals):
    #     # Add node back to active set
    #     active_nodes.add(node_id)
        
    #     # Create subgraph with active nodes and their edges
    #     if len(active_nodes) > 0:
    #         # Find vertices that correspond to active nodes
    #         active_vertex_indices = [i for i, v in enumerate(graph.vs) if v['static_id'] in active_nodes]
    #         if len(active_vertex_indices) > 0:
    #             subgraph = graph.induced_subgraph(active_vertex_indices)
              
    #     # Add edges involving this node if both endpoints are active
    #     edges_to_add = []
    #     for u, v in original_edges:
    #         orig_u = graph.vs[u]['static_id']
    #         orig_v = graph.vs[v]['static_id']
            
    #         if orig_u == node_id and orig_v in active_nodes:
    #             edges_to_add.append((orig_u, orig_v))
    #         elif orig_v == node_id and orig_u in active_nodes:
    #             edges_to_add.append((orig_u, orig_v))
        
    #     # Create subgraph with active nodes and their edges
    #     if len(active_nodes) > 0:
    #         subgraph = graph.induced_subgraph([i for i, v in enumerate(graph.vs) if v['static_id'] in active_nodes])
    #         if subgraph.vcount() > 0:
    #             lcc_size = get_lcc_size(subgraph)  # Absolute LCC size (not normalized)
    #             total_max_num += lcc_size
    
    # # Subtract final LCC size (when all nodes are back)
    # if len(active_nodes) > 0:
    #     final_lcc_size = get_lcc_size(graph)
    #     total_max_num -= final_lcc_size
    
    # # Normalize by n^2 as in C++ implementation
    # robustness = total_max_num / (n_init * n_init)
    
    return auc, robustness

# Usage
METHODS = {
    "Random": random_dismantling,
    "CoreHD": core_hd,
    "Spectral": spectral_dismantling,
    "SpectralA":spectral_dismantling_advance,
    "Degree": adaptive_degree,
    "BPD": bpd_dismantling,
    "Betweenness": adaptive_betweenness,
    "PageRank": adaptive_pagerank,
    "CI": adaptive_ci,
}

def baseline_dismantling(graph, methods,max_steps=None,visualize=False):
    ensure_static_id(graph)
    n_init = graph.vcount() 
    methods_results = {}
    color_map = ['blue','orange','green','grey']
    i = 0
    plt.figure(figsize=(6,4))
    for name, func in methods.items():
        # Get the sequence of nodes to remove
        removals = func(graph,max_steps=max_steps)

        auc, r = evaluate_sol(graph,removals)
        print(f"method {name}: AUC={auc}, Robustness={r}")

        methods_results[name] = removals

    if visualize:
        from visualize_dismantling import visualize_multiple_curve
        visualize_multiple_curve(graph,methods_results)

    return methods_results

#-----------------------------------------------------------------
# Usage
METHODS = {
    "Random": random_dismantling,
    "CoreHD": core_hd,
    "Spectral": spectral_dismantling,
    "SpectralA":spectral_dismantling_advance,
    "Degree": adaptive_degree,
    "BPD": bpd_dismantling,
    "Betweenness": adaptive_betweenness,
    "PageRank": adaptive_pagerank,
    "CI": adaptive_ci,
}

# Import FINDER methods
def finder_dismantling_wrapper(G, max_steps=None):
    from baseline_rl.finder import finder
    removals, score, MaxCCList = finder(G)
    print(len(set(removals)))
    return removals

METHODS.update({
    "FINDER": finder_dismantling_wrapper
})

# Import GND methods
try:
    from gnd_python import gnd_spectral_dismantling, gnd_weighted_dismantling
    from gnd_reinsertion import gnd_with_reinsertion
    GND_AVAILABLE = True
except ImportError:
    print("GND methods not available - gnd_python.py or gnd_reinsertion.py not found")
    GND_AVAILABLE = False

def gnd_dismantling_wrapper(G, max_steps=None):
    """Wrapper for GND weighted dismantling"""
    if not GND_AVAILABLE:
        return adaptive_degree(G, max_steps)
    return gnd_weighted_dismantling(G, max_steps)

def gnd_spectral_wrapper(G, max_steps=None):
    """Wrapper for GND spectral dismantling"""
    if not GND_AVAILABLE:
        return spectral_dismantling(G, max_steps)
    return gnd_spectral_dismantling(G, max_steps)

def gnd_with_reinsertion_wrapper(G, max_steps=None):
    """Wrapper for GND with reinsertion"""
    if not GND_AVAILABLE:
        return adaptive_degree(G, max_steps)
    return gnd_with_reinsertion(G, target_size_ratio=0.01, use_weighted=True, 
                               reinsertion_threshold=max(100, G.vcount()//10))


# Add GND methods if available
if GND_AVAILABLE:
    METHODS.update({
        "GND Weighted": gnd_dismantling_wrapper,
        "GND Spectral": gnd_spectral_wrapper,
        "GNDR": gnd_with_reinsertion_wrapper,
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
        ensure_static_id(graph)
        
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
        

