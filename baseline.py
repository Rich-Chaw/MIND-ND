import matplotlib.pyplot as plt
import numpy as np
import igraph as ig
from scipy.sparse.linalg import eigsh
from copy import deepcopy
from typing import Callable, Dict
import os
import time

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
    removed_nodes = []
    
    while True:
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
        removed_nodes.append(original_idx)
        temp_G.delete_vertices(idx_to_remove)
        
    return removed_nodes

def core_hd(G, max_steps=None):
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removed_nodes = []
    
    while True:
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
            
        removed_nodes.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)
        
    return removed_nodes

def adaptive_degree(G, max_steps=None):
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removed_nodes = []
    
    while True:
        if temp_G.vcount() <= 2 or temp_G.ecount() == 0: 
            break
        
        idx_to_remove = np.argmax(temp_G.degree())
        removed_nodes.append(temp_G.vs[idx_to_remove]['static_id'])
        temp_G.delete_vertices(idx_to_remove)
    return removed_nodes

# def adaptive_betweenness(G,max_steps=None):

# def adaptive_pagerank(G,max_steps=None):

# def adaptive_ci(G,max_steps=None):


def random_dismantling(G, max_steps=None):
    """Random dismantling for comparison"""
    temp_G = G.copy()
    ensure_static_id(temp_G)
    
    # Get all node IDs and shuffle them
    node_ids = [v['static_id'] for v in temp_G.vs]
    np.random.shuffle(node_ids)
    return node_ids

def bpd_dismantling(G, max_steps=None):
    """
    Belief Propagation Decimation (Min-Sum inspired) for Network Dismantling.
    Targets the Feedback Vertex Set (nodes that break cycles).
    """
    temp_G = G.copy()
    ensure_static_id(temp_G)
    removed_nodes = []
    
    # Parameters for the BP algorithm
    x = 10.0  # Sensitivity parameter (resembles inverse temperature)
    
    while True:
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
        removed_nodes.append(temp_G.vs[node_to_del_idx]['static_id'])
        temp_G.delete_vertices(node_to_del_idx)
        
    return removed_nodes

def create_dismantling_comparison_complete(graph, methods, output_dir="dismantling_analysis"):
    """
    Run dismantling methods and create both static plots and GIF animations
    """
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
    
    # Create GIF animations
    gif_paths = visualize_multiple_dynamic(graph, methods_results, 
                                         output_dir = os.path.join(output_dir, 'gifs'))
    
    return methods_results, save_path, gif_paths


def baseline_dismantling(graph, methods,max_steps=None,visualize=False):
    ensure_static_id(graph)
    n_init = graph.vcount() 
    methods_results = {}
    color_map = ['blue','orange','green','grey']
    i = 0
    plt.figure(figsize=(6,4))
    for name, func in methods.items():
        # Get the sequence of nodes to remove
        print(f"  Running {name}...")
        t1 = time.time()
        removals = func(graph,max_steps)
        t2 = time.time()
        print(f"method {name} dismantling time: ",t2-t1)
        
        methods_results[name] = removals

    if visualize:
        from visualize_dismantling import visualize_multiple_curve
        visualize_multiple_curve(graph,methods_results)

    return methods_results

# Usage
METHODS = {
    "Random":random_dismantling,
    "CoreHD": core_hd,
    "Spectral":spectral_dismantling,
    "Adaptive Degree": adaptive_degree,
    "BPD": bpd_dismantling
}


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
        results = create_dismantling_comparison_complete(graph, methods, output_dir)
