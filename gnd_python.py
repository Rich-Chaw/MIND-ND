"""
Python implementation of Generalized Network Dismantling (GND) algorithm
Based on the C++ implementation from the GND paper
"""

import numpy as np
import igraph as ig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import networkx as nx
from typing import List, Tuple, Dict
import time

def ensure_static_id(graph):
    """Ensure graph has static_id attribute"""
    if 'static_id' not in graph.vs.attributes():
        graph.vs['static_id'] = list(range(graph.vcount()))

def get_gcc_igraph(graph):
    """Get giant connected component using igraph"""
    if graph.vcount() == 0:
        return graph.copy()
    
    components = graph.connected_components()
    if len(components) == 0:
        return graph.copy()
    
    # Find largest component
    largest_idx = np.argmax(components.sizes())
    gcc_vertices = components[largest_idx]
    
    # Create subgraph with only GCC vertices
    gcc = graph.induced_subgraph(gcc_vertices)
    return gcc

def power_iteration_standard(adj_matrix, max_iter=None):
    """
    Standard power iteration for Laplacian L = D_max - A
    Following GND.cpp power_iteration function
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return np.array([])
    
    # Initialize random vector
    np.random.seed(42)  # For reproducibility
    x = np.random.uniform(-1, 1, n)
    y = np.zeros(n)
    
    # Get maximum degree
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    d_max = int(np.max(degrees))
    
    # Number of iterations: 30 * log(n) * sqrt(log(n))
    if max_iter is None:
        max_iter = max(10, int(30 * np.log(n) * np.sqrt(np.log(n))))
    
    for iteration in range(max_iter):
        # Multiply by Laplacian: y = (D_max * I - A) * x
        # y_i = d_max * x_i - sum_j A_ij * x_j
        y = d_max * x - adj_matrix.dot(x)
        
        # Second multiplication: x = (D_max * I - A) * y  
        x = d_max * y - adj_matrix.dot(y)
        
        # Orthonormalize
        x = orthonormalize(x)
        
    return x

def power_iteration_weighted(adj_matrix, max_iter=None):
    """
    Weighted power iteration for B = WA + AW - A
    Following GND.cpp power_iterationB function
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return np.array([])
    
    # Initialize random vector
    np.random.seed(42)
    x = np.random.uniform(-1, 1, n)
    y = np.zeros(n)
    
    # Compute degrees and weighted degrees
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # Compute db: d_i * (d_i - 1) + sum_j d_j for neighbors j
    db = np.zeros(n)
    for i in range(n):
        d_i = degrees[i]
        db[i] = d_i * (d_i - 1)
        # Add degrees of neighbors
        neighbors = adj_matrix[i].nonzero()[1]
        db[i] += np.sum(degrees[neighbors])
    
    d_max = int(np.max(degrees))
    d_max2 = int(np.max(db))
    d_max_combined = d_max * d_max + d_max2
    
    if max_iter is None:
        max_iter = max(10, int(30 * np.log(n) * np.sqrt(np.log(n))))
    
    for iteration in range(max_iter):
        # Multiply by weighted Laplacian
        y = multiply_by_weight_laplacian(adj_matrix, x, db, d_max_combined)
        x = multiply_by_weight_laplacian(adj_matrix, y, db, d_max_combined)
        x = orthonormalize(x)
        
    return x

def multiply_by_weight_laplacian(adj_matrix, x, db, d_max):
    """
    Multiply vector by weighted Laplacian matrix
    Following GND.cpp multiplyByWeightLaplacian function
    """
    n = len(x)
    y = np.zeros(n)
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    
    # First part: y_i = (d_i - 1) * sum_j A_ij * x_j
    for i in range(n):
        neighbors_sum = adj_matrix[i].dot(x)
        y[i] = (degrees[i] - 1) * neighbors_sum
    
    # Second part: add sum_j A_ij * d_j * x_j + (d_max - db_i) * x_i
    for i in range(n):
        neighbors = adj_matrix[i].nonzero()[1]
        for j in neighbors:
            y[i] += degrees[j] * x[j]
        y[i] += (d_max - db[i]) * x[i]
    
    return y

def orthonormalize(x):
    """
    Orthonormalize vector following GND.cpp orthonormalize function
    """
    n = len(x)
    if n == 0:
        return x
    
    # Remove component along uniform vector
    inner = np.sum(x) / np.sqrt(n)
    x = x - inner / np.sqrt(n)
    
    # Normalize
    norm = np.linalg.norm(x)
    if norm > 1e-10:
        x = x / norm
    
    return x

def vertex_cover_weighted(crossing_edges, degrees):
    """
    Weighted vertex cover using greedy algorithm
    Following GND.cpp vertex_cover_2 function
    """
    n = len(degrees)
    if len(crossing_edges) == 0:
        return np.zeros(n, dtype=int)
    
    # Build edge list for vertex cover
    edge_list = {i: [] for i in range(n)}
    for u, v in crossing_edges:
        edge_list[u].append(v)
        edge_list[v].append(u)
    
    flag = np.zeros(n, dtype=int)  # Removal order
    remove_order = 0
    current_degrees = degrees.copy()
    
    # Total edges to cover
    total_edges = sum(len(neighbors) for neighbors in edge_list.values()) // 2
    
    while total_edges > 0:
        # Compute degree in cover problem
        degree_cover = np.array([len(edge_list[i]) for i in range(n)])
        
        # Compute value = degree / degree_cover (lower is better)
        values = np.full(n, np.inf)
        for i in range(n):
            if degree_cover[i] > 0:
                values[i] = current_degrees[i] / degree_cover[i]
        
        # Find node with minimum value
        min_idx = np.argmin(values)
        if values[min_idx] == np.inf:
            break
            
        # Mark for removal
        remove_order += 1
        flag[min_idx] = remove_order
        
        # Remove edges incident to this node
        neighbors_to_remove = edge_list[min_idx].copy()
        edge_list[min_idx] = []
        
        for neighbor in neighbors_to_remove:
            if min_idx in edge_list[neighbor]:
                edge_list[neighbor].remove(min_idx)
        
        # Update current degrees (simulate node removal from original graph)
        current_degrees[min_idx] = 0
        for neighbor in neighbors_to_remove:
            if current_degrees[neighbor] > 0:
                current_degrees[neighbor] -= 1
        
        # Recalculate total edges
        total_edges = sum(len(neighbors) for neighbors in edge_list.values()) // 2
    
    return flag

def gnd_dismantling_step(graph, use_weighted=True):
    """
    Single step of GND algorithm: partition and find vertex cover
    """
    ensure_static_id(graph)
    
    if graph.vcount() <= 2 or graph.ecount() == 0:
        return []
    
    # Convert to adjacency matrix
    adj_matrix = csr_matrix(graph.get_adjacency().data)
    
    # Compute eigenvector
    if use_weighted:
        eigenvector = power_iteration_weighted(adj_matrix)
    else:
        eigenvector = power_iteration_standard(adj_matrix)
    
    if len(eigenvector) == 0:
        return []
    
    # Find crossing edges (edges between positive and negative eigenvector components)
    crossing_edges = []
    for edge in graph.es:
        u, v = edge.tuple
        if eigenvector[u] * eigenvector[v] < 0:  # Different signs
            crossing_edges.append((u, v))
    
    if len(crossing_edges) == 0:
        # No crossing edges, remove highest degree node
        degrees = graph.degree()
        max_degree_idx = np.argmax(degrees)
        return [graph.vs[max_degree_idx]['static_id']]
    
    # Solve vertex cover on crossing edges
    degrees = np.array(graph.degree())
    vertex_cover_flags = vertex_cover_weighted(crossing_edges, degrees)
    
    # Extract nodes to remove (in order)
    nodes_to_remove = []
    removal_order = [(i, vertex_cover_flags[i]) for i in range(len(vertex_cover_flags)) 
                     if vertex_cover_flags[i] > 0]
    removal_order.sort(key=lambda x: x[1])  # Sort by removal order
    
    for node_idx, _ in removal_order:
        nodes_to_remove.append(graph.vs[node_idx]['static_id'])
    
    return nodes_to_remove

def gnd_dismantling(graph, target_size_ratio=0.01, use_weighted=True, max_iterations=100):
    """
    Complete GND dismantling algorithm
    
    Args:
        graph: igraph Graph object
        target_size_ratio: Stop when GCC size < target_size_ratio * original_size
        use_weighted: Use weighted version (True) or standard version (False)
        max_iterations: Maximum number of iterations
    """
    ensure_static_id(graph)
    original_size = graph.vcount()
    target_size = max(1, int(target_size_ratio * original_size))
    
    temp_graph = graph.copy()
    removed_nodes = []
    iteration = 0
    
    print(f"GND Dismantling: Original size = {original_size}, Target size = {target_size}")
    
    while iteration < max_iterations:
        # Get current GCC
        gcc = get_gcc_igraph(temp_graph)
        gcc_size = gcc.vcount()
        
        print(f"Iteration {iteration}: GCC size = {gcc_size}")
        
        if gcc_size <= target_size:
            print(f"Target reached: GCC size {gcc_size} <= {target_size}")
            break
        
        if gcc_size <= 2 or gcc.ecount() == 0:
            print("GCC too small or no edges remaining")
            break
        
        # Perform one dismantling step
        step_removals = gnd_dismantling_step(gcc, use_weighted)
        
        if len(step_removals) == 0:
            print("No nodes to remove in this step")
            break
        
        # Map back to original graph indices and remove nodes
        nodes_to_remove_from_temp = []
        for static_id in step_removals:
            # Find vertex in temp_graph with this static_id
            for v in temp_graph.vs:
                if v['static_id'] == static_id:
                    nodes_to_remove_from_temp.append(v.index)
                    break
        
        if len(nodes_to_remove_from_temp) == 0:
            print("Could not map removal nodes back to temp graph")
            break
        
        # Remove nodes one by one (greedy order based on current degrees)
        for _ in range(len(nodes_to_remove_from_temp)):
            if temp_graph.vcount() == 0:
                break
                
            # Recalculate which nodes to remove (some may have been removed already)
            current_removals = []
            for static_id in step_removals:
                for v in temp_graph.vs:
                    if v['static_id'] == static_id:
                        current_removals.append(v.index)
                        break
            
            if len(current_removals) == 0:
                break
            
            # Choose node with minimum degree (for weighted) or maximum degree (for unweighted)
            degrees = temp_graph.degree(current_removals)
            if use_weighted:
                best_idx = current_removals[np.argmin(degrees)]
            else:
                best_idx = current_removals[np.argmax(degrees)]
            
            # Remove the node
            static_id = temp_graph.vs[best_idx]['static_id']
            removed_nodes.append(static_id)
            step_removals.remove(static_id)
            temp_graph.delete_vertices(best_idx)
        
        iteration += 1
    
    print(f"GND completed: Removed {len(removed_nodes)} nodes in {iteration} iterations")
    return removed_nodes

def gnd_spectral_dismantling(graph, max_steps=None):
    """
    GND-based spectral dismantling for comparison with other methods
    """
    return gnd_dismantling(graph, target_size_ratio=0.01, use_weighted=False)

def gnd_weighted_dismantling(graph, max_steps=None):
    """
    GND-based weighted dismantling for comparison with other methods
    """
    return gnd_dismantling(graph, target_size_ratio=0.01, use_weighted=True)

# Test the implementation
if __name__ == "__main__":
    print("Testing GND Python implementation...")
    
    # Create a test graph
    np.random.seed(42)
    test_graph = ig.Graph.Barabasi(n=100, m=2)
    
    print(f"Test graph: {test_graph.vcount()} nodes, {test_graph.ecount()} edges")
    
    # Test both versions
    print("\n=== Testing Weighted GND ===")
    start_time = time.time()
    removed_weighted = gnd_weighted_dismantling(test_graph)
    weighted_time = time.time() - start_time
    print(f"Weighted GND removed {len(removed_weighted)} nodes in {weighted_time:.2f}s")
    
    print("\n=== Testing Standard GND ===")
    start_time = time.time()
    removed_standard = gnd_spectral_dismantling(test_graph)
    standard_time = time.time() - start_time
    print(f"Standard GND removed {len(removed_standard)} nodes in {standard_time:.2f}s")