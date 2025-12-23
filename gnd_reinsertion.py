"""
Python implementation of GND reinsertion algorithm
Based on reinsertion.cpp from the GND project
"""

import numpy as np
import igraph as ig
from typing import List, Tuple, Dict
import networkx as nx

class UnionFind:
    """Disjoint set data structure for tracking connected components"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        self.size[px] += self.size[py]
        
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        return True
    
    def get_size(self, x):
        return self.size[self.find(x)]

def compute_component_change(graph, node_idx, present_nodes, union_find):
    """
    Compute how the giant component size would change if we add node_idx
    Returns (new_component_size, num_components_merged)
    """
    if node_idx in present_nodes:
        return 0, 0
    
    # Find all neighbors that are currently present
    neighbors = []
    for neighbor in graph.neighbors(node_idx):
        if neighbor in present_nodes:
            neighbors.append(neighbor)
    
    if len(neighbors) == 0:
        return 1, 0  # Just adds isolated node
    
    # Find unique components that would be merged
    components = set()
    total_size = 1  # Size of the new node itself
    
    for neighbor in neighbors:
        comp_root = union_find.find(neighbor)
        if comp_root not in components:
            components.add(comp_root)
            total_size += union_find.get_size(neighbor)
    
    num_components_merged = len(components)
    return total_size, num_components_merged

def gnd_reinsertion(graph, removed_nodes, threshold=1000000, sort_strategy=1):
    """
    Reinsertion algorithm following reinsertion.cpp
    
    Args:
        graph: Original igraph Graph
        removed_nodes: List of node indices that were removed (in removal order)
        threshold: Stop reinsertion if giant component exceeds this size
        sort_strategy: 0=keep order, 1=ascending degree, 2=descending degree
    
    Returns:
        final_removed_nodes: List of nodes that should remain removed after reinsertion
    """
    n = graph.vcount()
    
    # Convert to set for faster lookup
    removed_set = set(removed_nodes)
    
    # Initialize present nodes (all nodes except removed ones)
    present_nodes = set(range(n)) - removed_set
    
    # Initialize union-find for tracking components
    union_find = UnionFind(n)
    
    # Build initial connected components from present nodes
    edges = graph.get_edgelist()
    for u, v in edges:
        if u in present_nodes and v in present_nodes:
            union_find.union(u, v)
    
    # Calculate initial giant component size
    giant_size = 0
    if present_nodes:
        component_sizes = {}
        for node in present_nodes:
            root = union_find.find(node)
            if root not in component_sizes:
                component_sizes[root] = 0
            component_sizes[root] += 1
        giant_size = max(component_sizes.values()) if component_sizes else 0
    
    print(f"Initial giant component size: {giant_size}")
    print(f"Nodes to potentially reinsert: {len(removed_nodes)}")
    
    # Reinsertion process - try to reinsert nodes in reverse order
    final_removed = []
    
    for step in range(len(removed_nodes)):
        if giant_size >= threshold:
            print(f"Threshold reached: giant component size {giant_size} >= {threshold}")
            break
        
        # Find the best node to reinsert (one that increases giant component least)
        best_node = None
        best_size_increase = float('inf')
        best_components_merged = 0
        
        candidates = [node for node in removed_nodes if node not in final_removed]
        
        if not candidates:
            break
        
        for node in candidates:
            new_size, components_merged = compute_component_change(
                graph, node, present_nodes, union_find
            )
            
            if new_size < best_size_increase:
                best_node = node
                best_size_increase = new_size
                best_components_merged = components_merged
        
        if best_node is None:
            break
        
        # Add the best node back to the network
        present_nodes.add(best_node)
        
        # Update union-find structure
        neighbors = [n for n in graph.neighbors(best_node) if n in present_nodes and n != best_node]
        for neighbor in neighbors:
            union_find.union(best_node, neighbor)
        
        # Update giant component size
        giant_size = best_size_increase
        
        # Remove from candidates (it's been reinserted)
        removed_nodes = [node for node in removed_nodes if node != best_node]
        
        if step % 100 == 0:
            print(f"Step {step}: Reinserted node {best_node}, giant size now {giant_size}")
    
    # Remaining nodes in removed_nodes are the final set to remove
    final_removed = removed_nodes.copy()
    
    # Apply sorting strategy to final removed nodes
    if sort_strategy != 0 and final_removed:
        degrees = graph.degree()
        node_degrees = [(node, degrees[node]) for node in final_removed]
        
        if sort_strategy == 1:  # Ascending degree (better for weighted case)
            node_degrees.sort(key=lambda x: x[1])
        elif sort_strategy == 2:  # Descending degree (better for unweighted case)
            node_degrees.sort(key=lambda x: x[1], reverse=True)
        
        final_removed = [node for node, _ in node_degrees]
    
    print(f"Final nodes to remove after reinsertion: {len(final_removed)}")
    return final_removed

def gnd_with_reinsertion(graph, target_size_ratio=0.01, use_weighted=True, 
                        reinsertion_threshold=1000000, sort_strategy=1):
    """
    Complete GND algorithm with reinsertion
    
    Args:
        graph: igraph Graph object
        target_size_ratio: Initial dismantling target
        use_weighted: Use weighted GND version
        reinsertion_threshold: Stop reinsertion if component exceeds this
        sort_strategy: Sorting strategy for final removal order
    """
    from gnd_python import gnd_dismantling
    
    print("=== Phase 1: Initial Dismantling ===")
    removed_nodes = gnd_dismantling(graph, target_size_ratio, use_weighted)
    
    print(f"\n=== Phase 2: Reinsertion (threshold={reinsertion_threshold}) ===")
    final_removed = gnd_reinsertion(graph, removed_nodes, reinsertion_threshold, sort_strategy)
    
    return final_removed

# Test the reinsertion
if __name__ == "__main__":
    print("Testing GND Reinsertion...")
    
    # Create test graph
    np.random.seed(42)
    test_graph = ig.Graph.Barabasi(n=200, m=3)
    
    print(f"Test graph: {test_graph.vcount()} nodes, {test_graph.ecount()} edges")
    
    # Test complete algorithm with reinsertion
    final_removed = gnd_with_reinsertion(
        test_graph, 
        target_size_ratio=0.05,  # More aggressive initial dismantling
        use_weighted=True,
        reinsertion_threshold=50,  # Lower threshold for testing
        sort_strategy=1
    )
    
    print(f"\nFinal result: {len(final_removed)} nodes to remove")
    
    # Verify the result
    temp_graph = test_graph.copy()
    temp_graph.delete_vertices(final_removed)
    
    if temp_graph.vcount() > 0:
        gcc_size = max(temp_graph.connected_components().sizes())
        print(f"Resulting giant component size: {gcc_size}")
    else:
        print("All nodes removed")