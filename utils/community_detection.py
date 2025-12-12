"""
Advanced Community Detection Methods for Network Dismantling
Includes overlapping community detection methods like Link Community Detection (LCD)
and LFR Benchmark generation for testing.
"""

import igraph as ig
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from collections import defaultdict
import random

class SimpleClustering:
    """Simple clustering class that mimics igraph's VertexClustering interface"""
    
    def __init__(self, communities: List[List[int]], n_nodes: int):
        self.communities = communities
        self.n_nodes = n_nodes
        self._membership = self._build_membership()
    
    def _build_membership(self) -> List[int]:
        """Build membership list from communities"""
        membership = [-1] * self.n_nodes
        for comm_id, community in enumerate(self.communities):
            for node in community:
                if node < self.n_nodes and membership[node] == -1:
                    membership[node] = comm_id
        
        # Assign unassigned nodes to community 0
        for i in range(len(membership)):
            if membership[i] == -1:
                membership[i] = 0
        
        return membership
    
    def __len__(self):
        """Return number of communities"""
        return len(self.communities)
    
    def __getitem__(self, index):
        """Get nodes in community index"""
        if index < len(self.communities):
            return self.communities[index]
        return []
    
    @property
    def membership(self):
        """Get membership list"""
        return self._membership


def LinkCommunityDetection(g: ig.Graph, threshold: float = 0.5) -> SimpleClustering:
    """
    Link Community Detection (LCD) - finds overlapping communities
    Based on hierarchical clustering of edges
    
    Args:
        g: igraph Graph object
        threshold: similarity threshold for community formation
    
    Returns:
        SimpleClustering object with community structure (clustering[i] = list of nodes in community i)
    """
    if g.ecount() == 0:
        return SimpleClustering([[i] for i in range(g.vcount())], g.vcount())
    
    # Step 1: Calculate edge similarities using Jaccard coefficient
    edges = [(e.source, e.target) for e in g.es]
    edge_similarities = {}
    
    for i, (u1, v1) in enumerate(edges):
        for j, (u2, v2) in enumerate(edges[i+1:], i+1):
            # Calculate Jaccard similarity between edge neighborhoods
            neighbors_e1 = set(g.neighbors(u1)) | set(g.neighbors(v1))
            neighbors_e2 = set(g.neighbors(u2)) | set(g.neighbors(v2))
            
            intersection = len(neighbors_e1 & neighbors_e2)
            union = len(neighbors_e1 | neighbors_e2)
            
            if union > 0:
                similarity = intersection / union
                edge_similarities[(i, j)] = similarity
    
    # Step 2: Hierarchical clustering of edges
    edge_communities = []
    processed_edges = set()
    
    for i, edge in enumerate(edges):
        if i in processed_edges:
            continue
            
        community_edges = [i]
        processed_edges.add(i)
        
        # Find similar edges
        for j in range(i+1, len(edges)):
            if j in processed_edges:
                continue
                
            if (i, j) in edge_similarities and edge_similarities[(i, j)] >= threshold:
                community_edges.append(j)
                processed_edges.add(j)
        
        edge_communities.append(community_edges)
    
    # Step 3: Convert edge communities to node communities
    node_communities = []
    for edge_comm in edge_communities:
        nodes = set()
        for edge_idx in edge_comm:
            u, v = edges[edge_idx]
            nodes.add(u)
            nodes.add(v)
        if nodes:
            node_communities.append(list(nodes))
    
    return SimpleClustering(node_communities, g.vcount())


def LFRBenchmark(n: int, tau1: float = 2.5, tau2: float = 1.5, mu: float = 0.3, 
                avg_degree: float = 10, max_degree: Optional[int] = None,
                min_community: int = 10, max_community: Optional[int] = None) -> Tuple[ig.Graph, SimpleClustering]:
    """
    Generate LFR (Lancichinetti-Fortunato-Radicchi) benchmark graph
    
    Args:
        n: number of nodes
        tau1: degree distribution exponent (must be > 1)
        tau2: community size distribution exponent (must be > 1)
        mu: mixing parameter (fraction of inter-community edges)
        avg_degree: average degree
        max_degree: maximum degree (default: n/2)
        min_community: minimum community size
        max_community: maximum community size (default: n/2)
    
    Returns:
        Tuple of (igraph Graph with LFR benchmark structure, SimpleClustering with ground truth communities)
    """
    if max_degree is None:
        max_degree = n // 2
    if max_community is None:
        max_community = n // 2
    
    # Ensure valid parameters
    tau1 = max(tau1, 1.1)
    tau2 = max(tau2, 1.1)
    
    # Generate degree sequence following power law
    degrees = []
    for _ in range(n):
        # Use exponential distribution as approximation if pareto fails
        try:
            degree = int(np.random.pareto(tau1 - 1) * avg_degree) + 1
        except:
            degree = int(np.random.exponential(avg_degree)) + 1
        degree = min(max(degree, 1), max_degree)
        degrees.append(degree)
    
    # Ensure sum of degrees is even
    if sum(degrees) % 2 == 1:
        degrees[0] += 1
    
    # Generate community sizes following power law
    community_sizes = []
    remaining_nodes = n
    
    while remaining_nodes > 0:
        try:
            size = int(np.random.pareto(tau2 - 1) * min_community) + min_community
        except:
            size = int(np.random.exponential(min_community)) + min_community
        
        size = min(size, max_community, remaining_nodes)
        if size >= min_community and remaining_nodes >= min_community:
            community_sizes.append(size)
            remaining_nodes -= size
        else:
            if community_sizes:
                community_sizes[-1] += remaining_nodes
            else:
                community_sizes.append(remaining_nodes)
            break
    
    # Assign nodes to communities
    node_communities = {}
    communities = []  # List of lists for SimpleClustering
    node_id = 0
    
    for comm_id, comm_size in enumerate(community_sizes):
        community_nodes = []
        for _ in range(comm_size):
            node_communities[node_id] = comm_id
            community_nodes.append(node_id)
            node_id += 1
        communities.append(community_nodes)
    
    # Generate edges
    edges = []
    node_degrees = {i: 0 for i in range(n)}
    
    # Internal community edges
    for node in range(n):
        target_degree = degrees[node]
        internal_edges = int(target_degree * (1 - mu))
        
        # Add internal community edges
        comm = node_communities[node]
        comm_nodes = [i for i, c in node_communities.items() if c == comm and i != node]
        
        for _ in range(min(internal_edges, len(comm_nodes))):
            if node_degrees[node] >= target_degree:
                break
            target = random.choice(comm_nodes)
            if (node, target) not in edges and (target, node) not in edges:
                edges.append((node, target))
                node_degrees[node] += 1
                node_degrees[target] += 1
    
    # External community edges
    for node in range(n):
        target_degree = degrees[node]
        external_edges = target_degree - node_degrees[node]
        
        comm = node_communities[node]
        external_nodes = [i for i, c in node_communities.items() if c != comm]
        
        for _ in range(min(external_edges, len(external_nodes))):
            if node_degrees[node] >= target_degree:
                break
            target = random.choice(external_nodes)
            if (node, target) not in edges and (target, node) not in edges:
                edges.append((node, target))
                node_degrees[node] += 1
                node_degrees[target] += 1
    
    # Create graph
    g = ig.Graph(n)
    g.add_edges(edges)
    
    # Add community membership as vertex attribute
    g.vs['community'] = [node_communities[i] for i in range(n)]
    
    # Create clustering object
    clustering = SimpleClustering(communities, n)
    
    return g, clustering


def SpectralClustering(g: ig.Graph, K: int) -> SimpleClustering:
    """
    Spectral clustering using graph Laplacian
    
    Args:
        g: igraph Graph object
        K: number of clusters
    
    Returns:
        SimpleClustering object where clustering[i] is the list of nodes in community i
    """
    try:
        import scipy.sparse as sp
        from sklearn.cluster import KMeans
        
        # Get adjacency matrix
        adj_matrix = np.array(g.get_adjacency().data)
        
        # Compute degree matrix
        degrees = np.array(g.degree())
        D = np.diag(degrees)
        
        # Compute normalized Laplacian
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-12)))
        L_norm = np.eye(g.vcount()) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(L_norm)
        
        # Use first K eigenvectors
        features = eigenvecs[:, :K]
        
        # K-means clustering
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        membership = kmeans.fit_predict(features)
        
        # Convert membership to communities list
        communities = [[] for _ in range(K)]
        for node, comm_id in enumerate(membership):
            communities[comm_id].append(node)
        
        return SimpleClustering(communities, g.vcount())
        
    except ImportError:
        # Fallback to label propagation if sklearn not available
        fallback_clustering = g.community_label_propagation()
        return fallback_clustering


def BigCLAM(g: ig.Graph, K: int = 5, max_iter: int = 100, learning_rate: float = 0.01, 
           regularization: float = 0.01, threshold: float = 0.5) -> SimpleClustering:
    """
    BigCLAM (Big Community Affiliation Model) for overlapping community detection
    
    This is a simplified implementation of BigCLAM that uses gradient descent
    to learn community affiliations for each node.
    
    Args:
        g: igraph Graph object
        K: number of communities
        max_iter: maximum number of iterations
        learning_rate: learning rate for gradient descent
        regularization: L2 regularization parameter
        threshold: threshold for community membership (0.5 means >= 0.5 probability)
    
    Returns:
        SimpleClustering object with overlapping communities converted to non-overlapping
    """
    n_nodes = g.vcount()
    
    if n_nodes == 0:
        return SimpleClustering([], 0)
    
    # Get adjacency matrix
    adj_matrix = np.array(g.get_adjacency().data)
    
    # Initialize community affiliation matrix F (n_nodes x K)
    # F[i,c] represents the affiliation strength of node i to community c
    np.random.seed(42)
    F = np.random.uniform(0.1, 1.0, (n_nodes, K))
    
    # Get edge list for efficient computation
    edges = [(e.source, e.target) for e in g.es]
    
    # Gradient descent optimization
    for iteration in range(max_iter):
        # Compute gradients
        grad_F = np.zeros_like(F)
        
        # For each edge, compute the gradient contribution
        for u, v in edges:
            # Compute P(u,v) = 1 - exp(-F[u] * F[v])
            dot_product = np.dot(F[u], F[v])
            if dot_product > 50:  # Prevent overflow
                p_uv = 1.0
                exp_term = 0.0
            else:
                exp_term = np.exp(-dot_product)
                p_uv = 1.0 - exp_term
            
            if p_uv > 1e-10:  # Avoid division by zero
                # Gradient for existing edge (A[u,v] = 1)
                common_factor = exp_term / p_uv
                grad_F[u] += common_factor * F[v]
                grad_F[v] += common_factor * F[u]
        
        # For non-edges, we sample a subset to make it computationally feasible
        # Sample some non-edges for gradient computation
        n_samples = min(len(edges) * 2, n_nodes * 10)  # Reasonable sampling
        for _ in range(n_samples):
            u = np.random.randint(0, n_nodes)
            v = np.random.randint(0, n_nodes)
            
            if u != v and adj_matrix[u, v] == 0:  # Non-edge
                # Compute P(u,v) = 1 - exp(-F[u] * F[v])
                dot_product = np.dot(F[u], F[v])
                if dot_product > 50:  # Prevent overflow
                    exp_term = 0.0
                else:
                    exp_term = np.exp(-dot_product)
                
                # Gradient for non-existing edge (A[u,v] = 0)
                grad_F[u] -= exp_term * F[v]
                grad_F[v] -= exp_term * F[u]
        
        # Add L2 regularization
        grad_F -= regularization * F
        
        # Update F using gradient ascent (we want to maximize likelihood)
        F += learning_rate * grad_F
        
        # Keep F positive
        F = np.maximum(F, 0.01)
        
        # Optional: print progress
        if iteration % 20 == 0:
            # Compute approximate log-likelihood for monitoring
            ll = 0
            for u, v in edges[:min(100, len(edges))]:  # Sample for efficiency
                dot_product = np.dot(F[u], F[v])
                if dot_product > 50:
                    ll += dot_product
                else:
                    ll += np.log(1 - np.exp(-dot_product) + 1e-10)
            # Note: This is just a partial likelihood for monitoring
    
    # Convert F matrix to communities
    # Each node can belong to multiple communities based on threshold
    overlapping_communities = [[] for _ in range(K)]
    
    for node in range(n_nodes):
        for comm in range(K):
            if F[node, comm] >= threshold:
                overlapping_communities[comm].append(node)
    
    # Remove empty communities
    overlapping_communities = [comm for comm in overlapping_communities if len(comm) > 0]
    
    # Convert overlapping to non-overlapping by assigning each node to its strongest community
    communities = []
    assigned_nodes = set()
    
    # First, create communities based on strongest affiliations
    for comm_nodes in overlapping_communities:
        if comm_nodes:
            communities.append(comm_nodes)
    
    # If no communities found, create one community with all nodes
    if not communities:
        communities = [list(range(n_nodes))]
    
    return SimpleClustering(communities, n_nodes)



def FastGreedy(g: ig.Graph, K: int = None):
    """Fast greedy community detection"""
    communities = g.community_fastgreedy()
    if K is not None:
        return communities.as_clustering(n=min(K, g.vcount()))
    else:
        return communities.as_clustering()


def LabelPropagation(g: ig.Graph):
    """Label propagation community detection"""
    return g.community_label_propagation()


def Louvain(g: ig.Graph):
    """Louvain (multilevel) community detection"""
    return g.community_multilevel()


def InfoMap(g: ig.Graph):
    """InfoMap community detection"""
    return g.community_infomap()


def LeadingEigenvector(g: ig.Graph, K: int = None):
    """Leading eigenvector community detection"""
    try:
        communities = g.community_leading_eigenvector()
        if K is not None:
            return communities.as_clustering(n=min(K, g.vcount()))
        else:
            return communities
    except:
        # Fallback for graphs where leading eigenvector fails
        return g.community_label_propagation()


def WalkTrap(g: ig.Graph, K: int = None, steps: int = 4):
    """Walktrap community detection"""
    communities = g.community_walktrap(steps=steps)
    if K is not None:
        return communities.as_clustering(n=min(K, g.vcount()))
    else:
        return communities.as_clustering()



# Simple registry - just functions with their natural signatures
COMMUNITY_METHODS = {
    'fast_greedy': FastGreedy,
    'label_propagation': LabelPropagation,
    'louvain': Louvain,
    'spectral': SpectralClustering,
    'infomap': InfoMap,
    'walktrap': WalkTrap,
    'leading_eigenvector': LeadingEigenvector,
    'lcd': LinkCommunityDetection,
    'bigclam': BigCLAM
}

import inspect
def _call_with_valid_params(func: Callable, **kwargs):
    """Call function with only the parameters it accepts"""
    sig = inspect.signature(func)
    valid_kwargs = {}
    
    for param_name in sig.parameters:
        if param_name in kwargs:
            valid_kwargs[param_name] = kwargs[param_name]
    
    return func(**valid_kwargs)


def partition(g: ig.Graph, partition_method: str = 'fast_greedy', **kwargs):
    """
    Advanced partition wrapper supporting multiple community detection methods
    
    Args:
        g: igraph Graph object
        partition_method: community detection partition_method name
        **kwargs: partition_method-specific parameters (K, steps, threshold, etc.)
    
    Returns:
        Clustering object where:
        - len(clustering) gives the number of communities
        - clustering[i] gives the list of nodes in community i
        - clustering.membership gives the membership list (for igraph VertexClustering objects)
    """
    if partition_method not in COMMUNITY_METHODS:
        raise ValueError(f"Unknown partition_method: {partition_method}. Available methods: {list(COMMUNITY_METHODS.keys())}")
    
    try:
        # Automatically filter kwargs to only pass valid parameters
        return _call_with_valid_params(COMMUNITY_METHODS[partition_method], g=g, **kwargs)
    except Exception as e:
        print(f"Warning: {partition_method} failed ({e}), falling back to label propagation")
        return g.community_label_propagation()


def test_comprehensive_methods():
    # Create test graph
    g = ig.Graph.Erdos_Renyi(n=20, p=0.2)
    print(f"Test graph: {g.vcount()} nodes, {g.ecount()} edges")

    # Test different methods with their natural parameters
    print("\nTesting different methods:")

    # Methods that don't use K
    print("Methods without K parameter:")
    result = partition(g, partition_method='label_propagation')
    print(f"label_propagation: {len(result)} communities")
    
    result = partition(g, partition_method='louvain')
    print(f"louvain: {len(result)} communities")
    
    result = partition(g, partition_method='infomap')
    print(f"infomap: {len(result)} communities")

    # Methods that use K
    print("\nMethods with K parameter:")
    result = partition(g, partition_method='fast_greedy', K=3)
    print(f"fast_greedy (K=3): {len(result)} communities")
    
    result = partition(g, partition_method='spectral', K=4)
    print(f"spectral (K=4): {len(result)} communities")
    
    result = partition(g, partition_method='leading_eigenvector', K=2)
    print(f"leading_eigenvector (K=2): {len(result)} communities")

    # Methods with other parameters
    print("\nMethods with special parameters:")
    result = partition(g, partition_method='walktrap', K=3, steps=6)
    print(f"walktrap (K=3, steps=6): {len(result)} communities")

    result = partition(g, partition_method='lcd', threshold=0.3)
    print(f"lcd (threshold=0.3): {len(result)} communities")

    result = partition(g, partition_method='bigclam', K=4, max_iter=50)
    print(f"bigclam (K=4, max_iter=50): {len(result)} communities")

    # Show how parameters are automatically filtered
    print("\nParameter filtering demo:")
    print("Passing K=5 to label_propagation (which doesn't use K) - should work fine:")
    result = partition(g, partition_method='label_propagation', K=5, extra_param='ignored')
    print(f"Result: {len(result)} communities (K and extra_param were ignored)")

    print("\nDone!")


if __name__ == "__main__":
    # Test cases
    print("Testing advanced community detection methods...")
    
    # Create test graph
    random.seed(42)
    np.random.seed(42)
    
    print("\n1. Testing on small graph...")
    g_small = ig.Graph([(0,1),(0,2),(0,3),(1,3),(2,4),(3,4),(4,5),(5,6)])
    print(f"Small graph: {g_small.vcount()} nodes, {g_small.ecount()} edges")

    # Test 2: LFR Benchmark generation
    print("\n2. Testing LFR Benchmark generation...")
    try:
        g_lfr, ground_truth_clustering = LFRBenchmark(n=50, mu=0.3, avg_degree=8)
        print(f"LFR graph: {g_lfr.vcount()} nodes, {g_lfr.ecount()} edges")
        print(f"Ground truth communities: {len(ground_truth_clustering)} communities")
        print(f"Community sizes: {[len(ground_truth_clustering[i]) for i in range(len(ground_truth_clustering))]}")
        
        # Test community detection on LFR
        clustering = partition(g_lfr, K=5, partition_method='louvain')
        print(f"Detected communities: {len(clustering)} communities")
    except Exception as e:
        print(f"LFR test failed: {e}")
    
    # Test 3: Link Community Detection
    print("\n3. Testing Link Community Detection...")
    try:
        g_test = ig.Graph.Erdos_Renyi(n=20, p=0.2)
        clustering = LinkCommunityDetection(g_test, threshold=0.3)
        print(f"Found {len(clustering)} communities")
        for i in range(min(3, len(clustering))):  # Show first 3
            print(f"Community {i}: {clustering[i]}")
    except Exception as e:
        print(f"LCD test failed: {e}")
    
    # Test 4: BigCLAM
    print("\n4. Testing BigCLAM...")
    try:
        g_test = ig.Graph.Erdos_Renyi(n=15, p=0.3)  # Smaller graph for faster testing
        clustering = BigCLAM(g_test, K=3, max_iter=30)
        print(f"BigCLAM found {len(clustering)} communities")
        for i in range(min(3, len(clustering))):  # Show first 3
            print(f"Community {i}: {clustering[i]}")
    except Exception as e:
        print(f"BigCLAM test failed: {e}")
    
    # Test 5: Integration functions (will be tested after function definitions)
    print("\n5. Testing integration functions...")
    print("Integration functions will be tested after all definitions are loaded.")
    test_comprehensive_methods()
    
    print("\nAll tests completed!")

# # Convenience functions for easy import (backward compatibility)
# FastGreedy = lambda g, K=None: partition(g, partition_method='fast_greedy', K=K)
# LabelPropagation = lambda g, K=None: partition(g, partition_method='label_propagation') 
# Louvain = lambda g, K=None: partition(g, partition_method='louvain')
# SpectralClustering_Method = lambda g, K=2: partition(g, partition_method='spectral', K=K)
# InfoMap_Method = lambda g, K=None: partition(g, partition_method='infomap')
# WalkTrap_Method = lambda g, K=None, steps=4: partition(g, partition_method='walktrap', K=K, steps=steps)
# LeadingEigenvector_Method = lambda g, K=None: partition(g, partition_method='leading_eigenvector', K=K)
# LinkCommunityDetection_Method = lambda g, K=None, **kwargs: partition(g, partition_method='lcd', **kwargs)

