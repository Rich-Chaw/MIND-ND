"""
Advanced Heuristics for Network Dismantling
Includes Community Bridge Score and other global-aware centrality measures
"""

import igraph as ig
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional, Union
from collections import defaultdict

# ------------------------------------------------------------------------------------------
def Degree(g:ig.Graph,k:int):
    '''
    return the nodes with top-k highest degree in a graph
    '''
    # handle invalid 
    if k==0: k = 1
    k = int(k)  # ensure k is integer
    
    # get all nodes and their degrees
    degrees = [(node, g.degree(node)) for node in range(g.vcount())]
    
    # sort by degree in descending order and get top k
    degrees.sort(key=lambda x: x[1], reverse=True)
    
    # return the node indices of top k nodes
    return [node for node, degree in degrees[:k]]

def Betweenness(g:ig.Graph, k:int):
    '''
    Return the nodes with top-k highest betweenness centrality in a graph
    '''
    # handle invalid 
    if k==0: k = 1
    k = int(k)  # ensure k is integer

    # calculate betweenness centrality for all nodes at once (more efficient)
    betweenness_scores = g.betweenness()
    
    # create list of (node, betweenness_score) tuples
    scores = [(node, betweenness_scores[node]) for node in range(g.vcount())]
    
    # sort by degree in descending order and get top k
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # return the node indices of top k nodes
    return [node for node, score in scores[:k]]

def PageRank(g:ig.Graph, k:int):
    '''
    Return the nodes with top-k highest PageRank scores in a graph
    '''
    # handle invalid
    if k == 0: k = 1
    k = int(k)  # ensure k is integer
    
    # calculate PageRank scores for all nodes
    pagerank_scores = g.pagerank()
    
    # create list of (node, pagerank_score) tuples
    scores = [(node, pagerank_scores[node]) for node in range(g.vcount())]
    
    # sort by PageRank score in descending order and get top k
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # return the node indices of top k nodes
    return [node for node, score in scores[:k]]


def CollectiveInfluence(g:ig.Graph, k:int, radius:int=2):
    '''
    Return the nodes with top-k highest Collective Influence (CI) in a graph
    CI(i) = (k_i - 1) * sum((k_j - 1) for j in neighbors at distance radius)
    
    Args:
        g: igraph Graph object
        k: number of top nodes to return
        radius: distance for calculating collective influence (default=2)
    '''
    # handle invalid
    if k == 0: k = 1
    k = int(k)  # ensure k is integer
    
    ci_scores = []
    
    for node in range(g.vcount()):
        # degree of current node
        k_i = g.degree(node)
        
        if k_i == 0:
            ci_scores.append((node, 0))
            continue
        
        # find neighbors at distance 'radius'
        # get shortest paths from node to all other nodes
        distances = g.shortest_paths(source=node)[0]
        
        # sum of (degree - 1) for all nodes at distance 'radius'
        ci_sum = sum(g.degree(j) - 1 for j in range(g.vcount()) 
                     if distances[j] == radius)
        
        # CI(i) = (k_i - 1) * sum
        ci = (k_i - 1) * ci_sum
        ci_scores.append((node, ci))
    
    # sort by CI score in descending order and get top k
    ci_scores.sort(key=lambda x: x[1], reverse=True)
    
    # return the node indices of top k nodes
    return [node for node, score in ci_scores[:k]]

PRIVATE_COMMUNITY_HEURISTICS = {
    'degree': Degree,
    'betweenness': Betweenness,
    'pagerank': PageRank,
    'ci': CollectiveInfluence
}

#----------------------------------------------------------------------------------------------
def CommunityBridgeScore(g: ig.Graph, clustering, k: int = 1) -> List[int]:
    """
    Community Bridge Score (CBS) - considers global connectivity when selecting nodes
    
    CBS(u) = Degree_out(u, non-C_i) × sum(Membership(u, C_i) for v in C_i)
    
    This implementation assumes non-overlapping communities and uses a simplified version:
    CBS(u) = External_Degree(u) × Community_Size(C_i)
    
    Args:
        g: igraph Graph object
        clustering: clustering object where clustering[i] gives nodes in community i
        k: number of top nodes to return
    
    Returns:
        List of node indices with highest CBS scores
    """
    if k == 0:
        k = 1
    k = int(k)
    
    # Build node to community mapping
    node_to_community = {}
    community_sizes = {}
    
    for comm_id in range(len(clustering)):
        comm_nodes = clustering[comm_id]
        community_sizes[comm_id] = len(comm_nodes)
        for node in comm_nodes:
            node_to_community[node] = comm_id
    
    cbs_scores = []
    
    for node in range(g.vcount()):
        if node not in node_to_community:
            cbs_scores.append((node, 0))
            continue
            
        node_community = node_to_community[node]
        community_size = community_sizes[node_community]
        
        # Calculate external degree (connections to nodes outside the community)
        neighbors = g.neighbors(node)
        external_degree = 0
        
        for neighbor in neighbors:
            if neighbor not in node_to_community or node_to_community[neighbor] != node_community:
                external_degree += 1
        
        # CBS = External_Degree × Community_Size
        cbs = external_degree * community_size
        cbs_scores.append((node, cbs))
    
    # Sort by CBS score in descending order and get top k
    cbs_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [node for node, score in cbs_scores[:k]]


def GlobalAwareDegree(g: ig.Graph, clustering, k: int = 1, alpha: float = 0.7) -> List[int]:
    """
    Global-Aware Degree centrality that combines local and global information
    
    GAD(u) = alpha × Local_Degree(u) + (1-alpha) × External_Degree(u)
    
    Args:
        g: igraph Graph object
        clustering: clustering object
        k: number of top nodes to return
        alpha: weight for local vs external degree (0.7 means 70% local, 30% external)
    
    Returns:
        List of node indices with highest GAD scores
    """
    if k == 0:
        k = 1
    k = int(k)
    
    # Build node to community mapping
    node_to_community = {}
    for comm_id in range(len(clustering)):
        for node in clustering[comm_id]:
            node_to_community[node] = comm_id
    
    gad_scores = []
    
    for node in range(g.vcount()):
        if node not in node_to_community:
            gad_scores.append((node, g.degree(node)))
            continue
            
        node_community = node_to_community[node]
        neighbors = g.neighbors(node)
        
        local_degree = 0
        external_degree = 0
        
        for neighbor in neighbors:
            if neighbor in node_to_community and node_to_community[neighbor] == node_community:
                local_degree += 1
            else:
                external_degree += 1
        
        # GAD = alpha × Local_Degree + (1-alpha) × External_Degree
        gad = alpha * local_degree + (1 - alpha) * external_degree
        gad_scores.append((node, gad))
    
    # Sort by GAD score in descending order and get top k
    gad_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [node for node, score in gad_scores[:k]]


def CommunityBetweenness(g: ig.Graph, clustering, k: int = 1) -> List[int]:
    """
    Community-aware betweenness centrality that considers inter-community paths
    
    Args:
        g: igraph Graph object
        clustering: clustering object
        k: number of top nodes to return
    
    Returns:
        List of node indices with highest community betweenness scores
    """
    if k == 0:
        k = 1
    k = int(k)
    
    # Build node to community mapping
    node_to_community = {}
    for comm_id in range(len(clustering)):
        for node in clustering[comm_id]:
            node_to_community[node] = comm_id
    
    # Calculate standard betweenness
    betweenness_scores = g.betweenness()
    
    # Weight betweenness by inter-community connectivity
    weighted_scores = []
    
    for node in range(g.vcount()):
        if node not in node_to_community:
            weighted_scores.append((node, betweenness_scores[node]))
            continue
            
        node_community = node_to_community[node]
        neighbors = g.neighbors(node)
        
        # Count connections to different communities
        external_communities = set()
        for neighbor in neighbors:
            if neighbor in node_to_community:
                neighbor_community = node_to_community[neighbor]
                if neighbor_community != node_community:
                    external_communities.add(neighbor_community)
        
        # Weight betweenness by number of external communities connected
        community_diversity = len(external_communities)
        weighted_betweenness = betweenness_scores[node] * (1 + community_diversity)
        
        weighted_scores.append((node, weighted_betweenness))
    
    # Sort by weighted betweenness in descending order and get top k
    weighted_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [node for node, score in weighted_scores[:k]]


def InterCommunityPageRank(g: ig.Graph, clustering, k: int = 1, damping: float = 0.85) -> List[int]:
    """
    Inter-community PageRank that gives higher weight to nodes connecting different communities
    
    Args:
        g: igraph Graph object
        clustering: clustering object
        k: number of top nodes to return
        damping: PageRank damping factor
    
    Returns:
        List of node indices with highest inter-community PageRank scores
    """
    if k == 0:
        k = 1
    k = int(k)
    
    # Build node to community mapping
    node_to_community = {}
    for comm_id in range(len(clustering)):
        for node in clustering[comm_id]:
            node_to_community[node] = comm_id
    
    # Calculate standard PageRank
    pagerank_scores = g.pagerank(damping=damping)
    
    # Weight PageRank by inter-community connectivity
    weighted_scores = []
    
    for node in range(g.vcount()):
        if node not in node_to_community:
            weighted_scores.append((node, pagerank_scores[node]))
            continue
            
        node_community = node_to_community[node]
        neighbors = g.neighbors(node)
        
        # Calculate inter-community ratio
        total_neighbors = len(neighbors)
        external_neighbors = 0
        
        for neighbor in neighbors:
            if neighbor in node_to_community and node_to_community[neighbor] != node_community:
                external_neighbors += 1
        
        # Inter-community ratio
        if total_neighbors > 0:
            inter_ratio = external_neighbors / total_neighbors
        else:
            inter_ratio = 0
        
        # Weight PageRank by inter-community connectivity
        weighted_pagerank = pagerank_scores[node] * (1 + inter_ratio)
        weighted_scores.append((node, weighted_pagerank))
    
    # Sort by weighted PageRank in descending order and get top k
    weighted_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [node for node, score in weighted_scores[:k]]


def CommunityCollectiveInfluence(g: ig.Graph, clustering, k: int = 1, radius: int = 2, 
                                beta: float = 0.5) -> List[int]:
    """
    Community-aware Collective Influence that considers both local and global influence
    
    CCI(u) = beta × CI_local(u) + (1-beta) × CI_global(u)
    
    Args:
        g: igraph Graph object
        clustering: clustering object
        k: number of top nodes to return
        radius: radius for collective influence calculation
        beta: weight for local vs global influence
    
    Returns:
        List of node indices with highest community collective influence scores
    """
    if k == 0:
        k = 1
    k = int(k)
    
    # Build node to community mapping
    node_to_community = {}
    for comm_id in range(len(clustering)):
        for node in clustering[comm_id]:
            node_to_community[node] = comm_id
    
    cci_scores = []
    
    for node in range(g.vcount()):
        node_degree = g.degree(node)
        
        if node_degree == 0:
            cci_scores.append((node, 0))
            continue
        
        # Get nodes at distance 'radius'
        distances = g.shortest_paths(source=node)[0]
        nodes_at_radius = [i for i in range(g.vcount()) if distances[i] == radius]
        
        if node not in node_to_community:
            # Standard CI if node not in any community
            ci_sum = sum(g.degree(j) - 1 for j in nodes_at_radius)
            ci = (node_degree - 1) * ci_sum
            cci_scores.append((node, ci))
            continue
        
        node_community = node_to_community[node]
        
        # Separate local and global influence
        local_ci_sum = 0
        global_ci_sum = 0
        
        for j in nodes_at_radius:
            degree_j = g.degree(j) - 1
            if j in node_to_community and node_to_community[j] == node_community:
                local_ci_sum += degree_j
            else:
                global_ci_sum += degree_j
        
        # Community Collective Influence
        local_ci = (node_degree - 1) * local_ci_sum
        global_ci = (node_degree - 1) * global_ci_sum
        cci = beta * local_ci + (1 - beta) * global_ci
        
        cci_scores.append((node, cci))
    
    # Sort by CCI score in descending order and get top k
    cci_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [node for node, score in cci_scores[:k]]


# Registry of community-aware heuristics
GLOBAL_COMMUNITY_HEURISTICS = {
    'cbs': CommunityBridgeScore,
    'global_aware_degree': GlobalAwareDegree,
    'community_betweenness': CommunityBetweenness,
    'inter_community_pagerank': InterCommunityPageRank,
    'community_collective_influence': CommunityCollectiveInfluence,
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

def select(g: ig.Graph, clustering, heuristic: str = 'cbs', k: int = 1, **kwargs) -> List[int]:
    """
    Unified interface for community-aware node selection
    
    Args:
        g: igraph Graph object
        clustering: clustering object where clustering[i] gives nodes in community i
        heuristic: heuristic method name ('cbs', 'global_aware_degree', etc.)
        k: number of top nodes to return
        **kwargs: additional arguments for specific methods
    
    Returns:
        List of node indices with highest scores according to the specified method
    """
    
    if heuristic in PRIVATE_COMMUNITY_HEURISTICS:
        selected_nodes = set()
        for i in range(len(clustering)):
            comm_nodes = clustering[i]
            if comm_nodes:
                # Create a subgraph of just this community to apply heuristic
                comm_subgraph = g.subgraph(comm_nodes)
                # Use the provided heuristic method to find best node in this community
                best_nodes_in_comm = _call_with_valid_params(
                                        PRIVATE_COMMUNITY_HEURISTICS[heuristic], 
                                        g=comm_subgraph,
                                        k=k, 
                                        **kwargs)
                if best_nodes_in_comm:
                    best_node_in_comm_subgraph = best_nodes_in_comm[0]
                    # Map from community subgraph index -> LCC subgraph index -> current graph index
                    current_graph_node_id = comm_nodes[best_node_in_comm_subgraph]
                    selected_nodes.add(current_graph_node_id)
        return list(selected_nodes)

    if heuristic in GLOBAL_COMMUNITY_HEURISTICS:
        # Automatically filter kwargs to only pass valid parameters
        selected_nodes = _call_with_valid_params(
                                    GLOBAL_COMMUNITY_HEURISTICS[heuristic], 
                                    g=g,
                                    clustering=clustering, 
                                    k = k*len(clustering),
                                    **kwargs)
        return selected_nodes


def test_community_heuristics():
    """Test function for community-aware heuristics"""
    print("Testing community-aware heuristics...")
    
    # Create a test graph with clear community structure
    g = ig.Graph()
    g.add_vertices(12)
    
    # Community 1: nodes 0-3
    g.add_edges([(0,1), (0,2), (1,2), (1,3), (2,3)])
    
    # Community 2: nodes 4-7
    g.add_edges([(4,5), (4,6), (5,6), (5,7), (6,7)])
    
    # Community 3: nodes 8-11
    g.add_edges([(8,9), (8,10), (9,10), (9,11), (10,11)])
    
    # Inter-community bridges
    g.add_edges([(2,4), (3,8), (7,9)])  # Bridge nodes: 2, 3, 4, 7, 8, 9
    
    print(f"Test graph: {g.vcount()} nodes, {g.ecount()} edges")
    
    # Create simple clustering
    from community_detection import SimpleClustering
    communities = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    clustering = SimpleClustering(communities, g.vcount())
    
    print(f"Communities: {[clustering[i] for i in range(len(clustering))]}")
    
    # Test different heuristics
    print("\nTesting heuristics:")
    
    # Community Bridge Score
    cbs_nodes = CommunityBridgeScore(g, clustering, k=3)
    print(f"Community Bridge Score top 3: {cbs_nodes}")
    
    # Global-Aware Degree
    gad_nodes = GlobalAwareDegree(g, clustering, k=3)
    print(f"Global-Aware Degree top 3: {gad_nodes}")
    
    # Community Betweenness
    cb_nodes = CommunityBetweenness(g, clustering, k=3)
    print(f"Community Betweenness top 3: {cb_nodes}")
    
    # Inter-Community PageRank
    icpr_nodes = InterCommunityPageRank(g, clustering, k=3)
    print(f"Inter-Community PageRank top 3: {icpr_nodes}")
    
    # Community Collective Influence
    cci_nodes = CommunityCollectiveInfluence(g, clustering, k=3)
    print(f"Community Collective Influence top 3: {cci_nodes}")
    
    print("\nExpected bridge nodes: [2, 3, 4, 7, 8, 9]")
    print("Heuristics should identify these as important nodes.")



if __name__ == "__main__":
    test_community_heuristics()