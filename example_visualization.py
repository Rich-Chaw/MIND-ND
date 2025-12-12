"""
Example script showing how to use the visualization functions with the baseline_develop code
"""

import pickle
import igraph as ig
from utils.community_detection import partition
from utils.heuristics import select
from utils.visualization import plot_communities, plot_selected_nodes, plot_stage1_analysis

def example_stage1_visualization():
    """Example of visualizing Stage 1 (preprocess) results"""
    
    # Load graph (adjust path as needed)
    try:
        with open("graphs/real/FINDER/Crime.pkl", 'rb') as f:
            g = pickle.load(f)
        print(f"Loaded graph: {g.vcount()} nodes, {g.ecount()} edges")
    except FileNotFoundError:
        print("Crime.pkl not found, creating a test graph...")
        # Create a test graph if the real data is not available
        g = ig.Graph.Erdos_Renyi(n=50, p=0.1)
        print(f"Created test graph: {g.vcount()} nodes, {g.ecount()} edges")
    
    # Stage 1: Community Detection
    print("\n=== Stage 1: Community Detection ===")
    partition_method = 'fast_greedy'
    partition_config = {'K': 10}
    
    clustering = partition(g, 
                          partition_method=partition_method, 
                          **partition_config)
    
    print(f"Found {len(clustering)} communities")
    for i in range(min(5, len(clustering))):  # Show first 5
        print(f"Community {i}: {len(clustering[i])} nodes")
    
    # Visualize communities
    plot_communities(g, clustering, 
                    title=f"Community Detection ({partition_method})",
                    figsize=(14, 8),
                    save_path="community_detection_result.png")
    
    # Stage 2: Node Selection
    print("\n=== Stage 2: Node Selection ===")
    heuristic = 'cbs'  # Community Bridge Score
    heuristic_config = {}
    
    selected_nodes = select(g, clustering, 
                           heuristic=heuristic, 
                           k=1,  # Select 1 node per community
                           **heuristic_config)
    
    print(f"Selected {len(selected_nodes)} nodes using {heuristic}: {selected_nodes}")
    
    # Visualize selected nodes
    plot_selected_nodes(g, clustering, selected_nodes,
                       heuristic_name=f"{heuristic.upper()} (k=1)",
                       title="Node Selection Results",
                       figsize=(14, 8),
                       save_path="node_selection_result.png")
    
    # Comprehensive analysis
    print("\n=== Comprehensive Stage 1 Analysis ===")
    plot_stage1_analysis(g, partition_method, heuristic, 
                        clustering, selected_nodes,
                        figsize=(16, 10),
                        save_path="stage1_comprehensive_analysis.png")
    
    return g, clustering, selected_nodes


def compare_heuristics_example():
    """Example comparing different heuristics"""
    
    # Create or load graph
    try:
        with open("graphs/real/FINDER/Crime.pkl", 'rb') as f:
            g = pickle.load(f)
    except FileNotFoundError:
        g = ig.Graph.Erdos_Renyi(n=30, p=0.15)
    
    # Get communities
    clustering = partition(g, partition_method='louvain')
    
    # Test different heuristics
    heuristics = ['degree', 'betweenness', 'cbs', 'global_aware_degree']
    
    print(f"\n=== Comparing Heuristics on {g.vcount()}-node graph ===")
    
    for heuristic in heuristics:
        print(f"\nTesting {heuristic}...")
        try:
            selected_nodes = select(g, clustering, heuristic=heuristic, k=1)
            print(f"{heuristic}: selected {len(selected_nodes)} nodes: {selected_nodes}")
            
            # Visualize each heuristic
            plot_selected_nodes(g, clustering, selected_nodes,
                               heuristic_name=heuristic.upper(),
                               title=f"Node Selection: {heuristic}",
                               figsize=(12, 6),
                               save_path=f"selection_{heuristic}.png")
        except Exception as e:
            print(f"Error with {heuristic}: {e}")


def small_graph_example():
    """Example with a small, interpretable graph"""
    
    print("\n=== Small Graph Example ===")
    
    # Create a small graph with clear community structure
    g = ig.Graph()
    g.add_vertices(12)
    
    # Community 1: nodes 0-3 (dense)
    g.add_edges([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)])
    
    # Community 2: nodes 4-7 (dense)
    g.add_edges([(4,5), (4,6), (4,7), (5,6), (5,7), (6,7)])
    
    # Community 3: nodes 8-11 (less dense)
    g.add_edges([(8,9), (8,10), (9,11), (10,11)])
    
    # Inter-community bridges (these should be selected by good heuristics)
    g.add_edges([(2,4), (3,8), (7,9)])  # Bridge nodes: 2, 3, 4, 7, 8, 9
    
    print(f"Small graph: {g.vcount()} nodes, {g.ecount()} edges")
    
    # Manual clustering (we know the ground truth)
    from utils.community_detection import SimpleClustering
    communities = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    clustering = SimpleClustering(communities, g.vcount())
    
    print(f"Ground truth communities: {[clustering[i] for i in range(len(clustering))]}")
    
    # Test different heuristics
    heuristics_to_test = ['degree', 'cbs', 'global_aware_degree']
    
    for heuristic in heuristics_to_test:
        print(f"\nTesting {heuristic}...")
        selected_nodes = select(g, clustering, heuristic=heuristic, k=1)
        print(f"Selected nodes: {selected_nodes}")
        
        # Show comprehensive analysis
        plot_stage1_analysis(g, "ground_truth", heuristic, 
                            clustering, selected_nodes,
                            figsize=(16, 10))
    
    print("Expected: Good heuristics should select bridge nodes [2, 3, 4, 7, 8, 9]")


if __name__ == "__main__":
    print("Running visualization examples...")
    
    # Run examples
    try:
        print("1. Running main Stage 1 visualization example...")
        example_stage1_visualization()
        
        print("\n" + "="*50)
        print("2. Running heuristics comparison...")
        compare_heuristics_example()
        
        print("\n" + "="*50)
        print("3. Running small graph example...")
        small_graph_example()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nVisualization examples completed!")