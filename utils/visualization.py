"""
Visualization functions for community detection and node selection in network dismantling
"""

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Union
import random


def plot_communities(g: ig.Graph, clustering, title: str = "Community Detection Results", 
                    figsize: tuple = (12, 8), layout: str = 'auto', 
                    node_size: int = 300, edge_alpha: float = 0.6,
                    save_path: Optional[str] = None, show_labels: bool = True) -> None:
    """
    Visualize graph with community structure
    
    Args:
        g: igraph Graph object
        clustering: clustering object where clustering[i] gives nodes in community i
        title: plot title
        figsize: figure size (width, height)
        layout: layout algorithm ('auto', 'fr', 'kk', 'circle', 'grid', 'random')
        node_size: size of nodes
        edge_alpha: transparency of edges
        save_path: path to save the plot (optional)
        show_labels: whether to show node labels
    """
    if g.vcount() == 0:
        print("Empty graph - nothing to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Choose layout
    if layout == 'auto':
        if g.vcount() < 100:
            pos = g.layout_fruchterman_reingold()
        else:
            pos = g.layout_kamada_kawai()
    elif layout == 'fr':
        pos = g.layout_fruchterman_reingold()
    elif layout == 'kk':
        pos = g.layout_kamada_kawai()
    elif layout == 'circle':
        pos = g.layout_circle()
    elif layout == 'grid':
        pos = g.layout_grid()
    elif layout == 'random':
        pos = g.layout_random()
    else:
        pos = g.layout_fruchterman_reingold()
    
    # Convert igraph layout to numpy array
    pos_array = np.array(pos.coords)
    
    # Generate colors for communities
    n_communities = len(clustering)
    if n_communities <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_communities))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_communities, 20)))
        if n_communities > 20:
            # Generate additional colors
            additional_colors = plt.cm.Set3(np.linspace(0, 1, n_communities - 20))
            colors = np.vstack([colors, additional_colors])
    
    # Create node color mapping
    node_colors = ['lightgray'] * g.vcount()  # Default color for unassigned nodes
    community_info = {}
    
    for comm_id in range(len(clustering)):
        comm_nodes = clustering[comm_id]
        if comm_nodes:
            color = colors[comm_id % len(colors)]
            community_info[comm_id] = {
                'nodes': comm_nodes,
                'color': color,
                'size': len(comm_nodes)
            }
            for node in comm_nodes:
                if node < len(node_colors):
                    node_colors[node] = color
    
    # Plot 1: Original graph without community coloring
    ax1.set_title("Original Graph", fontsize=14, fontweight='bold')
    
    # Draw edges
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        ax1.plot(x_coords, y_coords, 'k-', alpha=edge_alpha, linewidth=0.5)
    
    # Draw nodes
    ax1.scatter(pos_array[:, 0], pos_array[:, 1], 
               c='lightblue', s=node_size, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add node labels if requested
    if show_labels and g.vcount() <= 50:  # Only show labels for small graphs
        for i in range(g.vcount()):
            ax1.annotate(str(i), (pos_array[i, 0], pos_array[i, 1]), 
                        fontsize=8, ha='center', va='center')
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Plot 2: Graph with community coloring
    ax2.set_title(f"{title}\n{n_communities} Communities", fontsize=14, fontweight='bold')
    
    # Draw edges
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        
        # Color edges differently if they connect different communities
        source_comm = None
        target_comm = None
        
        for comm_id, info in community_info.items():
            if edge.source in info['nodes']:
                source_comm = comm_id
            if edge.target in info['nodes']:
                target_comm = comm_id
        
        if source_comm is not None and target_comm is not None and source_comm != target_comm:
            # Inter-community edge
            ax2.plot(x_coords, y_coords, 'red', alpha=edge_alpha*1.5, linewidth=1.5)
        else:
            # Intra-community edge
            ax2.plot(x_coords, y_coords, 'gray', alpha=edge_alpha, linewidth=0.5)
    
    # Draw nodes with community colors
    ax2.scatter(pos_array[:, 0], pos_array[:, 1], 
               c=node_colors, s=node_size, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add node labels if requested
    if show_labels and g.vcount() <= 50:
        for i in range(g.vcount()):
            ax2.annotate(str(i), (pos_array[i, 0], pos_array[i, 1]), 
                        fontsize=8, ha='center', va='center')
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Add legend for communities
    if n_communities <= 15:  # Only show legend for reasonable number of communities
        legend_elements = []
        for comm_id, info in community_info.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=info['color'], markersize=10,
                                            label=f'Comm {comm_id} ({info["size"]} nodes)'))
        
        ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add statistics
    stats_text = f"Nodes: {g.vcount()}\nEdges: {g.ecount()}\nCommunities: {n_communities}"
    if community_info:
        avg_comm_size = np.mean([info['size'] for info in community_info.values()])
        stats_text += f"\nAvg Community Size: {avg_comm_size:.1f}"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_selected_nodes(g: ig.Graph, clustering, selected_nodes: List[int], 
                       heuristic_name: str = "Selected Nodes", 
                       title: str = "Node Selection Results",
                       figsize: tuple = (12, 8), layout: str = 'auto',
                       node_size: int = 300, edge_alpha: float = 0.6,
                       save_path: Optional[str] = None, show_labels: bool = True,
                       highlight_communities: bool = True) -> None:
    """
    Visualize graph with selected nodes highlighted
    
    Args:
        g: igraph Graph object
        clustering: clustering object where clustering[i] gives nodes in community i
        selected_nodes: list of selected node indices
        heuristic_name: name of the heuristic used for selection
        title: plot title
        figsize: figure size (width, height)
        layout: layout algorithm ('auto', 'fr', 'kk', 'circle', 'grid', 'random')
        node_size: size of nodes
        edge_alpha: transparency of edges
        save_path: path to save the plot (optional)
        show_labels: whether to show node labels
        highlight_communities: whether to show community structure
    """
    if g.vcount() == 0:
        print("Empty graph - nothing to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Choose layout (same as plot_communities)
    if layout == 'auto':
        if g.vcount() < 100:
            pos = g.layout_fruchterman_reingold()
        else:
            pos = g.layout_kamada_kawai()
    elif layout == 'fr':
        pos = g.layout_fruchterman_reingold()
    elif layout == 'kk':
        pos = g.layout_kamada_kawai()
    elif layout == 'circle':
        pos = g.layout_circle()
    elif layout == 'grid':
        pos = g.layout_grid()
    elif layout == 'random':
        pos = g.layout_random()
    else:
        pos = g.layout_fruchterman_reingold()
    
    pos_array = np.array(pos.coords)
    
    # Generate colors for communities if highlighting is enabled
    if highlight_communities:
        n_communities = len(clustering)
        if n_communities <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_communities))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(n_communities, 20)))
            if n_communities > 20:
                additional_colors = plt.cm.Set3(np.linspace(0, 1, n_communities - 20))
                colors = np.vstack([colors, additional_colors])
        
        # Create node color mapping
        node_colors = ['lightgray'] * g.vcount()
        community_info = {}
        
        for comm_id in range(len(clustering)):
            comm_nodes = clustering[comm_id]
            if comm_nodes:
                color = colors[comm_id % len(colors)]
                community_info[comm_id] = {
                    'nodes': comm_nodes,
                    'color': color,
                    'size': len(comm_nodes)
                }
                for node in comm_nodes:
                    if node < len(node_colors):
                        node_colors[node] = color
    else:
        node_colors = ['lightblue'] * g.vcount()
        community_info = {}
    
    # Plot 1: Communities (if highlighting enabled) or original graph
    if highlight_communities:
        ax1.set_title("Community Structure", fontsize=14, fontweight='bold')
    else:
        ax1.set_title("Original Graph", fontsize=14, fontweight='bold')
    
    # Draw edges
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        ax1.plot(x_coords, y_coords, 'gray', alpha=edge_alpha, linewidth=0.5)
    
    # Draw nodes
    ax1.scatter(pos_array[:, 0], pos_array[:, 1], 
               c=node_colors, s=node_size, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add node labels if requested
    if show_labels and g.vcount() <= 50:
        for i in range(g.vcount()):
            ax1.annotate(str(i), (pos_array[i, 0], pos_array[i, 1]), 
                        fontsize=8, ha='center', va='center')
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Plot 2: Selected nodes highlighted
    ax2.set_title(f"{title}\n{heuristic_name}: {len(selected_nodes)} nodes selected", 
                 fontsize=14, fontweight='bold')
    
    # Draw edges
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        
        # Highlight edges connected to selected nodes
        if edge.source in selected_nodes or edge.target in selected_nodes:
            ax2.plot(x_coords, y_coords, 'red', alpha=edge_alpha*1.5, linewidth=1.5)
        else:
            ax2.plot(x_coords, y_coords, 'gray', alpha=edge_alpha*0.5, linewidth=0.5)
    
    # Draw non-selected nodes
    non_selected = [i for i in range(g.vcount()) if i not in selected_nodes]
    if non_selected:
        if highlight_communities:
            non_selected_colors = [node_colors[i] for i in non_selected]
        else:
            non_selected_colors = 'lightblue'
        
        ax2.scatter(pos_array[non_selected, 0], pos_array[non_selected, 1], 
                   c=non_selected_colors, s=node_size, alpha=0.6, 
                   edgecolors='black', linewidth=0.5)
    
    # Draw selected nodes with special highlighting
    if selected_nodes:
        selected_pos = pos_array[selected_nodes]
        ax2.scatter(selected_pos[:, 0], selected_pos[:, 1], 
                   c='red', s=node_size*2, alpha=0.9, 
                   edgecolors='darkred', linewidth=2, marker='*')
    
    # Add node labels if requested
    if show_labels and g.vcount() <= 50:
        for i in range(g.vcount()):
            color = 'white' if i in selected_nodes else 'black'
            weight = 'bold' if i in selected_nodes else 'normal'
            ax2.annotate(str(i), (pos_array[i, 0], pos_array[i, 1]), 
                        fontsize=8, ha='center', va='center', 
                        color=color, weight=weight)
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                  markersize=15, label=f'Selected Nodes ({len(selected_nodes)})')
    ]
    
    if highlight_communities and len(community_info) <= 10:
        for comm_id, info in community_info.items():
            selected_in_comm = len([n for n in selected_nodes if n in info['nodes']])
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=info['color'], markersize=10,
                          label=f'Comm {comm_id} ({selected_in_comm}/{info["size"]} selected)')
            )
    
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add statistics
    stats_text = f"Nodes: {g.vcount()}\nEdges: {g.ecount()}\nSelected: {len(selected_nodes)}"
    
    if highlight_communities and community_info:
        stats_text += f"\nCommunities: {len(community_info)}"
        # Count selections per community
        selections_per_comm = {}
        for comm_id, info in community_info.items():
            selections_per_comm[comm_id] = len([n for n in selected_nodes if n in info['nodes']])
        
        if selections_per_comm:
            avg_selections = np.mean(list(selections_per_comm.values()))
            stats_text += f"\nAvg Selections/Comm: {avg_selections:.1f}"
    
    # Calculate degree statistics for selected nodes
    if selected_nodes:
        selected_degrees = [g.degree(node) for node in selected_nodes]
        avg_degree = np.mean(selected_degrees)
        stats_text += f"\nAvg Degree (Selected): {avg_degree:.1f}"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_stage1_analysis(g: ig.Graph, partition_method: str, heuristic: str, 
                        clustering, selected_nodes: List[int],
                        figsize: tuple = (16, 10), save_path: Optional[str] = None) -> None:
    """
    Comprehensive visualization of Stage 1 (Preprocess) analysis
    Shows both community detection and node selection results in one figure
    
    Args:
        g: igraph Graph object
        partition_method: name of partition method used
        heuristic: name of heuristic used for selection
        clustering: clustering object
        selected_nodes: list of selected node indices
        figsize: figure size (width, height)
        save_path: path to save the plot (optional)
    """
    if g.vcount() == 0:
        print("Empty graph - nothing to plot")
        return
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Choose layout
    if g.vcount() < 100:
        pos = g.layout_fruchterman_reingold()
    else:
        pos = g.layout_kamada_kawai()
    
    pos_array = np.array(pos.coords)
    
    # Generate colors for communities
    n_communities = len(clustering)
    if n_communities <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_communities))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_communities, 20)))
    
    node_colors = ['lightgray'] * g.vcount()
    community_info = {}
    
    for comm_id in range(len(clustering)):
        comm_nodes = clustering[comm_id]
        if comm_nodes:
            color = colors[comm_id % len(colors)]
            community_info[comm_id] = {
                'nodes': comm_nodes,
                'color': color,
                'size': len(comm_nodes)
            }
            for node in comm_nodes:
                if node < len(node_colors):
                    node_colors[node] = color
    
    # Plot 1: Original Graph
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original Graph", fontsize=12, fontweight='bold')
    
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        ax1.plot(x_coords, y_coords, 'gray', alpha=0.6, linewidth=0.5)
    
    ax1.scatter(pos_array[:, 0], pos_array[:, 1], 
               c='lightblue', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Plot 2: Community Detection
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f"Communities ({partition_method})\n{n_communities} communities", 
                 fontsize=12, fontweight='bold')
    
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        ax2.plot(x_coords, y_coords, 'gray', alpha=0.6, linewidth=0.5)
    
    ax2.scatter(pos_array[:, 0], pos_array[:, 1], 
               c=node_colors, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Plot 3: Selected Nodes
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title(f"Selected Nodes ({heuristic})\n{len(selected_nodes)} nodes", 
                 fontsize=12, fontweight='bold')
    
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        if edge.source in selected_nodes or edge.target in selected_nodes:
            ax3.plot(x_coords, y_coords, 'red', alpha=0.8, linewidth=1.5)
        else:
            ax3.plot(x_coords, y_coords, 'gray', alpha=0.4, linewidth=0.5)
    
    # Non-selected nodes
    non_selected = [i for i in range(g.vcount()) if i not in selected_nodes]
    if non_selected:
        ax3.scatter(pos_array[non_selected, 0], pos_array[non_selected, 1], 
                   c=[node_colors[i] for i in non_selected], s=100, alpha=0.6, 
                   edgecolors='black', linewidth=0.5)
    
    # Selected nodes
    if selected_nodes:
        selected_pos = pos_array[selected_nodes]
        ax3.scatter(selected_pos[:, 0], selected_pos[:, 1], 
                   c='red', s=200, alpha=0.9, edgecolors='darkred', 
                   linewidth=2, marker='*')
    
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    # Plot 4: Community Size Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title("Community Size Distribution", fontsize=12, fontweight='bold')
    
    comm_sizes = [info['size'] for info in community_info.values()]
    if comm_sizes:
        ax4.hist(comm_sizes, bins=min(10, len(comm_sizes)), alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Community Size')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Degree Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title("Degree Distribution", fontsize=12, fontweight='bold')
    
    degrees = g.degree()
    selected_degrees = [degrees[i] for i in selected_nodes] if selected_nodes else []
    
    ax5.hist(degrees, bins=min(20, max(degrees)+1), alpha=0.7, 
             color='lightblue', edgecolor='black', label='All nodes')
    
    if selected_degrees:
        ax5.hist(selected_degrees, bins=min(20, max(degrees)+1), alpha=0.8, 
                 color='red', edgecolor='darkred', label='Selected nodes')
    
    ax5.set_xlabel('Degree')
    ax5.set_ylabel('Frequency')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Selection Statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title("Selection Statistics", fontsize=12, fontweight='bold')
    
    # Count selections per community
    selections_per_comm = []
    comm_labels = []
    
    for comm_id, info in community_info.items():
        selected_in_comm = len([n for n in selected_nodes if n in info['nodes']])
        selections_per_comm.append(selected_in_comm)
        comm_labels.append(f'C{comm_id}')
    
    if selections_per_comm:
        bars = ax6.bar(range(len(selections_per_comm)), selections_per_comm, 
                      color=[info['color'] for info in community_info.values()],
                      alpha=0.7, edgecolor='black')
        
        ax6.set_xlabel('Community')
        ax6.set_ylabel('Selected Nodes')
        ax6.set_xticks(range(len(comm_labels)))
        ax6.set_xticklabels(comm_labels, rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Add overall statistics
    stats_text = f"""Graph Statistics:
Nodes: {g.vcount()}
Edges: {g.ecount()}
Communities: {n_communities}
Selected Nodes: {len(selected_nodes)}

Method: {partition_method}
Heuristic: {heuristic}"""
    
    if comm_sizes:
        stats_text += f"\nAvg Community Size: {np.mean(comm_sizes):.1f}"
    
    if selected_degrees:
        stats_text += f"\nAvg Degree (Selected): {np.mean(selected_degrees):.1f}"
        stats_text += f"\nAvg Degree (All): {np.mean(degrees):.1f}"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle(f"Stage 1 Analysis: {partition_method} + {heuristic}", 
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive plot saved to {save_path}")
    
    plt.show()


# Convenience function for quick testing
def test_visualization():
    """Test the visualization functions with a simple graph"""
    print("Testing visualization functions...")
    
    # Create test graph
    g = ig.Graph()
    g.add_vertices(12)
    
    # Community 1: nodes 0-3
    g.add_edges([(0,1), (0,2), (1,2), (1,3), (2,3)])
    
    # Community 2: nodes 4-7
    g.add_edges([(4,5), (4,6), (5,6), (5,7), (6,7)])
    
    # Community 3: nodes 8-11
    g.add_edges([(8,9), (8,10), (9,10), (9,11), (10,11)])
    
    # Inter-community bridges
    g.add_edges([(2,4), (3,8), (7,9)])
    
    # Create simple clustering
    from community_detection import SimpleClustering
    communities = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    clustering = SimpleClustering(communities, g.vcount())
    
    # Test selected nodes (bridge nodes)
    selected_nodes = [2, 3, 4, 7, 8, 9]
    
    print("1. Testing plot_communities...")
    plot_communities(g, clustering, title="Test Community Detection")
    
    print("2. Testing plot_selected_nodes...")
    plot_selected_nodes(g, clustering, selected_nodes, 
                       heuristic_name="Bridge Nodes", 
                       title="Test Node Selection")
    
    print("3. Testing plot_stage1_analysis...")
    plot_stage1_analysis(g, "test_method", "bridge_heuristic", 
                        clustering, selected_nodes)
    
    print("Visualization tests completed!")


if __name__ == "__main__":
    test_visualization()

def visualize_preprocess_step(g: ig.Graph, partition_method: str, partition_config: dict,
                             heuristic: str, heuristic_config: dict, max_iterations: int = 1,
                             figsize: tuple = (16, 10), save_prefix: str = None) -> tuple:
    """
    Integrated function to run and visualize one step of the preprocess function
    This is designed to be easily integrated into your baseline_develop.ipynb workflow
    
    Args:
        g: igraph Graph object
        partition_method: community detection method name
        partition_config: configuration for partition method
        heuristic: heuristic method name
        heuristic_config: configuration for heuristic method
        max_iterations: number of iterations (default 1 for single step visualization)
        figsize: figure size for plots
        save_prefix: prefix for saved plot files (optional)
    
    Returns:
        tuple: (clustering, selected_nodes, remaining_graph)
    """
    from .community_detection import partition
    from .heuristics import select
    from copy import deepcopy
    
    print(f"=== Preprocess Step Visualization ===")
    print(f"Graph: {g.vcount()} nodes, {g.ecount()} edges")
    print(f"Method: {partition_method} + {heuristic}")
    
    # Step 1: Community Detection
    print(f"\n1. Running community detection ({partition_method})...")
    clustering = partition(g, partition_method=partition_method, **partition_config)
    print(f"   Found {len(clustering)} communities")
    
    # Visualize communities
    plot_communities(g, clustering, 
                    title=f"Community Detection: {partition_method}",
                    figsize=(12, 6),
                    save_path=f"{save_prefix}_communities.png" if save_prefix else None)
    
    # Step 2: Node Selection
    print(f"\n2. Running node selection ({heuristic})...")
    selected_nodes = select(g, clustering, heuristic=heuristic, k=1, **heuristic_config)
    print(f"   Selected {len(selected_nodes)} nodes: {selected_nodes}")
    
    # Visualize selected nodes
    plot_selected_nodes(g, clustering, selected_nodes,
                       heuristic_name=heuristic.upper(),
                       title=f"Node Selection: {heuristic}",
                       figsize=(12, 6),
                       save_path=f"{save_prefix}_selection.png" if save_prefix else None)
    
    # Step 3: Comprehensive Analysis
    print(f"\n3. Generating comprehensive analysis...")
    plot_stage1_analysis(g, partition_method, heuristic, 
                        clustering, selected_nodes,
                        figsize=figsize,
                        save_path=f"{save_prefix}_analysis.png" if save_prefix else None)
    
    # Step 4: Create remaining graph (simulate removal)
    remaining_g = deepcopy(g)
    if selected_nodes:
        # Add static_id to track original nodes
        remaining_g.vs['static_id'] = list(range(g.vcount()))
        remaining_g.delete_vertices(selected_nodes)
        print(f"\n4. After removal: {remaining_g.vcount()} nodes, {remaining_g.ecount()} edges remaining")
    
    return clustering, selected_nodes, remaining_g


def quick_preprocess_viz(g_path: str, partition_method: str = 'fast_greedy', 
                        heuristic: str = 'cbs', K: int = 10) -> None:
    """
    Quick visualization function for testing - loads graph and runs single preprocess step
    
    Args:
        g_path: path to pickled graph file
        partition_method: community detection method
        heuristic: node selection heuristic
        K: number of communities (for methods that support it)
    """
    import pickle
    
    # Load graph
    with open(g_path, 'rb') as f:
        g = pickle.load(f)
    
    print(f"Loaded graph from {g_path}")
    
    # Run visualization
    clustering, selected_nodes, remaining_g = visualize_preprocess_step(
        g, 
        partition_method=partition_method,
        partition_config={'K': K} if 'K' in partition.__code__.co_varnames else {},
        heuristic=heuristic,
        heuristic_config={},
        save_prefix=f"quick_viz_{partition_method}_{heuristic}"
    )
    
    return clustering, selected_nodes, remaining_g