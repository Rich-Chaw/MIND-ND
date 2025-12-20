"""
Visualization functions for community detection and node selection in network dismantling
"""

import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Union, Tuple
import random
import math

def community_aware_layout(g: ig.Graph, clustering, 
                          layout_method: str = 'adaptive',
                          community_spacing: float = 5.0, 
                          intra_community_scale: float = 1.2,
                          force_iterations: int = 100,
                          repulsion_strength: float = 2.0) -> np.ndarray:
    """
    Advanced community-aware layout with multiple algorithms for better community separation
    
    Args:
        g: igraph Graph object
        clustering: clustering object where clustering[i] gives nodes in community i
        layout_method: 'adaptive', 'grid', 'force_atlas', 'multilevel', 'spring_block'
        community_spacing: distance between community centers
        intra_community_scale: scale factor for within-community layout
        force_iterations: iterations for force-directed algorithms
        repulsion_strength: strength of inter-community repulsion
    
    Returns:
        numpy array of shape (n_nodes, 2) with node positions
    """
    n_nodes = g.vcount()
    n_communities = len(clustering)
    
    if n_nodes == 0:
        return np.array([]).reshape(0, 2)
    
    if n_communities == 0:
        # Fallback to standard layout if no communities
        pos = g.layout_fruchterman_reingold()
        return np.array(pos.coords)
    
    # Choose layout method based on graph size and structure
    if layout_method == 'adaptive':
        if n_nodes <= 100:
            layout_method = 'spring_block'
        elif n_communities <= 6:
            layout_method = 'grid'
        else:
            layout_method = 'multilevel'
    
    # Calculate community centers using different strategies
    if layout_method == 'grid':
        community_centers = _calculate_grid_centers(n_communities, community_spacing)
    elif layout_method == 'multilevel':
        community_centers = _calculate_multilevel_centers(clustering, community_spacing)
    else:
        community_centers = _calculate_circular_centers(n_communities, community_spacing)
    
    # Initialize position array
    positions = np.zeros((n_nodes, 2))
    
    # Layout each community separately with improved algorithms
    for comm_id in range(n_communities):
        comm_nodes = clustering[comm_id]
        if not comm_nodes:
            continue
            
        comm_center = community_centers[comm_id]
        comm_size = len(comm_nodes)
        
        if comm_size == 1:
            positions[comm_nodes[0]] = comm_center
        else:
            sub_positions = _layout_single_community(
                g, comm_nodes, comm_size, intra_community_scale, force_iterations
            )
            
            # Apply community-specific scaling based on size
            scale_factor = min(1.0, math.sqrt(comm_size) / 10) * intra_community_scale
            sub_positions *= scale_factor
            
            # Center and translate to community position
            if len(sub_positions) > 0:
                sub_center = np.mean(sub_positions, axis=0)
                sub_positions -= sub_center
                sub_positions[:, 0] += comm_center[0]
                sub_positions[:, 1] += comm_center[1]
                
                # Assign positions
                for i, node_idx in enumerate(comm_nodes):
                    if i < len(sub_positions):
                        positions[node_idx] = sub_positions[i]
                    else:
                        positions[node_idx] = comm_center
    
    # Apply inter-community force adjustment for better separation
    if layout_method in ['force_atlas', 'spring_block']:
        positions = _apply_inter_community_forces(
            positions, clustering, community_centers, repulsion_strength, force_iterations
        )
    
    # Handle unassigned nodes
    positions = _handle_unassigned_nodes(positions, clustering, n_nodes, community_spacing)
    
    return positions


def _calculate_grid_centers(n_communities: int, spacing: float) -> List[Tuple[float, float]]:
    """Calculate community centers in a grid layout"""
    if n_communities <= 1:
        return [(0, 0)]
    
    # Calculate grid dimensions
    cols = int(math.ceil(math.sqrt(n_communities)))
    rows = int(math.ceil(n_communities / cols))
    
    centers = []
    for i in range(n_communities):
        row = i // cols
        col = i % cols
        
        # Center the grid
        x = (col - (cols - 1) / 2) * spacing
        y = (row - (rows - 1) / 2) * spacing
        centers.append((x, y))
    
    return centers


def _calculate_multilevel_centers(clustering, spacing: float) -> List[Tuple[float, float]]:
    """Calculate community centers using hierarchical placement based on community sizes"""
    n_communities = len(clustering)
    if n_communities <= 1:
        return [(0, 0)]
    
    # Sort communities by size (largest first)
    comm_sizes = [(i, len(clustering[i])) for i in range(n_communities)]
    comm_sizes.sort(key=lambda x: x[1], reverse=True)
    
    centers = [None] * n_communities
    
    # Place largest community at center
    largest_comm_id = comm_sizes[0][0]
    centers[largest_comm_id] = (0, 0)
    
    # Place other communities in concentric circles
    if n_communities > 1:
        # First ring: 2nd to 4th largest
        ring1_comms = comm_sizes[1:min(4, n_communities)]
        for i, (comm_id, _) in enumerate(ring1_comms):
            angle = 2 * math.pi * i / len(ring1_comms)
            x = spacing * math.cos(angle)
            y = spacing * math.sin(angle)
            centers[comm_id] = (x, y)
        
        # Second ring: remaining communities
        if n_communities > 4:
            ring2_comms = comm_sizes[4:]
            for i, (comm_id, _) in enumerate(ring2_comms):
                angle = 2 * math.pi * i / len(ring2_comms)
                x = spacing * 1.8 * math.cos(angle)
                y = spacing * 1.8 * math.sin(angle)
                centers[comm_id] = (x, y)
    
    return centers


def _calculate_circular_centers(n_communities: int, spacing: float) -> List[Tuple[float, float]]:
    """Calculate community centers in a circular layout (improved version)"""
    if n_communities <= 1:
        return [(0, 0)]
    
    centers = []
    
    if n_communities <= 6:
        # Small number of communities - use regular polygon
        for i in range(n_communities):
            angle = 2 * math.pi * i / n_communities
            x = spacing * math.cos(angle)
            y = spacing * math.sin(angle)
            centers.append((x, y))
    else:
        # Many communities - use multiple concentric circles
        inner_count = min(6, n_communities // 2)
        outer_count = n_communities - inner_count
        
        # Inner circle
        for i in range(inner_count):
            angle = 2 * math.pi * i / inner_count
            x = spacing * 0.6 * math.cos(angle)
            y = spacing * 0.6 * math.sin(angle)
            centers.append((x, y))
        
        # Outer circle
        for i in range(outer_count):
            angle = 2 * math.pi * i / outer_count
            x = spacing * 1.4 * math.cos(angle)
            y = spacing * 1.4 * math.sin(angle)
            centers.append((x, y))
    
    return centers


def _layout_single_community(g: ig.Graph, comm_nodes: List[int], comm_size: int, 
                           scale: float, iterations: int) -> np.ndarray:
    """Layout a single community using the best algorithm for its size"""
    try:
        subgraph = g.subgraph(comm_nodes)
        
        if comm_size <= 3:
            # Very small - manual circle
            positions = []
            for i in range(comm_size):
                if comm_size == 2:
                    angle = math.pi * i
                    radius = 0.5
                else:
                    angle = 2 * math.pi * i / comm_size
                    radius = 0.3
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions.append([x, y])
            return np.array(positions)
            
        elif comm_size <= 8:
            # Small - use circle layout with slight randomization
            pos = subgraph.layout_circle()
            positions = np.array(pos.coords)
            # Add small random perturbation to avoid perfect symmetry
            noise = np.random.normal(0, 0.1, positions.shape)
            positions += noise
            return positions
            
        elif comm_size <= 25:
            # Medium - use Fruchterman-Reingold with more iterations
            pos = subgraph.layout_fruchterman_reingold(niter=iterations)
            return np.array(pos.coords)
            
        elif comm_size <= 100:
            # Large - use Kamada-Kawai for better structure preservation
            try:
                pos = subgraph.layout_kamada_kawai()
                return np.array(pos.coords)
            except:
                # Fallback to FR with fewer iterations for speed
                pos = subgraph.layout_fruchterman_reingold(niter=max(50, iterations//2))
                return np.array(pos.coords)
        else:
            # Very large - use fast spring layout
            try:
                pos = subgraph.layout_drl()
                return np.array(pos.coords)
            except:
                # Final fallback
                pos = subgraph.layout_fruchterman_reingold(niter=30)
                return np.array(pos.coords)
                
    except Exception as e:
        # Ultimate fallback - manual circular placement
        positions = []
        for i in range(comm_size):
            angle = 2 * math.pi * i / comm_size
            radius = max(0.5, math.sqrt(comm_size) / 4)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions.append([x, y])
        return np.array(positions)


def _apply_inter_community_forces(positions: np.ndarray, clustering, 
                                community_centers: List[Tuple[float, float]], 
                                repulsion_strength: float, iterations: int) -> np.ndarray:
    """Apply force-directed adjustment to improve community separation"""
    n_nodes = len(positions)
    
    # Create community membership mapping
    node_to_comm = {}
    for comm_id in range(len(clustering)):
        for node in clustering[comm_id]:
            node_to_comm[node] = comm_id
    
    # Apply force iterations
    for _ in range(min(iterations, 20)):  # Limit iterations for performance
        forces = np.zeros_like(positions)
        
        # Inter-community repulsion
        for i in range(n_nodes):
            if i not in node_to_comm:
                continue
                
            comm_i = node_to_comm[i]
            
            for j in range(i + 1, n_nodes):
                if j not in node_to_comm:
                    continue
                    
                comm_j = node_to_comm[j]
                
                if comm_i != comm_j:  # Different communities
                    # Calculate repulsive force
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)
                    
                    if dist > 0.01:  # Avoid division by zero
                        force_magnitude = repulsion_strength / (dist ** 2)
                        force_direction = diff / dist
                        
                        forces[i] += force_magnitude * force_direction
                        forces[j] -= force_magnitude * force_direction
        
        # Apply forces with damping
        positions += forces * 0.1
    
    return positions


def _handle_unassigned_nodes(positions: np.ndarray, clustering, n_nodes: int, 
                           spacing: float) -> np.ndarray:
    """Handle nodes not assigned to any community"""
    assigned_nodes = set()
    for comm_id in range(len(clustering)):
        assigned_nodes.update(clustering[comm_id])
    
    unassigned_nodes = [i for i in range(n_nodes) if i not in assigned_nodes]
    
    if unassigned_nodes:
        # Place unassigned nodes in a separate outer ring
        for i, node_idx in enumerate(unassigned_nodes):
            angle = 2 * math.pi * i / len(unassigned_nodes)
            radius = spacing * 2.0  # Further out than communities
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[node_idx] = [x, y]
    
    return positions


def auto_node_size(n_nodes: int, base_size: int = 300) -> int:
    """
    Automatically calculate appropriate node size based on number of nodes
    
    Args:
        n_nodes: number of nodes in the graph
        base_size: base node size for small graphs
    
    Returns:
        appropriate node size
    """
    if n_nodes <= 20:
        return base_size
    elif n_nodes <= 50:
        return max(base_size // 2, 100)
    elif n_nodes <= 100:
        return max(base_size // 4, 50)
    elif n_nodes <= 200:
        return max(base_size // 6, 30)
    else:
        return max(base_size // 10, 20)


def plot_communities(g: ig.Graph, clustering, title: str = "Community Detection Results", 
                    figsize: tuple = (12, 8), layout: str = 'auto', 
                    node_size: Optional[int] = None, edge_alpha: float = 0.6,
                    save_path: Optional[str] = None, show_labels: bool = True,
                    community_spacing: float = 5.0, intra_community_scale: float = 1.2,
                    layout_method: str = 'adaptive', force_iterations: int = 100) -> None:
    """
    Visualize graph with community structure using improved layout algorithms
    
    Args:
        g: igraph Graph object
        clustering: clustering object where clustering[i] gives nodes in community i
        title: plot title
        figsize: figure size (width, height)
        layout: layout algorithm ('auto', 'community', 'fr', 'kk', 'circle', 'grid', 'random')
        node_size: size of nodes (auto-calculated if None)
        edge_alpha: transparency of edges
        save_path: path to save the plot (optional)
        show_labels: whether to show node labels
        community_spacing: distance between community centers (for community layout)
        intra_community_scale: scale factor for within-community layout
        layout_method: community layout method ('adaptive', 'grid', 'multilevel', 'spring_block')
        force_iterations: iterations for force-directed refinement
    """
    if g.vcount() == 0:
        print("Empty graph - nothing to plot")
        return
    
    # Auto-calculate node size if not provided
    if node_size is None:
        node_size = auto_node_size(g.vcount())
    
    # Auto-adjust label display for large graphs
    if show_labels and g.vcount() > 50:
        show_labels = False
        print(f"Note: Node labels disabled for large graph ({g.vcount()} nodes)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Choose layout
    if layout == 'auto' or layout == 'community':
        # Use improved community-aware layout for better community visualization
        pos_array = community_aware_layout(g, clustering, layout_method, community_spacing, 
                                         intra_community_scale, force_iterations)
    elif layout == 'fr':
        pos = g.layout_fruchterman_reingold()
        pos_array = np.array(pos.coords)
    elif layout == 'kk':
        pos = g.layout_kamada_kawai()
        pos_array = np.array(pos.coords)
    elif layout == 'circle':
        pos = g.layout_circle()
        pos_array = np.array(pos.coords)
    elif layout == 'grid':
        pos = g.layout_grid()
        pos_array = np.array(pos.coords)
    elif layout == 'random':
        pos = g.layout_random()
        pos_array = np.array(pos.coords)
    else:
        # Default to improved community-aware layout
        pos_array = community_aware_layout(g, clustering, layout_method, community_spacing, 
                                         intra_community_scale, force_iterations)
    
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
    
    # Draw edges (behind nodes)
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        ax1.plot(x_coords, y_coords, 'k-', alpha=edge_alpha, linewidth=0.5, zorder=1)
    
    # Draw nodes (on top of edges)
    ax1.scatter(pos_array[:, 0], pos_array[:, 1], 
               c='lightblue', s=node_size, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=2)
    
    # Add node labels if requested
    if show_labels and g.vcount() <= 50:  # Only show labels for small graphs
        for i in range(g.vcount()):
            ax1.annotate(str(i), (pos_array[i, 0], pos_array[i, 1]), 
                        fontsize=8, ha='center', va='center')
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Plot 2: Graph with community coloring
    ax2.set_title(f"{title}\n{n_communities} Communities", fontsize=14, fontweight='bold')
    
    # Draw edges in proper order: intra-community first, then inter-community
    # This ensures inter-community edges don't hide nodes
    
    # First pass: Draw all intra-community edges (background)
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        
        # Find communities for source and target
        source_comm = None
        target_comm = None
        
        for comm_id, info in community_info.items():
            if edge.source in info['nodes']:
                source_comm = comm_id
            if edge.target in info['nodes']:
                target_comm = comm_id
        
        # Only draw intra-community edges in this pass
        if source_comm is not None and target_comm is not None and source_comm == target_comm:
            ax2.plot(x_coords, y_coords, 'gray', alpha=edge_alpha, linewidth=0.5, zorder=1)
    
    # Second pass: Draw inter-community edges (but still behind nodes)
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        
        # Find communities for source and target
        source_comm = None
        target_comm = None
        
        for comm_id, info in community_info.items():
            if edge.source in info['nodes']:
                source_comm = comm_id
            if edge.target in info['nodes']:
                target_comm = comm_id
        
        # Only draw inter-community edges in this pass
        if source_comm is not None and target_comm is not None and source_comm != target_comm:
            ax2.plot(x_coords, y_coords, 'red', alpha=edge_alpha*0.8, linewidth=1.2, zorder=2)
    
    # Draw nodes with community colors (on top of all edges)
    ax2.scatter(pos_array[:, 0], pos_array[:, 1], 
               c=node_colors, s=node_size, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=3)
    
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
                       node_size: Optional[int] = None, edge_alpha: float = 0.6,
                       save_path: Optional[str] = None, show_labels: bool = True,
                       highlight_communities: bool = True,
                       community_spacing: float = 5.0, intra_community_scale: float = 1.2,
                       layout_method: str = 'adaptive', force_iterations: int = 100) -> None:
    """
    Visualize graph with selected nodes highlighted
    
    Args:
        g: igraph Graph object
        clustering: clustering object where clustering[i] gives nodes in community i
        selected_nodes: list of selected node indices
        heuristic_name: name of the heuristic used for selection
        title: plot title
        figsize: figure size (width, height)
        layout: layout algorithm ('auto', 'community', 'fr', 'kk', 'circle', 'grid', 'random')
        node_size: size of nodes (auto-calculated if None)
        edge_alpha: transparency of edges
        save_path: path to save the plot (optional)
        show_labels: whether to show node labels
        highlight_communities: whether to show community structure
        community_spacing: distance between community centers (for community layout)
        intra_community_scale: scale factor for within-community layout
    """
    if g.vcount() == 0:
        print("Empty graph - nothing to plot")
        return
    
    # Auto-calculate node size if not provided
    if node_size is None:
        node_size = auto_node_size(g.vcount())
    
    # Auto-adjust label display for large graphs
    if show_labels and g.vcount() > 50:
        show_labels = False
        print(f"Note: Node labels disabled for large graph ({g.vcount()} nodes)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Choose layout (same as plot_communities)
    if layout == 'auto' or layout == 'community':
        # Use improved community-aware layout for better community visualization
        pos_array = community_aware_layout(g, clustering, layout_method, community_spacing, 
                                         intra_community_scale, force_iterations)
    elif layout == 'fr':
        pos = g.layout_fruchterman_reingold()
        pos_array = np.array(pos.coords)
    elif layout == 'kk':
        pos = g.layout_kamada_kawai()
        pos_array = np.array(pos.coords)
    elif layout == 'circle':
        pos = g.layout_circle()
        pos_array = np.array(pos.coords)
    elif layout == 'grid':
        pos = g.layout_grid()
        pos_array = np.array(pos.coords)
    elif layout == 'random':
        pos = g.layout_random()
        pos_array = np.array(pos.coords)
    else:
        # Default to improved community-aware layout
        pos_array = community_aware_layout(g, clustering, layout_method, community_spacing, 
                                         intra_community_scale, force_iterations)
    
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
    
    # Draw edges (behind nodes)
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        ax1.plot(x_coords, y_coords, 'gray', alpha=edge_alpha, linewidth=0.5, zorder=1)
    
    # Draw nodes (on top of edges)
    ax1.scatter(pos_array[:, 0], pos_array[:, 1], 
               c=node_colors, s=node_size, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=2)
    
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
    
    # Draw edges in proper order to avoid hiding nodes
    # First pass: Draw non-highlighted edges (background)
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        
        # Only draw non-highlighted edges in this pass
        if edge.source not in selected_nodes and edge.target not in selected_nodes:
            ax2.plot(x_coords, y_coords, 'gray', alpha=edge_alpha*0.5, linewidth=0.5, zorder=1)
    
    # Second pass: Draw highlighted edges (but still behind nodes)
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        
        # Only draw highlighted edges in this pass
        if edge.source in selected_nodes or edge.target in selected_nodes:
            ax2.plot(x_coords, y_coords, 'red', alpha=edge_alpha*0.9, linewidth=1.2, zorder=2)
    
    # Draw non-selected nodes (on top of edges)
    non_selected = [i for i in range(g.vcount()) if i not in selected_nodes]
    if non_selected:
        if highlight_communities:
            non_selected_colors = [node_colors[i] for i in non_selected]
        else:
            non_selected_colors = 'lightblue'
        
        ax2.scatter(pos_array[non_selected, 0], pos_array[non_selected, 1], 
                   c=non_selected_colors, s=node_size, alpha=0.6, 
                   edgecolors='black', linewidth=0.5, zorder=3)
    
    # Draw selected nodes with special highlighting (on top of everything)
    if selected_nodes:
        selected_pos = pos_array[selected_nodes]
        ax2.scatter(selected_pos[:, 0], selected_pos[:, 1], 
                   c='red', s=node_size*2, alpha=0.9, 
                   edgecolors='darkred', linewidth=2, marker='*', zorder=4)
    
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
    
    # Choose layout - use improved community-aware layout for better visualization
    pos_array = community_aware_layout(g, clustering, layout_method='adaptive', 
                                     community_spacing=5.0, intra_community_scale=1.2)
    
    # Auto-calculate node size
    node_size = auto_node_size(g.vcount(), base_size=200)  # Smaller base for multi-panel plot
    
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
        ax1.plot(x_coords, y_coords, 'gray', alpha=0.6, linewidth=0.5, zorder=1)
    
    ax1.scatter(pos_array[:, 0], pos_array[:, 1], 
               c='lightblue', s=node_size, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Plot 2: Community Detection
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f"Communities ({partition_method})\n{n_communities} communities", 
                 fontsize=12, fontweight='bold')
    
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        ax2.plot(x_coords, y_coords, 'gray', alpha=0.6, linewidth=0.5, zorder=1)
    
    ax2.scatter(pos_array[:, 0], pos_array[:, 1], 
               c=node_colors, s=node_size, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=2)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Plot 3: Selected Nodes
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title(f"Selected Nodes ({heuristic})\n{len(selected_nodes)} nodes", 
                 fontsize=12, fontweight='bold')
    
    # Draw edges in proper order to avoid hiding nodes
    # First pass: Draw non-highlighted edges
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        if edge.source not in selected_nodes and edge.target not in selected_nodes:
            ax3.plot(x_coords, y_coords, 'gray', alpha=0.4, linewidth=0.5, zorder=1)
    
    # Second pass: Draw highlighted edges
    for edge in g.es:
        x_coords = [pos_array[edge.source][0], pos_array[edge.target][0]]
        y_coords = [pos_array[edge.source][1], pos_array[edge.target][1]]
        if edge.source in selected_nodes or edge.target in selected_nodes:
            ax3.plot(x_coords, y_coords, 'red', alpha=0.7, linewidth=1.2, zorder=2)
    
    # Non-selected nodes (on top of edges)
    non_selected = [i for i in range(g.vcount()) if i not in selected_nodes]
    if non_selected:
        ax3.scatter(pos_array[non_selected, 0], pos_array[non_selected, 1], 
                   c=[node_colors[i] for i in non_selected], s=node_size, alpha=0.6, 
                   edgecolors='black', linewidth=0.5, zorder=3)
    
    # Selected nodes (on top of everything)
    if selected_nodes:
        selected_pos = pos_array[selected_nodes]
        ax3.scatter(selected_pos[:, 0], selected_pos[:, 1], 
                   c='red', s=node_size*2, alpha=0.9, edgecolors='darkred', 
                   linewidth=2, marker='*', zorder=4)
    
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
