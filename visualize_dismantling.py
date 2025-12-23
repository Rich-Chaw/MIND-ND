import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np
import igraph as ig
from copy import deepcopy
import os

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

def get_curve_list(graph,removals,step_ratio=None):
    graph = graph.copy()
    ensure_static_id(graph)
    n_init = graph.vcount()
    if step_ratio:
        step_size = max(1,int(n_init*step_ratio))
    else:
        step_size = 1
    
    removed_sizes = [0]
    lcc_sizes = [n_init]
    i = 0
    while (i + step_size) <= len(removals):
        step_removals = removals[i:i+step_size]
        step_nodes_ids = [v.index for v in graph.vs if v['static_id'] in step_removals]
        graph.delete_vertices(step_nodes_ids)

        removed_sizes.append(removed_sizes[-1]+step_size)
        lcc_sizes.append(get_lcc_size(graph))
        i = i + step_size

    if i<len(removals):
        step_removals = removals[i:]
        step_nodes_ids = [v.index for v in graph.vs if v['static_id'] in step_removals]
        graph.delete_vertices(step_nodes_ids)

        removed_sizes.append(removed_sizes[-1]+step_size)
        lcc_sizes.append(get_lcc_size(graph))

    return lcc_sizes, removed_sizes


# ------------------------------------Static dismantling process visualization-----------------------------------------------
def create_dismantling_process(graph, removals, method_name, max_steps=6):
    """Create a step-by-step static visualization"""
    # Ensure static_id exists
    if 'static_id' not in graph.vs.attributes():
        graph.vs['static_id'] = list(range(graph.vcount()))
    
    n_init = graph.vcount()
    
    # Get layout
    if n_init <= 50:
        layout = graph.layout("fr")
    else:
        layout = graph.layout("kk")
    
    # Store original positions
    original_positions = {graph.vs[i]['static_id']: (layout[i][0], layout[i][1]) 
                         for i in range(graph.vcount())}
    
    # Calculate steps to show
    total_steps = min(len(removals), n_init)
    if total_steps > max_steps:
        step_indices = np.linspace(0, total_steps-1, max_steps, dtype=int)
    else:
        step_indices = list(range(total_steps))
    
    # Create subplots
    cols = 3
    rows = (len(step_indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f'{method_name} Dismantling Process', fontsize=16)
    
    for idx, step_idx in enumerate(step_indices):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Create graph at this step
        removed_nodes = removals[:step_idx+1] if step_idx > 0 else []
        current_graph = graph.copy()
        
        # Remove nodes
        for node_id in removed_nodes:
            for v in current_graph.vs:
                if v['static_id'] == node_id:
                    current_graph.delete_vertices(v.index)
                    break
        
        lcc_size = get_lcc_size(current_graph)
        ax.set_title(f'Step {step_idx}\nLCC: {lcc_size}, Remaining: {current_graph.vcount()}')
        ax.set_aspect('equal')
        
        # Draw remaining graph
        if current_graph.vcount() > 0:
            # Get positions for remaining nodes
            remaining_positions = []
            node_colors = []
            
            # Identify LCC
            components = current_graph.connected_components()
            if len(components) > 0:
                lcc_node_indices = set(components.giant().vs.indices)
            else:
                lcc_node_indices = set()
            
            for i, v in enumerate(current_graph.vs):
                static_id = v['static_id']
                pos = original_positions[static_id]
                remaining_positions.append(pos)
                
                # Color nodes: LCC in blue, others in gray
                if i in lcc_node_indices:
                    node_colors.append('blue')
                else:
                    node_colors.append('lightgray')
            
            # Draw edges
            for edge in current_graph.es:
                source_pos = remaining_positions[edge.source]
                target_pos = remaining_positions[edge.target]
                ax.plot([source_pos[0], target_pos[0]], 
                       [source_pos[1], target_pos[1]], 
                       'k-', alpha=0.4, linewidth=0.8)
            
            # Draw nodes
            if remaining_positions:
                x_coords = [pos[0] for pos in remaining_positions]
                y_coords = [pos[1] for pos in remaining_positions]
                ax.scatter(x_coords, y_coords, c=node_colors, s=40, alpha=0.8)
        
        # Draw removed nodes as red X
        for node_id in removed_nodes:
            if node_id in original_positions:
                pos = original_positions[node_id]
                ax.scatter(pos[0], pos[1], c='red', marker='x', s=60, alpha=0.8)
        
        # Set axis limits
        if original_positions:
            ax.set_xlim([min(pos[0] for pos in original_positions.values()) - 0.5,
                        max(pos[0] for pos in original_positions.values()) + 0.5])
            ax.set_ylim([min(pos[1] for pos in original_positions.values()) - 0.5,
                        max(pos[1] for pos in original_positions.values()) + 0.5])
        ax.axis('off')
    # Hide unused subplots
    for idx in range(len(step_indices), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_multiple_curve(graph, methods_results,step_ratio = None, save_path=None):
    """Create a static comparison plot"""
    
    plt.figure(figsize=(12, 8))
    n_init = graph.vcount()
    # colors = ['blue', 'green', 'orange', 'purple']
    colors = plt.cm.tab20(np.linspace(0, 2, 30))
    
    for i, (method_name, removals) in enumerate(methods_results.items()):
        lcc_sizes,removed_sizes = get_curve_list(graph,removals,step_ratio)
        x = np.array(removed_sizes) / n_init
        y = np.array(lcc_sizes) / n_init
        plt.plot(x, y, color=colors[i], linewidth=2, 
                marker='o', markersize=3, label=method_name, alpha=0.8)
    
    plt.xlabel('Fraction of Nodes Removed', fontsize=12)
    plt.ylabel('Normalized LCC Size', fontsize=12)
    plt.title('Dismantling Methods Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Static plot saved to: {save_path}")
        return save_path
    
    plt.show()
    



# ------------------------------------Dynamic dismantling process visualization-----------------------------------------------
def create_dismantling_gif(graph, removals, method_name, output_path="dismantling_animation.gif", 
                          step_size=1, layout_type="auto", figsize=(10, 8)):
    """
    Create a GIF animation showing the dismantling process step by step.
    
    Parameters:
    - graph: igraph.Graph object
    - removals: list of node IDs to remove in order
    - method_name: string name of the dismantling method
    - output_path: path to save the GIF
    - step_size: number of nodes to remove per frame
    - layout_type: layout algorithm ("auto", "fr", "kk", "circle", "grid")
    - figsize: figure size tuple
    """
    
    # Ensure static_id exists
    if 'static_id' not in graph.vs.attributes():
        graph.vs['static_id'] = list(range(graph.vcount()))
    
    # Create a copy for manipulation
    g = graph.copy()
    n_init = g.vcount()
    
    # Choose layout
    if layout_type == "auto":
        if n_init <= 100:
            layout = g.layout("fr")
        else:
            layout = g.layout("kk")
    else:
        layout = g.layout(layout_type)
    
    # Store original positions for all nodes
    original_positions = {g.vs[i]['static_id']: (layout[i][0], layout[i][1]) 
                         for i in range(g.vcount())}
    
    # Prepare frames data
    frames_data = []
    current_graph = graph.copy()
    removed_so_far = []
    
    # Initial frame
    frames_data.append({
        'graph': current_graph.copy(),
        'removed_nodes': [],
        'step': 0,
        'lcc_size': current_graph.connected_components().giant().vcount()
    })
    
    # Process removals step by step
    for i in range(0, len(removals), step_size):
        step_removals = removals[i:i+step_size]
        removed_so_far.extend(step_removals)
        
        # Find nodes to remove in current graph
        nodes_to_remove = []
        for node_id in step_removals:
            for v in current_graph.vs:
                if v['static_id'] == node_id:
                    nodes_to_remove.append(v.index)
                    break
        
        # Remove nodes
        if nodes_to_remove:
            current_graph.delete_vertices(nodes_to_remove)
        
        # Calculate LCC size
        if current_graph.vcount() > 0:
            components = current_graph.connected_components()
            if len(components) > 0:
                lcc_size = components.giant().vcount()
            else:
                lcc_size = 0
        else:
            lcc_size = 0
        
        frames_data.append({
            'graph': current_graph.copy(),
            'removed_nodes': removed_so_far.copy(),
            'step': i // step_size + 1,
            'lcc_size': lcc_size
        })
        
        # Stop if graph is empty
        if current_graph.vcount() == 0:
            break
    
    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    def animate(frame_idx):
        ax1.clear()
        ax2.clear()
        
        frame = frames_data[frame_idx]
        g_frame = frame['graph']
        removed_nodes = frame['removed_nodes']
        step = frame['step']
        lcc_size = frame['lcc_size']
        
        # Plot 1: Graph visualization
        ax1.set_title(f'{method_name} - Step {step}\nNodes removed: {len(removed_nodes)}/{n_init}')
        ax1.set_aspect('equal')
        
        if g_frame.vcount() > 0:
            # Get positions for remaining nodes
            remaining_positions = []
            node_colors = []
            node_sizes = []
            
            # Identify LCC
            components = g_frame.connected_components()
            if len(components) > 0:
                # Get the largest component indices
                lcc_nodes = set(components.giant().vs.indices)
            else:
                lcc_nodes = set()
            
            for i, v in enumerate(g_frame.vs):
                static_id = v['static_id']
                pos = original_positions[static_id]
                remaining_positions.append(pos)
                
                # Color nodes: LCC in blue, others in gray
                if i in lcc_nodes:
                    node_colors.append('blue')
                    node_sizes.append(50)
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(30)
            
            # Draw edges
            for edge in g_frame.es:
                source_pos = remaining_positions[edge.source]
                target_pos = remaining_positions[edge.target]
                ax1.plot([source_pos[0], target_pos[0]], 
                        [source_pos[1], target_pos[1]], 
                        'k-', alpha=0.3, linewidth=0.5)
            
            # Draw nodes
            if remaining_positions:
                x_coords = [pos[0] for pos in remaining_positions]
                y_coords = [pos[1] for pos in remaining_positions]
                ax1.scatter(x_coords, y_coords, c=node_colors, s=node_sizes, alpha=0.8)
        
        # Draw removed nodes as red X
        for node_id in removed_nodes:
            if node_id in original_positions:
                pos = original_positions[node_id]
                ax1.scatter(pos[0], pos[1], c='red', marker='x', s=100, alpha=0.7)
        
        ax1.set_xlim([min(pos[0] for pos in original_positions.values()) - 0.5,
                     max(pos[0] for pos in original_positions.values()) + 0.5])
        ax1.set_ylim([min(pos[1] for pos in original_positions.values()) - 0.5,
                     max(pos[1] for pos in original_positions.values()) + 0.5])
        ax1.axis('off')
        
        # Plot 2: LCC size over time
        steps = [f['step'] for f in frames_data[:frame_idx+1]]
        lcc_sizes = [f['lcc_size'] for f in frames_data[:frame_idx+1]]
        
        ax2.plot(steps, np.array(lcc_sizes) / n_init, 'b-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Normalized LCC Size')
        ax2.set_title('LCC Size Decay')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, len(frames_data))
        ax2.set_ylim(0, 1)
        
        # Add text info
        ax2.text(0.02, 0.98, f'Current LCC: {lcc_size}\nRemaining: {g_frame.vcount()}\nRemoved: {len(removed_nodes)}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Create animation
    # Limit frames to avoid memory issues
    max_frames = min(len(frames_data), 100)
    if len(frames_data) > max_frames:
        print(f"Note: Limiting animation to {max_frames} frames to avoid memory issues")
        # Sample frames evenly
        frame_indices = np.linspace(0, len(frames_data)-1, max_frames, dtype=int)
        frames_data = [frames_data[i] for i in frame_indices]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data), 
                                 interval=500, repeat=True, blit=False)
    
    # Save as GIF
    print(f"Saving animation to {output_path}...")
    try:
        anim.save(output_path, writer='pillow', fps=2, dpi=80)
        plt.close()
    except Exception as e:
        print(f"Error saving GIF: {e}")
        print("Trying with lower quality settings...")
        try:
            anim.save(output_path, writer='pillow', fps=1, dpi=60)
            plt.close()
        except Exception as e2:
            print(f"Failed to save GIF: {e2}")
            plt.close()
            return None

    return output_path

def visualize_multiple_dynamic(graph, methods_results, output_dir="dismantling_gifs"):
    """
    Create GIFs for multiple dismantling methods.
    
    Parameters:
    - graph: igraph.Graph object
    - methods_results: dict with method_name: removals_list
    - output_dir: directory to save GIFs
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"GIF animations saved to: {output_dir}")

    gif_paths = []
    
    for method_name, removals in methods_results.items():
        output_path = os.path.join(output_dir, f"{method_name}_dismantling.gif")
        gif_path = create_dismantling_gif(graph, removals, method_name, output_path)
        gif_paths.append(gif_path)
        print(f"Created GIF for {method_name}")
    
    return gif_paths

# Example usage function
def demo_dismantling_visualization():
    """
    Demo function showing how to use the visualization
    """
    # Create a sample graph
    np.random.seed(42)
    G = ig.Graph.GRG(n=50, radius=0.2)
    G.vs['static_id'] = list(range(G.vcount()))
    
    # Define some simple dismantling methods for demo
    def degree_dismantling(graph):
        temp_G = graph.copy()
        if 'static_id' not in temp_G.vs.attributes():
            temp_G.vs['static_id'] = list(range(temp_G.vcount()))
        
        removed_nodes = []
        while temp_G.vcount() > 0:
            degrees = temp_G.degree()
            if max(degrees) == 0:
                break
            idx_to_remove = np.argmax(degrees)
            removed_nodes.append(temp_G.vs[idx_to_remove]['static_id'])
            temp_G.delete_vertices(idx_to_remove)
        return removed_nodes
    
    def random_dismantling(graph):
        temp_G = graph.copy()
        if 'static_id' not in temp_G.vs.attributes():
            temp_G.vs['static_id'] = list(range(temp_G.vcount()))
        
        removed_nodes = []
        node_ids = [v['static_id'] for v in temp_G.vs]
        np.random.shuffle(node_ids)
        return node_ids
    
    # Get removal sequences
    methods_results = {
        'Degree': degree_dismantling(G),
        'Random': random_dismantling(G)
    }
    
    # Create visualizations
    gif_paths = visualize_multiple_dynamic(G, methods_results)
    
    print("Demo completed! Check the generated GIF files.")
    return gif_paths

if __name__ == "__main__":
    demo_dismantling_visualization()