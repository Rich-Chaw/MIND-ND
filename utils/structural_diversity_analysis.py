#!/usr/bin/env python3
"""
Structural Diversity Analysis for MIND-ND Synthetic Graphs
Visualizes Modularity vs Assortativity by graph type (ER, LPA, Copy)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_graphs(pkl_files):
    """Load and convert graphs from pkl files"""
    graphs = []
    
    for pkl_file in pkl_files:
        if not os.path.exists(pkl_file):
            continue
            
        with open(pkl_file, 'rb') as f:
            net_dict = pickle.load(f)
        for graph_dict in net_dict.values():
            try:
                adj_matrix = graph_dict['adj']
                g = ig.Graph.Adjacency(adj_matrix.tolist(), mode="undirected")
                topology = graph_dict['info']['topology']
                graphs.append((g, topology))
            except:
                continue

    return graphs

def calculate_properties(graphs):
    """Calculate modularity and assortativity for all graphs"""
    q_values, r_values, labels = [], [], []
    
    for i, graph in enumerate(graphs):
        if i % 1000 == 0:
            print(f"Processing {i+1}/{len(graphs)}")
            
        try:
            if graph.vcount() < 10:
                continue
                
            # Modularity via Leiden
            communities = graph.community_leiden(objective_function="modularity")
            q = communities.modularity
            
            # Assortativity
            r = graph.assortativity_degree()
            
            q_values.append(q)
            r_values.append(r)
            labels.append(graph['config']['topology'])
            
        except Exception as e:
            print(e)
            continue
    
    return np.array(q_values), np.array(r_values), np.array(labels)

def create_scatter_plot(q_values, r_values, labels, save_path=None):
    """Create scatter plot with topology labels"""
    print(f"\nCreating scatter plot with {len(q_values)} points...")

    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))
    
    plt.figure(figsize=(10, 6))
    
    for label in unique_labels:
        mask = labels == label
        plt.scatter(q_values[mask], r_values[mask], 
                   color=color_map[label], alpha=0.6, s=20, edgecolors='black',linewidth=0.3, label=label)
    
    plt.xlabel('Modularity (Q)')
    plt.ylabel('Assortativity (r)')
    plt.title('Structural Diversity: Modularity vs Assortativity by Graph Type')
    plt.legend(title='Graph Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add stats
    stats = f"n={len(q_values)} | Q∈[{q_values.min():.2f},{q_values.max():.2f}] | r∈[{r_values.min():.2f},{r_values.max():.2f}]"
    plt.figtext(0.02, 0.02, stats, fontsize=9, style='italic')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
    plt.show()

def create_heatmap(q_values, r_values, labels, save_path = None):
    """
    Create heatmap showing distribution density by topology
    """
    print("\nCreating topology-specific heatmaps...")
    
    # Convert to numpy arrays
    q_values = np.array(q_values)
    r_values = np.array(r_values)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    n_topologies = len(unique_labels)
    
    # Create subplots for each topology
    fig, axes = plt.subplots(1, n_topologies, figsize=(5*n_topologies, 4))
    if n_topologies == 1:
        axes = [axes]
    
    for i, topology in enumerate(unique_labels):
        mask = labels == topology
        q_topo = q_values[mask]
        r_topo = r_values[mask]
        
        # Create 2D histogram
        hist, q_edges, r_edges = np.histogram2d(q_topo, r_topo, bins=15)
        
        # Plot heatmap
        im = axes[i].imshow(hist.T, origin='lower', aspect='auto', cmap='viridis',
                            extent=[q_edges[0], q_edges[-1], r_edges[0], r_edges[-1]])
        
        axes[i].set_xlabel('Modularity (Q)')
        axes[i].set_ylabel('Assortativity (r)')
        axes[i].set_title(f'{topology} (n={np.sum(mask)})')
        axes[i].grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Count')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Heatmap saved to {save_path}")
    
def uniformity(q_values, r_values) -> float:
    """
    Quantify uniformity using 2D binning and coefficient of variation
    
    Returns:
        Uniformity coefficient of variation (lower is more uniform)
    """
    print("\nAnalyzing uniformity...")
    
    # Convert to numpy arrays
    q_values = np.array(q_values)
    r_values = np.array(r_values)
    
    # Create 2D histogram
    num_bins = 10
    hist, q_edges, r_edges = np.histogram2d(q_values, r_values, bins=num_bins)
    
    # Calculate uniformity metric (Coefficient of Variation)
    non_empty_bins = hist[hist > 0]
    if len(non_empty_bins) > 0:
        uniformity_cv = np.std(non_empty_bins) / np.mean(non_empty_bins)
    else:
        uniformity_cv = float('inf')
    
    print(f"Uniformity CV: {uniformity_cv:.4f} (lower is more uniform)")
    print(f"Non-empty bins: {len(non_empty_bins)}/{num_bins**2}")
    
    return uniformity_cv
    
def main():
    """Main analysis function"""
    pkl_files = [
        # '200_300_ER_LPA_COPY_2000.pkl',
        # '100_200_ER_LPA_COPY_10000.pkl'
        'switched_graphs.pkl'
        # 'combined_graphs.pkl'
    ]
    
    # Load and analyze
    graphs = load_graphs(pkl_files)
    print(f"Loaded {len(graphs)} graphs")
    
    q_values, r_values, labels = calculate_properties(graphs)
    
    # Visualize and summarize
    create_scatter_plot(q_values, r_values, labels)
    create_heatmap(q_values, r_values, labels)
    uniformity_cv = uniformity(q_values, r_values)
    
    print(f"Uniformity CV: {uniformity_cv:.4f}")
    for label in np.unique(labels):
        mask = labels == label
        q_sub, r_sub = q_values[mask], r_values[mask]
        print(f"\n{label}: Q={q_sub.mean():.3f}±{q_sub.std():.3f}, r={r_sub.mean():.3f}±{r_sub.std():.3f}")

if __name__ == "__main__":
    main()