#!/usr/bin/env python3
"""
Structural Diversity Analysis for MIND-ND Synthetic Graphs
Visualizes Modularity vs Assortativity by graph type (ER, LPA, Copy)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from collections import Counter
import warnings
import powerlaw
warnings.filterwarnings('ignore')

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

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
                g["config"] = {"topology": topology}
                graphs.append((g, topology))
            except:
                continue

    return graphs

def statistics(g):
    leiden_comm = g.community_leiden(
        objective_function="modularity", 
        weights=None, 
        resolution_parameter=1.0, 
        n_iterations=2
    )

    if g.is_connected():
        df = pd.DataFrame(columns=['Num_nodes','Num_edges','AvgDegree', 'Diam', 'AvgShortPath','Clustering Coffe','r','Q'])
        N = g.vcount()
        E = g.vcount()
        AD = np.mean(g.degree())
        CC = g.transitivity_avglocal_undirected() 
        Diam = g.diameter()
        AvgShortPath = g.average_path_length()
        r = g.assortativity_degree()
        Q = leiden_comm.modularity
        df.loc[len(df)] = [N,E,AD, Diam, AvgShortPath,CC, r,Q]
        return df
        # print(df)
    else: 
        print("unconnected graph")
        df = pd.DataFrame(columns=['Num_nodes','Num_edges','AvgDegree','Clustering Coffe','r','Q'])
        N = g.vcount()
        E = g.vcount()
        AD = np.mean(g.degree())
        CC = g.transitivity_avglocal_undirected() 
        r = g.assortativity_degree()
        Q = leiden_comm.modularity
        df.loc[len(df)] = [N,E,AD,CC, r,Q]
        return df

def analyze_powerlaw(graph):
    degrees = graph.degree()
    
    fit = powerlaw.Fit(degrees, discrete=True)
    
    # Calculate the Power-law Coefficient (alpha)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    
    # Significance Testing: Compare power_law vs exponential distribution
    # R is the loglikelihood ratio. Positive R favors the first distribution.
    R, p_value = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    
    print(f"--- Power-law Analysis ---")
    print(f"Alpha (Coefficient): {alpha:.4f}")
    print(f"xmin (Threshold): {xmin}")
    print(f"Loglikelihood Ratio (R): {R:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if R > 0 and p_value < 0.05:
        print("Result: Power-law is significantly more likely than Exponential.")
    else:
        print("Result: Power-law distribution is NOT statistically significant.")

    # 5. Visualization
    plt.figure() 
    fig = fit.plot_pdf(color='b', linewidth=2, label='Empirical Data')
    fit.power_law.plot_pdf(color='r', linestyle='--', ax=fig, label='Power-law Fit')
    plt.xlabel('Degree (k)')
    plt.ylabel('P(k)')
    plt.legend()
    plt.show()

def calculate_properties(graphs):
    """Calculate modularity, assortativity and clustering for all graphs; return a DataFrame."""
    rows = []
    
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
            
            # Clustering coefficient (average local transitivity)
            clustering = graph.transitivity_avglocal_undirected()
            
            rows.append({
                "Q": q,
                "r": r,
                "clustering": clustering,
                "label": graph["config"]["topology"],
            })
            
        except Exception as e:
            print(e)
            continue
    
    df = pd.DataFrame(rows)
    return df

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
    q_values, r_values, labels = calculate_properties(graphs)
    X = np.column_stack([q_values, r_values])
    
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
    q_values, r_values, labels = calculate_properties(graphs)
    
    print(f"\n{label}: Q={q_sub.mean():.3f}±{q_sub.std():.3f}, r={r_sub.mean():.3f}±{r_sub.std():.3f}")

if __name__ == "__main__":
    main()