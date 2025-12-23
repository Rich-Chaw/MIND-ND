#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis Script for MIND-ND Synthetic Graph Dataset

This script implements the five analysis methods outlined in dataset_analysis.md:
1. 2D Scatter Plot and Uniformity Quantification (Q vs. r)
2. Feature Representation Similarity
3. Graph Spectral Analysis
4. Small-World Property Quantification
5. Graphlets and Motif Counting

Author: Generated for MIND-ND project
Date: December 2024
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Graph libraries
import igraph as ig
import networkx as nx
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Will use PCA for dimensionality reduction.")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# PyTorch for MIND model
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.mind import MIND
from utils.graph_data import ig_to_data, Batch


class DatasetAnalyzer:
    """Comprehensive analyzer for synthetic graph datasets"""
    
    def __init__(self, data_dir: str = "train/200_300", model_path: str = "saved/mind.ckpt",mind_model = None):
        self.data_dir = data_dir
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load synthetic graphs
        self.synthetic_graphs = self._load_synthetic_graphs()
        print(f"Loaded {len(self.synthetic_graphs)} synthetic graphs from {data_dir}")
        
        # Load MIND model
        self.mind_model = mind_model
 
        # Results storage
        self.results = {}
        
    def _load_synthetic_graphs(self) -> List[ig.Graph]:
        """Load all synthetic graphs from the data directory"""
        graphs = []
        data_path = Path(self.data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
            
        for file_path in sorted(data_path.glob("*.pkl")):
            try:
                with open(file_path, 'rb') as f:
                    graph = pickle.load(f)
                    if isinstance(graph, ig.Graph):
                        graphs.append(graph)
                    else:
                        print(f"Warning: {file_path} does not contain an igraph.Graph object")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return graphs
    
    def analyze_structural_diversity(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Method 1: 2D Scatter Plot and Uniformity Quantification (Modularity Q vs. Assortativity r)
        """
        print("\n=== Analyzing Structural Diversity (Modularity Q vs Assortativity r) ===")
        
        q_values = []
        r_values = []
        
        for i, graph in enumerate(self.synthetic_graphs):
            # if i % 100 == 0:
            #     print(f"Processing graph {i+1}/{len(self.synthetic_graphs)}")
            
            try:
                # Calculate Modularity (Q) - requires community detection first
                # communities = graph.community_leiden(resolution_parameter=1.0)
                # q = graph.modularity(communities)
                leiden_comm = graph.community_leiden(
    objective_function="modularity", 
    weights=None, 
    resolution_parameter=1.0, 
    n_iterations=2
)
                q = leiden_comm.modularity
                
                # Calculate Assortativity (r)
                r = graph.assortativity_degree()
                
                q_values.append(q)
                r_values.append(r)
                
            except Exception as e:
                print(f"Error processing graph {i}: {e}")
                continue
        
        q_values = np.array(q_values)
        r_values = np.array(r_values)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(q_values, r_values, alpha=0.6, s=20)
        plt.xlabel('Modularity (Q)')
        plt.ylabel('Assortativity (r)')
        plt.title('Synthetic Dataset Structural Coverage')
        plt.grid(True, alpha=0.3)
        
        # Add density contours
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([q_values, r_values])
            kde = gaussian_kde(xy)
            x_grid = np.linspace(q_values.min(), q_values.max(), 50)
            y_grid = np.linspace(r_values.min(), r_values.max(), 50)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)
            plt.contour(X, Y, Z, levels=5, alpha=0.5)
        except:
            pass
        
        plt.tight_layout()
        plt.savefig('graphs/structural_diversity_scatter.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Quantify uniformity using 2D binning
        num_bins = 10
        hist, q_edges, r_edges = np.histogram2d(q_values, r_values, bins=num_bins)
        
        # Calculate uniformity metric (Coefficient of Variation)
        non_empty_bins = hist[hist > 0]
        if len(non_empty_bins) > 0:
            uniformity_cv = np.std(non_empty_bins) / np.mean(non_empty_bins)
        else:
            uniformity_cv = float('inf')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(hist.T, annot=True, fmt='g', cmap='viridis',
                   xticklabels=np.round(q_edges[:-1], 2),
                   yticklabels=np.round(r_edges[:-1], 2))
        plt.xlabel('Modularity (Q)')
        plt.ylabel('Assortativity (r)')
        plt.title(f'Structural Coverage Heatmap (CV = {uniformity_cv:.3f})')
        plt.tight_layout()
        plt.savefig('graphs/structural_diversity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['structural_diversity'] = {
            'uniformity_cv': uniformity_cv,
            'q_values': q_values,
            'r_values': r_values,
            'q_range': (q_values.min(), q_values.max()),
            'r_range': (r_values.min(), r_values.max())
        }
        
        print(f"Uniformity CV: {uniformity_cv:.4f} (lower is more uniform)")
        print(f"Q range: [{q_values.min():.3f}, {q_values.max():.3f}]")
        print(f"r range: [{r_values.min():.3f}, {r_values.max():.3f}]")
        
        return uniformity_cv, q_values, r_values   
 
    def analyze_feature_similarity(self, real_graphs: List[ig.Graph] = None) -> float:
        """
        Method 2: Feature Representation Similarity using MIND embeddings
        """
        print("\n=== Analyzing Feature Similarity ===")
        
        # If no real graphs provided, create some reference graphs
        if real_graphs is None:
            print("No real graphs provided. Creating reference graphs for comparison.")
            real_graphs = self._create_reference_graphs()
        
        # Extract embeddings from synthetic graphs
        synthetic_embeddings = []
        batch_size = 32  # Process in batches to avoid memory issues
        
        print("Extracting synthetic graph embeddings...")
        for i in range(0, len(self.synthetic_graphs), batch_size):
            batch_graphs = self.synthetic_graphs[i:i+batch_size]
            try:
                batch_data = [ig_to_data(g) for g in batch_graphs]
                batch_obj = Batch(self.device, batch_data)
                
                with torch.no_grad():
                    embeddings = self.mind_model(batch_obj)
                    # Global pooling to get graph-level embeddings
                    graph_embeddings = []
                    start_idx = 0
                    for j, g in enumerate(batch_graphs):
                        end_idx = start_idx + g.vcount()
                        graph_emb = embeddings[start_idx:end_idx].mean(dim=0)
                        graph_embeddings.append(graph_emb.cpu().numpy())
                        start_idx = end_idx
                    
                    synthetic_embeddings.extend(graph_embeddings)
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue
        
        # Extract embeddings from real graphs
        real_embeddings = []
        print("Extracting real graph embeddings...")
        for i in range(0, len(real_graphs), batch_size):
            batch_graphs = real_graphs[i:i+batch_size]
            try:
                batch_data = [ig_to_data(g) for g in batch_graphs]
                batch_obj = Batch(self.device, batch_data)
                
                with torch.no_grad():
                    embeddings = self.mind_model(batch_obj)
                    # Global pooling to get graph-level embeddings
                    graph_embeddings = []
                    start_idx = 0
                    for j, g in enumerate(batch_graphs):
                        end_idx = start_idx + g.vcount()
                        graph_emb = embeddings[start_idx:end_idx].mean(dim=0)
                        graph_embeddings.append(graph_emb.cpu().numpy())
                        start_idx = end_idx
                    
                    real_embeddings.extend(graph_embeddings)
            except Exception as e:
                print(f"Error processing real batch {i//batch_size}: {e}")
                continue
        
        if len(synthetic_embeddings) == 0 or len(real_embeddings) == 0:
            print("Warning: Could not extract embeddings. Skipping feature similarity analysis.")
            return 0.5
        
        # Combine embeddings
        synthetic_embeddings = np.array(synthetic_embeddings)
        real_embeddings = np.array(real_embeddings)
        
        # Sample if too many graphs
        max_samples = 1000
        if len(synthetic_embeddings) > max_samples:
            indices = np.random.choice(len(synthetic_embeddings), max_samples, replace=False)
            synthetic_embeddings = synthetic_embeddings[indices]
        if len(real_embeddings) > max_samples:
            indices = np.random.choice(len(real_embeddings), max_samples, replace=False)
            real_embeddings = real_embeddings[indices]
        
        combined_embeddings = np.vstack([synthetic_embeddings, real_embeddings])
        labels = np.hstack([
            np.zeros(len(synthetic_embeddings)),  # 0 for synthetic
            np.ones(len(real_embeddings))         # 1 for real
        ])
        
        # Apply UMAP for visualization
        if UMAP_AVAILABLE:
            print("Applying UMAP dimensionality reduction...")
            try:
                umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                projected_embeddings = umap_reducer.fit_transform(combined_embeddings)
                
                # Visualize
                plt.figure(figsize=(12, 8))
                colors = ['blue', 'red']
                labels_str = ['Synthetic', 'Real']
                
                for i, (color, label) in enumerate(zip(colors, labels_str)):
                    mask = labels == i
                    plt.scatter(projected_embeddings[mask, 0], projected_embeddings[mask, 1], 
                               c=color, alpha=0.6, s=20, label=label)
                
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')
                plt.title('Feature Space Similarity (UMAP Projection)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('graphs/feature_similarity_umap.png', dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"UMAP failed: {e}. Using PCA instead.")
                pca = PCA(n_components=2)
                projected_embeddings = pca.fit_transform(combined_embeddings)
                
                plt.figure(figsize=(12, 8))
                colors = ['blue', 'red']
                labels_str = ['Synthetic', 'Real']
                
                for i, (color, label) in enumerate(zip(colors, labels_str)):
                    mask = labels == i
                    plt.scatter(projected_embeddings[mask, 0], projected_embeddings[mask, 1], 
                               c=color, alpha=0.6, s=20, label=label)
                
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title('Feature Space Similarity (PCA Projection)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('graphs/feature_similarity_pca.png', dpi=300, bbox_inches='tight')
                plt.show()
        else:
            print("UMAP not available. Using PCA for dimensionality reduction...")
            pca = PCA(n_components=2)
            projected_embeddings = pca.fit_transform(combined_embeddings)
            
            plt.figure(figsize=(12, 8))
            colors = ['blue', 'red']
            labels_str = ['Synthetic', 'Real']
            
            for i, (color, label) in enumerate(zip(colors, labels_str)):
                mask = labels == i
                plt.scatter(projected_embeddings[mask, 0], projected_embeddings[mask, 1], 
                           c=color, alpha=0.6, s=20, label=label)
            
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('Feature Space Similarity (PCA Projection)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('graphs/feature_similarity_pca.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Quantify overlap using classifier
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                projected_embeddings, labels, test_size=0.3, random_state=42, stratify=labels
            )
            
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Classification accuracy: {accuracy:.4f} (lower means better overlap)")
            
            self.results['feature_similarity'] = {
                'classification_accuracy': accuracy,
                'num_synthetic': len(synthetic_embeddings),
                'num_real': len(real_embeddings)
            }
            
            return accuracy
            
        except Exception as e:
            print(f"Classification failed: {e}")
            return 0.5
    
    def _create_reference_graphs(self) -> List[ig.Graph]:
        """Create reference graphs for comparison when real graphs are not available"""
        reference_graphs = []
        
        # Create various types of reference graphs
        n_nodes = 200  # Similar to synthetic graphs
        
        # Erdős-Rényi graphs
        for p in [0.01, 0.05, 0.1, 0.2]:
            g = ig.Graph.Erdos_Renyi(n_nodes, p)
            reference_graphs.append(g)
        
        # Barabási-Albert graphs
        for m in [1, 2, 3, 5]:
            g = ig.Graph.Barabasi(n_nodes, m)
            reference_graphs.append(g)
        
        # Watts-Strogatz small-world graphs
        for p in [0.01, 0.1, 0.5]:
            g = ig.Graph.Watts_Strogatz(1, n_nodes, 4, p)
            reference_graphs.append(g)
        
        # Regular lattice
        g = ig.Graph.Lattice([int(np.sqrt(n_nodes)), int(np.sqrt(n_nodes))], circular=False)
        reference_graphs.append(g)
        
        return reference_graphs
    
    def analyze_graph_spectrum(self) -> Tuple[float, List[float]]:
        """
        Method 3: Graph Spectral Analysis
        """
        print("\n=== Analyzing Graph Spectrum ===")
        
        adj_spec = []
        lap_spec = []
        lambda2_values = []
        spectral_gaps = []
        
        for i, graph in enumerate(self.synthetic_graphs):
            if i % 100 == 0:
                print(f"Processing graph {i+1}/{len(self.synthetic_graphs)}")
            
            try:
                # Get adjacency matrix eigenvalues
                adj_matrix = np.array(graph.get_adjacency().data)
                adj_eigenvals = np.linalg.eigvals(adj_matrix)
                adj_eigenvals = np.sort(adj_eigenvals)[::-1]  # Descending order
                
                # Get Laplacian matrix eigenvalues
                laplacian = np.array(graph.laplacian(normalized=True))
                lap_eigenvals = np.linalg.eigvals(laplacian)
                lap_eigenvals = np.sort(lap_eigenvals)  # Ascending order
                
                # Store top 5 eigenvalues
                adj_spec.append(adj_eigenvals[:5] if len(adj_eigenvals) >= 5 else adj_eigenvals)
                lap_spec.append(lap_eigenvals[:5] if len(lap_eigenvals) >= 5 else lap_eigenvals)
                
                # Fiedler value (second smallest Laplacian eigenvalue)
                if len(lap_eigenvals) >= 2:
                    lambda2_values.append(lap_eigenvals[1])
                
                # Spectral gap (difference between largest and second largest adjacency eigenvalue)
                if len(adj_eigenvals) >= 2:
                    spectral_gaps.append(adj_eigenvals[0] - adj_eigenvals[1])
                
            except Exception as e:
                print(f"Error processing graph {i}: {e}")
                continue
        
        # Visualize Fiedler values distribution
        if lambda2_values:
            plt.figure(figsize=(10, 6))
            plt.hist(lambda2_values, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Fiedler Value (λ₂)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Graph Bisection Costs')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('graphs/fiedler_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Visualize spectral gaps
        if spectral_gaps:
            plt.figure(figsize=(10, 6))
            plt.hist(spectral_gaps, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Spectral Gap')
            plt.ylabel('Frequency')
            plt.title('Distribution of Adjacency Spectral Gaps')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('graphs/spectral_gap_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        avg_spectral_gap = np.mean(spectral_gaps) if spectral_gaps else 0
        
        self.results['spectral_analysis'] = {
            'avg_spectral_gap': avg_spectral_gap,
            'lambda2_values': lambda2_values,
            'lambda2_mean': np.mean(lambda2_values) if lambda2_values else 0,
            'lambda2_std': np.std(lambda2_values) if lambda2_values else 0
        }
        
        print(f"Average spectral gap: {avg_spectral_gap:.4f}")
        print(f"Fiedler value - Mean: {np.mean(lambda2_values):.4f}, Std: {np.std(lambda2_values):.4f}")
        
        return avg_spectral_gap, lambda2_values 
   
    def analyze_small_world_property(self) -> Tuple[float, float]:
        """
        Method 4: Small-World Property Quantification
        """
        print("\n=== Analyzing Small-World Property ===")
        
        sigma_values = []
        omega_values = []
        
        for i, graph in enumerate(self.synthetic_graphs):
            if i % 100 == 0:
                print(f"Processing graph {i+1}/{len(self.synthetic_graphs)}")
            
            try:
                # Skip if graph is too small or disconnected
                if graph.vcount() < 10 or not graph.is_connected():
                    continue
                
                # Calculate properties of the synthetic graph
                L_g = graph.average_path_length()
                C_g = graph.transitivity_avglocal_undirected()  # average local clustering coeffiecient
                
                if C_g is None or C_g == 0:
                    continue
                
                # Generate reference random graph with same N and average degree
                N = graph.vcount()
                k_avg = 2 * graph.ecount() / N  # Average degree
                p_rand = k_avg / (N - 1)
                g_rand = ig.Graph.Erdos_Renyi(N, p_rand)
                
                # Ensure random graph is connected
                if not g_rand.is_connected():
                    # Add edges to make it connected
                    components = g_rand.components()
                    for j in range(len(components) - 1):
                        v1 = components[j][0]
                        v2 = components[j + 1][0]
                        g_rand.add_edge(v1, v2)
                
                L_rand = g_rand.average_path_length()
                C_rand = g_rand.transitivity_avglocal_undirected()
                
                if C_rand is None or C_rand == 0 or L_rand == 0:
                    continue
                
                # Calculate Small-World Index (Sigma)
                sigma = (C_g / C_rand) / (L_g / L_rand)
                sigma_values.append(sigma)
                
                # Calculate Omega (alternative measure)
                # Create lattice reference (approximate)
                try:
                    # Create a ring lattice as approximation
                    k_lattice = int(k_avg)
                    if k_lattice >= 2 and k_lattice < N and k_lattice % 2 == 0:
                        g_lattice = ig.Graph.Ring(N, directed=False)
                        # Add additional edges to reach k_lattice neighbors
                        for i in range(2, k_lattice//2 + 1):
                            for v in range(N):
                                neighbor = (v + i) % N
                                if not g_lattice.are_connected(v, neighbor):
                                    g_lattice.add_edge(v, neighbor)
                        
                        C_lattice = g_lattice.transitivity_avglocal_undirected()
                        if C_lattice is not None and C_lattice > 0:
                            omega = (L_rand / L_g) - (C_g / C_lattice)
                            omega_values.append(omega)
                except:
                    pass
                
            except Exception as e:
                if i < 10:  # Only print first few errors
                    print(f"Error processing graph {i}: {e}")
                continue
        
        # Visualize Sigma distribution
        if sigma_values:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(sigma_values, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Small-World Index (σ)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Small-World Property (σ)')
            plt.axvline(x=1, color='red', linestyle='--', label='σ = 1 (random)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if omega_values:
                plt.subplot(1, 2, 2)
                plt.hist(omega_values, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Small-World Index (ω)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Small-World Property (ω)')
                plt.axvline(x=0, color='red', linestyle='--', label='ω = 0 (small-world)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('graphs/small_world_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        avg_sigma = np.mean(sigma_values) if sigma_values else 0
        avg_omega = np.mean(omega_values) if omega_values else 0
        
        self.results['small_world'] = {
            'avg_sigma': avg_sigma,
            'avg_omega': avg_omega,
            'sigma_values': sigma_values,
            'omega_values': omega_values,
            'num_analyzed': len(sigma_values)
        }
        
        print(f"Average Sigma: {avg_sigma:.4f} (>1 indicates small-world)")
        print(f"Average Omega: {avg_omega:.4f} (≈0 indicates small-world)")
        print(f"Analyzed {len(sigma_values)} graphs for small-world properties")
        
        return avg_sigma, avg_omega
    
    def analyze_graphlet_diversity(self, k_size: int = 4) -> Tuple[float, np.ndarray]:
        """
        Method 5: Graphlets and Motif Counting
        """
        print(f"\n=== Analyzing Graphlet Diversity (k={k_size}) ===")
        
        if k_size == 3:
            # For k=3, we can count triangles and other 3-node motifs
            motif_counts = []
            
            for i, graph in enumerate(self.synthetic_graphs):
                if i % 100 == 0:
                    print(f"Processing graph {i+1}/{len(self.synthetic_graphs)}")
                
                try:
                    # Count triangles
                    triangles = len(graph.cliques(min=3, max=3))
                    
                    # Count 3-stars (one central node connected to 3 others)
                    stars = 0
                    for v in range(graph.vcount()):
                        degree = graph.degree(v)
                        if degree >= 3:
                            # Number of ways to choose 3 neighbors from degree neighbors
                            stars += degree * (degree - 1) * (degree - 2) // 6
                    
                    # Count paths of length 2
                    paths = 0
                    for v in range(graph.vcount()):
                        neighbors = graph.neighbors(v)
                        for i, n1 in enumerate(neighbors):
                            for j, n2 in enumerate(neighbors[i+1:], i+1):
                                if not graph.are_connected(n1, n2):
                                    paths += 1
                    
                    total_motifs = triangles + stars + paths
                    if total_motifs > 0:
                        gfd = np.array([triangles, stars, paths]) / total_motifs
                        motif_counts.append(gfd)
                    
                except Exception as e:
                    if i < 10:
                        print(f"Error processing graph {i}: {e}")
                    continue
            
            motif_labels = ['Triangles', '3-Stars', '2-Paths']
            
        else:
            # For k=4 and higher, use a simplified approach
            print(f"Warning: Full graphlet analysis for k={k_size} is computationally expensive.")
            print("Using simplified motif counting...")
            
            motif_counts = []
            
            for i, graph in enumerate(self.synthetic_graphs[:min(100, len(self.synthetic_graphs))]):
                if i % 20 == 0:
                    print(f"Processing graph {i+1}")
                
                try:
                    # Count some basic motifs
                    cliques_3 = len(graph.cliques(min=3, max=3))  # Triangles
                    cliques_4 = len(graph.cliques(min=4, max=4))  # 4-cliques
                    
                    # Count stars
                    stars_3 = sum(1 for v in range(graph.vcount()) if graph.degree(v) >= 3)
                    stars_4 = sum(1 for v in range(graph.vcount()) if graph.degree(v) >= 4)
                    
                    total_motifs = cliques_3 + cliques_4 + stars_3 + stars_4
                    if total_motifs > 0:
                        gfd = np.array([cliques_3, cliques_4, stars_3, stars_4]) / total_motifs
                        motif_counts.append(gfd)
                    
                except Exception as e:
                    continue
            
            motif_labels = ['3-Cliques', '4-Cliques', '3-Stars', '4-Stars']
        
        if not motif_counts:
            print("No motifs could be counted.")
            return 0, np.array([])
        
        # Convert to matrix and analyze diversity
        gfd_matrix = np.array(motif_counts)
        motif_diversity_vector = np.std(gfd_matrix, axis=0)
        
        # Visualize motif diversity
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(motif_labels, motif_diversity_vector)
        plt.ylabel('Standard Deviation of Motif Frequency')
        plt.title(f'Local Structural Diversity (k={k_size} Graphlets)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Show average motif frequencies
        avg_frequencies = np.mean(gfd_matrix, axis=0)
        plt.bar(motif_labels, avg_frequencies)
        plt.ylabel('Average Motif Frequency')
        plt.title(f'Average Motif Distribution (k={k_size})')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'graphs/motif_diversity_k{k_size}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        overall_diversity_score = np.mean(motif_diversity_vector)
        
        self.results['graphlet_diversity'] = {
            'overall_diversity_score': overall_diversity_score,
            'motif_diversity_vector': motif_diversity_vector,
            'motif_labels': motif_labels,
            'avg_frequencies': np.mean(gfd_matrix, axis=0),
            'num_analyzed': len(motif_counts)
        }
        
        print(f"Overall diversity score: {overall_diversity_score:.4f}")
        print(f"Analyzed {len(motif_counts)} graphs for motif diversity")
        
        return overall_diversity_score, motif_diversity_vector
    
    def run_full_analysis(self, real_graphs: List[ig.Graph] = None) -> Dict[str, Any]:
        """Run all analysis methods and generate comprehensive report"""
        print("=" * 60)
        print("COMPREHENSIVE DATASET ANALYSIS")
        print("=" * 60)
        
        # Run all analyses
        try:
            self.analyze_structural_diversity()
        except Exception as e:
            print(f"Structural diversity analysis failed: {e}")
        
        try:
            self.analyze_feature_similarity(real_graphs)
        except Exception as e:
            print(f"Feature similarity analysis failed: {e}")
        
        try:
            self.analyze_graph_spectrum()
        except Exception as e:
            print(f"Spectral analysis failed: {e}")
        
        try:
            self.analyze_small_world_property()
        except Exception as e:
            print(f"Small-world analysis failed: {e}")
        
        try:
            self.analyze_graphlet_diversity(k_size=3)
        except Exception as e:
            print(f"Graphlet diversity analysis failed: {e}")
        
        # Generate summary report
        self._generate_summary_report()
        
        return self.results
    
    def _generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY REPORT")
        print("=" * 60)
        
        print(f"Dataset: {self.data_dir}")
        print(f"Total graphs analyzed: {len(self.synthetic_graphs)}")
        print(f"Model: {self.model_path}")
        print("-" * 60)
        
        # Structural Diversity
        if 'structural_diversity' in self.results:
            sd = self.results['structural_diversity']
            print(f"1. STRUCTURAL DIVERSITY (Q vs r)")
            print(f"   Uniformity CV: {sd['uniformity_cv']:.4f} (lower = more uniform)")
            print(f"   Modularity range: [{sd['q_range'][0]:.3f}, {sd['q_range'][1]:.3f}]")
            print(f"   Assortativity range: [{sd['r_range'][0]:.3f}, {sd['r_range'][1]:.3f}]")
        
        # Feature Similarity
        if 'feature_similarity' in self.results:
            fs = self.results['feature_similarity']
            print(f"2. FEATURE SIMILARITY")
            print(f"   Classification accuracy: {fs['classification_accuracy']:.4f} (lower = better overlap)")
            print(f"   Synthetic graphs: {fs['num_synthetic']}, Real graphs: {fs['num_real']}")
        
        # Spectral Analysis
        if 'spectral_analysis' in self.results:
            sa = self.results['spectral_analysis']
            print(f"3. SPECTRAL ANALYSIS")
            print(f"   Average spectral gap: {sa['avg_spectral_gap']:.4f}")
            print(f"   Fiedler value - Mean: {sa['lambda2_mean']:.4f}, Std: {sa['lambda2_std']:.4f}")
        
        # Small-World
        if 'small_world' in self.results:
            sw = self.results['small_world']
            print(f"4. SMALL-WORLD PROPERTIES")
            print(f"   Average Sigma: {sw['avg_sigma']:.4f} (>1 = small-world)")
            print(f"   Average Omega: {sw['avg_omega']:.4f} (≈0 = small-world)")
            print(f"   Graphs analyzed: {sw['num_analyzed']}")
        
        # Graphlet Diversity
        if 'graphlet_diversity' in self.results:
            gd = self.results['graphlet_diversity']
            print(f"5. GRAPHLET DIVERSITY")
            print(f"   Overall diversity score: {gd['overall_diversity_score']:.4f}")
            print(f"   Graphs analyzed: {gd['num_analyzed']}")
        
        print("-" * 60)
        print("Analysis complete! Check the graphs/ directory for visualization plots.")


def main():
    """Main function to run the analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze synthetic graph dataset')
    parser.add_argument('--data_dir', default='train/200_300', 
                       help='Directory containing synthetic graphs')
    parser.add_argument('--model_path', default='saved/mind.ckpt',
                       help='Path to pretrained MIND model')
    parser.add_argument('--real_graphs_dir', default=None,
                       help='Directory containing real graphs for comparison')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(args.data_dir, args.model_path)
    
    # Load real graphs if provided
    real_graphs = None
    if args.real_graphs_dir and os.path.exists(args.real_graphs_dir):
        real_graphs = []
        for file_path in Path(args.real_graphs_dir).glob("*.pkl"):
            try:
                with open(file_path, 'rb') as f:
                    graph = pickle.load(f)
                    if isinstance(graph, ig.Graph):
                        real_graphs.append(graph)
            except:
                continue
        print(f"Loaded {len(real_graphs)} real graphs for comparison")
    
    # Run analysis
    results = analyzer.run_full_analysis(real_graphs)
    
    # Save results
    with open('graphs/analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nResults saved to graphs/analysis_results.pkl")


if __name__ == "__main__":
    main()