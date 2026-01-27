import os
import tyro
import time
import torch
import random
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from typing import Optional
from collections import deque
from dataclasses import dataclass
from torch_scatter import scatter_add, scatter_mean
from datetime import datetime, timedelta
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

from env import DismantleEnv
from networks.dismantle import load_sac_dismantler
from utils import ReplayBuffer, PriorReplayBuffer, Batch, validate, validate_with_type_logging, ig_to_data, Discriminator, train_discriminator, DiscriminatorDataset
import torch.nn.functional as F
import gc
from utils import graph_models


@dataclass
class Args:
    use_tb: bool=False
    """record using tensorboard"""
    seed: int=0
    """random seed"""
    device: str='cuda:0'
    """the device to use"""
    gnn: str='hgnn_v3'

    ckpt_pth: Optional[str]=None
    """where checkpoint was saved"""

    num_features: int = 16
    """number of initial node features"""
    num_heads: int=4
    """number of message passings heads"""
    num_mps: int=6
    """number of message passings"""
    
    # Analysis parameters
    num_graphs: int = 2
    """number of graphs to generate for analysis"""
    graph_size_min: int = 50
    """minimum graph size"""
    graph_size_max: int = 100
    """maximum graph size"""


def compute_l2_distance(model1, model2):
    """
    Compute L2 (Frobenius norm) distance between parameters of two models
    """
    total_distance = 0.0
    param_count = 0
    
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 == name2:  # Ensure we're comparing the same parameters
            distance = torch.norm(param1 - param2, p='fro').item()
            total_distance += distance
            param_count += 1
            print(f"Layer {name1}: L2 distance = {distance:.6f}")
    
    avg_distance = total_distance / param_count if param_count > 0 else 0.0
    print(f"Average L2 distance: {avg_distance:.6f}")
    return total_distance, avg_distance


def compute_mean_average_distance(embeddings):
    """
    Compute Mean Average Distance (MAD) of embeddings using Cosine Distance.
    MAD = mean(1 - cosine_similarity(x_i, x_j))
    Measures diversity of representations. Higher is more diverse.
    """
    if embeddings.size(0) <= 1:
        return 0.0
        
    # Normalize embeddings for cosine similarity
    norm_emb = F.normalize(embeddings, p=2, dim=-1)
    
    # Compute similarity matrix: (N, N)
    # Note: For very large batches this might be memory intensive.
    sim_matrix = torch.mm(norm_emb, norm_emb.t())
    
    # Convert to distance: 1 - similarity
    dist_matrix = 1 - sim_matrix
    
    # Mean Average Distance (mean of all pairs)
    mad = dist_matrix.mean()
    return mad.item()


def compute_dirichlet_energy(x, edge_index):
    """
    Computes the Dirichlet Energy of node features x on graph edge_index.
    E(X) = 0.5 * sum_{(i,j) in E} ||x_i - x_j||^2 / N
    Measures signal smoothness/frequency. Lower means smoother (potentially oversmoothed).
    """
    N = x.size(0)
    
    # Calculate squared differences between connected nodes efficiently
    src = edge_index[:,0]
    dst = edge_index[:,1]
    
    # Sum of squared Euclidean distances for all edges
    # shape: (E, F) -> sum over F -> (E,)
    sq_diff = torch.sum((x[src] - x[dst])**2, dim=1)
    
    # Energy = 0.5 * sum(sq_diff)
    energy = 0.5 * sq_diff.sum()
    
    return energy.item() / N

def compute_cka(activations1, activations2):
    """
    Compute Centered Kernel Alignment (CKA) between two activation matrices
    
    Args:
        activations1: [N, D1] activation matrix from model 1
        activations2: [N, D2] activation matrix from model 2
    
    Returns:
        CKA similarity score (0-1, higher is more similar)
    """
    def center_gram_matrix(gram):
        """Center the Gram matrix"""
        n = gram.shape[0]
        H = torch.eye(n, device=gram.device) - torch.ones(n, n, device=gram.device) / n
        return H @ gram @ H
    
    # Compute Gram matrices (dot product matrices)
    gram1 = torch.mm(activations1, activations1.t())
    gram2 = torch.mm(activations2, activations2.t())
    
    # Center the Gram matrices
    gram1_centered = center_gram_matrix(gram1)
    gram2_centered = center_gram_matrix(gram2)
    
    # Compute CKA
    numerator = torch.trace(torch.mm(gram1_centered, gram2_centered))
    denominator = torch.sqrt(torch.trace(torch.mm(gram1_centered, gram1_centered)) * 
                           torch.trace(torch.mm(gram2_centered, gram2_centered)))
    
    cka = numerator / (denominator + 1e-8)
    return cka.item()


def generate_test_graphs(num_graphs, size_min, size_max, seed=42):
    """
    Generate a mix of SBM and BA graphs for testing
    """
    random.seed(seed)
    np.random.seed(seed)
    
    graphs = []
    graph_types = []
    
    for i in range(num_graphs):
        size = random.randint(size_min, size_max)
        
        if i % 2 == 0:  # SBM graphs
            p_in = random.uniform(0.1, 0.2)
            p_out = random.uniform(0.002, 0.008)
            num_blocks = random.randint(2, 4)
            g = graph_models.SBM(size, p_in, p_out, num_blocks)
            graph_types.append(f'SBM_{num_blocks}')
        else:  # BA graphs
            m = random.randint(2, min(8, size//10))
            g = graph_models.BA(size, m)
            graph_types.append(f'BA_{m}')
        
        # Ensure graph is connected and has reasonable size
        if g.is_connected() and g.vcount() >= 10:
            graphs.append(g)
        else:
            # Fallback to simple connected graph
            g = ig.Graph.Ring(max(10, size//2))
            graphs.append(g)
            graph_types[-1] = 'Ring'
    
    return graphs, graph_types


def extract_embeddings(model, batch):
    """
    Extract node and graph embeddings from a model
    """
    with torch.no_grad():
        # Get full embeddings from graph_embedding layer
        embeddings = model.graph_embedding(batch)  # [N, 2*K*F]
        
        # Split into node and graph embeddings (assuming MIND architecture)
        embed_dim = embeddings.shape[1] // 2
        node_embeddings = embeddings[:, :embed_dim]  # [N, K*F]
        graph_embeddings = embeddings[:, embed_dim:]  # [N, K*F]
        
        # Aggregate graph embeddings per graph
        graph_level_embeddings = scatter_mean(graph_embeddings, batch.batch_non_omni, dim=0)  # [B, K*F]
        
    return node_embeddings, graph_level_embeddings


def visualize_embeddings(embeddings, labels, title, method='PCA', save_path=None):
    """
    Visualize embeddings using PCA or t-SNE
    """
    embeddings_np = embeddings.cpu().numpy()
    
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings_np)
        explained_var = reducer.explained_variance_ratio_
        subtitle = f'PCA (explained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f})'
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, max(1,len(embeddings_np)//4)))
        reduced = reducer.fit_transform(embeddings_np)
        subtitle = 't-SNE'
    
    plt.figure(figsize=(10, 8))
    
    # Create color map for different graph types
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7, s=50)
    
    plt.title(f'{title}\n{subtitle}')
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def analyze_parameter_similarity(policy, qf1, qf2):
    """
    Analyze parameter similarity between policy and Q-networks
    """
    # Compare policy vs qf1
    total_dist_1, avg_dist_1 = compute_l2_distance(policy.graph_embedding, qf1.graph_embedding)
    
    # Compare policy vs qf2
    total_dist_2, avg_dist_2 = compute_l2_distance(policy.graph_embedding, qf2.graph_embedding)
    
    # Compare qf1 vs qf2
    total_dist_3, avg_dist_3 = compute_l2_distance(qf1.graph_embedding, qf2.graph_embedding)
    
    return {
        'policy_qf1': avg_dist_1,
        'policy_qf2': avg_dist_2,
        'qf1_qf2': avg_dist_3
    }


def analyze_activation_similarity(policy, qf1, qf2, test_graphs, device):
    """
    Analyze activation similarity using CKA
    """
    # Convert graphs to batch
    batch = Batch(device, [ig_to_data(g) for g in test_graphs])
    
    # Extract embeddings from all models
    policy_node_emb, policy_graph_emb = extract_embeddings(policy, batch)
    qf1_node_emb, qf1_graph_emb = extract_embeddings(qf1, batch)
    qf2_node_emb, qf2_graph_emb = extract_embeddings(qf2, batch)
    
    # Compute CKA for node embeddings
    cka_policy_qf1_node = compute_cka(policy_node_emb, qf1_node_emb)
    cka_policy_qf2_node = compute_cka(policy_node_emb, qf2_node_emb)
    cka_qf1_qf2_node = compute_cka(qf1_node_emb, qf2_node_emb)
    
    # Compute CKA for graph embeddings
    cka_policy_qf1_graph = compute_cka(policy_graph_emb, qf1_graph_emb)
    cka_policy_qf2_graph = compute_cka(policy_graph_emb, qf2_graph_emb)
    cka_qf1_qf2_graph = compute_cka(qf1_graph_emb, qf2_graph_emb)
    
    return {
        'node_embeddings': {
            'policy_qf1': cka_policy_qf1_node,
            'policy_qf2': cka_policy_qf2_node,
            'qf1_qf2': cka_qf1_qf2_node
        },
        'graph_embeddings': {
            'policy_qf1': cka_policy_qf1_graph,
            'policy_qf2': cka_policy_qf2_graph,
            'qf1_qf2': cka_qf1_qf2_graph
        }
    }

def analyze_network_embeddings(policy, test_graphs, device):
    """
    Analyze node and graph embeddings for smoothing metrics (MAD, Dirichlet Energy)
    """
    # Convert graphs to batch
    batch = Batch(device, [ig_to_data(g) for g in test_graphs])
    
    # Extract embeddings
    node_embeddings, graph_embeddings = extract_embeddings(policy, batch)
    
    # 1. Compute MAD for node and graph embeddings
    mad_node = compute_mean_average_distance(node_embeddings)
    mad_graph = compute_mean_average_distance(graph_embeddings)
    
    # 2. Compute Dirichlet Energy for node embeddings
    # We use the batch edge_index which represents the connectivity
    dir_energy = compute_dirichlet_energy(node_embeddings, batch.edge_index)
    
    return {
        'mad_node': mad_node,
        'mad_graph': mad_graph,
        'dirichlet_energy': dir_energy
    }

def visualize_network_embeddings(policy, test_graphs, graph_types, device, save_dir='visualizations'):
    """
    Visualize node and graph embeddings
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert graphs to batch
    batch = Batch(device, [ig_to_data(g) for g in test_graphs])
    
    # Extract embeddings
    node_embeddings, graph_embeddings = extract_embeddings(policy, batch)
    
    # Create labels for nodes (by graph type)
    node_labels = []
    for i, graph_type in enumerate(graph_types):
        graph_size = test_graphs[i].vcount()
        node_labels.extend([graph_type] * graph_size)
    
    # Visualize node embeddings
    print("\nVisualizing node embeddings...")
    visualize_embeddings(
        node_embeddings, node_labels, 
        'Node Embeddings by Graph Type', 
        method='PCA',
        save_path=os.path.join(save_dir, 'node_embeddings_pca.png')
    )
    
    visualize_embeddings(
        node_embeddings, node_labels, 
        'Node Embeddings by Graph Type', 
        method='TSNE',
        save_path=os.path.join(save_dir, 'node_embeddings_tsne.png')
    )
    
    # Visualize graph embeddings
    print("\nVisualizing graph embeddings...")
    visualize_embeddings(
        graph_embeddings, graph_types, 
        'Graph Embeddings by Graph Type', 
        method='PCA',
        save_path=os.path.join(save_dir, 'graph_embeddings_pca.png')
    )
    
    visualize_embeddings(
        graph_embeddings, graph_types, 
        'Graph Embeddings by Graph Type', 
        method='TSNE',
        save_path=os.path.join(save_dir, 'graph_embeddings_tsne.png')
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    
    # Load models
    print("\nLoading SAC dismantler models...")
    policy, qf1, qf2, qf1_target, qf2_target = load_sac_dismantler(
        args.num_features, args.num_heads, args.num_mps, args.gnn, device, args.ckpt_pth
    )
    
    if args.ckpt_pth:
        print(f"Loaded checkpoint: {args.ckpt_pth}")
    else:
        print("Using randomly initialized models")
    
    # Set models to evaluation mode
    policy.eval()
    qf1.eval()
    qf2.eval()
    
    # Generate test graphs
    print(f"\nGenerating {args.num_graphs} test graphs...")
    test_graphs, graph_types = generate_test_graphs(
        args.num_graphs, args.graph_size_min, args.graph_size_max, args.seed
    )
    
    print(f"Generated graphs: {len(test_graphs)}")
    print(f"Graph types: {set(graph_types)}")
    
    # # 1. Parameter similarity analysis
    # param_similarities = analyze_parameter_similarity(policy, qf1, qf2)
    
    # # 2. Activation similarity analysis
    # activation_similarities = analyze_activation_similarity(policy, qf1, qf2, test_graphs, device)
    
    # # 3. Embedding visualization
    # save_dir = f"visualizations/{args.gnn}_mps_{args.num_mps}"
    # visualize_network_embeddings(policy, test_graphs, graph_types, device, save_dir)

    # 4. Network Embedding Analysis (MAD, Dirichlet)
    embedding_metrics = analyze_network_embeddings(policy, test_graphs, device)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Parameter L2 distances:")
    for key, value in param_similarities.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nNode embedding CKA scores:")
    for key, value in activation_similarities['node_embeddings'].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nGraph embedding CKA scores:")
    for key, value in activation_similarities['graph_embeddings'].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nVisualization saved to 'visualizations/{args.gnn}' directory")
    print("Analysis complete!") 
