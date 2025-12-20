"""
Multi-Agent MIND with Global Coordinator Test Script
Implements the algorithm from multiagent_eval.md using METIS partitioning

Key Insight about Omni-Node:
- Each graph has a virtual omni-node connected to ALL nodes
- The omni-node aggregates global graph information through message passing
- Node features are concatenated as [local_features, omni_features] 
- This gives every node access to both local neighborhood and global graph context
- In multi-agent setting: Global Coordinator's omni-node provides global awareness
  to Partition Agents' local omni-nodes, enabling coordinated decision making
"""

import pickle
import igraph as ig
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Tuple, Dict
import time
from dataclasses import dataclass

# Import required modules
from utils.community_detection import partition
from networks.dismantle import SACPolicy
from utils.graph_data import Batch, ig_to_data


@dataclass
class MultiAgentConfig:
    """Configuration for Multi-Agent MIND"""
    K: int = 4  # Number of partition agents
    target_lcc_ratio: float = 0.1  # Target LCC size as fraction of original
    step_ratio: float = 0.0025  # Step ratio for node removal
    ckpt_path: str = 'saved/mind.ckpt'  # Path to MIND checkpoint
    device: str = 'cuda:0'  # Device for computation
    max_iterations: int = 1000  # Maximum iterations to prevent infinite loops


class GlobalCoordinator:
    """Global Coordinator Agent that computes global context and selects best nodes"""
    
    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load the MIND policy for global coordination
        self.policy = SACPolicy(
            num_features=16,
            num_heads=4,
            num_mps=6,
        ).to(self.device)
        self.policy.load_state_dict(torch.load(config.ckpt_path)['policy_state_dict'])
        self.policy.eval()
    
    def compute_global_omni_node(self, G: ig.Graph) -> torch.Tensor:
        """
        Compute global context Z_global that reflects the LCC of the entire graph G
        This extracts the omni-node embedding which contains global graph information
        """
        with torch.no_grad():
            batch_data = Batch(self.device, [ig_to_data(G)])
            
            # Get the graph embedding from MIND encoder
            # This processes the entire graph including the omni-node
            graph_embedding = self.policy.graph_embedding(batch_data)  # [N, 2*K*F]
            
            # The graph_embedding is for non-omni nodes only, but each node's embedding
            # contains [node_embedding_KF, omni_node_embedding_KF]
            # We want to extract the global omni-node embedding (second half)
            K_F = graph_embedding.shape[1] // 2  # K*F dimension
            
            # Extract the omni-node part from any node (they all have the same omni-node embedding)
            global_omni_embedding = graph_embedding[0, K_F:]  # [K*F] - second half
            
            return global_omni_embedding
    
    def select_best_node(self, candidate_nodes: List[Tuple[List[int], np.ndarray]], 
                        original_graph: ig.Graph) -> int:
        """
        Select the best node across all partition agents
        
        Args:
            candidate_nodes: List of (subgraph_nodes, probabilities) from each agent
            original_graph: The original graph for mapping back node IDs
            
        Returns:
            Node ID in the original graph to remove
        """
        best_score = -1
        best_node = None
        
        for subgraph_nodes, probs in candidate_nodes:
            if len(probs) > 0 and len(subgraph_nodes) > 0:
                # Find the best node in this partition
                best_local_idx = np.argmax(probs)
                best_local_score = probs[best_local_idx]
                
                if best_local_score > best_score:
                    best_score = best_local_score
                    # Map back to original graph node ID
                    best_node = subgraph_nodes[best_local_idx]
        
        return best_node


class PartitionAgent:
    """Partition Agent that evaluates nodes in its assigned subgraph"""
    
    def __init__(self, agent_id: int, config: MultiAgentConfig):
        self.agent_id = agent_id
        self.config = config
        self.device = torch.device(config.device)
        
        # Load the MIND policy for local evaluation
        self.policy = SACPolicy(
            num_features=16,
            num_heads=4,
            num_mps=6,
        ).to(self.device)
        self.policy.load_state_dict(torch.load(config.ckpt_path)['policy_state_dict'])
        self.policy.eval()
        
        # Global context integration layer (simple linear transformation)
        embedding_dim = 16 * 6 * 2  # num_features * num_mps * 2
        self.global_context_layer = torch.nn.Linear(embedding_dim * 2, embedding_dim).to(self.device)
        self.global_context_layer.eval()
    
    def evaluate(self, subgraph: ig.Graph, subgraph_nodes: List[int], 
                Z_global: torch.Tensor) -> Tuple[List[int], np.ndarray]:
        """
        Evaluate nodes in the subgraph with global awareness
        
        Args:
            subgraph: The partition subgraph
            subgraph_nodes: Original node IDs in this subgraph
            Z_global: Global context from coordinator (global omni-node embedding K*F)
            
        Returns:
            Tuple of (subgraph_nodes, probabilities)
        """
        if subgraph.vcount() == 0:
            return subgraph_nodes, np.array([])
        
        with torch.no_grad():
            batch_data = Batch(self.device, [ig_to_data(subgraph)])
            
            # Get local subgraph embeddings from MIND encoder
            local_embeddings = self.policy.graph_embedding(batch_data)  # [N_sub, 2*K*F]
            
            # Split into node embeddings and omni-node embeddings
            K_F = local_embeddings.shape[1] // 2  # K*F dimension
            local_node_embeddings = local_embeddings[:, :K_F]      # [N_sub, K*F] - first half
            local_omni_embeddings = local_embeddings[:, K_F:]      # [N_sub, K*F] - second half
            
            # Method 1: Replace local omni with weighted combination of local + global
            alpha = 0.7  # Weight for local context  
            beta = 0.3   # Weight for global context
            
            # Z_global is the global omni-node embedding [K*F]
            # Broadcast it to match the local omni embeddings shape
            global_omni_broadcast = Z_global.unsqueeze(0).expand(local_omni_embeddings.shape[0], -1)  # [N_sub, K*F]
            
            # Create enhanced omni embeddings
            enhanced_omni_embeddings = alpha * local_omni_embeddings + beta * global_omni_broadcast
            
            # Reconstruct the full embeddings with enhanced global context
            enhanced_embeddings = torch.cat([local_node_embeddings, enhanced_omni_embeddings], dim=1)  # [N_sub, 2*K*F]
            
            # Compute logits for each node using the policy's MLP
            logits = self.policy.mlp(enhanced_embeddings).flatten()  # [N_sub]
            
            # Convert to probabilities using softmax
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            
            return subgraph_nodes, probs


def get_lcc_size(graph: ig.Graph) -> int:
    """Get the size of the largest connected component"""
    if graph.vcount() == 0:
        return 0
    components = graph.connected_components()
    return max(components.sizes()) if components.sizes() else 0


def multiagent_mind_dismantling(graph: ig.Graph, config: MultiAgentConfig) -> Dict:
    """
    Multi-Agent MIND with Global Coordinator Algorithm
    
    Args:
        graph: Input graph to dismantle
        config: Configuration parameters
        
    Returns:
        Dictionary with results including removed nodes, LCC sizes, etc.
    """
    print(f"Starting Multi-Agent MIND with K={config.K} agents")
    
    # Initialize
    G = deepcopy(graph)
    n_init = G.vcount()
    G.vs["static_id"] = list(range(n_init))  # Track original node IDs
    
    # Create agents
    coordinator = GlobalCoordinator(config)
    agents = [PartitionAgent(i, config) for i in range(config.K)]
    
    # Tracking variables
    removed_nodes = []
    lcc_sizes = [get_lcc_size(G)]
    gcc_ratios = [lcc_sizes[0] / n_init]
    iteration = 0
    
    print(f"Initial graph: {G.vcount()} nodes, {G.ecount()} edges")
    print(f"Initial LCC size: {lcc_sizes[0]} ({gcc_ratios[0]:.3f})")
    
    # Main algorithm loop
    while (get_lcc_size(G) / n_init > config.target_lcc_ratio and 
           G.ecount() > 0 and 
           iteration < config.max_iterations):
        
        # Step 1: Partitioning using METIS
        try:
            # Use METIS for balanced partitioning
            clustering = partition(G, partition_method='metis', K=config.K, objective='cut')
            subgraphs_info = []
            
            for i in range(len(clustering)):
                if len(clustering[i]) > 0:
                    # Get subgraph and original node mapping
                    subgraph_nodes = clustering[i]
                    subgraph = G.subgraph(subgraph_nodes)
                    subgraphs_info.append((subgraph, subgraph_nodes))
                    
        except Exception as e:
            print(f"Warning: METIS partitioning failed ({e}), using fallback")
            # Fallback: simple random partitioning
            nodes = list(range(G.vcount()))
            np.random.shuffle(nodes)
            partition_size = len(nodes) // config.K
            subgraphs_info = []
            
            for i in range(config.K):
                start_idx = i * partition_size
                end_idx = (i + 1) * partition_size if i < config.K - 1 else len(nodes)
                subgraph_nodes = nodes[start_idx:end_idx]
                
                if len(subgraph_nodes) > 0:
                    subgraph = G.subgraph(subgraph_nodes)
                    subgraphs_info.append((subgraph, subgraph_nodes))
        
        if not subgraphs_info:
            print("No valid subgraphs found, stopping")
            break
        
        # Step 2: Global Coordinator computes Global Context
        Z_global = coordinator.compute_global_omni_node(G)
        
        # Step 3: Parallel Local Inference with Global Awareness
        candidate_nodes = []
        for i, (subgraph, subgraph_nodes) in enumerate(subgraphs_info):
            if i < len(agents):
                # Agent evaluates its partition with global context
                nodes, probs = agents[i].evaluate(subgraph, subgraph_nodes, Z_global)
                candidate_nodes.append((nodes, probs))
                
                # Debug info (only for first few iterations)
                if iteration < 3:
                    print(f"  Agent {i}: partition size={len(subgraph_nodes)}, max_prob={probs.max():.4f}")
        
        if iteration < 3:
            print(f"  Global omni-node embedding norm: {torch.norm(Z_global).item():.4f}")
        
        # Step 4: Consensus & Removal
        # Global Coordinator picks the best node across all agents
        v_target = coordinator.select_best_node(candidate_nodes, G)
        
        if v_target is None:
            print("No valid target node found, stopping")
            break
        
        # Remove the selected node
        original_id = G.vs[v_target]["static_id"]
        removed_nodes.append(original_id)
        G.delete_vertices(v_target)
        
        # Update tracking
        current_lcc = get_lcc_size(G)
        lcc_sizes.append(current_lcc)
        gcc_ratios.append(current_lcc / n_init)
        
        iteration += 1
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: LCC size = {current_lcc} ({current_lcc/n_init:.3f})")
    
    print(f"Completed in {iteration} iterations")
    print(f"Final LCC size: {lcc_sizes[-1]} ({gcc_ratios[-1]:.3f})")
    print(f"Nodes removed: {len(removed_nodes)}")
    
    return {
        'removed_nodes': removed_nodes,
        'lcc_sizes': lcc_sizes,
        'gcc_ratios': gcc_ratios,
        'iterations': iteration,
        'final_lcc_ratio': gcc_ratios[-1]
    }


def baseline_mind_dismantling(graph: ig.Graph, config: MultiAgentConfig) -> Dict:
    """
    Baseline MIND dismantling for comparison
    """
    print("Running baseline MIND dismantling...")
    
    G = deepcopy(graph)
    n_init = G.vcount()
    G.vs["static_id"] = list(range(n_init))
    
    # Load policy
    device = torch.device(config.device)
    policy = SACPolicy(
        num_features=16,
        num_heads=4,
        num_mps=6,
    ).to(device)
    policy.load_state_dict(torch.load(config.ckpt_path)['policy_state_dict'])
    policy.eval()
    
    # Calculate step size
    step_size = max(int(config.step_ratio * n_init), 1)
    
    removed_nodes = []
    lcc_sizes = [get_lcc_size(G)]
    gcc_ratios = [lcc_sizes[0] / n_init]
    
    while (get_lcc_size(G) / n_init > config.target_lcc_ratio and 
           G.ecount() > 0):
        
        with torch.no_grad():
            batch_data = Batch(device, [ig_to_data(G)])
            _, log_probs = policy.get_action(batch_data, val=True)
        
        probs = log_probs.exp().cpu().numpy()
        current_step = min(step_size, len(probs))
        top_k_indices = np.argsort(-probs)[:current_step]
        
        for idx in sorted(top_k_indices, reverse=True):
            if idx < G.vcount():
                original_id = G.vs[idx]["static_id"]
                removed_nodes.append(original_id)
                G.delete_vertices(idx)
                
                current_lcc = get_lcc_size(G)
                lcc_sizes.append(current_lcc)
                gcc_ratios.append(current_lcc / n_init)
                
                if current_lcc / n_init < config.target_lcc_ratio or G.ecount() == 0:
                    break
    
    return {
        'removed_nodes': removed_nodes,
        'lcc_sizes': lcc_sizes,
        'gcc_ratios': gcc_ratios,
        'iterations': len(removed_nodes),
        'final_lcc_ratio': gcc_ratios[-1]
    }


def plot_comparison(multiagent_results: Dict, baseline_results: Dict, 
                   graph_name: str = "Graph"):
    """Plot comparison between multi-agent and baseline approaches"""
    
    plt.figure(figsize=(12, 8))
    
    # Plot multi-agent results
    ma_x = np.arange(len(multiagent_results['gcc_ratios'])) / len(multiagent_results['removed_nodes']) if multiagent_results['removed_nodes'] else [0]
    ma_y = multiagent_results['gcc_ratios']
    plt.plot(ma_x, ma_y, 'b-', linewidth=2, marker='o', markersize=4, 
             label=f'Multi-Agent MIND (K={config.K})')
    
    # Plot baseline results  
    bl_x = np.arange(len(baseline_results['gcc_ratios'])) / len(baseline_results['removed_nodes']) if baseline_results['removed_nodes'] else [0]
    bl_y = baseline_results['gcc_ratios']
    plt.plot(bl_x, bl_y, 'r--', linewidth=2, marker='s', markersize=4, 
             label='Baseline MIND')
    
    plt.xlabel('Fraction of Dismantling Progress', fontsize=12)
    plt.ylabel('Normalized LCC Size', fontsize=12)
    plt.title(f'Multi-Agent vs Baseline MIND Comparison - {graph_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()
    
    # Print comparison statistics
    print(f"\n=== Comparison Results for {graph_name} ===")
    print(f"Multi-Agent MIND:")
    print(f"  - Nodes removed: {len(multiagent_results['removed_nodes'])}")
    print(f"  - Final LCC ratio: {multiagent_results['final_lcc_ratio']:.4f}")
    print(f"  - Iterations: {multiagent_results['iterations']}")
    
    print(f"Baseline MIND:")
    print(f"  - Nodes removed: {len(baseline_results['removed_nodes'])}")
    print(f"  - Final LCC ratio: {baseline_results['final_lcc_ratio']:.4f}")
    print(f"  - Iterations: {baseline_results['iterations']}")
    
    # Efficiency comparison
    ma_efficiency = multiagent_results['final_lcc_ratio'] / len(multiagent_results['removed_nodes']) if multiagent_results['removed_nodes'] else float('inf')
    bl_efficiency = baseline_results['final_lcc_ratio'] / len(baseline_results['removed_nodes']) if baseline_results['removed_nodes'] else float('inf')
    
    print(f"\nEfficiency (final_lcc_ratio / nodes_removed):")
    print(f"  - Multi-Agent: {ma_efficiency:.6f}")
    print(f"  - Baseline: {bl_efficiency:.6f}")
    print(f"  - Improvement: {((bl_efficiency - ma_efficiency) / bl_efficiency * 100):.2f}%")


if __name__ == "__main__":
    # Configuration
    config = MultiAgentConfig(
        K=4,  # Number of partition agents
        target_lcc_ratio=0.1,
        step_ratio=0.0025,
        ckpt_path='saved/mind.ckpt',
        device='cuda:0',
        max_iterations=1000
    )
    
    # Test graphs
    test_graphs = [
        "graphs/real/FINDER/Crime.pkl"
    ]
    
    for graph_path in test_graphs:
        try:
            print(f"\n{'='*60}")
            print(f"Testing on: {graph_path}")
            print(f"{'='*60}")
            
            # Load graph
            with open(graph_path, 'rb') as f:
                g = pickle.load(f)
            
            print(f"Graph loaded: {g.vcount()} nodes, {g.ecount()} edges")
            
            # Run multi-agent MIND
            print("\n--- Running Multi-Agent MIND ---")
            start_time = time.time()
            multiagent_results = multiagent_mind_dismantling(g, config)
            ma_time = time.time() - start_time
            print(f"Multi-Agent MIND completed in {ma_time:.2f} seconds")
            
            # Run baseline MIND
            print("\n--- Running Baseline MIND ---")
            start_time = time.time()
            baseline_results = baseline_mind_dismantling(g, config)
            bl_time = time.time() - start_time
            print(f"Baseline MIND completed in {bl_time:.2f} seconds")
            
            # Plot and compare results
            graph_name = graph_path.split('/')[-1].replace('.pkl', '')
            plot_comparison(multiagent_results, baseline_results, graph_name)
            
            print(f"\nTiming Comparison:")
            print(f"  - Multi-Agent MIND: {ma_time:.2f}s")
            print(f"  - Baseline MIND: {bl_time:.2f}s")
            print(f"  - Time ratio: {ma_time/bl_time:.2f}x")
            
        except Exception as e:
            print(f"Error processing {graph_path}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Multi-Agent MIND evaluation completed!")
    print(f"{'='*60}")