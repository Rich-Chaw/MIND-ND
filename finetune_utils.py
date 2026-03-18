import numpy as np
import torch
import igraph as ig
import gc
from torch_scatter import scatter_add
from utils import Batch, ig_to_data


def teacher_wrapper(graph, teacher_method='spectral', max_steps=None):
    from baseline import (
        spectral_dismantling,
        adaptive_betweenness,
        adaptive_ci,
        adaptive_greedy_lcc,
        adaptive_degree,
        adaptive_k_shell,
        random_dismantling,
    )
    # Hybrid: randomly pick one of several simple heuristics
    if teacher_method == 'Hybrid':
        import random
        # Degree, betweenness, k-shell (coreness)
        choices = ['degree', 'betweenness', 'k-shell']
        teacher_method = random.choice(choices)

    if teacher_method == 'spectral':
        try:
            removals = spectral_dismantling(graph, max_steps=max_steps)
        except Exception:
            removals = adaptive_betweenness(graph, max_steps=max_steps)
    elif teacher_method == 'betweenness':
        removals = adaptive_betweenness(graph, max_steps=max_steps)
    elif teacher_method == 'degree':
        removals = adaptive_degree(graph, max_steps=max_steps)
    elif teacher_method == 'k-shell':
        removals = adaptive_k_shell(graph, max_steps=max_steps)
    elif teacher_method == 'CI':
        removals = adaptive_ci(graph, max_steps=max_steps)
    elif teacher_method == 'greedy_lcc':
        removals = adaptive_greedy_lcc(graph, max_steps=max_steps)
    else:
        print(f"Unknown teacher method: {teacher_method}, using random")
        removals = random_dismantling(graph, max_steps=max_steps)
    return removals


def batch_to_igraphs(batch):
    """Convert a Batch object back to list of igraph objects"""
    graphs = []
    
    for i in range(batch.batch_size):
        # Get nodes and edges for this graph
        graph_mask = (batch.batch == i)
        # Remove the omni-node (last node in each graph)
        graph_nodes = graph_mask.sum().item() - 1

        # Get edges for this graph (excluding omni-node connections)
        graph_edges = []
        edge_mask = graph_mask[batch.edge_index[0]] & graph_mask[batch.edge_index[1]]
        
        if edge_mask.any():
            graph_edge_indices = batch.edge_index[:, edge_mask]
            
            # Convert global indices to local indices for this graph
            node_offset = (batch.batch == i).nonzero()[0].item()
            local_edges = graph_edge_indices - node_offset
            
            # Filter out omni-node connections (edges involving the last node)
            valid_edge_mask = (local_edges[0] < graph_nodes) & (local_edges[1] < graph_nodes)
            if valid_edge_mask.any():
                valid_edges = local_edges[:, valid_edge_mask]
                graph_edges = valid_edges.t().cpu().numpy().tolist()
        
        # Create igraph
        if graph_nodes > 0:
            g = ig.Graph(n=graph_nodes, edges=graph_edges, directed=False)
        else:
            g = ig.Graph(n=1, edges=[], directed=False)  # Minimum graph
        
        graphs.append(g)
    
    return graphs


def to_soft_probs(graph, current_step ,removals, temperature=1.0, epsilon=0.1):
    """
    Convert teacher action to epsilon-greedy soft probability distribution
    """

    n_nodes = graph.vcount()
    probs = np.full(n_nodes, np.full(n_nodes, epsilon / (n_nodes - 1) if n_nodes > 1 else 0.0))  # Small background probability
    
    # Get next few nodes in sequence
    remaining_nodes = removals[current_step+1:]
    if len(remaining_nodes) > 0:
        # Primary choice (next node)
        next_node = remaining_nodes[0]
        probs[next_node] = 0.7
        
        # Secondary choices (next 2-3 nodes)
        for i, node in enumerate(remaining_nodes[1:4]):  # Next 3 nodes
            if i < len(remaining_nodes) - 1:
                probs[node] = 0.3 / (i + 1)  # Decreasing probability
    
    # Apply temperature scaling
    # probs = np.exp(probs/temperature)
    probs = probs / probs.sum()
    return probs


    # n_nodes = graph.vcount()
    
    # if n_nodes == 0:
    #     # If no nodes, return empty array
    #     return np.array([])
    
    # # Epsilon-greedy distribution
    # optimal_action = removals[current_step]  # a* - the chosen action to remove
    
    # # Initialize with epsilon probability for non-optimal actions
    # probs = np.full(n_nodes, epsilon / (n_nodes - 1) if n_nodes > 1 else 0.0)
    
    # # Set optimal action probability
    # if optimal_action < n_nodes:  # Ensure action is valid
    #     probs[optimal_action] = 1.0 - epsilon
    
    # return probs


def compute_teacher_actions_on_demand(batch, teacher_method='spectral', temperature=2.0, device='cuda'):
    """
    Returns teacher log probabilities in the same format as student policy
    """
    from baseline import spectral_dismantling, adaptive_betweenness
    
    # Convert batch to igraphs
    batch_graphs = batch_to_igraphs(batch)
    
    # Initialize teacher logprobs for all non-omni nodes
    num_non_omni_nodes = batch.non_omni_mask.sum().item()
    teacher_logprobs = torch.full((num_non_omni_nodes,), -float('inf'), dtype=torch.float64, device=device)
    
    for i, graph in enumerate(batch_graphs):
        try:
            if graph.vcount() <= 2 or graph.ecount() == 0:
                continue
            
            removals = teacher_wrapper(graph, teacher_method, max_steps=5)
            # Convert to soft probabilities using epsilon-greedy
            soft_probs = to_soft_probs(graph,0, removals, temperature, epsilon=0.1)
            
            # Get indices for this graph's non-omni nodes - use explicit indexing
            node_mask = (batch.batch_non_omni == i)
            teacher_probs_tensor = torch.tensor(soft_probs, device=device)
            teacher_logprobs[node_mask] = torch.log(teacher_probs_tensor + 1e-8)

        except Exception as e:
            continue
    
    import gc
    del batch_graphs
    gc.collect()
    
    return teacher_logprobs

def teacher_step(obs_list, teacher_method='spectral'):
    """
    Compute teacher actions for a list of graph observations
    """
    act_arr = np.zeros(len(obs_list), dtype=np.int64)
    obs_next_list = []
    rew_arr = np.zeros(len(obs_list), dtype=np.float32)
    done_arr = np.zeros(len(obs_list), dtype=bool)
    
    for i, graph in enumerate(obs_list):
        if graph.vcount() <= 2 or graph.ecount() == 0:
            # Handle empty or very small graphs
            act_arr[i] = 0 if graph.vcount() > 0 else 0
            obs_next = graph.copy()
            if graph.vcount() > 0:
                obs_next.delete_vertices(0)
        else:
            # Generate teacher action using specified method
            removals = teacher_wrapper(graph, teacher_method, max_steps=1)
            teacher_action = removals[0]
            act_arr[i] = teacher_action
            obs_next = graph.copy()
            obs_next.delete_vertices(teacher_action)
            
            # Explicit cleanup of removals list
            del removals

        # Compute reward and done flag (following env.step logic)
        if obs_next.vcount() == 0:
            lcc_size = 0.0
            done_flag = True
        else:
            components = obs_next.connected_components()
            lcc_size = max(components.sizes()) if len(components.sizes()) > 0 else 0
            lcc_ratio = lcc_size / max(graph['n_init'], 1)
            done_flag = (lcc_ratio < 0.1 or obs_next.ecount() == 0)
        
            # Explicit cleanup of components
            del components
        
        rew_arr[i] = -lcc_size / max(graph['n_init'], 1)  # Negative LCC ratio as reward
        done_arr[i] = done_flag
        obs_next_list.append(obs_next)
        
        # Periodic garbage collection for large batches
        if i % 16 == 15:  # Every 16 graphs
            gc.collect()
            
    return act_arr, obs_next_list, rew_arr, done_arr


def compute_reward_shaping(obs_list, act_arr, shaping_method='betweenness', policy=None, discriminator=None, teacher_method='spectral', temperature=2.0, device='cuda'):
    """
    Unified reward shaping function supporting multiple methods
    
    Args:
        obs_list: List of graph observations
        act_arr: Array of selected actions
        shaping_method: 'betweenness', 'KL', or 'core'
        policy: Policy network (required for KL method)
        teacher_method: Teacher method for KL divergence ('spectral' or 'betweenness')
        temperature: Temperature for teacher probabilities
        device: Device for computations
    
    Returns:
        shaping_rewards: Array of shaping rewards
    """
    shaping_rewards = np.zeros(len(obs_list), dtype=np.float32)

    def _core2_size(g):
        """Size of 2-core: number of vertices with coreness >= 2."""
        if g.vcount() == 0 or g.ecount() == 0:
            return 0
        coreness = g.coreness()
        return sum(1 for k in coreness if k >= 2)

    if shaping_method == 'core':
        # Shaping reward = (Core_2(s) - Core_2(s')) / Core_2(s_0)
        for i, graph in enumerate(obs_list):
            if graph.vcount() <= 0 or graph.ecount() == 0:
                shaping_rewards[i] = 0.0
                continue
            core2_s0 = graph.get('core2_init', graph['n_init'])
            if core2_s0 <= 0:
                shaping_rewards[i] = 0.0
                continue
            core2_s = _core2_size(graph)
            # s' = graph after removing action node
            g_next = graph.copy()
            action_idx = int(act_arr[i])
            if action_idx >= g_next.vcount():
                shaping_rewards[i] = 0.0
                continue
            g_next.delete_vertices(action_idx)
            core2_s_next = _core2_size(g_next)
            shaping_rewards[i] = (core2_s - core2_s_next) / float(core2_s0)
            del g_next

    elif shaping_method == 'betweenness':
        # Betweenness centrality based reward shaping
        for i, graph in enumerate(obs_list):
            if graph.vcount() <= 2 or graph.ecount() == 0:
                shaping_rewards[i] = 0.0
                continue
                
            # Compute betweenness centrality
            betweenness = graph.betweenness()
            # Normalize betweenness scores
            max_bc = max(betweenness) if max(betweenness) > 0 else 1.0
            bc_normalized = [bc / max_bc for bc in betweenness]
            
            # Get shaping reward for the selected action
            action_idx = act_arr[i]
            shaping_rewards[i] = bc_normalized[action_idx]
    
    elif shaping_method == 'KL':
        # KL divergence based reward shaping
        if policy is None:
            print("Error: Policy required for KL-based reward shaping")
            return shaping_rewards
            
        try:
            # Convert obs_list to batch format for policy evaluation
            batch = Batch(device, [ig_to_data(g) for g in obs_list])
            
            # Get student policy probabilities
            with torch.no_grad():
                _, student_logp = policy.get_action(batch)
                student_probs = student_logp.exp()
            
            # Get teacher probabilities
            teacher_logp = compute_teacher_actions_on_demand(batch, teacher_method, temperature, device)
            teacher_probs = teacher_logp.exp()
            
            # Compute KL divergence per graph
            kl_per_node = teacher_probs * (teacher_logp - student_logp)
            b = batch.batch[batch.non_omni_mask]
            kl_per_graph = scatter_add(kl_per_node, b, dim_size=batch.batch_size)
            
            # Convert to numpy and use negative KL as reward (alignment = positive reward)
            kl_rewards = -kl_per_graph.cpu().numpy()
            
            # Normalize by graph size (n_init) to match main reward structure
            for i, graph in enumerate(obs_list):
                kl_rewards[i] = kl_rewards[i] / graph['n_init']
            
            # Apply mild clipping to prevent extreme values
            # kl_rewards = np.clip(kl_rewards, -1.0, 1.0)
            shaping_rewards = kl_rewards.astype(np.float32)
        
        except Exception as e:
            print(f"Error in KL-based reward shaping: {e}")
            shaping_rewards = np.zeros(len(obs_list), dtype=np.float32)
    
    elif shaping_method == 'POfD':
        g = Batch(device, [ig_to_data(g) for g in obs_list])
        e = policy.graph_embedding(g) #[N,2KF]
        batch_x = e[g.act_offsets + act_arr] 
        shaping_rewards = discriminator(e)

    else:
        print(f"Unknown shaping method: {shaping_method}. Using zero rewards.")
    
    return shaping_rewards