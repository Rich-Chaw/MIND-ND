import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import random
from typing import List, Tuple

from networks.dismantle import load_sac_dismantler
from networks.gnn import GNN_ENCODER
from env import DismantleEnv
from utils.graph_data import Batch, ig_to_data
from utils import graph_models

class TrajectoryBuffer:
    """
    Buffer for storing transitions (s, a, r, s') from a single trajectory.
    Each buffer represents one complete trajectory/episode.
    """
    def __init__(self):
        self.obs_list = []  # List of igraph graphs (states)
        self.act_list = []  # List of actions
        self.rew_list = []  # List of rewards
        self.obs_next_list = []  # List of igraph graphs (next states)
        self.done_list = []  # List of done flags
    
    def add(self, obs, act, rew, obs_next, done):
        """Add a transition (s, a, r, s', done) to the buffer"""
        self.obs_list.append(obs)
        self.act_list.append(act)
        self.rew_list.append(rew)
        self.obs_next_list.append(obs_next)
        self.done_list.append(done)
    
    def size(self):
        """Return the number of transitions in this trajectory"""
        return len(self.obs_list)
    
    def sample_sequence(self, seq_len, device):
        """
        Sample a continuous sequence of length seq_len from this trajectory.
        Returns: (obs_batch, act_tensor, rew_tensor, obs_next_batch)
        """
        if self.size() < seq_len:
            return None
        
        # Sample a random starting index
        start_idx = random.randint(0, self.size() - seq_len)
        
        # Extract sequence
        obs_seq = self.obs_list[start_idx:start_idx + seq_len]
        act_seq = self.act_list[start_idx:start_idx + seq_len]
        rew_seq = self.rew_list[start_idx:start_idx + seq_len]
        obs_next_seq = self.obs_next_list[start_idx:start_idx + seq_len]
        
        # Convert to tensors
        obs_batch = Batch(device, [ig_to_data(g) for g in obs_seq])
        obs_next_batch = Batch(device, [ig_to_data(g) for g in obs_next_seq])
        act_tensor = torch.tensor(act_seq, device=device, dtype=torch.long)
        rew_tensor = torch.tensor(rew_seq, device=device, dtype=torch.float32)
        
        return obs_batch, act_tensor, rew_tensor, obs_next_batch


class TaskEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=16):
        super(TaskEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # RNN encoder
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # VAE encoder: maps RNN output to latent space (mu, logvar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def encode(self, x):
        """
        x: [batch, seq_len, input_dim]
        Returns: mu, logvar both [batch, seq_len, latent_dim] - sequence-length embeddings
        """
        # Get RNN output (all hidden states)
        rnn_out, _ = self.rnn(x)  # [batch, seq_len, hidden_dim]
        # Use all hidden states for sequence-length embeddings
        h = rnn_out  # [batch, seq_len, hidden_dim]
        
        mu = self.fc_mu(h)  # [batch, seq_len, latent_dim]
        logvar = self.fc_logvar(h)  # [batch, seq_len, latent_dim]
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        x: [batch, seq_len, input_dim]
        Returns: z [batch, seq_len, latent_dim], mu [batch, seq_len, latent_dim], logvar [batch, seq_len, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def get_z(self, x):
        """
        Get deterministic latent representation (mean)
        x: [batch, seq_len, input_dim]
        Returns: mu [batch, seq_len, latent_dim] - sequence-length embeddings
        """
        mu, _ = self.encode(x)
        return mu


class RewardDecoder(nn.Module):
    """Decoder to predict reward from (state, action, task_embedding)"""
    def __init__(self, state_dim, latent_dim, hidden_dim=64):
        super(RewardDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + 1 + latent_dim, hidden_dim),  # state + action + z
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state_embedding, action, z):
        """
        state_embedding: [seq_len, state_dim] or [batch, seq_len, state_dim]
        action: [seq_len] or [batch, seq_len] or [seq_len, 1] or [batch, seq_len, 1]
        z: [seq_len, latent_dim] or [batch, seq_len, latent_dim] - per-timestep task embeddings
        Returns: predicted_reward [seq_len] or [batch, seq_len]
        """
        # Handle different input shapes
        if state_embedding.dim() == 2:
            # [seq_len, state_dim]
            seq_len = state_embedding.shape[0]
            if action.dim() == 1:
                action = action.unsqueeze(1)  # [seq_len, 1]
            elif action.dim() == 2 and action.shape[1] == 1:
                pass  # Already [seq_len, 1]
            x = torch.cat([state_embedding, action.float(), z], dim=1)  # [seq_len, state_dim + 1 + latent_dim]
            return self.mlp(x).squeeze(1)  # [seq_len]
        else:
            # [batch, seq_len, state_dim]
            batch_size, seq_len = state_embedding.shape[:2]
            if action.dim() == 2:
                action = action.unsqueeze(2)  # [batch, seq_len, 1]
            x = torch.cat([state_embedding, action.float(), z], dim=2)  # [batch, seq_len, state_dim + 1 + latent_dim]
            return self.mlp(x).squeeze(2)  # [batch, seq_len]


class StateDecoder(nn.Module):
    """Decoder to predict next state from (action, task_embedding)"""
    def __init__(self, state_dim, latent_dim, hidden_dim=128):
        super(StateDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 + latent_dim, hidden_dim),  # action + z
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # Predict next state
        )
    
    def forward(self, action, z):
        """
        action: [seq_len] or [batch, seq_len] or [seq_len, 1] or [batch, seq_len, 1]
        z: [seq_len, latent_dim] or [batch, seq_len, latent_dim] - per-timestep task embeddings
        Returns: predicted_next_state [seq_len, state_dim] or [batch, seq_len, state_dim]
        """
        # Handle different input shapes
        if z.dim() == 2:
            # [seq_len, latent_dim]
            seq_len = z.shape[0]
            if action.dim() == 1:
                action = action.unsqueeze(1)  # [seq_len, 1]
            elif action.dim() == 2 and action.shape[1] == 1:
                pass  # Already [seq_len, 1]
            x = torch.cat([action.float(), z], dim=1)  # [seq_len, 1 + latent_dim]
            return self.mlp(x)  # [seq_len, state_dim]
        else:
            # [batch, seq_len, latent_dim]
            batch_size, seq_len = z.shape[:2]
            if action.dim() == 2:
                action = action.unsqueeze(2)  # [batch, seq_len, 1]
            x = torch.cat([action.float(), z], dim=2)  # [batch, seq_len, 1 + latent_dim]
            return self.mlp(x)  # [batch, seq_len, state_dim]


def extract_embeddings_from_trajectories(gnn_encoder, task_encoder, test_trajectories_pl, 
                                         test_trajectories_sbm, sequence_length, device):
    """
    Extract task embeddings from test trajectories.
    
    Args:
        gnn_encoder: GNN encoder model
        task_encoder: Task encoder model
        test_trajectories_pl: List of PL/BA test trajectories
        test_trajectories_sbm: List of SBM test trajectories
        sequence_length: Sequence length for sampling
        device: Device to run on
        
    Returns:
        z_all: numpy array of embeddings [num_samples, latent_dim]
        labels_all: numpy array of labels [num_samples] (0 for PL/BA, 1 for SBM)
    """
    gnn_encoder.eval()
    task_encoder.eval()
    
    z_list = []
    labels_list = []
    
    with torch.no_grad():
        # Evaluate on PL/BA trajectories
        for traj in test_trajectories_pl:
            if traj.size() < sequence_length:
                continue
            seq_data = traj.sample_sequence(sequence_length, device)
            if seq_data is None:
                continue
            
            obs_seq, act_seq, rew_seq, obs_next_seq = seq_data
            
            # Encode states
            embeddings = gnn_encoder(obs_seq)
            embed_dim = embeddings.shape[1] // 2
            graph_emb = embeddings[:, embed_dim:]
            state_seq = scatter_mean(graph_emb, obs_seq.batch_non_omni, dim=0, dim_size=obs_seq.batch_size)  # [seq_len, KF]
            
            # Encode next states
            embeddings_next = gnn_encoder(obs_next_seq)
            graph_emb_next = embeddings_next[:, embed_dim:]
            state_next_seq = scatter_mean(graph_emb_next, obs_next_seq.batch_non_omni, dim=0, dim_size=obs_next_seq.batch_size)  # [seq_len, KF]
            
            # Create input sequence
            act_expanded = act_seq.unsqueeze(1).float()
            rew_expanded = rew_seq.unsqueeze(1)
            seq = torch.cat([state_seq, act_expanded, rew_expanded, state_next_seq], dim=1)
            seq = seq.unsqueeze(0)
            
            z = task_encoder.get_z(seq)  # [1, seq_len, latent_dim]
            # Use last timestep for evaluation
            z_last = z[:, -1, :]  # [1, latent_dim]
            z_list.append(z_last.cpu())
            labels_list.append(0)
        
        # Evaluate on SBM trajectories
        for traj in test_trajectories_sbm:
            if traj.size() < sequence_length:
                continue
            seq_data = traj.sample_sequence(sequence_length, device)
            if seq_data is None:
                continue
            
            obs_seq, act_seq, rew_seq, obs_next_seq = seq_data
            
            # Encode states
            embeddings = gnn_encoder(obs_seq)
            embed_dim = embeddings.shape[1] // 2
            graph_emb = embeddings[:, embed_dim:]
            state_seq = scatter_mean(graph_emb, obs_seq.batch_non_omni, dim=0, dim_size=obs_seq.batch_size)  # [seq_len, KF]
            
            # Encode next states
            embeddings_next = gnn_encoder(obs_next_seq)
            graph_emb_next = embeddings_next[:, embed_dim:]
            state_next_seq = scatter_mean(graph_emb_next, obs_next_seq.batch_non_omni, dim=0, dim_size=obs_next_seq.batch_size)  # [seq_len, KF]
            
            # Create input sequence
            act_expanded = act_seq.unsqueeze(1).float()
            rew_expanded = rew_seq.unsqueeze(1)
            seq = torch.cat([state_seq, act_expanded, rew_expanded, state_next_seq], dim=1)
            seq = seq.unsqueeze(0)
            
            z = task_encoder.get_z(seq)  # [1, seq_len, latent_dim]
            # Use last timestep for evaluation
            z_last = z[:, -1, :]  # [1, latent_dim]
            z_list.append(z_last.cpu())
            labels_list.append(1)
    
    if len(z_list) > 0:
        z_all = torch.cat(z_list, dim=0).numpy()
        labels_all = np.array(labels_list)
        return z_all, labels_all
    else:
        return None, None


def compute_silhouette_score(z_all, labels_all):
    """
    Compute silhouette score from embeddings and labels.
    
    Args:
        z_all: numpy array of embeddings [num_samples, latent_dim]
        labels_all: numpy array of labels [num_samples]
        
    Returns:
        score: Silhouette score (float) or None if cannot compute
    """
    if len(np.unique(labels_all)) > 1:
        score = silhouette_score(z_all, labels_all)
        return score
    else:
        return None


def visualize_tsne(z_all, labels_all, save_path):
    """
    Create and save t-SNE visualization.
    
    Args:
        z_all: numpy array of embeddings [num_samples, latent_dim]
        labels_all: numpy array of labels [num_samples]
        save_path: Path to save the visualization
    """
    print("Computing t-SNE visualization...")
    z_embedded = TSNE(n_components=2, random_state=42).fit_transform(z_all)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=labels_all, 
                        cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Graph Type (0: PL/BA, 1: SBM)')
    plt.title("Task Embedding Space (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved t-SNE visualization to {save_path}")
    plt.close()


def info_nce_loss(z_anchor, z_positive, z_negatives, temperature=0.1):
    """
    InfoNCE contrastive loss with multiple negative samples
    z_anchor: [batch, latent_dim]
    z_positive: [batch, latent_dim] - same graph type, different time
    z_negatives: [batch, num_negatives, latent_dim] - different graph type, multiple negatives
    """
    batch_size = z_anchor.shape[0]
    num_negatives = z_negatives.shape[1]
    
    # Normalize embeddings
    z_anchor = F.normalize(z_anchor, dim=1)  # [batch, latent_dim]
    z_positive = F.normalize(z_positive, dim=1)  # [batch, latent_dim]
    z_negatives = F.normalize(z_negatives, dim=2)  # [batch, num_negatives, latent_dim]
    
    # Positive similarity: [batch]
    pos_sim = torch.sum(z_anchor * z_positive, dim=1) / temperature  # [batch]
    
    # Negative similarities: [batch, num_negatives]
    # z_anchor: [batch, latent_dim] -> [batch, 1, latent_dim]
    # z_negatives: [batch, num_negatives, latent_dim]
    z_anchor_expanded = z_anchor.unsqueeze(1)  # [batch, 1, latent_dim]
    neg_sims = torch.sum(z_anchor_expanded * z_negatives, dim=2) / temperature  # [batch, num_negatives]
    
    # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    # For each anchor, compute: -log(exp(pos) / (exp(pos) + sum_i exp(neg_i)))
    pos_exp = torch.exp(pos_sim)  # [batch]
    neg_exp_sum = torch.sum(torch.exp(neg_sims), dim=1)  # [batch] - sum over negatives
    loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8))  # [batch]
    
    return loss.mean()


def collect_expert_trajectories(policy, graphs, device, num_episodes=1000, max_steps=200):
    """
    Collect full trajectories from expert policy and store them in TrajectoryBuffer objects.
    
    Args:
        policy: Expert policy to collect trajectories from
        graphs: List of graphs to run the policy on
        device: Device to run on
        num_episodes: Number of episodes (max number of graphs to use)
        max_steps: Maximum steps per episode
    
    Returns: List of TrajectoryBuffer objects, each containing one complete trajectory
    """
    policy.eval()
    trajectory_buffers = []  # List of TrajectoryBuffer 
    
    # note that the env gives 1 graph at a time, so we collect 1 trajectory at a time
    env = DismantleEnv(graph_data=graphs, batch_size=1, is_val=False, seed=42)
    
    for episode in range(min(num_episodes, len(graphs))):
        obs_list, _ = env.reset()
        if len(obs_list) == 0:
            continue
            
        obs = obs_list[0]
        trajectory_buffer = TrajectoryBuffer()
        
        for step in range(max_steps):
            # Get action from policy
            with torch.no_grad():
                batch = Batch(device, [ig_to_data(obs)])
                act, _ = policy.get_action(batch, val=False)
                act = act.item()
            
            # Step environment
            obs_next_list, rew_arr, done_arr, info_list = env.step(np.array([act]))
            reward = rew_arr[0]
            done = done_arr[0]
            obs_next = obs_next_list[0]
            
            # Store transition (s, a, r, s', done) in trajectory buffer
            trajectory_buffer.add(obs, act, reward, obs_next, done)
            
            if done:
                break
                
            obs = obs_next
        
        # Only add trajectory if it has at least one transition
        if trajectory_buffer.size() > 0:
            trajectory_buffers.append(trajectory_buffer)
    
    return trajectory_buffers


# Task Encoder: RNN/GRU + VAE Head
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_features', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_mps', type=int, default=6)
    parser.add_argument('--gnn', type=str, default='mind')
    parser.add_argument('--ckpt_pl', type=str, default='saved/mind/mind.ckpt')
    parser.add_argument('--ckpt_sbm', type=str, default='saved/mind/mind_SBM_16799.ckpt')
    parser.add_argument('--num_train_graphs', type=int, default=1000)
    parser.add_argument('--num_test_graphs', type=int, default=200)
    parser.add_argument('--sequence_length', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for task embedding')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_negatives', type=int, default=5, help='Number of negative samples for InfoNCE loss')
    parser.add_argument('--save_dir', type=str, default='saved/task_encoder')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Calculate input dimension: state_emb_dim + action(1) + reward(1) + next_state_emb_dim
    state_emb_dim = args.num_features * args.num_mps  # KF from graph embedding
    input_dim = state_emb_dim + 1 + 1 + state_emb_dim  # state + action + reward + next_state
    

    print("======== Task Encoder Training ========")
    print(f"Device: {device}")
    print(f"Input dimension: {input_dim} (state_emb: {state_emb_dim} + action: 1 + reward: 1 + next_state_emb: {state_emb_dim})")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Latent dimension: {args.latent_dim}")
    
    ### Step 1: Generate training and test graphs ###
    print("\n[1/4] Generating graphs...")
    graphs_pl = []
    graphs_sbm = []
    
    # Generate PL/BA graphs for training
    i = 0
    while i < args.num_train_graphs+args.num_test_graphs:
        N = random.randint(100, 200)
        m = random.choice([2, 3, 4, 5, 6, 7])
        gamma = 2.5 + random.random()
        g = graph_models.LPA(N, m, gamma)
        if g.is_connected():
            g['name'] = f'LPA_{N}_{m}_{gamma}_{i}'
            graphs_pl.append(g)
            i += 1
     
    # Generate SBM graphs for training
    i = 0
    while i < args.num_train_graphs+args.num_test_graphs:
        N = random.randint(100, 200)
        p_in = random.uniform(0.1, 0.2)
        p_out = random.uniform(0.001, 0.01)
        g = graph_models.SBM(N, p_in, p_out, num_blocks=random.randint(2, 4))
        if g.is_connected():
            g['name'] = f"SBM_{N}_{p_in}_{p_out}_{i}"
            graphs_sbm.append(g)
            i += 1 

    train_graphs_pl = graphs_pl[:args.num_train_graphs]
    train_graphs_sbm = graphs_sbm[:args.num_train_graphs]
    test_graphs_pl = graphs_pl[args.num_train_graphs:]
    test_graphs_sbm = graphs_sbm[args.num_train_graphs:]

    ### Step 2: Load expert policies ###
    print("\n[2/4] Loading expert policies...")
    policy_pl, _, _, _, _ = load_sac_dismantler(
        args.num_features, args.num_heads, args.num_mps, 
        args.gnn, device, ckpt_pth=args.ckpt_pl
    )
    policy_sbm, _, _, _, _ = load_sac_dismantler(
        args.num_features, args.num_heads, args.num_mps, 
        args.gnn, device, ckpt_pth=args.ckpt_sbm
    )
    print("Expert policies loaded")
    
    ### Step 3: Collect expert trajectories ###
    print("\n[3/4] Collecting expert trajectories...")
    train_trajectories_pl = collect_expert_trajectories(
        policy_pl, train_graphs_pl, device, 
        num_episodes=len(train_graphs_pl), max_steps=50
    )
    print(f"Collected {len(train_trajectories_pl)} PL/BA trajectories")
    
    train_trajectories_sbm = collect_expert_trajectories(
        policy_sbm, train_graphs_sbm, device,
        num_episodes=len(train_graphs_sbm), max_steps=50
    )
    print(f"Collected {len(train_trajectories_sbm)} SBM trajectories")
    
    # Filter trajectories that are long enough
    train_trajectories_pl = [t for t in train_trajectories_pl if t.size() >= args.sequence_length]
    train_trajectories_sbm = [t for t in train_trajectories_sbm if t.size() >= args.sequence_length]
    print(f"After filtering: {len(train_trajectories_pl)} PL/BA, {len(train_trajectories_sbm)} SBM trajectories")
    
    # Collect test trajectories
    test_trajectories_pl = collect_expert_trajectories(
        policy_pl, test_graphs_pl, device,
        num_episodes=len(test_graphs_pl), max_steps=200
    )
    test_trajectories_sbm = collect_expert_trajectories(
        policy_sbm, test_graphs_sbm, device,
        num_episodes=len(test_graphs_sbm), max_steps=200
    )
    test_trajectories_pl = [t for t in test_trajectories_pl if t.size() >= args.sequence_length]
    test_trajectories_sbm = [t for t in test_trajectories_sbm if t.size() >= args.sequence_length]
    print(f"Test trajectories: {len(test_trajectories_pl)} PL/BA, {len(test_trajectories_sbm)} SBM")
    
    ### Step 4: Train task encoder with GNN ###
    print("\n[4/4] Training task encoder...")
    
    # Initialize GNN encoder (trainable)
    gnn_encoder = GNN_ENCODER['rfgnn'](args.num_features, args.num_heads, args.num_mps).to(device)
    task_encoder = TaskEncoder(input_dim, args.hidden_dim, args.latent_dim).to(device)
    reward_decoder = RewardDecoder(state_emb_dim, args.latent_dim).to(device)
    state_decoder = StateDecoder(state_emb_dim, args.latent_dim).to(device)
    
    optimizer = torch.optim.Adam(
        list(gnn_encoder.parameters()) + 
        list(task_encoder.parameters()) + 
        list(reward_decoder.parameters()) +
        list(state_decoder.parameters()),
        lr=args.lr
    )
    
    # Training loop
    best_silhouette_score = -1.0  # Track best silhouette score for checkpoint saving
    best_epoch = 0
    
    for epoch in range(args.num_epochs):
        gnn_encoder.train()
        task_encoder.train()
        reward_decoder.train()
        state_decoder.train()
        
        total_loss = 0.0
        total_contra_loss = 0.0
        total_rew_recon_loss = 0.0
        total_state_recon_loss = 0.0
        total_vae_loss = 0.0
        num_batches = 0
        
        for batch_idx in range(args.batch_size):
            # Sample anchor and positive from same trajectory (same graph type)
            # Randomly choose PL or SBM
            if random.random() < 0.5:
                # Use PL trajectories
                if len(train_trajectories_pl) < 1:
                    continue
                traj = random.choice(train_trajectories_pl)
                # Negatives from SBM
                if len(train_trajectories_sbm) < args.num_negatives:
                    continue
                neg_trajectories = random.sample(train_trajectories_sbm, args.num_negatives)
            else:
                # Use SBM trajectories
                if len(train_trajectories_sbm) < 1:
                    continue
                traj = random.choice(train_trajectories_sbm)
                # Negatives from PL
                if len(train_trajectories_pl) < args.num_negatives:
                    continue
                neg_trajectories = random.sample(train_trajectories_pl, args.num_negatives)
            
            # Sample sequences: anchor and positive from same trajectory (different starting points)
            anchor_data = traj.sample_sequence(args.sequence_length, device)
            pos_data = traj.sample_sequence(args.sequence_length, device)
            
            if anchor_data is None or pos_data is None:
                continue

            # Sample multiple negative sequences
            neg_data_list = []
            for neg_traj in neg_trajectories:
                neg_data = neg_traj.sample_sequence(args.sequence_length, device)
                if neg_data is None:
                    continue
                neg_data_list.append(neg_data)
            
            if len(neg_data_list) < args.num_negatives:
                continue
            
            obs_anchor, act_anchor, rew_anchor, obs_next_anchor = anchor_data
            obs_pos, act_pos, rew_pos, obs_next_pos = pos_data
            
            # Encode states using GNN
            embeddings_anchor = gnn_encoder(obs_anchor)  # [N_anchor, 2KF]
            embeddings_pos = gnn_encoder(obs_pos)  # [N_pos, 2KF]
            
            # Get graph-level embeddings
            embed_dim = embeddings_anchor.shape[1] // 2
            graph_emb_anchor = embeddings_anchor[:, embed_dim:]  # [N_anchor, KF]
            graph_emb_pos = embeddings_pos[:, embed_dim:]  # [N_pos, KF]
            
            # Aggregate to graph-level (for each timestep in sequence)
            # Since each Batch contains seq_len graphs, we aggregate per graph [seq_len, KF]
            state_anchor = scatter_mean(graph_emb_anchor, obs_anchor.batch_non_omni, dim=0, dim_size=obs_anchor.batch_size)  # [seq_len, KF]
            state_pos = scatter_mean(graph_emb_pos, obs_pos.batch_non_omni, dim=0, dim_size=obs_pos.batch_size)  # [seq_len, KF]
            
            # Encode next states
            embeddings_next_anchor = gnn_encoder(obs_next_anchor)  # [N_next_anchor, 2KF]
            embeddings_next_pos = gnn_encoder(obs_next_pos)  # [N_next_pos, 2KF]
            graph_emb_next_anchor = embeddings_next_anchor[:, embed_dim:]  # [N_next_anchor, KF]
            graph_emb_next_pos = embeddings_next_pos[:, embed_dim:]  # [N_next_pos, KF]
            state_next_anchor = scatter_mean(graph_emb_next_anchor, obs_next_anchor.batch_non_omni, dim=0, dim_size=obs_next_anchor.batch_size)  # [seq_len, KF]
            state_next_pos = scatter_mean(graph_emb_next_pos, obs_next_pos.batch_non_omni, dim=0, dim_size=obs_next_pos.batch_size)  # [seq_len, KF]
            
            # Create input sequences: [seq_len, state_dim + 1 + 1 + state_dim] = [seq_len, input_dim]
            seq_anchor = torch.cat([state_anchor, 
                                    act_anchor.unsqueeze(1).float(), 
                                    rew_anchor.unsqueeze(1).float(),
                                    state_next_anchor], dim=1) 
            seq_pos = torch.cat([state_pos, 
                                 act_pos.unsqueeze(1).float(), 
                                 rew_pos.unsqueeze(1).float(),
                                 state_next_pos], dim=1)
            
            # Process negative sequences
            seq_neg_list = []
            for neg_data in neg_data_list:
                obs_neg, act_neg, rew_neg, obs_next_neg = neg_data
                
                # Encode states using GNN
                embeddings_neg = gnn_encoder(obs_neg)  # [N_neg, 2KF]
                graph_emb_neg = embeddings_neg[:, embed_dim:]  # [N_neg, KF]
                state_neg = scatter_mean(graph_emb_neg, obs_neg.batch_non_omni, dim=0, dim_size=obs_neg.batch_size)  # [seq_len, KF]
                
                # Encode next states
                embeddings_next_neg = gnn_encoder(obs_next_neg)  # [N_next_neg, 2KF]
                graph_emb_next_neg = embeddings_next_neg[:, embed_dim:]  # [N_next_neg, KF]
                state_next_neg = scatter_mean(graph_emb_next_neg, obs_next_neg.batch_non_omni, dim=0, dim_size=obs_next_neg.batch_size)  # [seq_len, KF]
                
                seq_neg = torch.cat([state_neg, 
                                    act_neg.unsqueeze(1).float(), 
                                    rew_neg.unsqueeze(1).float(),
                                    state_next_neg], dim=1)  # [seq_len, input_dim]
                seq_neg_list.append(seq_neg)
            # Stack negative sequences: [num_negatives, seq_len, input_dim]
            seq_negs = torch.stack(seq_neg_list)  # [num_negatives, seq_len, input_dim]
            
            # Add batch dimension: [1, seq_len, input_dim]
            seq_anchor = seq_anchor.unsqueeze(0)
            seq_pos = seq_pos.unsqueeze(0)
            # seq_negatives is already [num_negatives, seq_len, input_dim]
            
            # Forward pass through task encoder
            z_anchor, mu_anchor, logvar_anchor = task_encoder(seq_anchor)  # [1, seq_len, latent_dim]
            z_positive, _, _ = task_encoder(seq_pos)  # [1, seq_len, latent_dim]
            # Encode all negatives at once: [num_negatives, seq_len, input_dim] -> [num_negatives, seq_len, latent_dim]
            z_negatives, _, _ = task_encoder(seq_negs)  # [num_negatives, seq_len, latent_dim]
            
            # For contrastive loss, use the last timestep embedding (or mean over sequence)
            z_anchor_last = z_anchor[:, -1, :]  # [1, latent_dim]
            z_positive_last = z_positive[:, -1, :]  # [1, latent_dim]
            z_negatives_last = z_negatives[:, -1, :]  # [num_negatives, latent_dim]
            z_negatives_last = z_negatives_last.unsqueeze(0)  # [1, num_negatives, latent_dim]

            # Contrastive loss
            loss_contra = info_nce_loss(z_anchor_last, z_positive_last, z_negatives_last, temperature=0.1)
            
            # VAE KL divergence loss (average over sequence length)
            kl_loss = -0.5 * torch.sum(1 + logvar_anchor - mu_anchor.pow(2) - logvar_anchor.exp(), dim=2).mean()
            
            # Reconstruction loss: use sequence-length embeddings z_t for each timestep
            # z_anchor: [1, seq_len, latent_dim] -> squeeze to [seq_len, latent_dim]
            z_anchor_seq = z_anchor.squeeze(0)  # [seq_len, latent_dim]
            
            # Reward Loss: For each step t, predict r_t using z_t
            # pred_reward_t = Decoder(s_t, a_t, z_t)
            pred_rewards = reward_decoder(state_anchor, act_anchor, z_anchor_seq)  # [seq_len]
            actual_rewards = rew_anchor  # [seq_len]
            loss_rew_recon = F.mse_loss(pred_rewards, actual_rewards)
            
            # State Loss: For each step t, predict s'_t using z_t
            # pred_s'_t = Decoder(a_t, z_t)
            pred_states_next = state_decoder(act_anchor, z_anchor_seq)  # [seq_len, KF]
            actual_states_next = state_next_anchor  # [seq_len, KF]
            loss_state_recon = F.mse_loss(pred_states_next, actual_states_next)
            
            # Total loss
            total_batch_loss = loss_contra + 0.1 * kl_loss + loss_rew_recon + loss_state_recon
            
            # Backward pass
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_contra_loss += loss_contra.item()
            total_rew_recon_loss += loss_rew_recon.item()
            total_state_recon_loss += loss_state_recon.item()
            total_vae_loss += kl_loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_contra = total_contra_loss / num_batches
            avg_rew_recon = total_rew_recon_loss / num_batches
            avg_state_recon = total_state_recon_loss / num_batches
            avg_vae = total_vae_loss / num_batches
            print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f} "
                  f"(Contra: {avg_contra:.4f}, RewRecon: {avg_rew_recon:.4f}, StateRecon: {avg_state_recon:.4f}, VAE: {avg_vae:.4f})")
        
        # Evaluate and save best checkpoint at end of each epoch
        if (epoch + 1) % 1 == 0:  # Evaluate every epoch
            reward_decoder.eval()
            state_decoder.eval()
            
            # Extract embeddings
            z_all, labels_all = extract_embeddings_from_trajectories(
                gnn_encoder, task_encoder, test_trajectories_pl, test_trajectories_sbm,
                args.sequence_length, device
            )
            
            # Compute silhouette score and save best checkpoint
            if z_all is not None:
                score = compute_silhouette_score(z_all, labels_all)
                
                if score is not None:
                    print(f"Epoch {epoch+1} - Silhouette Score: {score:.4f}", end="")
                    
                    # Save best checkpoint
                    if score > best_silhouette_score:
                        best_silhouette_score = score
                        best_epoch = epoch + 1
                        os.makedirs(args.save_dir, exist_ok=True)
                        torch.save({
                            'gnn_encoder_state_dict': gnn_encoder.state_dict(),
                            'task_encoder_state_dict': task_encoder.state_dict(),
                            'reward_decoder_state_dict': reward_decoder.state_dict(),
                            'state_decoder_state_dict': state_decoder.state_dict(),
                            'input_dim': input_dim,
                            'hidden_dim': args.hidden_dim,
                            'latent_dim': args.latent_dim,
                            'num_features': args.num_features,
                            'num_heads': args.num_heads,
                            'num_mps': args.num_mps,
                            'gnn_type': args.gnn,
                            'epoch': epoch + 1,
                            'silhouette_score': score,
                        }, os.path.join(args.save_dir, 'task_encoder_best.ckpt'))
                        print(f" (NEW BEST! Saved checkpoint)")
                    else:
                        print(f" (Best: {best_silhouette_score:.4f} at epoch {best_epoch})")
                else:
                    print(f"Epoch {epoch+1} - Warning: Only one class in test set, cannot compute silhouette score")
    
    ### ================== Evaluate task encoder (final evaluation)============================= ###
    print("======== Task Encoder best ckpt Evaluating ========")
    
    # Load best checkpoint
    best_ckpt_path = os.path.join(args.save_dir, 'task_encoder_best.ckpt')
    if os.path.exists(best_ckpt_path):
        print(f"Loading best checkpoint from {best_ckpt_path}...")
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        gnn_encoder.load_state_dict(best_ckpt['gnn_encoder_state_dict'])
        task_encoder.load_state_dict(best_ckpt['task_encoder_state_dict'])
        if 'epoch' in best_ckpt:
            print(f"Best checkpoint: epoch {best_ckpt['epoch']} with silhouette score {best_ckpt.get('silhouette_score', 'N/A'):.4f}")
    else:
        print(f"Warning: Best checkpoint not found at {best_ckpt_path}, using current model state")
    
    # Extract embeddings using modular function
    z_all, labels_all = extract_embeddings_from_trajectories(
        gnn_encoder, task_encoder, test_trajectories_pl, test_trajectories_sbm,
        args.sequence_length, device
    )
    
    if z_all is not None:
        # Create t-SNE visualization
        tsne_path = os.path.join(args.save_dir, 'task_embedding_tsne.png')
        visualize_tsne(z_all, labels_all, tsne_path)
        
        print("\nEvaluation completed!")
    else:
        print("Warning: No test data available for evaluation")