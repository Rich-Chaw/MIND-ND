import os
import json
import torch
import torch.nn as nn
from torch_scatter import scatter_log_softmax, scatter_max, scatter_mean

from utils.graph_data import Batch
from .gnn_interface import GNN_ENCODER
from .hypernetwork import Hypernetwork, FiLMGenerator


def load_sac_dismantler(F=16, H=3, K=6, gnn=None, device=None, ckpt_pth=None, positional_encoding=None, handcrafted_features=False):
    if ckpt_pth != None:
        config_dir = os.path.dirname(ckpt_pth)
        if os.path.exists(os.path.join(config_dir, 'args.json')):
            with open(os.path.join(config_dir, 'args.json')) as f:
                args = json.load(f)
        F = args['num_features']
        H = args['num_heads']
        K = args['num_mps']
        gnn = args['gnn']
        positional_encoding = args['positional_encoding']
        handcrafted_features = args['handcrafted_features']
    
    policy = SACPolicy(F, H, K, gnn, positional_encoding, handcrafted_features).to(device)
    qf1 = SACQNetwork(F, H, K, gnn, positional_encoding, handcrafted_features).to(device)
    qf2 = SACQNetwork(F, H, K, gnn, positional_encoding, handcrafted_features).to(device)
    qf1_target = SACQNetwork(F, H, K, gnn, positional_encoding, handcrafted_features).to(device)
    qf2_target = SACQNetwork(F, H, K, gnn, positional_encoding, handcrafted_features).to(device)
    if ckpt_pth != None:
        ckpt = torch.load(ckpt_pth, map_location=device)
        policy.load_state_dict(ckpt['policy_state_dict'])
        qf1.load_state_dict(ckpt['qf1_state_dict'])
        qf2.load_state_dict(ckpt['qf2_state_dict'])
        qf1_target.load_state_dict(ckpt['qf1_target_state_dict'])
        qf2_target.load_state_dict(ckpt['qf2_target_state_dict'])
    else:
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

    return policy, qf1, qf2, qf1_target, qf2_target



class SACPolicy(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, gnn, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.graph_embedding = GNN_ENCODER[gnn](num_features, num_heads, num_mps, positional_encoding=positional_encoding, handcrafted_features=handcrafted_features)
        e_size = (num_features*num_mps)*2

        self.mlp = nn.Sequential(
            nn.Linear(e_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, g:Batch):
        e = self.graph_embedding(g) #[N,2KF]
        logits = self.mlp(e).flatten()
        return logits
    
    def get_action(self, g:Batch, val=False):
        '''
        return 
            act: action for each graph [B]
            log_probs: [N] without omni code
        '''
        logits = self(g)
        log_probs = scatter_log_softmax(logits, g.batch_non_omni, dim_size=g.batch_size)
        if val:
            _, act = scatter_max(log_probs, g.batch_non_omni, dim_size=g.batch_size)
        else:
            # Gumbel-Max trick
            gumbel_noise = -torch.empty_like(log_probs).exponential_().log()
            gumbel_logits = log_probs + gumbel_noise
            _, act = scatter_max(gumbel_logits, g.batch_non_omni, dim_size=g.batch_size)  # (B,)
        act -= g.act_offsets
        return act, log_probs



class SACQNetwork(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, gnn, positional_encoding=None, handcrafted_features=False):
        super().__init__()
        self.graph_embedding = GNN_ENCODER[gnn](num_features, num_heads, num_mps, positional_encoding=positional_encoding, handcrafted_features=handcrafted_features)
        e_size = (num_features*num_mps)*2

        self.mlp = nn.Sequential(
            nn.Linear(e_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, g:Batch):
        e = self.graph_embedding(g)
        q_vals = self.mlp(e).flatten()
        return q_vals

class PPOPolicy(SACPolicy):
    pass


class PPOVNetwork(nn.Module):
    def __init__(self, num_features, num_heads, num_mps, gnn):
        super().__init__()
        self.graph_embedding = GNN_ENCODER[gnn](num_features, num_heads, num_mps)
        # Extract only the graph embedding part (second half)
        e_size = num_features * num_mps

        self.mlp = nn.Sequential(
            nn.Linear(e_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, g: Batch):
        e = self.graph_embedding(g)  # [N, 2KF]
        # Extract only the graph embedding part (second half)
        graph_e = e[:, e.shape[1]//2:]  # [N, KF]
        node_values = self.mlp(graph_e).flatten()  # [N]
        v_vals = scatter_mean(node_values, g.batch_non_omni, dim_size=g.batch_size) #[B]
        return v_vals

def load_ppo_dismantler(F, H, K, device, ckpt_pth=None):
    policy = PPOPolicy(F, H, K).to(device)
    vf = PPOVNetwork(F, H, K).to(device)
    
    if ckpt_pth is not None:
        ckpt = torch.load(ckpt_pth)
        policy.load_state_dict(ckpt['policy_state_dict'])
        vf.load_state_dict(ckpt['vf_state_dict'])
    
    return policy, vf

def load_dqn_dismantler(F=16, H=3, K=6, gnn=None, device=None, ckpt_pth=None, positional_encoding=None, handcrafted_features=False):
    if ckpt_pth != None:
        config_dir = os.path.dirname(ckpt_pth)
        if os.path.exists(os.path.join(config_dir, 'args.json')):
            with open(os.path.join(config_dir, 'args.json')) as f:
                args = json.load(f)
            F = args['num_features']
            H = args['num_heads']
            K = args['num_mps']
            gnn = args['gnn']
            positional_encoding = args['positional_encoding']
            handcrafted_features = args['handcrafted_features']
    
    qf = SACQNetwork(F, H, K, gnn, positional_encoding, handcrafted_features).to(device)
    qf_target = SACQNetwork(F, H, K, gnn, positional_encoding, handcrafted_features).to(device)

    if ckpt_pth is not None:
        ckpt = torch.load(ckpt_pth, map_location=device)
        if "qf_state_dict" in ckpt:
            qf.load_state_dict(ckpt["qf_state_dict"])
            qf_target.load_state_dict(ckpt.get("qf_target_state_dict", ckpt["qf_state_dict"]))
        elif "qf1_state_dict" in ckpt:
            qf.load_state_dict(ckpt["qf1_state_dict"])
            qf_target.load_state_dict(ckpt.get("qf1_target_state_dict", ckpt["qf1_state_dict"]))
        else:
            raise KeyError(f"Checkpoint {ckpt_pth} does not contain DQN-compatible Q-network weights.")
    else:
        qf_target.load_state_dict(qf.state_dict())

    return qf, qf_target

class DQNPolicy(torch.nn.Module):
    def __init__(self, qf: SACQNetwork):
        super().__init__()
        self.qf = qf

    def forward(self, g: Batch):
        return self.qf(g)

    def get_action(self, g: Batch, epsilon: float = 0.0, val: bool = False):
        q_vals = self.qf(g)
        _, greedy_act = scatter_max(q_vals, g.batch_non_omni, dim_size=g.batch_size)
        greedy_act = greedy_act - g.act_offsets

        if val or epsilon <= 0.0:
            return greedy_act, q_vals

        random_mask = torch.rand(g.batch_size, device=q_vals.device) < epsilon
        if random_mask.any():
            num_actions = (g.num_nodes_b - 1).clamp(min=1)
            random_act = (torch.rand(g.batch_size, device=q_vals.device) * num_actions.float()).long()
            act = torch.where(random_mask, random_act, greedy_act)
            return act, q_vals
        return greedy_act, q_vals

def load_dismantler(ckpt_pth=None,device=torch.device('cpu')):
    # defalut value when no args.config
    # device: str='cuda:0'
    F = 16; H = 4; K = 6
    gnn = 'mind'
    
    if 'sac' in ckpt_pth:
        policy, _, _, _, _ = load_sac_dismantler(F, H, K, gnn, device, ckpt_pth)
    else:
        qf, _ = load_dqn_dismantler(F, H, K, gnn, device, ckpt_pth)
        policy = DQNPolicy(qf)
        
    return policy


# ========== Task-Adaptive Networks with Shared Hypernetwork ==========

class SACPolicyWithHypernetwork(nn.Module):
    """
    SAC Policy with shared backbone and hypernetwork-generated last layer.
    """
    def __init__(self, num_features, num_heads, num_mps, gnn, hypernetwork):
        super().__init__()
        self.graph_embedding = GNN_ENCODER[gnn](num_features, num_heads, num_mps)
        e_size = (num_features*num_mps)*2
        self.mlp = nn.Sequential(
            nn.Linear(e_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.hypernetwork = hypernetwork
    
    def forward(self, g: Batch, z=None):
        """
        Forward pass with optional task embedding.
        If z is None, uses a default zero embedding (for compatibility).
        
        Args:
            g: Batch of graphs
            z: Task embedding [latent_dim] or [batch_size, latent_dim]
        """
        e = self.graph_embedding(g)  # [N, 2KF]
        features = self.mlp(e)  # [N, 256]
        if z is not None:
            # Generate last layer weights from task embedding
            weight, bias = self.hypernetwork(z)
            # Two supported modes:
            # - z: [latent_dim] -> weight: [1, 256], bias: [1] (shared across graphs)
            # - z: [B, latent_dim] -> weight: [B, 256], bias: [B] (one per graph)
            if weight.dim() == 2 and weight.shape[0] == 1:
                logits = torch.matmul(features, weight.t()) + bias  # [N, 1]
                logits = logits.flatten()  # [N]
            else:
                # Per-graph weights: index each node by its graph id
                b = g.batch_non_omni  # [N]
                w = weight[b]  # [N, 256]
                bb = bias[b]   # [N] or [N, 1]
                logits = (features * w).sum(dim=1) + bb.view(-1)
        else:
            # Fallback: use zero task embedding
            weight, bias = self.hypernetwork(torch.zeros(self.hypernetwork.latent_dim, device=features.device))
            logits = torch.matmul(features, weight.t()) + bias
            logits = logits.flatten()
        
        return logits
    
    def get_action(self, g: Batch, z=None, val=False):
        """
        Get action with optional task embedding.
        """
        logits = self(g, z) # [N, 1]
        log_probs = scatter_log_softmax(logits, g.batch_non_omni, dim_size=g.batch_size)
        if val:
            _, act = scatter_max(log_probs, g.batch_non_omni, dim_size=g.batch_size)
        else:
            # Gumbel-Max trick
            gumbel_noise = -torch.empty_like(log_probs).exponential_().log()
            gumbel_logits = log_probs + gumbel_noise
            _, act = scatter_max(gumbel_logits, g.batch_non_omni, dim_size=g.batch_size)
        act -= g.act_offsets
        return act, log_probs


class SACQNetworkWithHypernetwork(nn.Module):
    """
    SAC Q-Network with shared backbone and hypernetwork-generated last layer.
    """
    def __init__(self,num_features, num_heads, num_mps, gnn, hypernetwork):
        super().__init__()
        self.graph_embedding = GNN_ENCODER[gnn](num_features, num_heads, num_mps)
        e_size = (num_features*num_mps)*2
        self.mlp = nn.Sequential(
            nn.Linear(e_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.hypernetwork = hypernetwork
    
    def forward(self, g: Batch, z=None):
        """
        Forward pass with optional task embedding.
        """
        e = self.graph_embedding(g)  # [N, 2KF]
        features = self.mlp(e)  # [N, 256]
        
        if z is not None:
            # Generate last layer weights from task embedding
            weight, bias = self.hypernetwork(z)
            if weight.dim() == 2 and weight.shape[0] == 1:
                q_vals = torch.matmul(features, weight.t()) + bias  # [N, 1]
                q_vals = q_vals.flatten()  # [N]
            else:
                b = g.batch_non_omni  # [N]
                w = weight[b]  # [N, 256]
                bb = bias[b]   # [N] or [N, 1]
                q_vals = (features * w).sum(dim=1) + bb.view(-1)
        else:
            # Fallback: use zero task embedding
            weight, bias = self.hypernetwork(torch.zeros(self.hypernetwork.latent_dim, device=features.device))
            q_vals = torch.matmul(features, weight.t()) + bias
            q_vals = q_vals.flatten()
        
        return q_vals


def load_sac_dismantler_with_hypernet(F, H, K, gnn, device, latent_dim, ckpt_pth=None):
    """
    Load SAC networks with shared backbone and hypernetworks.
    
    Args:
        F: num_features
        H: num_heads
        K: num_mps
        gnn: GNN type
        device: Device
        latent_dim: Task embedding dimension
        ckpt_pth: Optional checkpoint path
    """
    # Create hypernetworks for policy and Q-networks
    hypernet = Hypernetwork(latent_dim, hidden_dim=128, output_dim=1, input_dim=256).to(device)
    
    # Create networks (same F, H, K, gnn as base SAC; shared hypernet for last layer)
    policy = SACPolicyWithHypernetwork(F, H, K, gnn, hypernet).to(device)
    qf1 = SACQNetworkWithHypernetwork(F, H, K, gnn, hypernet).to(device)
    qf2 = SACQNetworkWithHypernetwork(F, H, K, gnn, hypernet).to(device)
    qf1_target = SACQNetworkWithHypernetwork(F, H, K, gnn, hypernet).to(device)
    qf2_target = SACQNetworkWithHypernetwork(F, H, K, gnn, hypernet).to(device)
    
    if ckpt_pth is not None:
        ckpt = torch.load(ckpt_pth)
        policy.load_state_dict(ckpt['policy_state_dict'])
        qf1.load_state_dict(ckpt['qf1_state_dict'])
        qf2.load_state_dict(ckpt['qf2_state_dict'])
        qf1_target.load_state_dict(ckpt['qf1_target_state_dict'])
        qf2_target.load_state_dict(ckpt['qf2_target_state_dict'])
        # Load hypernetwork
        hypernet.load_state_dict(ckpt.get('hypernet_state_dict', {}))
    else:
        # Initialize target networks with same weights
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
    
    return policy, qf1, qf2, qf1_target, qf2_target, hypernet


# ========== Shared GNN + Separate MLP Heads + FiLM Conditioning ==========

def _apply_film_mlp(e, film_list, linear_layers, batch_idx):
    """
    Run a 3-layer MLP with FiLM after the first two hidden layers.
    linear_layers: [Linear(e_size, 256), Linear(256, 256), Linear(256, 1)]
    film_list: list of (gamma [B, 256], beta [B, 256]), length 2
    batch_idx: [N] graph index per node
    """
    h = linear_layers[0](e)
    h = torch.relu(h)
    g, b = film_list[0]
    h = g[batch_idx] * h + b[batch_idx]

    h = linear_layers[1](h)
    h = torch.relu(h)
    g, b = film_list[1]
    h = g[batch_idx] * h + b[batch_idx]

    out = linear_layers[2](h)
    return out.flatten()


class SACPolicyFiLM(nn.Module):
    """
    SAC Policy: shared GNN + 3-layer Policy MLP head with FiLM conditioning.
    """
    def __init__(self, shared_gnn, e_size, film_generator):
        super().__init__()
        self.graph_embedding = shared_gnn
        self.film_generator = film_generator
        self.mlp = nn.ModuleList([
            nn.Linear(e_size, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 1),
        ])

    def forward(self, g: Batch, z=None):
        e = self.graph_embedding(g)
        b = g.batch_non_omni
        if z is not None:
            policy_film, _ = self.film_generator(z)
            return _apply_film_mlp(e, policy_film, self.mlp, b)
        z0 = torch.zeros(1, self.film_generator.latent_dim, device=e.device)
        policy_film, _ = self.film_generator(z0)
        return _apply_film_mlp(e, policy_film, self.mlp, b)

    def get_action(self, g: Batch, z=None, val=False):
        logits = self(g, z)
        log_probs = scatter_log_softmax(logits, g.batch_non_omni, dim_size=g.batch_size)
        if val:
            _, act = scatter_max(log_probs, g.batch_non_omni, dim_size=g.batch_size)
        else:
            gumbel_noise = -torch.empty_like(log_probs).exponential_().log()
            gumbel_logits = log_probs + gumbel_noise
            _, act = scatter_max(gumbel_logits, g.batch_non_omni, dim_size=g.batch_size)
        act -= g.act_offsets
        return act, log_probs


class SACQNetworkFiLM(nn.Module):
    """
    SAC Q-Network: shared GNN + 3-layer Q MLP head with FiLM conditioning.
    """
    def __init__(self, shared_gnn, e_size, film_generator):
        super().__init__()
        self.graph_embedding = shared_gnn
        self.film_generator = film_generator
        self.mlp = nn.ModuleList([
            nn.Linear(e_size, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 1),
        ])

    def forward(self, g: Batch, z=None):
        e = self.graph_embedding(g)
        if z is not None:
            _, critic_film = self.film_generator(z)
            b = g.batch_non_omni
            return _apply_film_mlp(e, critic_film, self.mlp, b)
        z0 = torch.zeros(1, self.film_generator.latent_dim, device=e.device)
        _, critic_film = self.film_generator(z0)
        b = g.batch_non_omni
        return _apply_film_mlp(e, critic_film, self.mlp, b)


def load_sac_dismantler_with_film(F, H, K, gnn, device, latent_dim, ckpt_pth=None):
    """
    Load SAC with shared GNN, separate policy/Q MLP heads, and FiLM conditioning.
    Returns: policy, qf1, qf2, qf1_target, qf2_target, film_generator
    """
    e_size = (F * K) * 2
    shared_gnn = GNN_ENCODER[gnn](F, H, K)
    film_generator = FiLMGenerator(
        latent_dim, hidden_dim=128, film_hidden_dim=256, num_film_layers=2
    ).to(device)

    shared_gnn = shared_gnn.to(device)
    policy = SACPolicyFiLM(shared_gnn, e_size, film_generator).to(device)
    qf1 = SACQNetworkFiLM(shared_gnn, e_size, film_generator).to(device)
    qf2 = SACQNetworkFiLM(shared_gnn, e_size, film_generator).to(device)
    qf1_target = SACQNetworkFiLM(shared_gnn, e_size, film_generator).to(device)
    qf2_target = SACQNetworkFiLM(shared_gnn, e_size, film_generator).to(device)

    if ckpt_pth is not None:
        ckpt = torch.load(ckpt_pth, map_location=device)
        policy.load_state_dict(ckpt.get("policy_state_dict", {}), strict=False)
        qf1.load_state_dict(ckpt.get("qf1_state_dict", {}), strict=False)
        qf2.load_state_dict(ckpt.get("qf2_state_dict", {}), strict=False)
        qf1_target.load_state_dict(ckpt.get("qf1_target_state_dict", qf1.state_dict()))
        qf2_target.load_state_dict(ckpt.get("qf2_target_state_dict", qf2.state_dict()))
        if "film_generator_state_dict" in ckpt:
            film_generator.load_state_dict(ckpt["film_generator_state_dict"])
    else:
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

    return policy, qf1, qf2, qf1_target, qf2_target, film_generator