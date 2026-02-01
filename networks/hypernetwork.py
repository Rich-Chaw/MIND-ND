import torch
import torch.nn as nn


class Hypernetwork(nn.Module):
    """
    Hypernetwork that generates weights and biases for the last layer
    of policy and Q-networks from task embedding.
    """
    def __init__(self, latent_dim, hidden_dim=128, output_dim=1, input_dim=256):
        """
        Args:
            latent_dim: Dimension of task embedding z
            hidden_dim: Hidden dimension of hypernetwork
            output_dim: Output dimension of target layer (1 for both policy and Q)
            input_dim: Input dimension of target layer (256 for both)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        # Generate weights: [latent_dim] -> [output_dim, input_dim]
        # bias=False so that zero z produces zero weight (no task -> no modulation)
        self.weight_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * input_dim)
        )
        
        # Generate bias: [latent_dim] -> [output_dim]
        # bias=False so that zero z produces zero bias
        self.bias_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        """
        Generate weights and bias for last layer from task embedding.

        Args:
            z: Task embedding [latent_dim] or [batch_size, latent_dim]

        Returns:
            weight: [1, input_dim] when z is [latent_dim];
                    [batch_size, input_dim] when z is [batch_size, latent_dim]
            bias: [output_dim] when z is [latent_dim]; [batch_size] when z is [batch_size, latent_dim]
        """
        # Ensure z is 2D: [batch_size, latent_dim]
        if z.dim() == 1:
            z = z.unsqueeze(0)
        batch_size = z.shape[0]

        # Weights: [batch_size, output_dim * input_dim] -> [batch_size, output_dim, input_dim]
        weight_flat = self.weight_net(z)  # [B, output_dim * input_dim]
        weight = weight_flat.view(batch_size, self.output_dim, self.input_dim)
        # For output_dim=1: [B, 1, input_dim] -> squeeze to [B, input_dim] so consumers get [B, 256]
        if self.output_dim == 1:
            weight = weight.squeeze(1)  # [B, input_dim]

        # Bias: [batch_size, output_dim]
        bias = self.bias_net(z)  # [B, output_dim]
        if self.output_dim == 1:
            bias = bias.squeeze(1)  # [B]

        return weight, bias


class FiLMGenerator(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) generator: from task embedding z, outputs
    (gamma, beta) pairs for conditioning MLP hidden layers. One set for Policy MLP,
    one set for Critic (Q) MLP. Applied as: out = gamma * hidden + beta (per layer).
    """
    def __init__(self, latent_dim, hidden_dim=128, film_hidden_dim=256, num_film_layers=2):
        """
        Args:
            latent_dim: Task embedding dimension
            hidden_dim: Hidden size of the generator MLP
            film_hidden_dim: Hidden size of the target MLP layers to modulate (256)
            num_film_layers: Number of hidden layers to modulate (2 for 3-layer MLP: 256, 256)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.film_hidden_dim = film_hidden_dim
        self.num_film_layers = num_film_layers
        # Per-layer: gamma and beta, each [film_hidden_dim]. Two heads: policy and critic.
        out_per_head = num_film_layers * film_hidden_dim * 2  # gamma and beta per layer
        self._out_per_head = out_per_head

        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_per_head),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_per_head),
        )

    def forward(self, z):
        """
        Args:
            z: [batch_size, latent_dim]

        Returns:
            policy_film: list of (gamma, beta) each [B, film_hidden_dim], length num_film_layers
            critic_film: list of (gamma, beta) each [B, film_hidden_dim], length num_film_layers
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.shape[0]
        D = self.film_hidden_dim
        L = self.num_film_layers

        p_flat = self.policy_net(z)   # [B, L*D*2]
        c_flat = self.critic_net(z)  # [B, L*D*2]

        policy_film = []
        for i in range(L):
            g = p_flat[:, i * D * 2 : i * D * 2 + D].view(B, D)
            b = p_flat[:, i * D * 2 + D : (i + 1) * D * 2].view(B, D)
            policy_film.append((g, b))
        critic_film = []
        for i in range(L):
            g = c_flat[:, i * D * 2 : i * D * 2 + D].view(B, D)
            b = c_flat[:, i * D * 2 + D : (i + 1) * D * 2].view(B, D)
            critic_film.append((g, b))

        return policy_film, critic_film

