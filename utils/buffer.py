import torch
import numpy as np

from .graph_data import Batch
from .graph_data import ig_to_data



class ReplayBuffer():
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        
        self.ptr = 0
        self.full = False
        self.obs_buffer = np.empty(buffer_size, dtype=object)
        self.act_buffer = np.empty(buffer_size, dtype=np.int64)
        self.obs_next_buffer = np.empty(buffer_size, dtype=object)
        self.rew_buffer = np.empty(buffer_size, dtype=np.float32)
        self.done_buffer = np.empty(buffer_size, dtype=bool)
        
    def add(self, obs_list, act_arr, obs_next_list, rew_arr, done_arr):
        num_t = len(obs_list)
        idx = np.arange(self.ptr, self.ptr+num_t)%self.buffer_size
        self.obs_buffer[idx] = np.array([ig_to_data(g) for g in obs_list])
        self.act_buffer[idx] = act_arr
        self.obs_next_buffer[idx] = np.array([ig_to_data(g) for g in obs_next_list])
        self.rew_buffer[idx] = rew_arr
        self.done_buffer[idx] = done_arr

        end = self.ptr + num_t
        if end >= self.buffer_size:
            self.full = True
        self.ptr = end % self.buffer_size
            
    def sample(self, batch_size):
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.ptr) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.ptr, size=batch_size)
            
        obs = Batch(self.device, self.obs_buffer[batch_inds].tolist())
        obs_next = Batch(self.device, self.obs_next_buffer[batch_inds].tolist())
        act = torch.tensor(self.act_buffer[batch_inds], device=self.device, dtype=torch.long)
        rew = torch.tensor(self.rew_buffer[batch_inds], device=self.device, dtype=torch.float32)
        done = torch.tensor(self.done_buffer[batch_inds], device=self.device, dtype=torch.float32)
        
        return obs, act, obs_next, rew, done

class PriorReplayBuffer():
    def __init__(self, buffer_size, device, gamma=1.0, alpha=0.3, beta=1.0, lambda_grad=1.0, teacher_bonus=1.0, epsilon=1e-6):
        self.buffer_size = buffer_size
        self.device = device
        
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.lambda_grad = lambda_grad
        self.teacher_bonus = teacher_bonus
        self.epsilon = epsilon

        self.fix_ptr = 0
        self.ptr = 0
        self.full = False
        self.obs_buffer = np.empty(buffer_size, dtype=object)
        self.act_buffer = np.empty(buffer_size, dtype=np.int64)
        self.obs_next_buffer = np.empty(buffer_size, dtype=object)
        self.rew_buffer = np.empty(buffer_size, dtype=np.float32)
        self.done_buffer = np.empty(buffer_size, dtype=bool)
        self.lcc_ratio_buffer = np.empty(buffer_size, dtype=np.float32)
        self.td_error_buffer = np.empty(buffer_size, dtype=np.float32)
        self.from_teacher_buffer = np.empty(buffer_size, dtype=bool)
        self.policy_grad_norm_buffer = np.empty(buffer_size, dtype=np.float32)
        
    def _compute_lcc_ratio(self, obs_list):
        """Compute LCC size ratio (current/original) for each observation"""
        lcc_ratios = []
        for obs in obs_list:
            try:
                n_init = obs['n_init']
                if obs.vcount() == 0:
                    current_lcc_size = 0
                else:
                    components = obs.connected_components()
                    current_lcc_size = max(components.sizes()) if len(components.sizes()) > 0 else 0
                ratio = current_lcc_size / max(n_init, 1)
                lcc_ratios.append(ratio)
            except Exception as e:
                print(f"error {e} in buffer lcc!")
                lcc_ratios.append(0.5)
        return np.array(lcc_ratios, dtype=np.float32)
    
    def add(self, obs_list, act_arr, obs_next_list, rew_arr, done_arr, td_errors=None, from_teacher=False, policy_grad_norms=None, fixed=False):
        """Add experiences to buffer with permanent teacher storage"""
        num_t = len(obs_list)
        lcc_ratios = self._compute_lcc_ratio(obs_list)
        
        if fixed:
            # Store teacher experiences permanently starting from fix_ptr
            idx = np.arange(self.fix_ptr, self.fix_ptr + num_t)
            if self.fix_ptr + num_t > self.buffer_size:
                print(f"Warning: Not enough space for teacher experiences")
                return
            
            self.fix_ptr += num_t
            # Ensure ptr is at least at fix_ptr
            if self.ptr < self.fix_ptr:
                self.ptr = self.fix_ptr
        else:
            # Store student experiences in circular buffer, but start from fix_ptr
            if self.ptr < self.fix_ptr:
                self.ptr = self.fix_ptr
            
            # Use circular indexing but ensure we don't go below fix_ptr
            idx = []
            for i in range(num_t):
                idx.append(self.ptr)
                self.ptr += 1
                if self.ptr >= self.buffer_size:
                    self.ptr = self.fix_ptr  # Wrap to fix_ptr, not 0
                    self.full = True
            idx = np.array(idx)
        
        # Store the data
        self.obs_buffer[idx] = np.array([ig_to_data(g) for g in obs_list])
        self.act_buffer[idx] = act_arr
        self.obs_next_buffer[idx] = np.array([ig_to_data(g) for g in obs_next_list])
        self.rew_buffer[idx] = rew_arr
        self.done_buffer[idx] = done_arr
        self.lcc_ratio_buffer[idx] = lcc_ratios
        self.from_teacher_buffer[idx] = from_teacher

        # Store TD-errors
        if td_errors is not None:
            self.td_error_buffer[idx] = td_errors
        else:
            self.td_error_buffer[idx] = 0.0
            
        # Store policy gradient norms
        if policy_grad_norms is not None:
            self.policy_grad_norm_buffer[idx] = policy_grad_norms
        else:
            self.policy_grad_norm_buffer[idx] = 0.0
    
    def sample(self, batch_size, priority=None, return_indices=False, return_tags=False):
        """
        Sample experiences with optional priority-based sampling and importance sampling weights
        """
        if self.full:
            available_size = self.buffer_size
            batch_inds = np.arange(self.buffer_size)
        else:
            available_size = self.ptr
            batch_inds = np.arange(self.ptr)
        
        if available_size == 0:
            return None
        
        weights = None  # Initialize weights
        
        if available_size > 0:
            # Compute priorities
            if priority == "LCC":
                lcc_ratios = self.lcc_ratio_buffer[batch_inds]
                priorities = np.ones(available_size)  # Base priority
                priorities += lcc_ratios  # LCC bonus (higher LCC = higher priority)
            elif priority == "TDE":
                # TD-error based priority (absolute difference)
                td_errors = np.abs(self.td_error_buffer[batch_inds])
                priorities = td_errors + self.epsilon  # Small epsilon to avoid zero priorities
            elif priority == "DDPGfD":
                # DDPGfD priority: P(i) = |TD_error| + λ*||∇policy|| + D_t + ε
                td_errors = np.abs(self.td_error_buffer[batch_inds])
                policy_grad_norms = self.policy_grad_norm_buffer[batch_inds]
                teacher_bonus_arr = self.from_teacher_buffer[batch_inds].astype(np.float32) * self.teacher_bonus
                priorities = (td_errors + self.lambda_grad * policy_grad_norms + teacher_bonus_arr + self.epsilon) ** self.alpha
            elif priority == "TEACHER":
                # Prioritize teacher experiences
                from_teacher = self.from_teacher_buffer[batch_inds].astype(np.float32)
                priorities = from_teacher + 0.1  # Teacher gets higher priority, student gets base priority
            else:
                priorities = np.ones(available_size)
            
            # Normalize priorities to probabilities
            probs = priorities / priorities.sum()
            
            # Sample with replacement based on priorities
            selected_indices = np.random.choice(batch_inds, size=min(batch_size, available_size), 
                                              p=probs, replace=True)
            
            # Calculate importance sampling weights: w_i = (1/N * 1/P(i))^β
            if priority is not None:
                selected_priorities = priorities[np.searchsorted(batch_inds, selected_indices)]
                weights = ((1.0 / available_size) * (1.0 / selected_priorities)) ** self.beta
                weights = weights / weights.max()                       # Normalize weights by max weight for stability
                weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
            else:
                weights = torch.ones(len(selected_indices), device=self.device, dtype=torch.float32)

        obs = Batch(self.device, self.obs_buffer[selected_indices].tolist())
        obs_next = Batch(self.device, self.obs_next_buffer[selected_indices].tolist())
        act = torch.tensor(self.act_buffer[selected_indices], device=self.device, dtype=torch.long)
        rew = torch.tensor(self.rew_buffer[selected_indices], device=self.device, dtype=torch.float32)
        done = torch.tensor(self.done_buffer[selected_indices], device=self.device, dtype=torch.float32)
        
        samples = [obs, act, obs_next, rew, done, weights]
        
        if return_indices:
            samples.append(selected_indices)
        if return_tags:
            tags = torch.tensor(self.from_teacher_buffer[selected_indices], device=self.device, dtype=torch.float32)
            samples.append(tags)
            
        return samples
    
    def update_td_errors(self, indices, td_errors):
        """Update TD-errors for specific buffer indices"""
        if len(indices) == len(td_errors):
            self.td_error_buffer[indices] = td_errors
    
    def update_policy_grad_norms(self, indices, policy_grad_norms):
        """Update policy gradient norms for specific buffer indices"""
        if len(indices) == len(policy_grad_norms):
            self.policy_grad_norm_buffer[indices] = policy_grad_norms

class RolloutBuffer:
    """
    Buffer for storing complete trajectories (episodes) for PPO training.
    Each trajectory contains a sequence of transitions until episode termination.
    """
    def __init__(self, device):
        self.device = device
        self.trajectories = []  # List of trajectories, each is a dict with lists of transitions
        
    def add_trajectory(self, obs_list, act_list, rew_list, obs_next_list, done_list):
        """
        Add a complete trajectory to the buffer.
        
        Args:
            obs_list: List of observations (igraph graphs) for the trajectory
            act_list: List of actions (numpy array or list)
            rew_list: List of rewards (numpy array or list)
            obs_next_list: List of next observations (igraph graphs)
            done_list: List of done flags (numpy array or list)
        """
        trajectory = {
            'obs': obs_list,
            'act': act_list,
            'rew': rew_list,
            'obs_next': obs_next_list,
            'done': done_list
        }
        self.trajectories.append(trajectory)

    def sample(self, batch_size):
        """
        Sample a batch of trajectories from the buffer and flatten to transitions
        """
        if len(self.trajectories) == 0:
            return []
        
        # Sample trajectories with replacement
        num_trajs = min(batch_size, len(self.trajectories))
        indices = np.random.choice(len(self.trajectories), size=num_trajs, replace=False)
        
        obs_list = []
        obs_next_list = []
        act_list = []
        rew_list = []
        done_list = []
        for idx in indices:
            traj = self.trajectories[idx]
            obs_list.extend([g for g in traj["obs"]])
            obs_next_list.extend([g for g in traj["obs_next"]])
            act_list.extend(traj["act"])
            rew_list.extend(traj["rew"])
            done_list.extend(traj["done"])
            
        # Convert to tensors and Batch objects
        obs = Batch(self.device, [ig_to_data(g) for g in obs_list])
        obs_next = Batch(self.device, [ig_to_data(g) for g in obs_next_list])
        act = torch.tensor(act_list, device=self.device, dtype=torch.long)
        rew = torch.tensor(rew_list, device=self.device, dtype=torch.float32)
        done = torch.tensor(done_list, device=self.device, dtype=torch.float32)
            
        return obs, act, rew, obs_next, done

    
    def sample_trajectories(self, batch_size):
        """
        Sample a batch of trajectories from the buffer.
        
        Args:
            batch_size: Number of trajectories to sample
            
        Returns:
            List of trajectory dictionaries, each containing:
            - obs: Batch object
            - act: torch.Tensor of actions
            - rew: torch.Tensor of rewards
            - obs_next: Batch object
            - done: torch.Tensor of done flags
        """
        if len(self.trajectories) == 0:
            return []
        
        # Sample trajectories with replacement
        num_trajs = min(batch_size, len(self.trajectories))
        indices = np.random.choice(len(self.trajectories), size=num_trajs, replace=False)
        
        sampled_trajs = []
        for idx in indices:
            traj = self.trajectories[idx]
            
            # Convert to tensors and Batch objects
            obs_batch = Batch(self.device, [ig_to_data(g) for g in traj['obs']])
            obs_next_batch = Batch(self.device, [ig_to_data(g) for g in traj['obs_next']])
            act_tensor = torch.tensor(traj['act'], device=self.device, dtype=torch.long)
            rew_tensor = torch.tensor(traj['rew'], device=self.device, dtype=torch.float32)
            done_tensor = torch.tensor(traj['done'], device=self.device, dtype=torch.float32)
            
            sampled_trajs.append({
                'obs': obs_batch,
                'act': act_tensor,
                'rew': rew_tensor,
                'obs_next': obs_next_batch,
                'done': done_tensor
            })
        
        return sampled_trajs
    
    def clear(self):
        """Clear all trajectories from the buffer."""
        self.trajectories = []
    
    def size(self):
        """Return the number of trajectories in the buffer."""
        return len(self.trajectories)
    
    