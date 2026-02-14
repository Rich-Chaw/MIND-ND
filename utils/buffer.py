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
        self.from_teacher_buffer = np.empty(buffer_size, dtype=bool)
        self.td_error_buffer = np.empty(buffer_size, dtype=np.float32)
        self.policy_grad_norm_buffer = np.empty(buffer_size, dtype=np.float32)
    
    def add(self, obs_list, act_arr, obs_next_list, rew_arr, done_arr, td_errors=None, from_teacher=False, policy_grad_norms=None, fixed=False):
        """Add experiences to buffer with permanent teacher storage"""
        num_t = len(obs_list)
        
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
        Sample experiences with optional priority-based sampling
        """
        if self.full:
            available_size = self.buffer_size
            batch_inds = np.arange(self.buffer_size)
        else:
            available_size = self.ptr
            batch_inds = np.arange(self.ptr)
        
        if available_size == 0:
            return None
        
        if available_size > 0:
            # Compute priorities
            if priority == "TDE":
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
            
        obs = Batch(self.device, self.obs_buffer[selected_indices].tolist())
        obs_next = Batch(self.device, self.obs_next_buffer[selected_indices].tolist())
        act = torch.tensor(self.act_buffer[selected_indices], device=self.device, dtype=torch.long)
        rew = torch.tensor(self.rew_buffer[selected_indices], device=self.device, dtype=torch.float32)
        done = torch.tensor(self.done_buffer[selected_indices], device=self.device, dtype=torch.float32)
        
        samples = [obs, act, obs_next, rew, done]
        
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

    def get_state_dict(self):
        """Return a dict of buffer state_dict for saving (e.g. inside a checkpoint)."""
        return {
            "buffer_size": self.buffer_size,
            "ptr": self.ptr,
            "fix_ptr": self.fix_ptr,
            "full": self.full,
            "obs_buffer": self.obs_buffer,
            "act_buffer": self.act_buffer,
            "obs_next_buffer": self.obs_next_buffer,
            "rew_buffer": self.rew_buffer,
            "done_buffer": self.done_buffer,
            "from_teacher_buffer": self.from_teacher_buffer,
            "td_error_buffer": self.td_error_buffer,
            "policy_grad_norm_buffer": self.policy_grad_norm_buffer,
        }

    def load_state_dict(self, state_dict):
        """
        Restore buffer from a state dict (from get_state() or from a checkpoint).
        If saved buffer is smaller than self.buffer_size, only the used slice is copied.
        """
        self.ptr = int(state_dict["ptr"])
        self.fix_ptr = int(state_dict["fix_ptr"])
        self.full = bool(state_dict["full"])
        saved_size = int(state_dict.get("buffer_size", len(state_dict["act_buffer"])))
        n = min(saved_size, self.buffer_size, self.ptr if not self.full else self.buffer_size)
        if n <= 0:
            return
        idx = np.arange(n)
        self.obs_buffer[idx] = state_dict["obs_buffer"][idx]
        self.act_buffer[idx] = state_dict["act_buffer"][idx]
        self.obs_next_buffer[idx] = state_dict["obs_next_buffer"][idx]
        self.rew_buffer[idx] = state_dict["rew_buffer"][idx]
        self.done_buffer[idx] = state_dict["done_buffer"][idx]
        self.from_teacher_buffer[idx] = state_dict["from_teacher_buffer"][idx]
        self.td_error_buffer[idx] = state_dict["td_error_buffer"][idx]
        self.policy_grad_norm_buffer[idx] = state_dict["policy_grad_norm_buffer"][idx]

    def save(self, path):
        """Save buffer state_dict to a file (torch.save)."""
        torch.save(self.get_state(), path)

    def load(self, path):
        """
        Load buffer state_dict from a file (torch.save) or from a path.
        Can also be used with a state_dict dict: buffer.load_state(torch.load(path)).
        """
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "act_buffer" in state_dict:
            self.load_state(state_dict)
        else:
            raise ValueError(f"Invalid buffer checkpoint at {path}: expected state_dict dict with 'act_buffer'.")


class RolloutBuffer:
    """
    Buffer for storing complete trajectories (episodes) for PPO training.
    Each trajectory contains a sequence of transitions until episode termination.
    """
    def __init__(self, device, buffer_size=None):
        """
        Args:
            device: Device to store tensors on
            buffer_size: Optional maximum number of trajectories to store. If None, stores all trajectories.
        """
        self.device = device
        self.buffer_size = buffer_size
        # List of trajectories, each is a dict with lists of transitions
        if buffer_size is not None:
            self.trajectories = [None] * buffer_size
        else:
            self.trajectories = []
        self.ptr = 0
        self.full = False
        
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
        
        if self.buffer_size is not None:
            # Use circular buffer if size is limited
            self.trajectories[self.ptr] = trajectory
            self.ptr = (self.ptr + 1) % self.buffer_size
            if self.ptr == 0:
                self.full = True
        else:
            # No size limit - append to list
            self.trajectories.append(trajectory)

    def sample(self, batch_size):
        """
        Sample a batch of trajectories from the buffer and flatten to transitions
        """
        if self.size() == 0:
            return []
        
        # Sample trajectories with replacement
        num_trajs = min(batch_size, self.size())
        indices = np.random.choice(self.size(), size=num_trajs, replace=False)
        
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
        if self.size() == 0:
            return []
        
        # Sample trajectories with replacement
        num_trajs = min(batch_size, self.size())
        indices = np.random.choice(self.size(), size=num_trajs, replace=False)
        
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
        if self.buffer_size is not None:
            return self.buffer_size if self.full else self.ptr
        else:
            return len(self.trajectories)
    
    def sample_sequences_with_context(self, batch_size, seq_len, gnn_encoder):
        """
        Sample batch_size sequences of length seq_len from trajectories.
        Returns context sequences (first seq_len-1 transitions) and training transitions (last transition).
        
        Args:
            batch_size: Number of sequences to sample
            seq_len: Length of sequence (last transition is for training, rest is context)
            gnn_encoder: GNN encoder to encode states for context
            
        Returns:
            context_seqs: [batch_size, seq_len-1, input_dim] - context sequences
            obs_b: Batch - observations for training (last transition)
            act_b: [batch_size] - actions for training
            obs_next_b: Batch - next observations for training
            rew_b: [batch_size] - rewards for training
            done_b: [batch_size] - done flags for training
        """
        if self.size() == 0:
            return None
        
        # Filter trajectories that have at least seq_len transitions
        valid_trajectories = [i for i in range(self.size()) 
                            if len(self.trajectories[i]['obs']) >= seq_len]
        
        if len(valid_trajectories) == 0:
            return None
        
        # Sample trajectory indices
        sampled_indices = np.random.choice(valid_trajectories, size=batch_size, replace=True)
        
    
        obs_list = []
        act_list = []
        rew_list = []
        obs_next_list = []
        done_list = []
        
        # Collect context graphs across the whole minibatch for a single GNN pass
        # We'll flatten in (sample_idx, time_idx) order so we can reshape back easily.
        flat_context_obs = []
        flat_context_obs_next = []
        flat_context_act = []
        flat_context_rew = []
        context_len = seq_len - 1
        
        for traj_idx in sampled_indices:
            traj = self.trajectories[traj_idx]
            traj_len = len(traj['obs'])
            
            # Sample a random starting index such that we have seq_len transitions
            start_idx = np.random.randint(0, traj_len - seq_len + 1)
            
            # Extract sequence
            obs_seq = traj['obs'][start_idx:start_idx + seq_len]
            act_seq = traj['act'][start_idx:start_idx + seq_len]
            rew_seq = traj['rew'][start_idx:start_idx + seq_len]
            obs_next_seq = traj['obs_next'][start_idx:start_idx + seq_len]
            
            # Context: first seq_len-1 transitions
            context_obs = obs_seq[:-1]
            context_act = act_seq[:-1]
            context_rew = rew_seq[:-1]
            context_obs_next = obs_next_seq[:-1]
            
            # Training sample: last transition
            obs_list.append(obs_seq[-1])
            act_list.append(act_seq[-1])
            rew_list.append(rew_seq[-1])
            obs_next_list.append(obs_next_seq[-1])
            done_list.append(traj['done'][start_idx + seq_len - 1])

            # Accumulate flattened context for later batch encoding
            flat_context_obs.extend(context_obs)
            flat_context_obs_next.extend(context_obs_next)
            flat_context_act.extend(context_act)
            flat_context_rew.extend(context_rew)
        
        # Encode all context states in one GNN pass (much faster than per-graph loops)
        with torch.no_grad():
            # Current states
            obs_ctx_b = Batch(self.device, [ig_to_data(g) for g in flat_context_obs])
            emb_ctx = gnn_encoder(obs_ctx_b)  # [N_ctx, 2KF]
            embed_dim = emb_ctx.shape[1] // 2
            graph_emb_ctx = emb_ctx[:, embed_dim:]  # [N_ctx, KF]
            from torch_scatter import scatter_mean
            state_ctx = scatter_mean(
                graph_emb_ctx,
                obs_ctx_b.batch_non_omni,
                dim=0,
                dim_size=obs_ctx_b.batch_size,
            )  # [batch_size*(seq_len-1), KF]

            # Next states
            obs_next_ctx_b = Batch(self.device, [ig_to_data(g) for g in flat_context_obs_next])
            emb_next_ctx = gnn_encoder(obs_next_ctx_b)  # [N_next_ctx, 2KF]
            graph_emb_next_ctx = emb_next_ctx[:, embed_dim:]  # [N_next_ctx, KF]
            state_next_ctx = scatter_mean(
                graph_emb_next_ctx,
                obs_next_ctx_b.batch_non_omni,
                dim=0,
                dim_size=obs_next_ctx_b.batch_size,
            )  # [batch_size*(seq_len-1), KF]

        # Reshape back to [batch_size, seq_len-1, KF]
        state_ctx = state_ctx.view(batch_size, context_len, -1)
        state_next_ctx = state_next_ctx.view(batch_size, context_len, -1)

        # Actions/rewards to tensors and reshape to [batch_size, seq_len-1]
        act_ctx = torch.tensor(flat_context_act, device=self.device, dtype=torch.float32).view(batch_size, context_len)
        rew_ctx = torch.tensor(flat_context_rew, device=self.device, dtype=torch.float32).view(batch_size, context_len)

        # Build context sequence: [batch_size, seq_len-1, input_dim] where input_dim = KF + 1 + 1 + KF
        context_seqs = torch.cat(
            [
                state_ctx,
                act_ctx.unsqueeze(2),
                rew_ctx.unsqueeze(2),
                state_next_ctx,
            ],
            dim=2,
        )
        
        # Create Batch objects for training samples
        obs_b = Batch(self.device, [ig_to_data(g) for g in obs_list])
        obs_next_b = Batch(self.device, [ig_to_data(g) for g in obs_next_list])
        act_b = torch.tensor(act_list, device=self.device, dtype=torch.long)
        rew_b = torch.tensor(rew_list, device=self.device, dtype=torch.float32)
        done_b = torch.tensor(done_list, device=self.device, dtype=torch.float32)
        
        return context_seqs, obs_b, act_b, obs_next_b, rew_b, done_b
    
    