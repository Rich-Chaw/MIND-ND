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

class FinetuneBuffer():
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
        self.lcc_ratio_buffer = np.empty(buffer_size, dtype=np.float32)
        self.td_error_buffer = np.empty(buffer_size, dtype=np.float32)
        
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
            except:
                lcc_ratios.append(0.5)
        return np.array(lcc_ratios, dtype=np.float32)
    
    def add(self, obs_list, act_arr, obs_next_list, rew_arr, done_arr, td_errors=None):
        """Add experiences to buffer"""
        num_t = len(obs_list)
        lcc_ratios = self._compute_lcc_ratio(obs_list)
        
        idx = np.arange(self.ptr, self.ptr + num_t) % self.buffer_size
        
        self.obs_buffer[idx] = np.array([ig_to_data(g) for g in obs_list])
        self.act_buffer[idx] = act_arr
        self.obs_next_buffer[idx] = np.array([ig_to_data(g) for g in obs_next_list])
        self.rew_buffer[idx] = rew_arr
        self.done_buffer[idx] = done_arr
        self.lcc_ratio_buffer[idx] = lcc_ratios
        
        # Store additional data for TD-error and difference priorities
        if td_errors is not None:
            self.td_error_buffer[idx] = td_errors
        else:
            self.td_error_buffer[idx] = 0.0  # Default value       
        
        end = self.ptr + num_t
        if end >= self.buffer_size:
            self.full = True
        self.ptr = end % self.buffer_size
    
    def sample(self, batch_size, use_priority=True, priority=None, return_indices=False):
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
            return None, None, None, None, None
        
        if use_priority and available_size > 0:
            # Compute priorities
            if priority == "LCC":
                lcc_ratios = self.lcc_ratio_buffer[batch_inds]
                priorities = np.ones(available_size)  # Base priority
                priorities += lcc_ratios  # LCC bonus (higher LCC = higher priority)
                probs = priorities / priorities.sum()
            elif priority == "TDE":
                # TD-error based priority (absolute difference)
                td_errors = np.abs(self.td_error_buffer[batch_inds])
                priorities = td_errors + 1e-6  # Small epsilon to avoid zero priorities
                probs = priorities / priorities.sum()
            elif priority == "R":
                reward = self.rew_buffer[batch_inds]
                priorities = reward  # Base priority
                probs = priorities / priorities.sum()
            else:
                priorities = np.ones(available_size)
                probs = priorities / priorities.sum()

            # Sample with replacement based on priorities
            selected_indices = np.random.choice(batch_inds, size=min(batch_size, available_size), 
                                              p=probs, replace=True)
        else:
            # Uniform random sampling (like original ReplayBuffer)
            if self.full:
                selected_indices = (np.random.randint(1, self.buffer_size, size=batch_size) + self.ptr) % self.buffer_size
            else:
                selected_indices = np.random.randint(0, self.ptr, size=batch_size)
        
        obs = Batch(self.device, self.obs_buffer[selected_indices].tolist())
        obs_next = Batch(self.device, self.obs_next_buffer[selected_indices].tolist())
        act = torch.tensor(self.act_buffer[selected_indices], device=self.device, dtype=torch.long)
        rew = torch.tensor(self.rew_buffer[selected_indices], device=self.device, dtype=torch.float32)
        done = torch.tensor(self.done_buffer[selected_indices], device=self.device, dtype=torch.float32)
        
        if return_indices:
            return obs, act, obs_next, rew, done, selected_indices
        else:
            return obs, act, obs_next, rew, done
    
    def update_td_errors(self, indices, td_errors):
        """Update TD-errors for specific buffer indices"""
        if len(indices) == len(td_errors):
            self.td_error_buffer[indices] = td_errors
