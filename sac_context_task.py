import os
import tyro
import time
import torch
import random
import numpy as np
import igraph as ig
from typing import Optional
from collections import deque
from dataclasses import dataclass
from torch_scatter import scatter_add, scatter_mean
from datetime import datetime, timedelta
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from env import DismantleEnv
from networks.dismantle import load_sac_dismantler, load_sac_dismantler_with_hypernet
from utils import ReplayBuffer, PriorReplayBuffer, RolloutBuffer, Batch, validate, validate_with_context_task_encoder, ig_to_data, Discriminator, train_discriminator, DiscriminatorDataset
from task_encoder import TaskEncoder
from networks.gnn import GNN_ENCODER
import torch.nn.functional as F
import gc



@dataclass
class Args:
    use_tb: bool=False
    """record using tensorboard"""
    seed: int=0
    """random seed"""
    device: str='cuda:0'
    """the device to use"""
    gnn: str='rfgnn'
    num_envs: int=64
    """number of parallel environments,default 64"""
    total_steps: int=200000
    """number of training steps (transitions = steps*num_envs)"""
    buffer_size: int=20000
    """size of the replay buffer, default 2000000"""
    batch_size: int=64
    """batch size for updating network, default 512"""
    val_frequency: int=200
    """validation frequency, default 1000"""
    save_frequency: int=1000
    """save frequency, default 1000"""
    learning_starts: int= 2000
    """timestep to start learning, default 2000"""
    learning_rate: float=3e-4
    """learning rate for the policy and the Q networks, original 3e-4"""
    tau: float=1.0
    """target smoothing factor,default 1.0"""
    alpha: float=0.005
    """intensity of entropy regularization"""
    gamma: float=0.99
    """Discount factor"""
    num_updates: int=16
    """number of network updates at each step"""
    target_frequency: int=200
    """the frequency for updating the target networks"""

    ckpt_pth: Optional[str]=None
    """where ckeckpoint was saved"""

    num_features: int = 16
    """number of initial node features"""
    num_heads: int=4
    """number of message passings heads"""
    num_mps: int=6
    """number of message passings"""
    normalize: bool=True
    """apply instance normalization"""

    task_encoder_ckpt_pth: Optional[str]='saved/task_encoder/task_encoder_best.ckpt'
    """path to task encoder checkpoint"""
    context_sequence_length: int=10
    """sequence length for task context"""

    train_dir: str = 'graphs/train/100_200_SBM_DCSBM_LPA_COPY_6000'
    valid_dir: str = 'graphs/valid/100_200_SBM_DCSBM_LPA_COPY_60'

from finetune_utils import teacher_wrapper, teacher_step, compute_reward_shaping

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    run_path = f"sac_task_train_{time_string}"
    device = torch.device(args.device)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.use_tb:
        writer = SummaryWriter(f"runs/{run_path}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" 
                for key, value in vars(args).items()])),
        )

    print(f'Training starts at {time_string}')
    print(f'Device is {device}. Seed set to {args.seed}')

    # Setup environments based on finetuning method
    env = DismantleEnv(
        data_dir=args.train_dir, 
        batch_size=args.num_envs, 
        is_val=False, 
        seed=args.seed,
        remove_scc=False
    )
    
    env_val = DismantleEnv(
        data_dir=args.valid_dir, 
        batch_size=args.num_envs, 
        is_val=True, 
        seed=args.seed
    )
    
    # Use RolloutBuffer to store complete trajectories
    buffer = RolloutBuffer(device, buffer_size=args.buffer_size)
    
    # Load task encoder
    print(f"Loading task encoder from {args.task_encoder_ckpt_pth}...")
    task_encoder_ckpt = torch.load(args.task_encoder_ckpt_pth, map_location=device)
    latent_dim = task_encoder_ckpt['latent_dim']
    hidden_dim = task_encoder_ckpt['hidden_dim']
    state_emb_dim = args.num_features * args.num_mps  # KF
    input_dim = state_emb_dim + 1 + 1 + state_emb_dim  # state + action + reward + next_state
    
    # Load GNN encoder (used for encoding states in context)
    gnn_encoder = GNN_ENCODER['rfgnn'](args.num_features, args.num_heads, args.num_mps).to(device)
    gnn_encoder.load_state_dict(task_encoder_ckpt['gnn_encoder_state_dict'])
    gnn_encoder.eval()
    
    # Load task encoder
    task_encoder = TaskEncoder(input_dim, hidden_dim, latent_dim).to(device)
    task_encoder.load_state_dict(task_encoder_ckpt['task_encoder_state_dict'])
    task_encoder.eval()
    print(f"Task encoder loaded: latent_dim={latent_dim}, hidden_dim={hidden_dim}")
    
    # Track ongoing trajectories for each environment
    # Each trajectory is a list of transition dicts (like ppo_finetune_prior.py)
    # Will be initialized in main loop
    
    # Load pretrained network with task encoder support
    policy, qf1, qf2, qf1_target, qf2_target, shared_backbone = load_sac_dismantler_with_hypernet(
        args.num_features, args.num_heads, args.num_mps, args.gnn, device, latent_dim, args.ckpt_pth
    )
    print(f"Loaded checkpoint for finetuning: {args.ckpt_pth}")
    
    # Setup optimizers with appropriate learning rates
    lr = args.learning_rate
    print(f'Using learning rate: {lr}')
    
    # Only optimize trainable parameters
    # Include shared backbone and hypernetworks
    q_params = (
        list(shared_backbone.parameters()) + 
        list(qf1.hypernetwork.parameters()) + 
        list(qf2.hypernetwork.parameters())
    )
    policy_params = (
        list(shared_backbone.parameters()) + 
        list(policy.hypernetwork.parameters())
    )
    
    q_optimizer = torch.optim.Adam(q_params, lr=lr, eps=1e-4)
    policy_optimizer = torch.optim.Adam(policy_params, lr=lr, eps=1e-4)
    
    num_eps, num_updates = 0, 0
    auc_buffer = deque(maxlen=20)
    start_time = time.time()
    
    #### MAIN LOOP ####
    obs_list, _ = env.reset()
    # Initialize trajectories as lists of transition dicts
    ongoing_trajectories = [[] for _ in range(args.num_envs)]
    
    for global_step in range(args.total_steps): # args.num_envs transitions at each global step
        
        #### DISMANTLE ####
        if global_step<args.learning_starts and args.ckpt_pth==None:
            act_arr = env.sample_act()
        else:
            with torch.no_grad():
                # Get task embedding from ongoing trajectories (batched GNN: 2 forwards total instead of O(num_envs * context_len * 2))
                num_envs = len(obs_list)
                context_seqs_tensor = torch.zeros(
                    num_envs, args.context_sequence_length, input_dim,
                    device=device, dtype=torch.float32
                )
                flat_context_obs = []
                flat_context_obs_next = []
                flat_context_act = []
                flat_context_rew = []
                graph_to_env_row = []  # (env_idx, row_idx) for each flat graph

                for env_idx, traj in enumerate(ongoing_trajectories):
                    context_len = min(args.context_sequence_length, len(traj))
                    if context_len == 0:
                        continue
                    context_transitions = traj[-context_len:]
                    start_row = args.context_sequence_length - context_len
                    for t_idx, t in enumerate(context_transitions):
                        flat_context_obs.append(t['obs'])
                        flat_context_obs_next.append(t['obs_next'])
                        flat_context_act.append(t['act'])
                        flat_context_rew.append(t['rew'])
                        graph_to_env_row.append((env_idx, start_row + t_idx))

                if len(flat_context_obs) > 0:
                    # Two batched GNN passes for all context graphs (instead of per-env, per-timestep)
                    obs_ctx_b = Batch(device, [ig_to_data(g) for g in flat_context_obs])
                    emb_ctx = gnn_encoder(obs_ctx_b)
                    embed_dim = emb_ctx.shape[1] // 2
                    graph_emb_ctx = emb_ctx[:, embed_dim:]
                    state_ctx = scatter_mean(
                        graph_emb_ctx,
                        obs_ctx_b.batch_non_omni,
                        dim=0,
                        dim_size=obs_ctx_b.batch_size,
                    )

                    obs_next_ctx_b = Batch(device, [ig_to_data(g) for g in flat_context_obs_next])
                    emb_next_ctx = gnn_encoder(obs_next_ctx_b)
                    graph_emb_next_ctx = emb_next_ctx[:, embed_dim:]
                    state_next_ctx = scatter_mean(
                        graph_emb_next_ctx,
                        obs_next_ctx_b.batch_non_omni,
                        dim=0,
                        dim_size=obs_next_ctx_b.batch_size,
                    )

                    act_ctx = torch.tensor(flat_context_act, device=device, dtype=torch.float32)
                    rew_ctx = torch.tensor(flat_context_rew, device=device, dtype=torch.float32)
                    for graph_idx, (env_idx, row_idx) in enumerate(graph_to_env_row):
                        context_seqs_tensor[env_idx, row_idx, :state_emb_dim] = state_ctx[graph_idx]
                        context_seqs_tensor[env_idx, row_idx, state_emb_dim] = act_ctx[graph_idx]
                        context_seqs_tensor[env_idx, row_idx, state_emb_dim + 1] = rew_ctx[graph_idx]
                        context_seqs_tensor[env_idx, row_idx, state_emb_dim + 2 : state_emb_dim * 2 + 2] = state_next_ctx[graph_idx]

                z_batch = task_encoder.get_z(context_seqs_tensor)
                z_batch_last = z_batch[:, -1, :]

                act_arr, _ = policy.get_action(
                    Batch(device, [ig_to_data(g) for g in obs_list]),
                    z=z_batch_last,
                )
            act_arr = act_arr.detach().cpu().numpy()

        obs_next_list, rew_arr, done_arr, info_list = env.step(act_arr)
        
        # Store transitions in trajectories for each environment
        for env_idx in range(len(obs_list)):
            ongoing_trajectories[env_idx].append({
                'obs': obs_list[env_idx],
                'act': act_arr[env_idx],
                'rew': rew_arr[env_idx],
                'obs_next': obs_next_list[env_idx],
                'done': done_arr[env_idx]
            })
            
            # If episode finished, save trajectory and reset
            if done_arr[env_idx] and len(ongoing_trajectories[env_idx]) > 0:
                # Extract trajectory data
                traj = ongoing_trajectories[env_idx]
                traj_obs = [t['obs'] for t in traj]
                traj_act = [t['act'] for t in traj]
                traj_rew = [t['rew'] for t in traj]
                traj_obs_next = [t['obs_next'] for t in traj]
                traj_done = [t['done'] for t in traj]
                
                # Add to rollout buffer
                buffer.add_trajectory(traj_obs, traj_act, traj_rew, traj_obs_next, traj_done)
                
                # Reset trajectory for this environment
                ongoing_trajectories[env_idx] = []
        
        obs_next_list, _ = env.reset_async(done_arr)
        obs_list = obs_next_list

        for logger in info_list: # info_dict is empty unless some eps are finished
            num_eps += 1
            auc_buffer.append(logger.auc/logger.n_init)
        
        if global_step % args.val_frequency == 0 and global_step >= args.learning_starts:
            val_auc_list = validate_with_context_task_encoder(
                env_val,
                policy,
                task_encoder,
                gnn_encoder,
                args.context_sequence_length,
                input_dim,
                state_emb_dim,
                latent_dim,
                device,
            )[0]
            auc_val_avg = sum(val_auc_list)/len(val_auc_list)
            print(f'At step {global_step}, Avg. Validation AUC is {auc_val_avg:.4f}')
            
            if args.use_tb:
                writer.add_scalar("val/val_avg_auc", auc_val_avg, global_step)

        if (global_step+1) % args.save_frequency == 0 and global_step>=args.learning_starts:
            directory = os.path.join('saved', run_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                "shared_backbone_state_dict": shared_backbone.state_dict(),
                "policy_hypernet_state_dict": policy.hypernetwork.state_dict(),
                "q1_hypernet_state_dict": qf1.hypernetwork.state_dict(),
                "q2_hypernet_state_dict": qf2.hypernetwork.state_dict(),
                "qf1_target_hypernet_state_dict": qf1_target.hypernetwork.state_dict(),
                "qf2_target_hypernet_state_dict": qf2_target.hypernetwork.state_dict(),
                "latent_dim": latent_dim
            }, os.path.join(directory, f'{global_step}.ckpt'))
            
        if (global_step + 1) % 50 == 0: #log train AUC
            time_relative = str(timedelta(seconds=time.time() - start_time)).split('.')[0]
            auc_avg = sum(auc_buffer)/max(len(auc_buffer), 1)
            print(f"[{time_relative} | {num_eps} episodes | {global_step} steps] Avg. AUC = {auc_avg:.3f} (Priority Finetuning)")
            if args.use_tb:
                writer.add_scalar("train/AUC", auc_avg, global_step)


        if global_step > args.learning_starts:
            for update_idx in range(args.num_updates):
                # Sample sequences from rollout buffer with context
                # Returns context sequences and training transitions
                sample_result = buffer.sample_sequences_with_context(
                    args.batch_size, 
                    args.context_sequence_length + 1,  # seq_len: context + 1 training sample
                    gnn_encoder
                )
                
                if sample_result is None:
                    continue
                
                context_seqs, obs_b, act_b, obs_next_b, rew_b, done_b = sample_result
                
                # Get task embedding z from context sequences
                with torch.no_grad():
                    # context_seqs: [batch_size, seq_len-1, input_dim]
                    # Get task embedding for each context sequence
                    z_batch = task_encoder.get_z(context_seqs)  # [batch_size, seq_len-1, latent_dim]
                    # Use last timestep embedding (or mean over sequence)
                    z_batch_last = z_batch[:, -1, :]  # [batch_size, latent_dim]

                # ---------------  CRITIC Training--------------------------------
                with torch.no_grad():
                    _, logp_next_b = policy.get_action(obs_next_b, z=z_batch_last)
                    
                    qf1_next_b = qf1_target(obs_next_b, z=z_batch_last)
                    qf2_next_b = qf2_target(obs_next_b, z=z_batch_last)
                    qf_next_b = torch.min(qf1_next_b, qf2_next_b)-args.alpha*logp_next_b
                    
                    # use E[Q(s',a')|a'] instead of using MC
                    b = obs_next_b.batch[obs_next_b.non_omni_mask]
                    v_next_b = scatter_add(logp_next_b.exp()*qf_next_b, b, dim_size=obs_next_b.batch_size)
                    q_target_b = rew_b.flatten() + (1-done_b.flatten()) * args.gamma * v_next_b
                    
                    # Explicit cleanup of intermediate tensors
                    del logp_next_b, qf1_next_b, qf2_next_b, qf_next_b, v_next_b
                
                # use Q-values only for the taken actions
                act_b += obs_b.act_offsets
                q1_b = qf1(obs_b, z=z_batch_last).gather(0, act_b).flatten()
                q2_b = qf2(obs_b, z=z_batch_last).gather(0, act_b).flatten()
                
                # Apply importance sampling weights to Q-loss
                q1_loss = (mse_loss(q1_b, q_target_b, reduction='none')).mean()
                q2_loss = (mse_loss(q2_b, q_target_b, reduction='none')).mean()
                q_loss = q1_loss + q2_loss
                
                q_optimizer.zero_grad(); q_loss.backward(); q_optimizer.step()
                
                # ---------------------- ACTOR Training ----------------------------
                _, logp_b = policy.get_action(obs_b, z=z_batch_last)
                
                # Standard SAC actor loss
                with torch.no_grad():
                    qf1_b = qf1(obs_b, z=z_batch_last)
                    qf2_b = qf2(obs_b, z=z_batch_last)
                v_b = logp_b.exp()*(args.alpha*logp_b - torch.min(qf1_b, qf2_b))
                b = obs_b.batch[obs_b.non_omni_mask]
                
                # original: policy_loss = scatter_add(v_b, b, dim_size=obs_b.batch_size).mean()
                policy_loss_per_graph = scatter_add(v_b, b, dim_size=obs_b.batch_size)
                policy_loss = policy_loss_per_graph.mean()
                
                policy_optimizer.zero_grad(); policy_loss.backward(); policy_optimizer.step()
                
                # Explicit cleanup of training tensors
                del obs_b, act_b, obs_next_b, rew_b, done_b
                
                if args.use_tb and num_updates%args.target_frequency == 0:
                    writer.add_scalar("losses/q1(s,a)", q1_b.mean().item(), global_step)
                    writer.add_scalar("losses/q2(s,a)", q2_b.mean().item(), global_step)
                    writer.add_scalar("losses/q_loss", q_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/policy_loss", -policy_loss.item(), global_step)
                num_updates += 1
            
            if global_step%args.target_frequency == 0:
                # Update target networks (shared backbone and hypernetworks)
                for param, target_param in zip(shared_backbone.parameters(), qf1_target.shared_backbone.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.hypernetwork.parameters(), qf1_target.hypernetwork.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.hypernetwork.parameters(), qf2_target.hypernetwork.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Additional cleanup for large steps
            if global_step % 100 == 0:
                # Clear any lingering references
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                gc.collect()  # Double collection for stubborn references

# Example usage:
# python sac_task.py --use_tb --device cuda:0 