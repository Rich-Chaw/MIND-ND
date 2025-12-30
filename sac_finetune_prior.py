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
from torch_scatter import scatter_add
from datetime import datetime, timedelta
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from env import DismantleEnv
from networks.dismantle import load_dismantler
from utils import ReplayBuffer, FinetuneBuffer, Batch, validate, ig_to_data
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
    num_envs: int=64
    """number of parallel environments,default 64"""
    total_steps: int=20000
    """number of training steps (transitions = steps*num_envs), default 200000"""
    buffer_size: int=1000000
    """size of the replay buffer, default 2000000"""
    batch_size: int=64
    """batch size for updating network, default 512"""
    val_frequency: int=200
    """validation frequency, default 1000"""
    save_frequency: int=200
    """save frequency, default 1000"""
    learning_starts: int= 1000
    """timestep to start learning, default 2000"""
    learning_rate: float=3e-5
    """learning rate for the policy and the Q networks, original 3e-4"""
    tau: float=1.0
    """target smoothing factor"""
    alpha: float=0.005
    """intensity of entropy regularization"""
    gamma: float=0.99
    """Discount factor"""
    num_updates: int=12
    """number of network updates at each step, original 16"""
    target_frequency: int=100
    """the frequency for updating the target networks, default 200"""

    ckpt_pth: Optional[str]='saved/mind.ckpt'
    """where ckeckpoint was saved"""
    pretrained_ckpt_pth: Optional[str]='saved/mind.ckpt'

    '''options'''
    distillation: bool=True
    distill_coeff: float=0.8
    """distillation coefficient lambda for preventing forgetting (λ1)"""

    freeze_gnn: bool=False
    """freeze GNN encoder layers for transfer_learning """

    replay: bool=False
    replay_ratio: float=0.2
    """ratio of original data to mix with new data """
    
    # Teacher method settings
    teacher_method: str = 'spectral'
    """Teacher method: 'spectral', 'betweenness'"""
    teacher_distill: bool = True
    """Add teacher experience to buffer"""
    
    # Warmup settings
    warmup: bool = False
    """Enable warmup phase with teacher supervision"""
    warmup_steps: int = 1000
    """Number of warmup steps for teacher supervision"""
    warmup_lr: float = 1e-4
    """Learning rate for warmup phase"""
    warmup_soft: bool = False
    """Use soft teacher probabilities in warmup phase"""
    warmup_top_k: int = 5
    """Number of top-k actions to consider for soft teacher"""
    warmup_temperature: float = 1.0
    """Temperature for soft teacher distribution"""
 
    num_features: int = 16
    """number of initial node features"""
    num_heads: int=4
    """number of message passings heads"""
    num_mps: int=6
    """number of message passings"""
    normalize: bool=True
    """apply instance normalization"""

    # pretrain directories
    train_dir: str = 'graphs/train/100_200_ER_LPA_COPY_10000'
    valid_dir: str = 'graphs/valid'

    # Finetuning directories
    ft_train_dir: str = 'graphs/train/100_200_SBM_2000'
    ft_valid_dir: str = 'graphs/valid'

def teacher_wrapper(graph, teacher_method='spectral', max_steps=None):
    from baseline import spectral_dismantling, spectral_dismantling_advance, adaptive_betweenness, random_dismantling
    if teacher_method == 'spectral':
        try:
            removals = spectral_dismantling(graph, max_steps=max_steps)
        except Exception:
            removals = adaptive_betweenness(graph, max_steps=max_steps)
    elif teacher_method == 'spectral_advanced':
        removals = spectral_dismantling_advance(graph, max_steps=max_steps)
    elif teacher_method == 'betweenness':
        removals = adaptive_betweenness(graph, max_steps=max_steps)
    else:
        print(f"Unknown teacher method: {teacher_method}, using random")
        removals = random_dismantling(graph, max_steps=max_steps)
    return removals

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
            removals = teacher_wrapper(graph.copy(), teacher_method, max_steps=1)
            teacher_action = removals[0]
            act_arr[i] = teacher_action
            obs_next = graph.copy()
            obs_next.delete_vertices(teacher_action)

        # Compute reward and done flag (following env.step logic)
        if obs_next.vcount() == 0:
            lcc_size = 0.0
            done_flag = True
        else:
            components = obs_next.connected_components()
            lcc_size = max(components.sizes()) if len(components.sizes()) > 0 else 0
            lcc_ratio = lcc_size / max(graph['n_init'], 1)
            done_flag = (lcc_ratio < 0.1 or obs_next.ecount() == 0)
            rew_arr[i] = -lcc_size / max(graph['n_init'], 1)  # Negative LCC ratio as reward
            done_arr[i] = done_flag
            obs_next_list.append(obs_next)
            
    return act_arr, obs_next_list, rew_arr, done_arr


def get_soft_teacher_distribution_from_removals(graph, removals, temperature=1.0):
    """
    Generate soft teacher distribution based on pre-computed removals
    """
    if graph.vcount() <= 2:
        probs = np.ones(graph.vcount()) / graph.vcount()
        return probs
    try:
        # Initialize uniform distribution
        probs = np.zeros(graph.vcount())
        
        # Assign probabilities based on teacher ranking
        for rank, static_id in enumerate(removals):
            # Find current index of this node
            for j, v in enumerate(graph.vs):
                if v['static_id'] == static_id:
                    # Exponential decay based on rank (rank 0 = highest priority)
                    probs[j] = np.exp(-rank / temperature)
                    break
        
        # Normalize to probabilities
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            # Fallback to uniform if no valid actions found
            probs = np.ones(graph.vcount()) / graph.vcount()
            
    except Exception as e:
        print(f"Soft teacher distribution failed: {e}")
        probs = np.ones(graph.vcount()) / graph.vcount()
    return probs

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Ensure checkpoint path is provided for finetuning
    if args.pretrained_ckpt_pth is None:
        raise ValueError("Pretrained Checkpoint path (--pretained_ckpt_pth) must be provided for finetuning!")
    
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    run_path = f"finetune_{time_string}"
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

    print(f'Finetuning starts at {time_string}')
    print(f'Device is {device}. Seed set to {args.seed}')

    # Setup environments based on finetuning method
    if args.replay:
        # Load and mix original and new data
        from utils import load_g
        
        # Load new synthetic data
        new_graphs = [load_g(os.path.join(args.ft_train_dir, p), 
                            f'new_{os.path.splitext(os.path.basename(p))[0]}') 
                     for p in sorted(os.listdir(args.ft_train_dir))]
        
        # Load original training data (sample based on replay ratio)
        orig_files = sorted(os.listdir(args.train_dir))
        n_orig = int(len(new_graphs) * args.replay_ratio)
        selected_orig_files = random.sample(orig_files, min(n_orig, len(orig_files)))
        orig_graphs = [load_g(os.path.join(args.train_dir, p), 
                             f'orig_{os.path.splitext(os.path.basename(p))[0]}') 
                      for p in selected_orig_files]
        
        # Combine datasets
        mixed_graphs = new_graphs + orig_graphs
        print(f'Experience replay: {len(new_graphs)} new + {len(orig_graphs)} original = {len(mixed_graphs)} total graphs')
        
        env = DismantleEnv(
            graph_data=mixed_graphs,
            batch_size=args.num_envs,
            is_val=False,
            seed=args.seed,
            remove_scc=False
        )
    else:
        # Use only new synthetic data for distillation and transfer learning
        env = DismantleEnv(
            data_dir=args.ft_train_dir, 
            batch_size=args.num_envs, 
            is_val=False, 
            seed=args.seed,
            remove_scc=False
        )
    
    env_val = DismantleEnv(
        data_dir=args.ft_valid_dir, 
        batch_size=args.num_envs, 
        is_val=True, 
        seed=args.seed
    )
    
    buffer = FinetuneBuffer(args.buffer_size, device)

    # Load pretrained network 
    policy, qf1, qf2, qf1_target, qf2_target = load_dismantler(args.num_features, args.num_heads, args.num_mps, device, args.ckpt_pth)
    print(f"Loaded checkpoint for fineturning: {args.ckpt_pth}")
    if args.freeze_gnn:
        # Freeze GNN encoder layers, only train MLP
        for param in policy.graph_embedding.parameters():
            param.requires_grad = False
        for param in qf1.graph_embedding.parameters():
            param.requires_grad = False
        for param in qf2.graph_embedding.parameters():
            param.requires_grad = False
        for param in qf1_target.graph_embedding.parameters():
            param.requires_grad = False
        for param in qf2_target.graph_embedding.parameters():
            param.requires_grad = False
        
        frozen_params = sum(p.numel() for p in policy.graph_embedding.parameters())
        trainable_params = sum(p.numel() for p in policy.mlp.parameters())
        print(f'---- GNN encoder frozen: {frozen_params} parameters')
        print(f'---- MLP trainable: {trainable_params} parameters')
     

    policy_pretrain = None
    
    if args.distillation:
        # Create pretrain network (frozen for distillation)
        policy_pretrain, _, _, _, _ = load_dismantler(args.num_features, args.num_heads, args.num_mps, device, args.pretrained_ckpt_pth)
        
        # Freeze pretrain network parameters
        for param in policy_pretrain.parameters():
            param.requires_grad = False
        policy_pretrain.eval()
        
        print(f'Loaded pretrained checkpoint: {args.pretrained_ckpt_pth}')
        print(f'---- pretrain policy network frozen with {sum(p.numel() for p in policy_pretrain.parameters())} parameters')
        print(f'---- Teacher method: {args.teacher_method} (priority-based sampling)')
    
    
    # Setup optimizers with appropriate learning rates
    lr = args.learning_rate
    print(f'Using reduced learning rate: {lr}')
    
    # Only optimize trainable parameters
    q_params = [p for p in list(qf1.parameters()) + list(qf2.parameters()) if p.requires_grad]
    policy_params = [p for p in policy.parameters() if p.requires_grad]
    
    q_optimizer = torch.optim.Adam(q_params, lr=lr, eps=1e-4)
    policy_optimizer = torch.optim.Adam(policy_params, lr=lr, eps=1e-4)

    num_eps, num_updates = 0, 0
    auc_buffer = deque(maxlen=20)
    start_time = time.time()

    #### WARMUP PHASE ####
    if args.warmup:
        from baseline import ensure_static_id
        print(f"Starting warmup phase for {args.warmup_steps} steps...")
  
        # Pre-compute teacher trajectories for all training graphs
        teacher_trajectories = []
        
        for i, graph in enumerate(env.graph_data[:10]):
            if i % 100 == 0:
                print(f"  Processing graph {i}/{len(env.graph_data)}")
            
            if graph.vcount() <= 1:
                continue
            
            ensure_static_id(graph)
            # Get complete teacher solution for trajectory progression
            removals = teacher_wrapper(graph.copy(), args.teacher_method)
            trajectory = []
            temp_graph = graph.copy()
            
            for step, removal_static_id in enumerate(removals):
                vertex_idx =  [i for i, v in enumerate(temp_graph.vs) if v['static_id'] == removal_static_id][0]
                
                if args.warmup_soft:
                    # Generate soft teacher distribution using remaining removals
                    remaining_removals = removals[step:step+args.warmup_top_k]
                    soft_probs = get_soft_teacher_distribution_from_removals(
                        temp_graph, remaining_removals, args.warmup_temperature
                    )
                    # Store (state, soft_probs) pair
                    trajectory.append((temp_graph.copy(), soft_probs))
                else:
                    # Store (state, hard_action) pair
                    trajectory.append((temp_graph.copy(), vertex_idx)) 
                temp_graph.delete_vertices(vertex_idx)
            teacher_trajectories.append(trajectory)
        
        # Warmup training
        warmup_optimizer = torch.optim.Adam(policy_params, lr=args.warmup_lr, eps=1e-4)
        
        for warmup_step in range(args.warmup_steps):
            # Sample batch directly from teacher trajectories
            batch_states = []
            batch_targets = []
            for _ in range(args.batch_size):
                traj_idx = np.random.randint(len(teacher_trajectories))
                trajectory = teacher_trajectories[traj_idx]
                if len(trajectory) == 0:
                    continue
                # Randomly select a step from this trajectory
                step_idx = np.random.randint(len(trajectory))
                state, target = trajectory[step_idx]
                batch_states.append(state)
                batch_targets.append(target)
            
            if len(batch_states) == 0:
                continue
                
            # Convert to batch format
            obs_b = Batch(device, [ig_to_data(g) for g in batch_states])
            # Get student policy logits
            _, student_logp_b = policy.get_action(obs_b)
            
            if args.warmup_soft:
                # Soft teacher supervision with KL divergence
                batch_soft_probs = np.concatenate([target for target in batch_targets])  # Shape: [N]
                teacher_prob_b = torch.tensor(batch_soft_probs, device=device, dtype=torch.float32)
                
                # KL divergence loss: KL(teacher || student)
                warmup_loss = F.kl_div(
                    student_logp_b,  # Already log probabilities
                    teacher_prob_b, 
                    reduction='batchmean'
                )
            else:
                # Hard teacher supervision with negative log-likelihood
                batch_actions = np.array(batch_targets)
                teacher_acts_batch = torch.tensor(batch_actions, device=device, dtype=torch.long)
                # Add action offsets for proper indexing
                teacher_acts_batch += obs_b.act_offsets
                
                # Negative log-likelihood loss
                warmup_loss = -student_logp_b[teacher_acts_batch].mean()
            
            # Update student policy
            warmup_optimizer.zero_grad()
            warmup_loss.backward()
            warmup_optimizer.step()
            
            if warmup_step % 100 == 0:
                print(f"Warmup step {warmup_step}/{args.warmup_steps}, Loss: {warmup_loss.item():.4f}")
                if args.use_tb:
                    writer.add_scalar("warmup/loss", warmup_loss.item(), warmup_step)
        
        # Validate student performance after warmup
        val_auc_list = validate(env_val, policy)[0]
        auc_val_avg = sum(val_auc_list)/len(val_auc_list)
        print(f'After warmup, Avg. Validation AUC is {auc_val_avg:.4f}')
        if args.use_tb:
            writer.add_scalar("warmup/val_auc", auc_val_avg, args.warmup_steps)
        
        # Clear memory
        del teacher_trajectories
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    #### MAIN LOOP ####
    obs_list, _ = env.reset()
    for global_step in range(args.total_steps): # args.num_envs transitions at each global step
        
        #### DISMANTLE ####
        if global_step<args.learning_starts and args.ckpt_pth==None:
            act_arr = env.sample_act()
        else:
            with torch.no_grad():
                act_arr, _ = policy.get_action(Batch(device, [ig_to_data(g) for g in obs_list]))
            act_arr = act_arr.detach().cpu().numpy()

        obs_next_list, rew_arr, done_arr, info_list = env.step(act_arr)

        buffer.add(obs_list, act_arr, obs_next_list, rew_arr, done_arr, is_teacher=False)
        
        # When teacher_distill is active, also generate and add teacher experiences
        if args.teacher_distill:
            tc_act_arr, tc_obs_next_list, tc_rew_arr, tc_done_arr = teacher_step(
                obs_list, teacher_method=args.teacher_method
            )
            buffer.add(obs_list, tc_act_arr, tc_obs_next_list, tc_rew_arr, tc_done_arr, is_teacher=True)

        obs_next_list, _ = env.reset_async(done_arr)
        
        obs_list = obs_next_list

        for logger in info_list: # info_dict is empty unless some eps are finished
            num_eps += 1
            auc_buffer.append(logger.auc/logger.n_init)
        
        if global_step % args.val_frequency == 0 and global_step >= args.learning_starts:
            val_auc_list = validate(env_val, policy)[0]
            auc_val_avg = sum(val_auc_list)/len(val_auc_list)
            print(f'At step {global_step}, Avg. Validation AUC is {auc_val_avg:.4f}')
            
            if args.use_tb:
                writer.add_scalar("val/val_avg_auc", auc_val_avg, global_step)

        if (global_step+1) % args.save_frequency == 0 and global_step>=args.learning_starts:
            directory = os.path.join('saved', run_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({"policy_state_dict": policy.state_dict(), 
                "qf1_state_dict": qf1.state_dict(),
                "qf2_state_dict": qf2.state_dict(),
                "qf1_target_state_dict": qf1_target.state_dict(),
                "qf2_target_state_dict": qf2_target.state_dict()
                }, os.path.join(directory, f'{global_step}.ckpt'))
            
        if (global_step + 1) % 50 == 0:
            time_relative = str(timedelta(seconds=time.time() - start_time)).split('.')[0]
            auc_avg = sum(auc_buffer)/len(auc_buffer)
            print(f"[{time_relative} | {num_eps} episodes | {global_step} steps] Avg. AUC = {auc_avg:.3f} (Priority Finetuning)")
            if args.use_tb:
                writer.add_scalar("train/AUC", auc_avg, global_step)
            
        if global_step > args.learning_starts:
            for _ in range(args.num_updates):
                # Sample with priority-based sampling
                obs_b, act_b, obs_next_b, rew_b, done_b = buffer.sample(
                    args.batch_size, 
                    use_priority=True
                )
                
                # CRITIC Training
                with torch.no_grad():
                    _, logp_next_b = policy.get_action(obs_next_b)
                    
                    qf1_next_b = qf1_target(obs_next_b)
                    qf2_next_b = qf2_target(obs_next_b)
                    qf_next_b = torch.min(qf1_next_b, qf2_next_b)-args.alpha*logp_next_b
                    
                    # use E[Q(s',a')|a'] instead of using MC
                    b = obs_next_b.batch[obs_next_b.non_omni_mask]
                    v_next_b = scatter_add(logp_next_b.exp()*qf_next_b, b, dim_size=obs_next_b.batch_size)
                    q_target_b = rew_b.flatten() + (1-done_b.flatten()) * args.gamma * v_next_b
                
                # use Q-values only for the taken actions
                act_b += obs_b.act_offsets
                q1_b = qf1(obs_b).gather(0, act_b).flatten()
                q2_b = qf2(obs_b).gather(0, act_b).flatten()
                q_loss = mse_loss(q1_b, q_target_b) + mse_loss(q2_b, q_target_b)
                
                q_optimizer.zero_grad(); q_loss.backward(); q_optimizer.step()
                
                # ACTOR Training based on finetuning method
                _, logp_b = policy.get_action(obs_b)
                
                # Standard SAC actor loss
                with torch.no_grad():
                    qf1_b = qf1(obs_b)
                    qf2_b = qf2(obs_b)
                v_b = logp_b.exp()*(args.alpha*logp_b - torch.min(qf1_b, qf2_b))
                b = obs_b.batch[obs_b.non_omni_mask]
                policy_loss = scatter_add(v_b, b, dim_size=obs_b.batch_size).mean()
                
                if args.distillation:
                    # MIND teacher distillation (prevent forgetting)
                    with torch.no_grad():
                        _, pretrain_logp_b = policy_pretrain.get_action(obs_b)
                    
                    # KL divergence loss: KL(pretrain || student)
                    student_prob_b = logp_b.exp()
                    pretrain_prob_b = pretrain_logp_b.exp()
                    
                    # KL divergence per graph, then average
                    kl_loss_per_node = pretrain_prob_b * (pretrain_logp_b - logp_b)
                    kl_loss_per_graph = scatter_add(kl_loss_per_node, b, dim_size=obs_b.batch_size)
                    distill_loss = kl_loss_per_graph.mean()
                    
                    # Combined loss (no teacher distillation loss - handled by priority sampling)
                    enhanced_policy_loss = policy_loss + args.distill_coeff * distill_loss
                    
                else:
                    # For transfer_learning and experience_replay, use standard SAC loss
                    enhanced_policy_loss = policy_loss
                    distill_loss = torch.tensor(0.0, device=device)
                
                policy_optimizer.zero_grad(); enhanced_policy_loss.backward(); policy_optimizer.step()
                
                if args.use_tb and num_updates%args.target_frequency == 0:
                    writer.add_scalar("losses/q1(s,a)", q1_b.mean().item(), global_step)
                    writer.add_scalar("losses/q2(s,a)", q1_b.mean().item(), global_step)
                    writer.add_scalar("losses/q_loss", q_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/policy_loss", -policy_loss.item(), global_step)
                    writer.add_scalar("losses/enhanced_policy_loss", -enhanced_policy_loss.item(), global_step)
                    if args.distillation:
                        writer.add_scalar("losses/distill_loss", distill_loss.item(), global_step)
                    
                num_updates += 1
            
            if global_step%args.target_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Add after each training loop iteration
            if global_step % 100 == 0:  # Periodic cleanup
                torch.cuda.empty_cache()  # If using CUDA
                import gc
                gc.collect()
# Example usage:
# Priority-based Finetuning with Spectral Teacher:
# python sac_finetune_prior.py --use_tb --device cuda:0 --distillation --distill_coeff 0.8 --teacher_method betweenness --teacher_distill

# Option 1 - Priority Distillation (default):
# python sac_finetune_prior.py --distillation --distill_coeff 0.8

# Option 2 - Transfer Learning with Priority Buffer:
# python sac_finetune_prior.py --freeze_gnn 

# Option 3 - Experience Replay with Priority:
# python sac_finetune_prior.py --replay --replay_ratio 0.2

# Option 4 - Warmup with Hard Teacher Supervision:
# python sac_finetune_prior.py --warmup --warmup_steps 1000 --warmup_lr 1e-4 --teacher_method spectral

# Option 5 - Warmup with Soft Teacher Supervision:
# python sac_finetune_prior.py --warmup --warmup_soft --warmup_top_k 5 --warmup_temperature 1.0 --teacher_method spectral

# Background execution:
# nohup python -u sac_finetune_prior.py --use_tb --device cuda:0 --ckpt_pth saved/mind.ckpt --distillation --warmup --warmup_soft > finetune_prior.out 2>&1 &