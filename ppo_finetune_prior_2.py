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
from networks.dismantle import load_ppo_dismantler
from utils import ReplayBuffer, PriorReplayBuffer, RolloutBuffer, Batch, validate, validate_with_type_logging, ig_to_data, Discriminator, train_discriminator, DiscriminatorDataset
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
    """number of parallel environments"""
    total_steps: int=20000
    """number of training steps (transitions = steps*num_envs)"""
    # buffer_size: int=500000
    # """size of the replay buffer, default 2000000"""
    batch_size: int=4
    """batch size for updating network"""
    val_frequency: int=200
    """validation frequency"""
    save_frequency: int=200
    """save frequency"""
    # learning_starts: int= 2000
    # """timestep to start learning, default 2000"""
    learning_rate: float=3e-5
    """learning rate for the policy and the value networks"""
    gamma: float=0.99
    """Discount factor"""
    gae_lambda: float=0.95
    """GAE lambda parameter for advantage estimation"""
    num_updates: int=10
    """number of network updates at each trajectory"""
    
    # PPO specific parameters
    clip_epsilon: float=0.2
    """PPO clip epsilon for clipped objective"""
    value_coeff: float=0.5
    """Coefficient for value loss"""
    entropy_coeff: float=0.01
    """Coefficient for entropy bonus"""
    max_grad_norm: float=0.5
    """Maximum gradient norm for clipping"""

    ckpt_pth: Optional[str]=None
    """where ckeckpoint was saved"""
    pretrained_ckpt_pth: Optional[str]='saved/mind.ckpt'

    '''options'''
    distillation: bool=False
    distill_coeff: float=0.8
    """distillation coefficient lambda for preventing forgetting (λ1)"""

    freeze_gnn: bool=False
    """freeze GNN encoder layers for transfer_learning """

    replay: bool=False
    replay_ratio: float=0.2
    """ratio of original data to mix with new data """
    
    # Teacher method settings
    teacher_method: Optional[str] = None
    """Teacher method: 'spectral', 'betweenness'"""
    teacher_distill: bool = False
    """Add teacher experience to buffer"""
    
    # Priority sampling settings
    priority_type: Optional[str] = None
    """Priority type for sampling: 'LCC', 'TDE', 'R', 'TEACHER', 'DDPGfD'"""
    
    # demeonstration setting
    demonstrate: bool = False
    """Save demonstration in teacher buffer before training"""
    num_demos: int = 1000

    # Warmup settings
    warmup: bool = False
    """Enable warmup phase with teacher supervision"""
    warmup_steps: int = 200
    """Number of warmup steps for teacher supervision"""
    warmup_lr: float = 3e-5
    """Learning rate for warmup phase"""
    
    discriminator: bool = False

    # Reward shaping settings
    reward_shaping: bool = False
    """Enable reward shaping"""
    shaping_method: str = 'KL'
    """Reward shaping method: 'betweenness' or 'KL'"""
    shaping_decay_iters: int = 200
    """Number of steps to decay reward shaping coefficient"""
    shaping_coeff: float = 0.1
    """Initial reward shaping coefficient"""

    regularization: bool = False
    regular_coeff: float = 1e-5

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
    ft_train_dir: str = 'graphs/train/50_100_SBM_2000'
    ft_valid_dir: str = 'graphs/valid/valid_20260104'

from finetune_utils import teacher_wrapper, teacher_step, compute_reward_shaping, batch_to_igraphs

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Ensure checkpoint path is provided for finetuning
    if args.pretrained_ckpt_pth is None:
        raise ValueError("Pretrained Checkpoint path (--pretrained_ckpt_pth) must be provided for finetuning!")
    
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    run_path = f"ppo_finetune_prior_{time_string}"
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

    print(f'PPO Finetuning starts at {time_string}')
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
    
    buffer = RolloutBuffer(device)

    # Load pretrained network 
    policy, vf = load_ppo_dismantler(args.num_features, args.num_heads, args.num_mps, device, args.ckpt_pth)
    print(f"Loaded checkpoint for finetuning: {args.ckpt_pth}")
    
    # Store original freeze_gnn setting for later restoration
    original_freeze_gnn = args.freeze_gnn
    # Always freeze GNN during warmup phase (following guide.md step 3)
    if args.warmup:
        args.freeze_gnn = True
        args.demonstrate = True
        print("---- Freezing GNN during warmup phase (following guide.md)")
    
    if args.freeze_gnn:
        # Freeze GNN encoder layers, only train MLP
        for param in policy.graph_embedding.parameters():
            param.requires_grad = False
        for param in vf.graph_embedding.parameters():
            param.requires_grad = False
        
        frozen_params = sum(p.numel() for p in policy.graph_embedding.parameters())
        trainable_params = sum(p.numel() for p in policy.mlp.parameters())
        print(f'---- GNN encoder frozen: {frozen_params} parameters')
        print(f'---- MLP trainable: {trainable_params} parameters')
     
    if args.demonstrate:
        # Collect teacher demonstrations as full trajectories and store in rollout buffer
        teacher_buffer = RolloutBuffer(device)
        obs_list, _ = env.reset()
        env_trajs = [[] for _ in range(args.num_envs)]
        num_eps = 0
        while num_eps < args.num_demos:
            act_arr = []
            for graph in obs_list:
                removals = teacher_wrapper(graph, args.teacher_method, max_steps=1)
                act_arr.append(removals[0])
            
            obs_next_list, rew_arr, done_arr, info_list = env.step(np.array(act_arr))
            
            for env_idx in range(args.num_envs):
                env_trajs[env_idx].append({
                    'obs': obs_list[env_idx],
                    'act': act_arr[env_idx],
                    'rew': rew_arr[env_idx],
                    'obs_next': obs_next_list[env_idx],
                    'done': done_arr[env_idx]
                })
                
                if done_arr[env_idx] and len(env_trajs[env_idx]) > 0:
                    traj_obs = [t['obs'] for t in env_trajs[env_idx]]
                    traj_act = [t['act'] for t in env_trajs[env_idx]]
                    traj_rew = [t['rew'] for t in env_trajs[env_idx]]
                    traj_obs_next = [t['obs_next'] for t in env_trajs[env_idx]]
                    traj_done = [t['done'] for t in env_trajs[env_idx]]
                    
                    teacher_buffer.add_trajectory(traj_obs, traj_act, traj_rew, traj_obs_next, traj_done)
                    env_trajs[env_idx] = []
            
            obs_next_list, _ = env.reset_async(done_arr)
            obs_list = obs_next_list
            
            # Count finished episodes from env info
            for _ in info_list:
                num_eps += 1
        
        # Cleanup
        del act_arr, obs_next_list, rew_arr, done_arr
        print(f"Saved {teacher_buffer.size()} demonstrations in buffer")


    policy_pretrain = None
    if args.distillation:
        # Create pretrain network (frozen for distillation)
        policy_pretrain, _ = load_ppo_dismantler(args.num_features, args.num_heads, args.num_mps, device, args.pretrained_ckpt_pth)
        
        # Freeze pretrain network parameters
        for param in policy_pretrain.parameters():
            param.requires_grad = False
        policy_pretrain.eval()
        
        print(f'Loaded pretrained checkpoint: {args.pretrained_ckpt_pth}')
        print(f'---- pretrain policy network frozen with {sum(p.numel() for p in policy_pretrain.parameters())} parameters')
        print(f'---- Teacher method: {args.teacher_method} (priority-based sampling: {args.priority_type})')
    
    
    # Setup optimizers with appropriate learning rates
    lr = args.learning_rate
    print(f'Using reduced learning rate: {lr}')
    
    # Only optimize trainable parameters
    vf_params = [p for p in vf.parameters() if p.requires_grad]
    policy_params = [p for p in policy.parameters() if p.requires_grad]
    
    vf_optimizer = torch.optim.Adam(vf_params, lr=lr, eps=1e-4)
    policy_optimizer = torch.optim.Adam(policy_params, lr=lr, eps=1e-4)
    
    discriminator = Discriminator(2 * args.num_features * args.num_mps).to(device)
    print(f"Initialized discriminator with embedding size: {2 * args.num_features * args.num_mps}")

    num_eps, num_updates = 0, 0
    auc_buffer = deque(maxlen=20)
    start_time = time.time()

    #### WARMUP PHASE ####
    if args.warmup:
        warmup_vf_optimizer = torch.optim.Adam(vf_params, lr=args.warmup_lr, eps=1e-4)
        warmup_policy_optimizer = torch.optim.Adam(policy_params, lr=args.warmup_lr, eps=1e-4)
        
        print(f"Starting warmup phase for {args.warmup_steps} steps, warmup_lr = {args.warmup_lr}...")
        
        # Track best warmup checkpoint
        warmup_directory = os.path.join('saved', run_path, 'warmup')
        if not os.path.exists(warmup_directory):
            os.makedirs(warmup_directory)
        best_warmup_auc = float('inf')
        best_warmup_ckpt_path = None
        
        for warmup_step in range(args.warmup_steps):
            # Sample teacher trajectories for warmup (supervised)
            obs_b, act_b, rew_b, obs_next_b, done_b = teacher_buffer.sample(args.batch_size)

            if obs_b.batch_size == 0:
                continue
            
            # Value update (teacher targets)
            with torch.no_grad():
                v_next_b = vf(obs_next_b)
                v_target_b = rew_b.flatten() + (1 - done_b.flatten()) * args.gamma * v_next_b
            
            v_b = vf(obs_b)
            warmup_vf_loss = mse_loss(v_b, v_target_b)
            
            warmup_vf_optimizer.zero_grad()
            warmup_vf_loss.backward()
            warmup_vf_optimizer.step()

            # Policy supervised update (negative log likelihood of teacher actions)
            _, student_logp_b = policy.get_action(obs_b)
            teacher_acts_batch = act_b + obs_b.act_offsets
            warmup_policy_loss = -student_logp_b[teacher_acts_batch].mean()
            
            warmup_policy_optimizer.zero_grad()
            warmup_policy_loss.backward()
            warmup_policy_optimizer.step()
            
            if warmup_step % 10 == 0:
                print(f"Warmup step {warmup_step}/{args.warmup_steps}, Policy Loss: {warmup_policy_loss.item():.4f}, VF Loss: {warmup_vf_loss.item():.4f}")
                if args.use_tb:
                    writer.add_scalar("warmup/warmup_vf_loss", warmup_vf_loss.item(), warmup_step)
        
                # Validate student performance after warmup
                val_auc_list = validate_with_type_logging(env_val, policy)[0]
                auc_val_avg = sum(val_auc_list)/len(val_auc_list)
                print(f'Warmup step {warmup_step}/{args.warmup_steps}, Avg. Validation AUC is {auc_val_avg:.4f}')
                if args.use_tb:
                    writer.add_scalar("warmup/val_auc", auc_val_avg, warmup_step)
                
                # Save best warmup checkpoint (lower AUC is better)
                if auc_val_avg < best_warmup_auc:
                    best_warmup_auc = auc_val_avg
                    
                    # Remove previous best checkpoint if exists
                    if best_warmup_ckpt_path and os.path.exists(best_warmup_ckpt_path):
                        os.remove(best_warmup_ckpt_path)
                        print(f"Removed previous best warmup checkpoint: {best_warmup_ckpt_path}")
                    
                    # Save new best checkpoint
                    best_warmup_ckpt_path = os.path.join(warmup_directory, f'warmup_best_step_{warmup_step}_auc_{auc_val_avg:.4f}.ckpt')
                    torch.save({
                        "policy_state_dict": policy.state_dict(), 
                        "vf_state_dict": vf.state_dict(),
                        "warmup_step": warmup_step,
                        "best_auc": auc_val_avg
                    }, best_warmup_ckpt_path)
                    print(f"Saved new best warmup checkpoint: {best_warmup_ckpt_path} (AUC: {auc_val_avg:.4f})")
            
            # Cleanup batch tensors
            del obs_b, act_b, obs_next_b, rew_b, done_b
            del v_b, v_next_b, v_target_b, warmup_vf_loss
            del student_logp_b, teacher_acts_batch, warmup_policy_loss
        
        # Clear memory after warmup
        if 'val_auc_list' in locals():
            del val_auc_list
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Unfreeze GNN after warmup (following guide.md step 4)
        if args.warmup and not original_freeze_gnn:
            print("---- Unfreezing GNN after warmup phase")
            for param in policy.graph_embedding.parameters():
                param.requires_grad = True
            for param in vf.graph_embedding.parameters():
                param.requires_grad = True
            
            # Update optimizers to include newly unfrozen parameters
            vf_params = [p for p in vf.parameters() if p.requires_grad]
            policy_params = [p for p in policy.parameters() if p.requires_grad]
            vf_optimizer = torch.optim.Adam(vf_params, lr=lr, eps=1e-4)
            policy_optimizer = torch.optim.Adam(policy_params, lr=lr, eps=1e-4)
            
            unfrozen_params = sum(p.numel() for p in policy.graph_embedding.parameters())
            print(f'---- GNN encoder unfrozen: {unfrozen_params} parameters')

    #### MAIN LOOP ####
    obs_list, _ = env.reset()
    env_trajectories = [[] for _ in range(args.num_envs)]  # List of lists, one per environment
    for global_step in range(args.total_steps):
        #### DISMANTLE ####
        with torch.no_grad():
            act_arr, _ = policy.get_action(Batch(device, [ig_to_data(g) for g in obs_list]))
        act_arr = act_arr.detach().cpu().numpy()

        obs_next_list, rew_arr, done_arr, info_list = env.step(act_arr)

        # Apply reward shaping if enabled (following guide.md)
        if args.reward_shaping:
            # Compute decay factor: β(t) starts high and decays to 0
            decay_progress = min(global_step / args.shaping_decay_iters, 1.0)
            beta_t = args.shaping_coeff * (1.0 - decay_progress)
            
            if beta_t > 0.001:  # Only compute if coefficient is significant
                # Compute reward shaping using specified method
                rew_shaping = compute_reward_shaping(
                    obs_list, act_arr,
                    shaping_method=args.shaping_method,
                    policy=policy,
                    discriminator=discriminator,
                    teacher_method=args.teacher_method,
                    temperature=1.0,
                    device=device
                )
                
                # Apply shaping: R_total = R_LCC + β(t) * shaping_reward
                rew_arr = rew_arr + beta_t * rew_shaping
                
                # Log shaping coefficient periodically
                if global_step % 100 == 0:
                    if args.use_tb:
                        writer.add_scalar("shaping/beta_coefficient", beta_t, global_step)
                        writer.add_scalar(f"shaping/{args.shaping_method.lower()}_reward_avg", np.mean(rew_shaping), global_step)

        # Store transitions in trajectories for each environment
        for env_idx in range(args.num_envs):
            env_trajectories[env_idx].append({
                'obs': obs_list[env_idx],
                'act': act_arr[env_idx],
                'rew': rew_arr[env_idx],
                'obs_next': obs_next_list[env_idx],
                'done': done_arr[env_idx]
            })
            
            # If episode finished, save trajectory and reset
            if done_arr[env_idx] and len(env_trajectories[env_idx]) > 0:
                # Extract trajectory data
                traj_obs = [t['obs'] for t in env_trajectories[env_idx]]
                traj_act = [t['act'] for t in env_trajectories[env_idx]]
                traj_rew = [t['rew'] for t in env_trajectories[env_idx]]
                traj_obs_next = [t['obs_next'] for t in env_trajectories[env_idx]]
                traj_done = [t['done'] for t in env_trajectories[env_idx]]
                
                # Add to rollout buffer
                buffer.add_trajectory(traj_obs, traj_act, traj_rew, traj_obs_next, traj_done)

                # Reset trajectory for this environment
                env_trajectories[env_idx] = []

        obs_next_list, _ = env.reset_async(done_arr)
        obs_list = obs_next_list
            
        # Track completed episodes
        for logger in info_list: # info_dict is empty unless some eps are finished
            num_eps += 1
            auc_buffer.append(logger.auc/logger.n_init)
        
        if global_step % args.val_frequency == 0:
            val_auc_list = validate_with_type_logging(env_val, policy)[0]
            auc_val_avg = sum(val_auc_list)/len(val_auc_list)
            print(f'At step {global_step}, Avg. Validation AUC is {auc_val_avg:.4f}')
            
            if args.use_tb:
                writer.add_scalar("val/val_avg_auc", auc_val_avg, global_step)

        if (global_step+1) % args.save_frequency == 0:
            directory = os.path.join('saved', run_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({"policy_state_dict": policy.state_dict(), 
                "vf_state_dict": vf.state_dict()
                }, os.path.join(directory, f'{global_step}.ckpt'))
            
        if (global_step + 1) % 50 == 0: #log train AUC
            time_relative = str(timedelta(seconds=time.time() - start_time)).split('.')[0]
            auc_avg = sum(auc_buffer)/len(auc_buffer)
            print(f"[{time_relative} | {num_eps} episodes | {global_step} steps] Avg. AUC = {auc_avg:.3f} (PPO Priority Finetuning)")
            if args.use_tb:
                writer.add_scalar("train/AUC", auc_avg, global_step)

        if args.discriminator:
            # TODO: sample trajectorise and create dataset
            t_obs_b, t_act_b, t_rew_b, t_obs_next_b, t_done_b = teacher_buffer.sample(args.batch_size)
            s_obs_b, s_act_b, s_rew_b, s_obs_next_b, s_done_b = buffer.sample(args.batch_size)
            
            from finetune_utils import batch_to_igraphs
            graphs = batch_to_igraphs(obs_b)
            dataset = DiscriminatorDataset(graphs, act_b, labels, seed=args.seed, device=device)

            avg_loss = train_discriminator(
                dataset, 
                discriminator, 
                encoder = policy.graph_embedding,
                batch_size=64, 
                num_epochs=5, 
                lr=0.001
            )   
            if args.use_tb and avg_loss is not None:
                writer.add_scalar("discriminator/loss", avg_loss, global_step)
                
            print(f"Step {global_step}: Trained discriminator, avg loss: {avg_loss:.4f}")

        #### TRAINING: Sample trajectories and update networks ####
        if buffer.size() >= args.batch_size:
            sampled_trajectories = buffer.sample_trajectories(args.batch_size)

            if len(sampled_trajectories) == 0:
                continue

            all_advantages = []
            all_v_target = []
            all_old_logp_b = [] # To store old log_probs for the entire flattened batch

            for traj_data in sampled_trajectories:
                obs_b_traj = traj_data['obs']
                act_b_traj = traj_data['act']
                rew_b_traj = traj_data['rew']
                obs_next_b_traj = traj_data['obs_next']
                done_b_traj = traj_data['done']

                with torch.no_grad():
                    v_b_traj = vf(obs_b_traj).flatten()
                    v_next_b_traj = vf(obs_next_b_traj).flatten()
                    v_target_b_traj = torch.zeros_like(rew_b_traj, device=device)
                    advantages_traj = torch.zeros_like(rew_b_traj, device=device)
                    
                    last_advantage = 0
                    for t in reversed(range(len(rew_b_traj))):
                        # Calculate TD target (V_target = R + gamma * V_next * (1 - done))
                        # Detach v_next_b_traj and v_b_traj to avoid computing gradients through them for targets
                        next_value = v_next_b_traj[t] * (1 - done_b_traj[t])
                        v_target_b_traj[t] = rew_b_traj[t] + args.gamma * next_value

                        # Calculate TD-error (delta = V_target - V_current)
                        delta = v_target_b_traj[t] - v_b_traj[t]
                        
                        # Calculate GAE (A_t = delta_t + gamma * lambda * A_{t+1})
                        last_advantage = delta + args.gamma * args.gae_lambda * last_advantage * (1 - done_b_traj[t])
                        advantages_traj[t] = last_advantage
                
                # After calculating GAE for each trajectory, we need the old log_probs for the entire flattened batch
                # So we collect them here.
                with torch.no_grad():
                    _, old_logp_b_traj = policy.get_action(obs_b_traj)
                    
                all_advantages.append(advantages_traj)
                all_v_target.append(v_target_b_traj)
                all_old_logp_b.append(old_logp_b_traj)

            # Concatenate all trajectory data into single batches
            advantages = torch.cat(all_advantages)
            v_target_b = torch.cat(all_v_target)
            old_logp_b_flat = torch.cat(all_old_logp_b) # Flat log_probs

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # This part is tricky if Batch doesn't have a simple concat method.
            flat_obs_list = []
            flat_act_list = []
            # Assuming obs_b.graph_data_list holds the underlying graph data
            for traj_data in sampled_trajectories:
                flat_obs_list.extend(batch_to_igraphs(traj_data['obs']))
                flat_act_list.extend(traj_data['act'].cpu().numpy()) # Convert to numpy for concatenation

            obs_b = Batch(device, [ig_to_data(g) for g in flat_obs_list])
            act_b = torch.tensor(flat_act_list, device=device, dtype=torch.long)

            old_logp_b = old_logp_b_flat

            # Multiple updates on the same trajectory (PPO epochs)
            for update_idx in range(args.num_updates):
                # ---------------  VALUE NETWORK loss--------------------------------
                v_b_current = vf(obs_b)
                v_loss = mse_loss(v_b_current.flatten(), v_target_b.detach())
                
                # Add L2 regularization
                if args.regularization:
                    v_l2_reg = sum(torch.norm(p, p=2) ** 2 for p in vf.parameters() if p.requires_grad)
                    v_loss = v_loss + args.regular_coeff * v_l2_reg
                
                # ---------------------- ACTOR loss (PPO Clipped Objective) ----------------------------
                # Get current policy log probabilities
                _, logp_b = policy.get_action(obs_b)
            
                act_b_offset = act_b + obs_b.act_offsets
                # Compute importance sampling ratio
                ratio = torch.exp(logp_b[act_b_offset] - old_logp_b[act_b_offset])
                
                # PPO clipped objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
                policy_loss_per_graph = -torch.min(surr1, surr2)
                policy_loss = policy_loss_per_graph.mean()
                
                # Entropy bonus
                b = obs_b.batch[obs_b.non_omni_mask]
                entropy_per_node = -logp_b.exp() * logp_b
                entropy_per_graph = scatter_add(entropy_per_node, b, dim_size=obs_b.batch_size)
                entropy_bonus = entropy_per_graph.mean()
        
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
                    
                    # Combined loss
                    enhanced_policy_loss = policy_loss - args.entropy_coeff * entropy_bonus + args.distill_coeff * distill_loss
                    
                    # Explicit cleanup of distillation tensors
                    del pretrain_logp_b, student_prob_b, pretrain_prob_b, kl_loss_per_node, kl_loss_per_graph
                    
                else:
                    # For transfer_learning and experience_replay, use standard PPO loss
                    enhanced_policy_loss = policy_loss - args.entropy_coeff * entropy_bonus
                    distill_loss = torch.tensor(0.0, device=device)
                
                # Add L2 regularization
                if args.regularization:
                    policy_l2_reg = sum(torch.norm(p, p=2) ** 2 for p in policy.parameters() if p.requires_grad)
                    enhanced_policy_loss = enhanced_policy_loss + args.regular_coeff * policy_l2_reg
                
                # Update Value network and Policy network
                vf_optimizer.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(vf_params, args.max_grad_norm)
                vf_optimizer.step()

                policy_optimizer.zero_grad()
                enhanced_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm)
                policy_optimizer.step()
                
                # Logging (only on last update of first trajectory to avoid spam)
                if args.use_tb and update_idx == args.num_updates - 1:
                    writer.add_scalar("losses/v(s)", v_b_current.mean().item(), global_step)
                    writer.add_scalar("losses/v_loss", v_loss.item(), global_step)
                    writer.add_scalar("losses/policy_loss", -policy_loss.item(), global_step)
                    writer.add_scalar("losses/enhanced_policy_loss", -enhanced_policy_loss.item(), global_step)
                    writer.add_scalar("losses/entropy", entropy_bonus.item(), global_step)
                    writer.add_scalar("losses/advantages", advantages.mean().item(), global_step)
                    if args.distillation:
                        writer.add_scalar("losses/distill_loss", distill_loss.item(), global_step)
                
            # Cleanup trajectory tensors
            del obs_b, act_b, advantages, v_target_b, old_logp_b
            
            # Clear rollout buffer after training (on-policy: use data once)
            buffer.clear()
            
        # Additional cleanup for large steps
        if global_step % 50 == 0:
            # Clear any lingering references
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            gc.collect()  # Double collection for stubborn references

# Example usage:
# Priority-based Finetuning with Spectral Teacher:
# python ppo_finetune_prior.py --use_tb --device cuda:0 --distillation --distill_coeff 0.8 --teacher_method betweenness --teacher_distill

# Option 1 - LCC Priority Distillation (default):
# python ppo_finetune_prior.py --distillation --distill_coeff 0.8 --priority_type LCC

# Option 2 - TD-Error Priority Distillation:
# python ppo_finetune_prior.py --distillation --distill_coeff 0.8 --priority_type TDE

# Option 3 - DDPGfD Priority with Importance Sampling:
# python ppo_finetune_prior.py --distillation --priority_type DDPGfD

# Option 4 - Transfer Learning with Priority Buffer:
# python ppo_finetune_prior.py --freeze_gnn --priority_type TDE

# Option 5 - Experience Replay with Priority:
# python ppo_finetune_prior.py --replay --replay_ratio 0.2 --priority_type DDPGfD

# Option 6 - Warmup with Hard Teacher Supervision (GNN frozen during warmup):
# python ppo_finetune_prior.py --warmup --warmup_steps 1000 --warmup_lr 1e-4 --teacher_method spectral --priority_type LCC

# Option 7 - Reward Shaping with Betweenness Centrality:
# python ppo_finetune_prior.py --reward_shaping --shaping_method betweenness --shaping_coeff 0.1 --shaping_decay_iters 5000

# Option 8 - Reward Shaping with KL Divergence:
# python ppo_finetune_prior.py --reward_shaping --shaping_method KL --teacher_method spectral --shaping_coeff 0.1 --shaping_decay_iters 5000

# Background execution with DDPGfD:
# nohup python -u ppo_finetune_prior.py --use_tb --device cuda:0 --ckpt_pth saved/mind.ckpt --demonstrate --teacher_method betweenness --priority_type DDPGfD --regularization > finetune_PPO_DDPGfD.out 2>&1 &

