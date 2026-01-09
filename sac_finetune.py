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
from networks.dismantle import load_sac_dismantler
from utils import FinetuneBuffer, Batch, validate,validate_with_type_logging, ig_to_data
import torch.nn.functional as F



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
    buffer_size: int=500000
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
    target_frequency: int=200
    """the frequency for updating the target networks, default 200"""

    ckpt_pth: Optional[str]='saved/mind.ckpt'
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
    teacher_method: str = 'spectral'
    """Teacher method: 'spectral', 'betweenness'"""
    teacher_distill: bool = False
    """Use teacher method distillation"""
    teacher_coeff: float = 0.8
    """Coefficient for teacher method distillation loss (λ2)"""
    teacher_temperature: float = 2.0
    """Temperature for soft probability generation from teacher"""
    teacher_decay: bool = False
    """Enable linear decay for teacher coefficient"""
    decay_rate: float = 0.0001
    """Decay rate for teacher coefficient linear decay"""
    
    # Priority sampling settings
    priority_type: Optional[str] = None
    """Priority type for sampling: 'LCC', 'TDE', 'R'"""
    
    # Reward shaping settings
    reward_shaping: bool = False
    """Enable reward shaping with KL divergence between teacher and policy"""
    shaping_method: str = 'KL'
    """Reward shaping method: 'betweenness' or 'KL'"""
    shaping_decay_steps: int = 5000
    """Number of steps to decay reward shaping coefficient"""
    shaping_coeff: float = 0.1
    """Initial reward shaping coefficient"""
    
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
    # ft_train_dir: str = 'graphs/train/100_200_SBM_2000'
    # ft_train_dir: str = 'graphs/train/50_100_BR_1000'
    ft_train_dir: str = 'graphs/train/50_100_SBM_2000'
    ft_valid_dir: str = 'graphs/valid/valid_20260104'

from finetune_utils import compute_teacher_actions_on_demand, compute_reward_shaping

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
    policy, qf1, qf2, qf1_target, qf2_target = load_sac_dismantler(args.num_features, args.num_heads, args.num_mps, device, args.ckpt_pth)
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
        policy_pretrain, _, _, _, _ = load_sac_dismantler(args.num_features, args.num_heads, args.num_mps, device, args.pretrained_ckpt_pth)
        
        # Freeze pretrain network parameters
        for param in policy_pretrain.parameters():
            param.requires_grad = False
        policy_pretrain.eval()
        
        print(f'Loaded pretrained checkpoint: {args.pretrained_ckpt_pth}')
        print(f'---- pretrain policy network frozen with {sum(p.numel() for p in policy_pretrain.parameters())} parameters')
        
        if args.teacher_distill:
            print(f'---- Teacher method: {args.teacher_method}')
            print(f'---- Priority sampling: {args.priority_type}')
    
    
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

        # Apply reward shaping if enabled (following guide.md)
        if args.reward_shaping:
            # Compute decay factor: β(t) starts high and decays to 0
            decay_progress = min(global_step / args.shaping_decay_steps, 1.0)
            beta_t = args.shaping_coeff * (1.0 - decay_progress)
            
            if beta_t > 0.001:  # Only compute if coefficient is significant
                # Compute reward shaping using specified method
                rew_shaping = compute_reward_shaping(
                    obs_list, act_arr, 
                    shaping_method=args.shaping_method,
                    policy=policy,
                    teacher_method=args.teacher_method, 
                    temperature=args.teacher_temperature, 
                    device=device
                )
                rew_arr = rew_arr + beta_t * rew_shaping
                
                # Log shaping coefficient periodically
                if global_step % 100 == 0:
                    method_name = args.shaping_method.upper()
                    print(f"Step {global_step}: Reward shaping β(t) = {beta_t:.4f}, avg {method_name} reward = {np.mean(rew_shaping):.4f}")
                    if args.use_tb:
                        writer.add_scalar("shaping/beta_coefficient", beta_t, global_step)
                        writer.add_scalar(f"shaping/{args.shaping_method}_reward_avg", np.mean(rew_shaping), global_step)
    
        buffer.add(obs_list, act_arr, obs_next_list, rew_arr, done_arr)

        obs_next_list, _ = env.reset_async(done_arr)
        
        obs_list = obs_next_list

        for logger in info_list: # info_dict is empty unless some eps are finished
            num_eps += 1
            auc_buffer.append(logger.auc/logger.n_init)
        
        if global_step % args.val_frequency == 0 and global_step >= args.learning_starts:
            val_auc_list = validate_with_type_logging(env_val, policy)[0]
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
            print(f"[{time_relative} | {num_eps} episodes | {global_step} steps] Avg. AUC = {auc_avg:.3f} (Finetuning)")
            if args.use_tb:
                writer.add_scalar("train/AUC", auc_avg, global_step)
            
        if global_step > args.learning_starts:
            for _ in range(args.num_updates):
                # Sample from buffer with priority sampling
                samples = buffer.sample(
                    args.batch_size,
                    use_priority=True,
                    priority=args.priority_type,
                    return_indices=True
                )
                
                if len(samples) == 7:
                    obs_b, act_b, obs_next_b, rew_b, done_b, weights, sample_indices = samples
                else:
                    obs_b, act_b, obs_next_b, rew_b, done_b, weights = samples
                    sample_indices = None
                
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
                
                # Compute TD-errors for priority buffer update (use Q1 for simplicity)
                with torch.no_grad():
                    td_errors = torch.abs(q_target_b - q1_b).cpu().numpy()
                    # Update TD-errors in buffer
                    if sample_indices is not None:
                        buffer.update_td_errors(sample_indices, td_errors)
                
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
                
                enhanced_policy_loss = policy_loss
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

                    enhanced_policy_loss += args.distill_coeff*distill_loss
                    
                # Teacher method distillation
                if args.teacher_distill and num_updates%50 == 0:
                    teacher_logp_b = compute_teacher_actions_on_demand(obs_b, args.teacher_method, args.teacher_temperature, device)
                    
                    # Same KL divergence computation as MIND distillation
                    teacher_prob_b = teacher_logp_b.exp()
                    student_prob_b = logp_b.exp()
                    
                    # KL divergence per node, then per graph
                    teacher_kl_per_node = teacher_prob_b * (teacher_logp_b - logp_b)
                    teacher_kl_per_graph = scatter_add(teacher_kl_per_node, b, dim_size=obs_b.batch_size)
                    teacher_distill_loss = teacher_kl_per_graph.mean()

                    # Apply linear decay to teacher coefficient if enabled
                    if args.teacher_decay:
                        teacher_coeff = max(0, args.teacher_coeff - global_step * args.decay_rate)
                    
                    # Combined loss
                    enhanced_policy_loss += args.teacher_coeff * teacher_distill_loss
                
                policy_optimizer.zero_grad(); enhanced_policy_loss.backward(); policy_optimizer.step()
                
                if args.use_tb and num_updates%args.target_frequency == 0:
                    writer.add_scalar("losses/q1(s,a)", q1_b.mean().item(), global_step)
                    writer.add_scalar("losses/q2(s,a)", q2_b.mean().item(), global_step)
                    writer.add_scalar("losses/q_loss", q_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/enhanced_policy_loss", -enhanced_policy_loss.item(), global_step)
                    writer.add_scalar("losses/policy_loss", -policy_loss.item(), global_step)
                    if args.distillation:
                        writer.add_scalar("losses/distill_loss", distill_loss.item(), global_step)  # Backward compatibility
                    if args.teacher_distill:
                        writer.add_scalar("losses/teacher_distill_loss", teacher_distill_loss.item(), global_step)
                    
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
# Multi-Teacher Distillation with Spectral Method and Priority Sampling:
# python sac_finetune.py --use_tb --device cuda:0 --distillation --teacher_distill --teacher_method spectral --distill_coeff 0.5 --teacher_coeff 0.8 --teacher_temperature 2.0 --priority_type LCC
# python sac_finetune.py --use_tb --device cuda:0 --distillation --teacher_distill --teacher_method betweenness --distill_coeff 0.5 --teacher_coeff 0.8 --teacher_temperature 2.0 --priority_type TDE

# Option 1 - Distillation with LCC Priority (default):
# python sac_finetune.py --distillation --distill_coeff 0.5 --priority_type LCC

# Option 2 - Distillation with TD-Error Priority:
# python sac_finetune.py --distillation --distill_coeff 0.5 --priority_type TDE

# Option 3 - Transfer Learning with Priority Sampling:
# python sac_finetune.py --freeze_gnn --priority_type TDE

# Option 4 - Experience Replay with Priority:
# python sac_finetune.py --replay --replay_ratio 0.2 --priority_type LCC

# Option 5 - Reward Shaping with KL Divergence:
# python sac_finetune.py --reward_shaping --shaping_method KL --shaping_coeff 0.1 --shaping_decay_steps 5000 --teacher_method spectral --priority_type LCC

# Option 6 - Reward Shaping with Betweenness Centrality:
# python sac_finetune.py --reward_shaping --shaping_method betweenness --shaping_coeff 0.1 --shaping_decay_steps 5000 --priority_type LCC

# Background execution:
# nohup python -u sac_finetune.py --use_tb --device cuda:0 --ckpt_pth saved/mind.ckpt --distillation --priority_type LCC > finetune.out 2>&1 &