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
from utils import ReplayBuffer, PriorReplayBuffer, Batch, validate, validate_with_type_logging, ig_to_data, Discriminator, train_discriminator, DiscriminatorDataset
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
    gnn: str='mind'
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
    learning_rate: float=1e-4
    """learning rate for the policy and the Q networks, original 3e-4"""
    tau: float=1.0
    """target smoothing factor,default 1.0"""
    alpha: float=0.005
    """intensity of entropy regularization"""
    gamma: float=0.99
    """Discount factor"""
    num_updates: int=12
    """number of network updates at each step, original 16"""
    target_frequency: int=50
    """the frequency for updating the target networks, default 200"""

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
    """Save demonstation in teacher buffer before training"""
    num_demos: int = 1000

    # Warmup settings
    warmup: bool = False
    """Enable warmup phase with teacher supervision"""
    warmup_steps: int = 200
    """Number of warmup steps for teacher supervision"""
    warmup_lr: float = 3e-5
    """Learning rate for warmup phase"""
    
    discriminator: bool = False
    discriminator_train_frequency: int = 50

    # Reward shaping settings
    reward_shaping: bool = False
    """Enable reward shaping"""
    shaping_method: str = 'KL'
    """Reward shaping method: 'betweenness' or 'KL'"""
    shaping_decay_steps: int = 5000
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
    train_dir: str = 'graphs/train/100_200_ER_LPA_COPY_rw_10000'
    valid_dir: str = 'graphs/valid/valid'

    # Finetuning directories
    # ft_train_dir: str = 'graphs/train/50_100_SBM_2000'
    # ft_valid_dir: str = 'graphs/valid/50_100_SBM_30'
    ft_train_dir: str = 'graphs/train/100_200_ER_LPA_COPY_rw_10000'
    ft_valid_dir: str = 'graphs/valid/valid'

from finetune_utils import teacher_wrapper, teacher_step, compute_reward_shaping

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Ensure checkpoint path is provided for finetuning
    if args.pretrained_ckpt_pth is None:
        raise ValueError("Pretrained Checkpoint path (--pretained_ckpt_pth) must be provided for finetuning!")
    
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    run_path = f"finetune_prior_{time_string}"
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
    
    buffer = PriorReplayBuffer(args.buffer_size, device)

    # Load pretrained network 
    policy, qf1, qf2, qf1_target, qf2_target = load_sac_dismantler(args.num_features, args.num_heads, args.num_mps, args.gnn, device, args.ckpt_pth)
    print(f"Loaded checkpoint for fineturning: {args.ckpt_pth}")
    
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
     
    if args.demonstrate:
        obs_list, _ = env.reset()
        num_eps = 0
        while num_eps < args.num_demos:
            act_arr = []
            for graph in obs_list:
                removals = teacher_wrapper(graph, args.teacher_method, max_steps=1)
                teacher_action = removals[0]
                act_arr.append(teacher_action)
            
            obs_next_list, rew_arr, done_arr, info_list = env.step(np.array(act_arr))
                
            # Add to teacher buffer
            buffer.add(obs_list, act_arr, obs_next_list, rew_arr, done_arr, from_teacher=True,fixed=True)
            num_eps += len(info_list)

            obs_next_list, _ = env.reset_async(done_arr)
            obs_list = obs_next_list

        # Cleanup
        del act_arr, obs_next_list, rew_arr, done_arr
        
        # log buffer size
        print(f"Saved {buffer.ptr} transitions from demonstration in buffer")


    policy_pretrain = None
    if args.distillation:
        # Create pretrain network (frozen for distillation)
        policy_pretrain, _, _, _, _ = load_sac_dismantler(args.num_features, args.num_heads, args.num_mps,args.gnn, device, args.pretrained_ckpt_pth)
        
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
    q_params = [p for p in list(qf1.parameters()) + list(qf2.parameters()) if p.requires_grad]
    policy_params = [p for p in policy.parameters() if p.requires_grad]
    
    q_optimizer = torch.optim.Adam(q_params, lr=lr, eps=1e-4)
    policy_optimizer = torch.optim.Adam(policy_params, lr=lr, eps=1e-4)
    
    discriminator = Discriminator(2 * args.num_features * args.num_mps).to(device)
    print(f"Initialized discriminator with embedding size: {2 * args.num_features * args.num_mps}")

    num_eps, num_updates = 0, 0
    auc_buffer = deque(maxlen=20)
    start_time = time.time()

    #### WARMUP PHASE ####
    if args.warmup:
        warmup_q_optimizer = torch.optim.Adam(q_params, lr=args.warmup_lr, eps=1e-4)
        warmup_policy_optimizer = torch.optim.Adam(policy_params, lr=args.warmup_lr, eps=1e-4)
        
        print(f"Starting warmup phase for {args.warmup_steps} steps, warmup_lr = {args.warmup_lr}...")
        
        # Track best warmup checkpoint
        warmup_directory = os.path.join('saved', run_path, 'warmup')
        if not os.path.exists(warmup_directory):
            os.makedirs(warmup_directory)
        best_warmup_auc = float('inf')
        best_warmup_ckpt_path = None
        
        for warmup_step in range(args.warmup_steps):
            samples = buffer.sample(
                args.batch_size,
                return_indices=False
            )
                   
            if len(samples) == 6:
                obs_b, act_b, obs_next_b, rew_b, done_b, weights = samples
            else:
                # Fallback for old format
                obs_b, act_b, obs_next_b, rew_b, done_b = samples
                weights = torch.ones(obs_b.batch_size, device=device)
            
            # Update Q-networks (following main loop pattern)
            with torch.no_grad():
                _, logp_next_b = policy.get_action(obs_next_b)
                
                qf1_next_b = qf1_target(obs_next_b)
                qf2_next_b = qf2_target(obs_next_b)
                qf_next_b = torch.min(qf1_next_b, qf2_next_b) - args.alpha * logp_next_b
                
                # use E[Q(s',a')|a'] instead of using MC
                b = obs_next_b.batch[obs_next_b.non_omni_mask]
                v_next_b = scatter_add(logp_next_b.exp() * qf_next_b, b, dim_size=obs_next_b.batch_size)
                q_target_b = rew_b.flatten() + (1 - done_b.flatten()) * args.gamma * v_next_b
                
                # Explicit cleanup of intermediate tensors
                del logp_next_b, qf1_next_b, qf2_next_b, qf_next_b, v_next_b
            
            # use Q-values only for the taken actions
            act_b_offset = act_b + obs_b.act_offsets
            q1_b = qf1(obs_b).gather(0, act_b_offset).flatten()
            q2_b = qf2(obs_b).gather(0, act_b_offset).flatten()
            warmup_q_loss = mse_loss(q1_b, q_target_b) + mse_loss(q2_b, q_target_b)
            
            # Update Q-networks
            warmup_q_optimizer.zero_grad()
            warmup_q_loss.backward()
            warmup_q_optimizer.step()

            # Update policy
            # Hard teacher supervision with negative log-likelihood
            _, student_logp_b = policy.get_action(obs_b)
            
            # Convert actions to proper format for indexing
            teacher_acts_batch = act_b + obs_b.act_offsets
            
            # Negative log-likelihood loss
            warmup_policy_loss = -student_logp_b[teacher_acts_batch].mean()
            
            # Update student policy
            warmup_policy_optimizer.zero_grad()
            warmup_policy_loss.backward()
            warmup_policy_optimizer.step()
            
            # Update target networks periodically
            if warmup_step % args.target_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            
            if warmup_step % 10 == 0:
                # print(f"Warmup step {warmup_step}/{args.warmup_steps}, Policy Loss: {warmup_policy_loss.item():.4f}")
                print(f"Warmup step {warmup_step}/{args.warmup_steps}, Policy Loss: {warmup_policy_loss.item():.4f}, Q Loss: {warmup_q_loss.item():.4f}")
                if args.use_tb:
                    # writer.add_scalar("warmup/policy_loss", warmup_policy_loss.item(), warmup_step)
                    writer.add_scalar("warmup/warmup_q_loss", warmup_q_loss.item(), warmup_step)
        
                # Validate student performance after warmup
                val_auc_list = validate(env_val, policy)[0]
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
                        "qf1_state_dict": qf1.state_dict(),
                        "qf2_state_dict": qf2.state_dict(),
                        "qf1_target_state_dict": qf1_target.state_dict(),
                        "qf2_target_state_dict": qf2_target.state_dict(),
                        "warmup_step": warmup_step,
                        "best_auc": auc_val_avg
                    }, best_warmup_ckpt_path)
                    print(f"Saved new best warmup checkpoint: {best_warmup_ckpt_path} (AUC: {auc_val_avg:.4f})")
            
            # Cleanup batch tensors
            del obs_b, act_b, obs_next_b, rew_b, done_b
            del act_b_offset, q1_b, q2_b, q_target_b, warmup_q_loss
            del student_logp_b, teacher_acts_batch, warmup_policy_loss
        
        # Clear memory after warmup (keep teacher_trajectories for future development)
        if 'val_auc_list' in locals():
            del val_auc_list
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Unfreeze GNN after warmup (following guide.md step 4)
        if args.warmup and not original_freeze_gnn:
            print("---- Unfreezing GNN after warmup phase")
            for param in policy.graph_embedding.parameters():
                param.requires_grad = True
            for param in qf1.graph_embedding.parameters():
                param.requires_grad = True
            for param in qf2.graph_embedding.parameters():
                param.requires_grad = True
            for param in qf1_target.graph_embedding.parameters():
                param.requires_grad = True
            for param in qf2_target.graph_embedding.parameters():
                param.requires_grad = True
            
            # Update optimizers to include newly unfrozen parameters
            q_params = [p for p in list(qf1.parameters()) + list(qf2.parameters()) if p.requires_grad]
            policy_params = [p for p in policy.parameters() if p.requires_grad]
            q_optimizer = torch.optim.Adam(q_params, lr=lr, eps=1e-4)
            policy_optimizer = torch.optim.Adam(policy_params, lr=lr, eps=1e-4)
            
            unfrozen_params = sum(p.numel() for p in policy.graph_embedding.parameters())
            print(f'---- GNN encoder unfrozen: {unfrozen_params} parameters')

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

        buffer.add(obs_list, act_arr, obs_next_list, rew_arr, done_arr, from_teacher=False)
        
        # When teacher_distill is active, also generate and add teacher experiences
        if args.teacher_distill:
            tc_act_arr, tc_obs_next_list, tc_rew_arr, tc_done_arr = teacher_step(
                obs_list, teacher_method=args.teacher_method
            )
            buffer.add(obs_list, tc_act_arr, tc_obs_next_list, tc_rew_arr, tc_done_arr, from_teacher=True) 
            # Explicit cleanup of teacher data
            del tc_act_arr, tc_obs_next_list, tc_rew_arr, tc_done_arr

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
            
        if (global_step + 1) % 50 == 0: #log train AUC
            time_relative = str(timedelta(seconds=time.time() - start_time)).split('.')[0]
            auc_avg = sum(auc_buffer)/len(auc_buffer)
            print(f"[{time_relative} | {num_eps} episodes | {global_step} steps] Avg. AUC = {auc_avg:.3f} (Priority Finetuning)")
            if args.use_tb:
                writer.add_scalar("train/AUC", auc_avg, global_step)

        if args.discriminator and (global_step + 1) % args.discriminator_train_frequency == 0 and global_step > args.learning_starts:
            # Sample experiences with tags to identify teacher vs student
            samples = buffer.sample(2000, return_tags=True)
            if samples is not None and len(samples) >= 7:
                obs_b, act_b, obs_next_b, rew_b, done_b, weights, tags = samples
                # Convert tags to binary labels (1=teacher, 0=student)
                labels = tags.cpu().numpy().astype(np.float32)
                # Create dataset for discriminator training
                from finetune_utils import batch_to_igraphs
                graphs = batch_to_igraphs(obs_b)
                dataset = DiscriminatorDataset(graphs, act_b, labels, seed=args.seed, device=device)
                
                # Train discriminator
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
                
                # Cleanup
                del obs_b, act_b, obs_next_b, rew_b, done_b, weights, tags
                del labels, dataset

        if global_step > args.learning_starts:
            for update_idx in range(args.num_updates):
                samples = buffer.sample(
                    args.batch_size,
                    priority=args.priority_type,
                    return_indices=True
                )
                
                if len(samples) == 7:
                    obs_b, act_b, obs_next_b, rew_b, done_b, weights, sample_indices = samples
                elif len(samples) == 6:
                    obs_b, act_b, obs_next_b, rew_b, done_b, weights = samples

                # ---------------  CRITIC Training--------------------------------
                with torch.no_grad():
                    _, logp_next_b = policy.get_action(obs_next_b)
                    
                    qf1_next_b = qf1_target(obs_next_b)
                    qf2_next_b = qf2_target(obs_next_b)
                    qf_next_b = torch.min(qf1_next_b, qf2_next_b)-args.alpha*logp_next_b
                    
                    # use E[Q(s',a')|a'] instead of using MC
                    b = obs_next_b.batch[obs_next_b.non_omni_mask]
                    v_next_b = scatter_add(logp_next_b.exp()*qf_next_b, b, dim_size=obs_next_b.batch_size)
                    q_target_b = rew_b.flatten() + (1-done_b.flatten()) * args.gamma * v_next_b
                    
                    # Explicit cleanup of intermediate tensors
                    del logp_next_b, qf1_next_b, qf2_next_b, qf_next_b, v_next_b
                
                # use Q-values only for the taken actions
                act_b += obs_b.act_offsets
                q1_b = qf1(obs_b).gather(0, act_b).flatten()
                q2_b = qf2(obs_b).gather(0, act_b).flatten()
                
                # Apply importance sampling weights to Q-loss
                q1_loss = (weights * mse_loss(q1_b, q_target_b, reduction='none')).mean()
                q2_loss = (weights * mse_loss(q2_b, q_target_b, reduction='none')).mean()
                q_loss = q1_loss + q2_loss
                
                # Add L2 regularization
                if args.regularization:
                    q_l2_reg = sum(torch.norm(p, p=2) ** 2 for p in qf1.parameters() if p.requires_grad) + \
                               sum(torch.norm(p, p=2) ** 2 for p in qf2.parameters() if p.requires_grad)
                    q_loss = q_loss + args.regular_coeff * q_l2_reg
                
                # Compute TD-errors for priority buffer update (use Q1 for simplicity)
                with torch.no_grad():
                    td_errors = torch.abs(q_target_b - q1_b).cpu().numpy()
                    # Update TD-errors in the appropriate buffer
                    if sample_indices is not None:
                        buffer.update_td_errors(sample_indices, td_errors)
                
                q_optimizer.zero_grad(); q_loss.backward(); q_optimizer.step()
                
                # ---------------------- ACTOR Training ----------------------------
                _, logp_b = policy.get_action(obs_b)
                
                # Standard SAC actor loss
                with torch.no_grad():
                    qf1_b = qf1(obs_b)
                    qf2_b = qf2(obs_b)
                v_b = logp_b.exp()*(args.alpha*logp_b - torch.min(qf1_b, qf2_b))
                b = obs_b.batch[obs_b.non_omni_mask]
                
                # original: policy_loss = scatter_add(v_b, b, dim_size=obs_b.batch_size).mean()
                policy_loss_per_graph = scatter_add(v_b, b, dim_size=obs_b.batch_size)
                policy_loss = (weights * policy_loss_per_graph).mean()
                
                # Compute policy gradient norm for DDPGfD priority update (after backward pass)
                if sample_indices is not None:
                    # Capture gradient norm after backward pass (more efficient)
                    policy_grad_norm = 0.0
                    for param in policy.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            policy_grad_norm += param_norm.item() ** 2
                    policy_grad_norm = policy_grad_norm ** (1. / 2)
                    
                    # Update policy gradient norms in buffer
                    policy_grad_norms = np.full(len(sample_indices), policy_grad_norm, dtype=np.float32)
                    buffer.update_policy_grad_norms(sample_indices, policy_grad_norms)
                
                if args.distillation:
                    # MIND teacher distillation (prevent forgetting)
                    with torch.no_grad():
                        _, pretrain_logp_b = policy_pretrain.get_action(obs_b)
                    
                    # KL divergence loss: KL(pretrain || student)
                    student_prob_b = logp_b.exp()
                    pretrain_prob_b = pretrain_logp_b.exp()
                    
                    # KL divergence per graph, then average with importance sampling weights
                    kl_loss_per_node = pretrain_prob_b * (pretrain_logp_b - logp_b)
                    kl_loss_per_graph = scatter_add(kl_loss_per_node, b, dim_size=obs_b.batch_size)
                    distill_loss = (weights * kl_loss_per_graph).mean()
                    
                    # Combined loss (no teacher distillation loss - handled by priority sampling)
                    enhanced_policy_loss = policy_loss + args.distill_coeff * distill_loss
                    
                    # Explicit cleanup of distillation tensors
                    del pretrain_logp_b, student_prob_b, pretrain_prob_b, kl_loss_per_node, kl_loss_per_graph
                    
                else:
                    # For transfer_learning and experience_replay, use standard SAC loss
                    enhanced_policy_loss = policy_loss
                    distill_loss = torch.tensor(0.0, device=device)
                    
                # Add L2 regularization
                if args.regularization:
                    policy_l2_reg = sum(torch.norm(p, p=2) ** 2 for p in policy.parameters() if p.requires_grad)
                    enhanced_policy_loss = enhanced_policy_loss + args.regular_coeff * policy_l2_reg
                
                policy_optimizer.zero_grad(); enhanced_policy_loss.backward(); policy_optimizer.step()
                
                # Explicit cleanup of training tensors
                del obs_b, act_b, obs_next_b, rew_b, done_b
                if sample_indices is not None:
                    del sample_indices
                
                if args.use_tb and num_updates%args.target_frequency == 0:
                    writer.add_scalar("losses/q1(s,a)", q1_b.mean().item(), global_step)
                    writer.add_scalar("losses/q2(s,a)", q2_b.mean().item(), global_step)
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

            # Additional cleanup for large steps
            if global_step % 100 == 0:
                # Clear any lingering references
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                gc.collect()  # Double collection for stubborn references
# Example usage:
# Priority-based Finetuning with Spectral Teacher:
# python sac_finetune_prior.py --use_tb --device cuda:0 --distillation --distill_coeff 0.8 --teacher_method betweenness --teacher_distill

# Option 1 - LCC Priority Distillation (default):
# python sac_finetune_prior.py --distillation --distill_coeff 0.8 --priority_type LCC

# Option 2 - TD-Error Priority Distillation:
# python sac_finetune_prior.py --distillation --distill_coeff 0.8 --priority_type TDE

# Option 3 - DDPGfD Priority with Importance Sampling:
# python sac_finetune_prior.py --distillation --priority_type DDPGfD --DDPGfD_lambda 1.0 --DDPGfD_teacher_bonus 1.0 --DDPGfD_beta 0.4

# Option 4 - Transfer Learning with Priority Buffer:
# python sac_finetune_prior.py --freeze_gnn --priority_type TDE

# Option 5 - Experience Replay with Priority:
# python sac_finetune_prior.py --replay --replay_ratio 0.2 --priority_type DDPGfD

# Option 6 - Warmup with Hard Teacher Supervision (GNN frozen during warmup):
# python sac_finetune_prior.py --warmup --warmup_steps 1000 --warmup_lr 1e-4 --teacher_method spectral --priority_type LCC

# Option 7 - Reward Shaping with Betweenness Centrality:
# python sac_finetune_prior.py --reward_shaping --shaping_method betweenness --shaping_coeff 0.1 --shaping_decay_steps 5000

# Option 8 - Reward Shaping with KL Divergence:
# python sac_finetune_prior.py --reward_shaping --shaping_method KL --teacher_method spectral --shaping_coeff 0.1 --shaping_decay_steps 5000


# Background execution with DDPGfD:
# nohup python -u sac_finetune_prior.py --use_tb --device cuda:0 --ckpt_pth saved/mind.ckpt --demonstrate --teacher_method betweenness --priority_type DDPGfD --regularization > finetune_DDPGfD.out 2>&1 &