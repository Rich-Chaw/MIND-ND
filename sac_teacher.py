import os
import tyro
import time
import torch
import random
import numpy as np
import igraph as ig
from typing import Optional, List
from dataclasses import dataclass, field
from collections import deque
from torch_scatter import scatter_add, scatter_max
from datetime import datetime, timedelta
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from env import DismantleEnv
from networks.dismantle import load_sac_dismantler
from utils import ReplayBuffer, PriorReplayBuffer, Batch, validate, validate, ig_to_data
import torch.nn.functional as F
import gc
import json



@dataclass
class Args:
    use_tb: bool=False
    """record using tensorboard"""
    seed: int=0
    """random seed"""
    device: str='cuda:0'
    """the device to use"""
    gnn: str='hgnn_v3'
    num_envs: int=64
    """number of parallel environments,default 64"""
    total_steps: int=50000
    """number of training steps (transitions = steps*num_envs), default 200000"""
    buffer_size: int=1500000
    """size of the replay buffer, default 2000000"""
    batch_size: int=64
    """batch size for updating network, default 512"""
    val_frequency: int=200
    """validation frequency, default 1000"""
    save_frequency: int=1000
    """save frequency, default 1000"""
    learning_starts: int= 1000
    """timestep to start learning, default 2000"""
    learning_rate: float=3e-4
    """learning rate for the policy and the Q networks, original 3e-4"""
    tau: float=1.0
    """target smoothing factor,default 1.0"""
    alpha: float=0.005
    """intensity of entropy regularization"""
    alpha_adaptive: bool = False
    """If True, tune alpha by minimizing J(alpha) = E[-alpha*log pi(a|s) - alpha*H_bar] in log-space."""
    target_entropy: float = -1.0
    """Target entropy H_bar for adaptive alpha (used when alpha_adaptive is True)."""
    gamma: float=0.99
    """Discount factor"""
    num_updates: int=12
    """number of network updates at each step, original 16"""
    target_frequency: int=200
    """the frequency for updating the target networks, default 200"""

    # ckpt_pth: Optional[str]='./saved/hgnn_v3/sac_teacher_degree_20260318_150529/24999.ckpt'
    ckpt_pth: Optional[str]=None
    """where ckeckpoint was saved"""
    
    reward_type: Optional[int] = 0
    """0: -LCC_t/N   1:(LCC_t-1 - LCC_t) / N """

    # Teacher method settings
    teacher_method: Optional[str] = None
    """Teacher method: 'spectral', 'betweenness', 'CI'"""
    teacher_distill: bool = False
    """Add teacher experience to buffer"""
    
    # Priority sampling settings
    priority_type: Optional[str] = None
    """Priority type for sampling: 'LCC', 'TDE', 'R', 'TEACHER', 'DDPGfD'"""
    
    # demeonstration setting
    demo: bool = False
    """Save demonstation in teacher buffer before training"""
    demo_dir: List[str] = field(default_factory=lambda: [
        # 'graphs/demo/100_200_LFR_3000',
        # 'graphs/demo/100_200_LPA_Copy_ER_3000'
        # 'graphs/train/100_150_SBM_DCSBM_LPA_COPY_ER_6000'
        # 'graphs/train/100_200_BA_5000'
        'graphs/train/100_200_DCSBM_5000'
    ])
    num_demos: int = 1000
    """Number of demonstrations to save"""
    bc: bool = False
    """Use behavior cloning to train the policy (supervised on teacher demos)"""
    bc_steps: int = 10000
    """Number of gradient steps for behavior cloning when training from buffer"""
    bc_supervise_q: bool = True
    """If True, also train Q1/Q2 on demos (TD + large-margin + L2), aligned with dqn_teacher pretrain."""
    bc_margin_value: float = 0.8
    """Margin l for expert-action large-margin loss on Q during BC"""
    bc_margin_coeff: float = 1.0
    """Weight λ1 for large-margin Q loss during BC"""
    bc_l2_coeff: float = 1e-5
    """Weight λ2 for L2 regularization on Q1/Q2 during BC"""

    # Reward shaping settings
    reward_shaping: bool = False
    """Enable reward shaping"""
    shaping_method: str = 'KL'
    """Reward shaping method: 'betweenness' or 'KL'"""
    shaping_decay: bool = False
    shaping_decay_steps: int = 10000
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
    positional_encoding: Optional[str]=None
    """node initial features: None = all ones, 'RW' = random walk return-probability encoding"""
    handcrafted_features: bool = False
    """if True, use 5 handcrafted features: degree, avg_degree_neighbor, local_clustering, k_core, 1 (num_features forced to 5)"""

    # dataset directories
    train_dir: List[str] = field(default_factory=lambda: [
        # 'graphs/train/100_200_BA_5000',
        # 'graphs/train/100_200_LFR_5000',
        # 'graphs/train/100_200_LPA_Copy_ER_5000'
        'graphs/train/100_200_DCSBM_5000'
        # 'graphs/train/100_150_SBM_DCSBM_LPA_COPY_ER_6000',
    ])
    valid_dir: List[str] = field(default_factory=lambda: [
        'graphs/valid/valid'
    ])

from finetune_utils import teacher_wrapper, teacher_step, compute_reward_shaping

def create_run_path_and_save_args(args):
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    run_path = f"{args.gnn}/sac_teacher"
    # if args.teacher_method:
    #     run_path += f"_{args.teacher_method}"
    if args.priority_type:
        run_path += f"_{args.priority_type}"
    run_path += f"_{time_string}"

    directory = os.path.join('saved', run_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    return run_path, time_string


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    run_path, time_string = create_run_path_and_save_args(args)
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

    
    env = DismantleEnv(
        data_dir=args.train_dir, 
        batch_size=args.num_envs, 
        is_val=False, 
        seed=args.seed,
        remove_scc=False,
        reward_type=args.reward_type
    )
    
    env_val = DismantleEnv(
        data_dir=args.valid_dir, 
        batch_size=args.num_envs, 
        is_val=True, 
        seed=args.seed
    )

    if args.demo_dir:
        env_demo = DismantleEnv(
            data_dir=args.demo_dir, 
            batch_size=args.num_envs, 
            is_val=False, 
            seed=args.seed,
            remove_scc=False,
            reward_type=args.reward_type
        )
    else:
        env_demo = env
        
    buffer = PriorReplayBuffer(args.buffer_size, device)

    # Load pretrained network
    num_features = 5 if args.handcrafted_features else args.num_features
    num_heads = 1 if args.handcrafted_features else args.num_heads  # 5 % 4 != 0 for MIND
    policy, qf1, qf2, qf1_target, qf2_target = load_sac_dismantler(num_features, num_heads, args.num_mps, args.gnn, device, args.ckpt_pth, args.positional_encoding, args.handcrafted_features)
    print(f"Loaded checkpoint: {args.ckpt_pth}")
    print(f"Teacher method: {args.teacher_method}")
    print(f"Priority sampling: {args.priority_type}")
     
    if args.demo:
        if args.ckpt_pth:
            ckpt = torch.load(args.ckpt_pth, map_location=device)
            if "buffer_state_dict" in ckpt:
                buffer.load_state_dict(ckpt["buffer_state_dict"])
                print(f"Loaded {buffer.ptr} transitions from checkpoint in buffer.")
        
        else:
            obs_list, _ = env_demo.reset()
            num_eps = 0
            while num_eps < args.num_demos:
                act_arr = []
                for graph in obs_list:
                    removals = teacher_wrapper(graph, args.teacher_method, max_steps=1)
                    teacher_action = removals[0]
                    act_arr.append(teacher_action)
                
                obs_next_list, rew_arr, done_arr, info_list = env_demo.step(np.array(act_arr))
                    
                # Add to teacher buffer
                buffer.add(obs_list, act_arr, obs_next_list, rew_arr, done_arr, from_teacher=True,fixed=True)
                num_eps += len(info_list)

                obs_next_list, _ = env_demo.reset_async(done_arr)
                obs_list = obs_next_list

            # Cleanup
            del act_arr, obs_next_list, rew_arr, done_arr
        
            # log buffer size
            print(f"Saved {buffer.ptr} transitions from demonstration in buffer")

        if args.bc:
            print(f"Training behavior cloning for {args.bc_steps} steps using teacher demonstrations.")
            bc_optimizer = torch.optim.Adam(
                [p for p in policy.parameters() if p.requires_grad],
                lr=args.learning_rate,
                eps=1e-4,
            )
            bc_q_optimizer = None
            if args.bc_supervise_q:
                bc_q_optimizer = torch.optim.Adam(
                     [p for p in list(qf1.parameters()) + list(qf2.parameters()) if p.requires_grad], 
                    lr=args.learning_rate, 
                    eps=1e-4)
                print(
                    f"BC also supervises Q networks."
                )

            for bc_step in range(args.bc_steps):
                # Sample transitions
                samples = buffer.sample(args.batch_size)
                if samples is None:
                    print("No demonstrations in buffer; stopping behavior cloning.")
                    break

                obs_b, act_b, obs_next_b, rew_b, done_b = samples

                # Compute log-probabilities over actions from the policy
                _, logp_nodes = policy.get_action(obs_b, val=True)

                # Map per-graph action indices to node indices
                act_nodes = act_b + obs_b.act_offsets
                logp_selected = logp_nodes[act_nodes]

                bc_loss = -logp_selected.mean()

                bc_optimizer.zero_grad()
                bc_loss.backward()
                bc_optimizer.step()

                # ----- Q supervision on demos (same spirit as dqn_teacher pretrain) -----
                if bc_q_optimizer is not None:
                    alpha_bc = torch.tensor(args.alpha, device=device, dtype=torch.float32)
                    with torch.no_grad():
                        _, logp_next_b = policy.get_action(obs_next_b)
                        qf1_next_b = qf1_target(obs_next_b)
                        qf2_next_b = qf2_target(obs_next_b)
                        qf_next_b = torch.min(qf1_next_b, qf2_next_b) - alpha_bc * logp_next_b
                        b_next = obs_next_b.batch[obs_next_b.non_omni_mask]
                        v_next_b = scatter_add(
                            logp_next_b.exp() * qf_next_b, b_next, dim_size=obs_next_b.batch_size
                        )
                        q_target_b = rew_b.flatten() + (1 - done_b.flatten()) * args.gamma * v_next_b

                    q1_all = qf1(obs_b)
                    q2_all = qf2(obs_b)
                    q1_pred = q1_all.gather(0, act_nodes).flatten()
                    q2_pred = q2_all.gather(0, act_nodes).flatten()
                    td_loss = mse_loss(q1_pred, q_target_b) + mse_loss(q2_pred, q_target_b)

                    margin = torch.full_like(q1_all, args.bc_margin_value)
                    margin.index_fill_(0, act_nodes, 0.0)
                    aug1 = q1_all + margin
                    aug2 = q2_all + margin
                    max1, _ = scatter_max(aug1, obs_b.batch_non_omni, dim_size=obs_b.batch_size)
                    max2, _ = scatter_max(aug2, obs_b.batch_non_omni, dim_size=obs_b.batch_size)
                    q1_e = q1_all.gather(0, act_nodes)
                    q2_e = q2_all.gather(0, act_nodes)
                    margin_loss = (max1 - q1_e).mean() + (max2 - q2_e).mean()

                    l2_q = sum(p.pow(2).sum() for p in qf1.parameters() if p.requires_grad) + sum(
                        p.pow(2).sum() for p in qf2.parameters() if p.requires_grad
                    )
                    q_bc_loss = td_loss + args.bc_margin_coeff * margin_loss + args.bc_l2_coeff * l2_q

                    bc_q_optimizer.zero_grad()
                    q_bc_loss.backward()
                    bc_q_optimizer.step()

                if bc_step % 100 == 0:
                    msg = f"[BC] step {bc_step}/{args.bc_steps}, policy_loss={bc_loss.item():.4f}"
                    if bc_q_optimizer is not None:
                        msg += f", q_loss={q_bc_loss.item():.4f} (td={td_loss.item():.4f}, margin={margin_loss.item():.4f})"
                    print(msg)
                    if args.use_tb:
                        writer.add_scalar("bc/policy_loss", bc_loss.item(), bc_step)
                        if bc_q_optimizer is not None:
                            writer.add_scalar("bc/q_total_loss", q_bc_loss.item(), bc_step)
                            writer.add_scalar("bc/q_td_loss", td_loss.item(), bc_step)
                            writer.add_scalar("bc/q_margin_loss", margin_loss.item(), bc_step)

            # Align target critics with online after BC (online Q was trained; targets were fixed for bootstrap)
            if bc_q_optimizer is not None:
                qf1_target.load_state_dict(qf1.state_dict())
                qf2_target.load_state_dict(qf2.state_dict())
                print("qf1_target / qf2_target synchronized with qf1 / qf2 after BC.")

            directory = os.path.join('saved', run_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                    "buffer_state_dict": buffer.get_state_dict(),
                    "policy_state_dict": policy.state_dict(),
                    "qf1_state_dict": qf1.state_dict(),
                    "qf2_state_dict": qf2.state_dict(),
                    "qf1_target_state_dict": qf1_target.state_dict(),
                    "qf2_target_state_dict": qf2_target.state_dict(),
                },
                os.path.join(directory, "bc.ckpt"),
            )
            print(f"Saved behavior cloning checkpoint to {os.path.join(directory, 'bc.ckpt')}")

            # Cleanup
            del obs_b, act_b, obs_next_b, rew_b, done_b
    
    # Setup optimizers with appropriate learning rates
    lr = args.learning_rate
    print(f'Using learning rate: {lr}')
    
    # Only optimize trainable parameters
    q_params = [p for p in list(qf1.parameters()) + list(qf2.parameters()) if p.requires_grad]
    policy_params = [p for p in policy.parameters() if p.requires_grad]
    
    q_optimizer = torch.optim.Adam(q_params, lr=lr, eps=1e-4)
    policy_optimizer = torch.optim.Adam(policy_params, lr=lr, eps=1e-4)

    log_alpha = None
    alpha_optimizer = None
    if args.alpha_adaptive:
        log_alpha = torch.tensor(np.log(args.alpha), device=device, dtype=torch.float32, requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr, eps=1e-4)
        print(f"Alpha adaptive: target_entropy H_bar = {args.target_entropy}")
    
    num_eps, num_updates = 0, 0
    auc_buffer = deque(maxlen=20)
    start_time = time.time()

    #### MAIN LOOP ####
    obs_list, _ = env.reset()
    for global_step in range(args.total_steps): # args.num_envs transitions at each global step
        
        #### DISMANTLE ####
        if global_step<args.learning_starts and args.ckpt_pth==None and not args.bc:
            act_arr = env.sample_act()
        else:
            with torch.no_grad():
                act_arr, _ = policy.get_action(Batch(device, [ig_to_data(g) for g in obs_list]))
            act_arr = act_arr.detach().cpu().numpy()

        obs_next_list, rew_arr, done_arr, info_list = env.step(act_arr)

        # Apply reward shaping if enabled (following guide.md)
        if args.reward_shaping:
            # Compute decay factor: β(t) starts high and decays to 0
            if args.shaping_decay:
                decay_progress = min(global_step / args.shaping_decay_steps, 1.0)
                beta_t = args.shaping_coeff * (1.0 - decay_progress)
            else:
                beta_t = args.shaping_coeff
            if beta_t > 0.001:  # Only compute if coefficient is significant
                # Compute reward shaping using specified method
                rew_shaping = compute_reward_shaping(
                    obs_list, act_arr,
                    shaping_method=args.shaping_method,
                    policy=policy,
                    discriminator=None,
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
            auc_avg = sum(auc_buffer)/max(len(auc_buffer), 1)
            print(f"[{time_relative} | {num_eps} episodes | {global_step} steps] Avg. AUC = {auc_avg:.3f} (Priority Finetuning)")
            if args.use_tb:
                writer.add_scalar("train/AUC", auc_avg, global_step)

        if global_step > args.learning_starts:
            for update_idx in range(args.num_updates):
                samples = buffer.sample(
                    args.batch_size,
                    priority=args.priority_type,
                    return_indices=True
                )
                
                if len(samples) == 6:
                    obs_b, act_b, obs_next_b, rew_b, done_b, sample_indices = samples
                elif len(samples) == 5:
                    obs_b, act_b, obs_next_b, rew_b, done_b = samples

                # Current alpha (log-space when adaptive for numerical stability)
                if args.alpha_adaptive:
                    alpha = log_alpha.exp()
                else:
                    alpha = torch.tensor(args.alpha, device=device, dtype=torch.float32)

                # ---------------  CRITIC Training--------------------------------
                with torch.no_grad():
                    _, logp_next_b = policy.get_action(obs_next_b)
                    
                    qf1_next_b = qf1_target(obs_next_b)
                    qf2_next_b = qf2_target(obs_next_b)
                    qf_next_b = torch.min(qf1_next_b, qf2_next_b) - alpha.detach() * logp_next_b
                    
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
                
                # Apply importance sampling to Q-loss
                q1_loss = (mse_loss(q1_b, q_target_b, reduction='none')).mean()
                q2_loss = (mse_loss(q2_b, q_target_b, reduction='none')).mean()
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
                v_b = logp_b.exp() * (alpha.detach() * logp_b - torch.min(qf1_b, qf2_b))
                b = obs_b.batch[obs_b.non_omni_mask]
                
                # original: policy_loss = scatter_add(v_b, b, dim_size=obs_b.batch_size).mean()
                policy_loss_per_graph = scatter_add(v_b, b, dim_size=obs_b.batch_size)
                policy_loss = (policy_loss_per_graph).mean()
                
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
                    
                # Add L2 regularization
                if args.regularization:
                    policy_l2_reg = sum(torch.norm(p, p=2) ** 2 for p in policy.parameters() if p.requires_grad)
                    policy_loss = policy_loss + args.regular_coeff * policy_l2_reg
                
                policy_optimizer.zero_grad(); policy_loss.backward(); policy_optimizer.step()

                # ---------------  ADAPTIVE ALPHA (minimize J(alpha) = E[-alpha*log pi(a|s) - alpha*H_bar]) ---------------
                if args.alpha_adaptive:
                    logp_taken = logp_b.gather(0, act_b).flatten().detach()
                    alpha_loss = (alpha * (-logp_taken - args.target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    if args.use_tb and num_updates % args.target_frequency == 0:
                        writer.add_scalar("entropy/entropy",logp_b.mean().item(), global_step)
                        writer.add_scalar("entropy/alpha", alpha.item(), global_step)
                        writer.add_scalar("entropy/alpha_loss", alpha_loss.item(), global_step)
                
                # Explicit cleanup of training tensors
                del obs_b, act_b, obs_next_b, rew_b, done_b
                if sample_indices is not None:
                    del sample_indices
                
                if args.use_tb and num_updates%args.target_frequency == 0:
                    writer.add_scalar("losses/q1(s,a)", q1_b.mean().item(), global_step)
                    writer.add_scalar("losses/q2(s,a)", q2_b.mean().item(), global_step)
                    writer.add_scalar("losses/q_loss", q_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/policy_loss", -policy_loss.item(), global_step)
                    
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
# nohup python -u sac_teacher.py --use_tb --device cuda:0 --teacher_method betweenness --demo --reward_shaping > sac_teacher.out 2>&1 &