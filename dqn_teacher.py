import os
import tyro
import time
import json
import torch
import random
import numpy as np
from typing import Optional, List
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from torch.nn.functional import smooth_l1_loss
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter_max

from env import DismantleEnv
from networks.dismantle import SACQNetwork, DQNPolicy, load_dqn_dismantler
from utils import ReplayBuffer, PriorReplayBuffer, Batch, validate, ig_to_data
from finetune_utils import teacher_wrapper, teacher_step, compute_reward_shaping


@dataclass
class Args:
    use_tb: bool = False
    """record using tensorboard"""
    seed: int = 0
    """random seed"""
    device: str = "cuda:0"
    """the device to use"""
    algo: str = "dqn"
    """training algorithm: dqn or ddqn"""
    gnn: str = "gcn"
    num_envs: int = 64
    """number of parallel environments, default 64"""
    total_steps: int = 200000
    """number of training steps (transitions = steps*num_envs)"""
    buffer_size: int = 2000000
    """size of the replay buffer"""
    batch_size: int = 128
    """batch size for updating network"""
    val_frequency: int = 1000
    """validation frequency"""
    save_frequency: int = 1000
    """save frequency"""
    learning_starts: int = 2000
    """timestep to start learning"""
    learning_rate: float = 3e-4
    """learning rate for the Q network"""
    tau: float = 1.0
    """target smoothing factor"""
    gamma: float = 0.99
    """discount factor"""
    num_updates: int = 16
    """number of network updates at each step"""
    target_frequency: int = 200
    """the frequency for updating the target network"""
    ckpt_pth: Optional[str] = None
    """where checkpoint was saved"""
    num_features: int = 16
    """number of initial node features"""
    num_heads: int = 4
    """number of message passings heads"""
    num_mps: int = 6
    """number of message passings"""
    normalize: bool = True
    """apply instance normalization"""
    positional_encoding: Optional[str] = None
    """node initial features: None = all ones, 'RW' = random walk return-probability encoding"""
    handcrafted_features: bool = False
    """if True, use 5 handcrafted features"""
    reward_type: Optional[int] = 0
    """0: -LCC_t/N   1:(LCC_t-1 - LCC_t) / N"""
    eps_start: float = 1.0
    """initial epsilon for epsilon-greedy exploration"""
    eps_end: float = 0.05
    """final epsilon for epsilon-greedy exploration"""
    eps_decay_steps: int = 100000
    """steps for linear epsilon decay"""
    max_grad_norm: float = 10.0
    """gradient clipping norm; <=0 disables clipping"""

    # Teacher / demonstration settings
    teacher_method: Optional[str] = None
    """Teacher method: 'spectral', 'betweenness', 'CI','Hybrid' etc."""
    demo: bool = False
    """Collect teacher demonstrations before training"""
    demo_dir: List[str] = field(default_factory=lambda: [
        "graphs/train/100_200_BA_5000",
    ])
    num_demos: int = 1000
    """Number of demonstration episodes to collect"""
    demo_ckpt: Optional[str] = None
    """Optional checkpoint that contains a saved teacher buffer state_dict"""

    # Reward shaping settings
    reward_shaping: bool = False
    """Enable reward shaping"""
    shaping_method: str = "KL"
    """Reward shaping method: 'betweenness' or 'KL'"""
    shaping_decay_steps: int = 10000
    """Number of steps to decay reward shaping coefficient"""
    shaping_coeff: float = 0.1
    """Initial reward shaping coefficient"""

    # Pretraining (DQNfD-style) settings
    pretrain: bool = False
    pretrain_steps: int = 10000
    """Number of gradient steps for DQN pretraining on demonstrations"""
    margin_value: float = 0.8
    """Margin l in the large-margin classification loss"""
    margin_coeff: float = 1.0
    """λ1: weight for the large-margin loss J_E"""
    l2_coeff: float = 1e-5
    """λ2: weight for L2 regularization loss J_L2"""

    train_dir: List[str] = field(default_factory=lambda: [
        "graphs/train/100_200_BA_5000",
    ])
    valid_dir: List[str] = field(default_factory=lambda: [
        "graphs/valid/valid"
    ])



def create_run_path_and_save_args(args):
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    run_path = f"{args.gnn}/{args.algo}"
    run_path += f"_teacher_{time_string}"

    directory = os.path.join("saved", run_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join("saved", run_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    return run_path, time_string


def linear_schedule(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    mix = min(step / decay_steps, 1.0)
    return start + mix * (end - start)

def get_next_q(obs_next_b, qf, qf_target, algo):
    with torch.no_grad():
        target_q_next = qf_target(obs_next_b)
        if algo == "ddqn":
            online_q_next = qf(obs_next_b)
            _, next_act = scatter_max(online_q_next, obs_next_b.batch_non_omni, dim_size=obs_next_b.batch_size)
            next_q = target_q_next.gather(0, next_act).flatten()
        elif algo == "dqn":
            next_q, _ = scatter_max(target_q_next, obs_next_b.batch_non_omni, dim_size=obs_next_b.batch_size)
        else:
            raise ValueError(f"Unknown algo={algo}, expected 'dqn' or 'ddqn'")

        next_q = torch.where(torch.isfinite(next_q), next_q, torch.zeros_like(next_q))
        return next_q


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.algo = args.algo.lower()
    if args.algo not in {"dqn", "ddqn"}:
        raise ValueError(f"Unknown algo={args.algo}, expected 'dqn' or 'ddqn'")

    run_path, time_string = create_run_path_and_save_args(args)
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.use_tb:
        writer = SummaryWriter(f"runs/{run_path}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    print(f"Training starts at {time_string}")
    print(f"Device is {device}. Seed set to {args.seed}")
    print(f"Algorithm is {args.algo.upper()}")

    env = DismantleEnv(
        data_dir=args.train_dir,
        batch_size=args.num_envs,
        is_val=False,
        seed=args.seed,
        remove_scc=False,
        reward_type=args.reward_type,
    )
    env_val = DismantleEnv(
        data_dir=args.valid_dir,
        batch_size=args.num_envs,
        is_val=True,
        seed=args.seed,
    )

    # Demo environment (if provided), otherwise reuse training env
    if args.demo_dir:
        env_demo = DismantleEnv(
            data_dir=args.demo_dir,
            batch_size=args.num_envs,
            is_val=False,
            seed=args.seed,
            remove_scc=False,
            reward_type=args.reward_type,
        )
    else:
        env_demo = env

    # Use PriorReplayBuffer so we can pin teacher demonstrations
    buffer = PriorReplayBuffer(args.buffer_size, device)

    num_features = 5 if args.handcrafted_features else args.num_features
    num_heads = 1 if args.handcrafted_features else args.num_heads
    qf, qf_target = load_dqn_dismantler(
        num_features,
        num_heads,
        args.num_mps,
        args.gnn,
        device,
        args.ckpt_pth,
        args.positional_encoding,
        args.handcrafted_features,
    )
    policy = DQNPolicy(qf)
    optimizer = torch.optim.Adam(qf.parameters(), lr=args.learning_rate, eps=1e-4)

    # ------------------------------------------------------------------
    # Demonstration collection and DQN pretraining on demos (Q-learning
    # + large-margin classification + L2 regularization)
    # ------------------------------------------------------------------
    if args.demo:
        if args.demo_ckpt and os.path.isfile(args.demo_ckpt):
            ckpt = torch.load(args.demo_ckpt, map_location=device)
            if "buffer_state_dict" in ckpt:
                buffer.load_state_dict(ckpt["buffer_state_dict"])
                print(f"Loaded {buffer.ptr} transitions from demo checkpoint into buffer.")
        else:
            obs_list, _ = env_demo.reset()
            num_eps_demo = 0
            print(f"Collecting {args.num_demos} demonstration episodes with teacher={args.teacher_method} ...")
            while num_eps_demo < args.num_demos:
                act_arr = []
                for graph in obs_list:
                    removals = teacher_wrapper(graph, args.teacher_method, max_steps=1)
                    teacher_action = removals[0]
                    act_arr.append(teacher_action)

                obs_next_list, rew_arr, done_arr, info_list = env_demo.step(np.array(act_arr))

                # Store as fixed teacher demonstrations
                buffer.add(obs_list, act_arr, obs_next_list, rew_arr, done_arr, from_teacher=True, fixed=True)
                num_eps_demo += len(info_list)

                obs_next_list, _ = env_demo.reset_async(done_arr)
                obs_list = obs_next_list

            print(f"Saved {buffer.ptr} transitions from demonstrations in buffer.")

        # ------------------- DQN Pretraining on D_demo -------------------
        if args.pretrain:
            qf.train()
            print(f"Starting DQN pretraining on demonstrations for {args.pretrain_steps} steps ...")
            for pre_step in range(args.pretrain_steps):
                samples = buffer.sample(args.batch_size)
                if samples is None:
                    print("No demonstrations in buffer; stopping pretraining.")
                    break

                obs_b, act_b, obs_next_b, rew_b, done_b = samples

                # 1) DQN TD loss J_DQN
                with torch.no_grad():
                    next_q = get_next_q(obs_next_b, qf, qf_target, args.algo)
                    q_target_b = rew_b.flatten() + (1 - done_b.flatten()) * args.gamma * next_q

                expert_act_nodes = act_b + obs_b.act_offsets
                q_all = qf(obs_b)
                q_pred_b = q_all.gather(0, expert_act_nodes).flatten()
                td_loss = smooth_l1_loss(q_pred_b, q_target_b)

                # 2) Large-margin classification loss J_E
                # Margin l(a_E, a): 0 if a == a_E else margin_value
                margin = torch.full_like(q_all, args.margin_value)
                margin.index_fill_(0, expert_act_nodes, 0.0)
                augmented_q = q_all + margin

                # max_a (Q(s,a) + l) per graph
                max_aug_q, _ = scatter_max(augmented_q, obs_b.batch_non_omni, dim_size=obs_b.batch_size)
                q_expert = q_all.gather(0, expert_act_nodes)
                margin_loss = (max_aug_q - q_expert).mean()

                # 3) L2 regularization J_L2
                l2_loss = sum(p.pow(2).sum() for p in qf.parameters() if p.requires_grad)

                total_loss = td_loss + args.margin_coeff * margin_loss + args.l2_coeff * l2_loss

                optimizer.zero_grad()
                total_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(qf.parameters(), args.max_grad_norm)
                optimizer.step()

                if pre_step % 100 == 0:
                    print(f"[PRE] step {pre_step}/{args.pretrain_steps}, total loss={total_loss.item():.4f}")
                    if args.use_tb:
                        writer.add_scalar("pretrain/td_loss", td_loss.item(), pre_step)
                        writer.add_scalar("pretrain/margin_loss", margin_loss.item(), pre_step)
                        writer.add_scalar("pretrain/l2_loss", l2_loss.item(), pre_step)
                        writer.add_scalar("pretrain/total_loss", total_loss.item(), pre_step)

    num_eps, num_updates = 0, 0
    auc_buffer = deque(maxlen=20)
    start_time = time.time()

    obs_list, _ = env.reset()
    for global_step in range(args.total_steps):
        if global_step < args.learning_starts and args.ckpt_pth is None:
            act_arr = env.sample_act()
            epsilon = args.eps_start
        else:
            epsilon = linear_schedule(global_step, args.eps_start, args.eps_end, args.eps_decay_steps)
            with torch.no_grad():
                act_arr, _ = policy.get_action(Batch(device, [ig_to_data(g) for g in obs_list]), epsilon=epsilon)
            act_arr = act_arr.detach().cpu().numpy()

        obs_next_list, rew_arr, done_arr, info_list = env.step(act_arr)

        # Optional reward shaping (R_total = R_env + β(t)*F)
        if args.reward_shaping and args.teacher_method is not None:
            decay_progress = min(global_step / args.shaping_decay_steps, 1.0)
            beta_t = args.shaping_coeff * (1.0 - decay_progress)

            if beta_t > 0.001:
                rew_shaping = compute_reward_shaping(
                    obs_list,
                    act_arr,
                    shaping_method=args.shaping_method,
                    policy=policy,
                    discriminator=None,
                    teacher_method=args.teacher_method,
                    temperature=1.0,
                    device=device,
                )
                rew_arr = rew_arr + beta_t * rew_shaping

                if args.use_tb and global_step % 100 == 0:
                    writer.add_scalar("shaping/beta_coefficient", beta_t, global_step)
                    writer.add_scalar(f"shaping/{args.shaping_method.lower()}_reward_avg", float(np.mean(rew_shaping)), global_step)

        buffer.add(obs_list, act_arr, obs_next_list, rew_arr, done_arr)

        obs_next_list, _ = env.reset_async(done_arr)
        obs_list = obs_next_list

        for logger in info_list:
            num_eps += 1
            auc_buffer.append(logger.auc / logger.n_init)

        if global_step % args.val_frequency == 0 and global_step >= args.learning_starts:
            val_auc_list = validate(env_val, policy)[0]
            auc_val_avg = sum(val_auc_list) / len(val_auc_list)
            print(f"At step {global_step}, Avg. Validation AUC is {auc_val_avg:.4f}")

            if args.use_tb:
                writer.add_scalar("val/val_avg_auc", auc_val_avg, global_step)

        if (global_step + 1) % args.save_frequency == 0 and global_step >= args.learning_starts:
            directory = os.path.join("saved", run_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(
                {
                    "qf_state_dict": qf.state_dict(),
                    "qf_target_state_dict": qf_target.state_dict(),
                },
                os.path.join(directory, f"{global_step}.ckpt"),
            )

        if (global_step + 1) % 50 == 0:
            time_relative = str(timedelta(seconds=time.time() - start_time)).split(".")[0]
            auc_avg = sum(auc_buffer) / max(len(auc_buffer), 1)
            print(f"[{time_relative} | {num_eps} episodes | {global_step} steps] Avg. AUC = {auc_avg:.3f}, eps = {epsilon:.3f}")
            if args.use_tb:
                writer.add_scalar("train/AUC", auc_avg, global_step)
                writer.add_scalar("train/epsilon", epsilon, global_step)

        if global_step > args.learning_starts:
            qf.train()
            for _ in range(args.num_updates):
                obs_b, act_b, obs_next_b, rew_b, done_b = buffer.sample(args.batch_size)

                with torch.no_grad():
                    next_q = get_next_q(obs_next_b, qf, qf_target, args.algo)
                    q_target_b = rew_b.flatten() + (1 - done_b.flatten()) * args.gamma * next_q

                act_b = act_b + obs_b.act_offsets
                q_pred_b = qf(obs_b).gather(0, act_b).flatten()
                q_loss = smooth_l1_loss(q_pred_b, q_target_b)

                optimizer.zero_grad()
                q_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(qf.parameters(), args.max_grad_norm)
                optimizer.step()

                if args.use_tb and num_updates % 200 == 0:
                    writer.add_scalar("losses/q_mean", q_pred_b.mean().item(), global_step)
                    writer.add_scalar("losses/q_target_mean", q_target_b.mean().item(), global_step)
                    writer.add_scalar("losses/q_loss", q_loss.item(), global_step)

                num_updates += 1

            if global_step % args.target_frequency == 0:
                for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    if args.use_tb:
        writer.close()

# nohup python -u dqn.py --algo ddqn --use_tb --device cuda:0 > train.out 2>&1 &
