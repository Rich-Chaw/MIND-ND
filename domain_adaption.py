import os
import json
import time
import random
import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

import tyro
import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_log_softmax, scatter_mean

from env import DismantleEnv
from networks.dismantle import load_sac_dismantler
from utils import load_g, AdaptionBuffer, Batch, ig_to_data, validate
from utils.sample import sample_real_subgraphs


@dataclass
class Args:
    ckpt_pth: str = 'saved/rfgnn/sac_teacher_20260421_065917/19999.ckpt'
    train_synth_dir: List[str] = field(default_factory=lambda: ["graphs/train/100_150_SBM_DCSBM_LPA_COPY_ER_6000"])
    train_real_dir: List[str] = field(default_factory=lambda: ["graphs/domain"])

    seed: int = 0
    device: str = "cuda:0"
    num_envs: int = 64
    total_steps: int = 20000
    learning_starts: int = 500
    num_updates: int = 6
    batch_size: int = 128
    buffer_size: int = 200000
    save_frequency: int = 1000
    val_frequency: int = 100

    # real subgraph sampling
    num_real_subgraphs: int = 2000
    subgraph_min_nodes: int = 200
    subgraph_max_nodes: int = 400
    
    ratio: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])  # [rw, mhrw, ff]
    val_ratio: float = 0.01

    # adaptation objective
    lambda_coral: float = 0.1
    lr_gnn: float = 1e-4
    lr_mlp: float = 1e-5


def load_graphs_from_dirs(dirs: List[str]):
    graphs = []
    for d in dirs:
        for p in sorted(os.listdir(d)):
            full_path = os.path.join(d, p)
            if not os.path.isfile(full_path):
                continue
            name = f"{os.path.basename(d)}_{os.path.splitext(os.path.basename(p))[0]}"
            g = load_g(full_path, name)
            if g is not None:
                graphs.append(g)
    return graphs


def coral_loss(x_synth: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    if x_synth.size(0) < 2 or x_real.size(0) < 2:
        return torch.tensor(0.0, device=x_synth.device)
    x_s = x_synth - x_synth.mean(dim=0, keepdim=True)
    x_r = x_real - x_real.mean(dim=0, keepdim=True)
    cov_s = (x_s.T @ x_s) / (x_s.size(0) - 1)
    cov_r = (x_r.T @ x_r) / (x_r.size(0) - 1)
    return ((cov_s - cov_r) ** 2).mean()


def create_run_dir(args: Args):
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("saved", f"domain_adapt_{time_str}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    return run_dir


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device(args.device)
    run_dir = create_run_dir(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) load synth + sample real subgraphs
    synth_graphs = load_graphs_from_dirs(args.train_synth_dir)
    for g in synth_graphs:
        g["type"] = "synth"

    real_graphs = load_graphs_from_dirs(args.train_real_dir)
    sampled_real = sample_real_subgraphs(
        real_graphs,
        num_subgraphs=args.num_real_subgraphs,
        min_nodes=args.subgraph_min_nodes,
        max_nodes=args.subgraph_max_nodes,
        ratio=args.ratio
    )

    mixed_graphs = synth_graphs + sampled_real
    if len(mixed_graphs) < 2:
        raise ValueError("Need at least 2 graphs in mixed_graphs to split train/val.")
    random.shuffle(mixed_graphs)
    val_size = max(1, int(len(mixed_graphs) * args.val_ratio))
    val_size = min(val_size, len(mixed_graphs) - 1)
    val_graphs = mixed_graphs[:val_size]
    train_graphs = mixed_graphs[val_size:]
    print(
        f"synth={len(synth_graphs)} real_subgraphs={len(sampled_real)} "
        f"train={len(train_graphs)} val={len(val_graphs)}"
    )

    # 2) env and adaptation buffer
    env = DismantleEnv(
        graph_data=train_graphs,
        batch_size=args.num_envs,
        is_val=False,
        seed=args.seed,
        remove_scc=False,
    )
    env_val = DismantleEnv(
        graph_data=val_graphs,
        batch_size=args.num_envs,
        is_val=True,
        seed=args.seed,
    )
    buffer = AdaptionBuffer(args.buffer_size, device)

    # 3) load checkpoint, keep teacher fixed, only adapt policy
    policy, qf1, qf2, qf1_target, qf2_target = load_sac_dismantler(
        gnn=None, device=device, ckpt_pth=args.ckpt_pth
    )
    teacher = copy.deepcopy(policy)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # no Q-network tuning
    del qf1, qf2, qf1_target, qf2_target

    optimizer = torch.optim.Adam(
        [
            {"params": policy.graph_embedding.parameters(), "lr": args.lr_gnn},
            {"params": policy.mlp.parameters(), "lr": args.lr_mlp},
        ],
        eps=1e-4,
    )

    ckpt_raw = torch.load(args.ckpt_pth, map_location="cpu")
    t0 = time.time()
    obs_list, _ = env.reset()
    loss = torch.tensor(0.0, device=device)
    loss_task = torch.tensor(0.0, device=device)
    loss_coral = torch.tensor(0.0, device=device)
    for step in range(args.total_steps):
        # collect trajectory data like sac.py
        if step < args.learning_starts:
            act_arr = env.sample_act()
        else:
            act_batch = Batch(device, [ig_to_data(g) for g in obs_list])
            with torch.no_grad():
                act_arr, _ = policy.get_action(act_batch)
            act_arr = act_arr.detach().cpu().numpy()

        obs_next_list, _, done_arr, _ = env.step(act_arr)
        buffer.add(obs_list)
        obs_next_list, _ = env.reset_async(done_arr)
        obs_list = obs_next_list

        if step >= args.learning_starts:
            for _ in range(args.num_updates):
                sampled = buffer.sample_balanced(args.batch_size)
                if sampled[0] is None:
                    continue
                batch, is_synth_graph = sampled
                is_real_graph = ~is_synth_graph
                node_graph_id = batch.batch_non_omni
                synth_node_mask = is_synth_graph[node_graph_id]

                # KD on synth
                student_logits = policy(batch)
                student_logp = scatter_log_softmax(student_logits, node_graph_id, dim_size=batch.batch_size)
                with torch.no_grad():
                    teacher_logits = teacher(batch)
                    teacher_logp = scatter_log_softmax(teacher_logits, node_graph_id, dim_size=batch.batch_size)
                    teacher_prob = teacher_logp.exp()
                if synth_node_mask.any():
                    loss_task = F.kl_div(
                        student_logp[synth_node_mask],
                        teacher_prob[synth_node_mask],
                        reduction="batchmean",
                    )
                else:
                    loss_task = torch.tensor(0.0, device=device)

                # CORAL on graph-level embeddings
                e = policy.graph_embedding(batch)
                g_emb = scatter_mean(e, node_graph_id, dim=0, dim_size=batch.batch_size)
                loss_coral = coral_loss(g_emb[is_synth_graph], g_emb[is_real_graph])

                loss = loss_task + args.lambda_coral * loss_coral
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if step % args.val_frequency == 0 and step >= args.learning_starts:
            val_auc_list = validate(env_val, policy)[0]
            auc_val_avg = sum(val_auc_list) / max(len(val_auc_list), 1)
            print(f"At step {step}, Avg. Validation AUC is {auc_val_avg:.4f}")

        if (step + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"[step {step+1}/{args.total_steps}] loss={loss.item():.6f} task={loss_task.item():.6f} coral={loss_coral.item():.6f} t={dt:.1f}s")

        if (step + 1) % args.save_frequency == 0:
            out = dict(ckpt_raw)
            out["policy_state_dict"] = policy.state_dict()
            torch.save(out, os.path.join(run_dir, f"{step+1}.ckpt"))

    out = dict(ckpt_raw)
    out["policy_state_dict"] = policy.state_dict()
    torch.save(out, os.path.join(run_dir, "final.ckpt"))
    print(f"saved to {run_dir}")
