"""
BC (Behavior Cloning) pipeline with HM-GNN (v3 or v5):
  Phase 1 (30 min): Supervised pretrain — HM-GNN learns to regress CI (Collective Influence) per node.
  Phase 2 (1 h):    Behavior cloning — policy imitates greedy/CI teacher on small graphs.
  Phase 3:         Evaluate on validation set.
"""
import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from env import DismantleEnv
from networks.dismantle import load_sac_dismantler
from utils import Batch, validate, ig_to_data
from finetune_utils import teacher_wrapper


# ---------- CI (Collective Influence) for regression target ----------
def compute_ci_per_node(graph) -> np.ndarray:
    """
    Compute CI(v) = (k_v - 1) * sum_{u in ball(v,2)} (k_u - 1) for each node.
    Returns array of shape (n,) for n = graph.vcount().
    """
    n = graph.vcount()
    if n == 0:
        return np.array([], dtype=np.float32)
    ci_scores = np.zeros(n, dtype=np.float64)
    for v in graph.vs:
        neighbors_1 = set(graph.neighbors(v.index))
        neighbors_2 = set()
        for n1 in neighbors_1:
            neighbors_2.update(graph.neighbors(n1))
        ball_2 = neighbors_1.union(neighbors_2) - {v.index}
        k_v = len(neighbors_1)
        ci_scores[v.index] = (k_v - 1) * sum(
            len(graph.neighbors(u)) - 1 for u in ball_2
        )
    return ci_scores.astype(np.float32)


def normalize_ci_target(ci_arr: np.ndarray) -> np.ndarray:
    """Log-scale and optional standardization for stable regression."""
    x = np.log1p(np.maximum(ci_arr, 0.0))
    return x.astype(np.float32)


# ---------- Args ----------
@dataclass
class Args:
    seed: int = 0
    device: str = "cuda:0"
    gnn: str = "hm_gnn_v5"
    """GNN encoder: hm_gnn_v5 or hm_gnn_v3"""

    # Phase 1: CI pretrain
    pretrain_minutes: float = 10
    pretrain_batch_size: int = 32
    pretrain_lr: float = 1e-3

    # Phase 2: BC
    bc_minutes: float = 25
    bc_batch_size: int = 64
    bc_lr: float = 3e-4
    teacher_method: str = "greedy_lcc"
    """Teacher: 'CI' (Collective Influence) or 'greedy_lcc' (true greedy by LCC drop)."""
    num_demos: int = 8000
    """Number of (s,a) transitions to collect from teacher for BC."""

    # Model
    num_features: int = 16
    num_heads: int = 4
    num_mps: int = 6
    positional_encoding: Optional[str] = None
    handcrafted_features: bool = True

    # Data
    train_dir: List[str] = field(
        default_factory=lambda: ["graphs/train/100_200_LPA_Copy_ER_5000"]
    )
    valid_dir: List[str] = field(default_factory=lambda: ["graphs/valid/valid"])

    # Save
    save_dir: str = "saved/bc"
    ckpt_phase1: Optional[str] = None
    """Load Phase 1 checkpoint to skip pretrain or start BC from it."""


def create_run_path_and_save_args(args: Args):
    now = datetime.now()
    time_string = now.strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(args.save_dir, f"{args.gnn}_bc_{time_string}")
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    return run_path, time_string


def main():
    args = tyro.cli(Args)
    run_path, time_string = create_run_path_and_save_args(args)
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    print(f"BC run at {time_string}, device={device}, gnn={args.gnn}")
    print(f"Phase 1: CI pretrain {args.pretrain_minutes} min")
    print(f"Phase 2: BC (teacher={args.teacher_method}) {args.bc_minutes} min")

    num_features = 5 if args.handcrafted_features else args.num_features
    num_heads = 1 if args.handcrafted_features else args.num_heads
    policy, qf1, qf2, qf1_t, qf2_t = load_sac_dismantler(
        num_features,
        num_heads,
        args.num_mps,
        args.gnn,
        device,
        ckpt_pth=args.ckpt_phase1,
        positional_encoding=args.positional_encoding,
        handcrafted_features=args.handcrafted_features,
    )
    e_size = (num_features * args.num_mps) * 2
    ci_head = nn.Linear(e_size, 1).to(device)
    ci_optimizer = torch.optim.Adam(
        list(policy.graph_embedding.parameters()) + list(ci_head.parameters()),
        lr=args.pretrain_lr,
    )

    env_train = DismantleEnv(
        data_dir=args.train_dir,
        batch_size=args.pretrain_batch_size,
        is_val=False,
        seed=args.seed,
        remove_scc=False,
    )
    env_val = DismantleEnv(
        data_dir=args.valid_dir,
        batch_size=args.pretrain_batch_size,
        is_val=True,
        seed=args.seed,
    )

    # ---------- Phase 1: CI regression (30 min) ----------
    phase1_end = time.time() + args.pretrain_minutes * 60
    step_ci = 0
    if args.ckpt_phase1 is None:
        print("\n--- Phase 1: CI pretrain ---")
        policy.train()
        ci_head.train()
        while time.time() < phase1_end:
            obs_list, _ = env_train.reset()
            if len(obs_list) == 0:
                continue
            batch_graphs = obs_list[: args.pretrain_batch_size]

            target_list = []
            for g in batch_graphs:
                ci = compute_ci_per_node(g)
                target_list.append(normalize_ci_target(ci))
            target_ci = torch.tensor(
                np.concatenate(target_list), device=device, dtype=torch.float32
            )
            g_batch = Batch(device, [ig_to_data(g) for g in batch_graphs])
            e = policy.graph_embedding(g_batch)
            pred_ci = ci_head(e).squeeze(-1)
            loss_ci = F.mse_loss(pred_ci, target_ci)
            ci_optimizer.zero_grad()
            loss_ci.backward()
            ci_optimizer.step()
            step_ci += 1
            if step_ci % 100 == 0:
                print(f"  [CI] step {step_ci}, loss={loss_ci.item():.4f}")
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "ci_head_state_dict": ci_head.state_dict(),
            },
            os.path.join(run_path, "phase1_ci.ckpt"),
        )
        print(f"  Phase 1 done. Saved to {run_path}/phase1_ci.ckpt")
    else:
        ckpt = torch.load(args.ckpt_phase1, map_location=device)
        if "ci_head_state_dict" in ckpt:
            ci_head.load_state_dict(ckpt["ci_head_state_dict"])
        print(f"  Loaded Phase 1 from {args.ckpt_phase1}")

    # Drop CI head; train full policy with BC
    bc_optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.bc_lr, eps=1e-4
    )

    # ---------- Phase 2: Collect teacher demos ----------
    print("\n--- Phase 2: Collect teacher demos ---")
    demo_obs_list = []
    demo_act_list = []
    env_demo = DismantleEnv(
        data_dir=args.train_dir,
        batch_size=min(32, args.pretrain_batch_size),
        is_val=False,
        seed=args.seed + 1,
        remove_scc=False,
    )
    num_collected = 0
    obs_list, _ = env_demo.reset()
    while num_collected < args.num_demos:
        if len(obs_list) == 0:
            obs_list, _ = env_demo.reset()
        for g in obs_list:
            if g.vcount() <= 2 or g.ecount() == 0:
                continue
            try:
                removals = teacher_wrapper(g, args.teacher_method, max_steps=1)
                a = removals[0]
                demo_obs_list.append(g.copy())
                demo_act_list.append(a)
                num_collected += 1
                if num_collected >= args.num_demos:
                    break
            except Exception:
                continue
        if num_collected >= args.num_demos:
            break
        obs_list, _ = env_demo.reset()
    print(f"  Collected {len(demo_obs_list)} (s,a) demos.")

    # ---------- Phase 2: BC training (1 h) ----------
    phase2_end = time.time() + args.bc_minutes * 60
    step_bc = 0
    indices = np.arange(len(demo_obs_list))
    print("\n--- Phase 2: BC training ---")
    policy.train()
    while time.time() < phase2_end:
        np.random.shuffle(indices)
        for start in range(0, len(indices), args.bc_batch_size):
            if time.time() >= phase2_end:
                break
            idx_batch = indices[start : start + args.bc_batch_size]
            obs_b = [demo_obs_list[i] for i in idx_batch]
            act_b = np.array([demo_act_list[i] for i in idx_batch], dtype=np.int64)
            g_batch = Batch(device, [ig_to_data(g) for g in obs_b])
            _, logp_nodes = policy.get_action(g_batch, val=True)
            act_nodes = torch.tensor(act_b, device=device) + g_batch.act_offsets
            logp_selected = logp_nodes[act_nodes]
            bc_loss = -logp_selected.mean()
            bc_optimizer.zero_grad()
            bc_loss.backward()
            bc_optimizer.step()
            step_bc += 1
            if step_bc % 200 == 0:
                print(f"  [BC] step {step_bc}, loss={bc_loss.item():.4f}")
    torch.save(
        {"policy_state_dict": policy.state_dict()},
        os.path.join(run_path, "phase2_bc.ckpt"),
    )
    print(f"  Phase 2 done. Saved to {run_path}/phase2_bc.ckpt")

    # ---------- Phase 3: Validation ----------
    print("\n--- Phase 3: Validation ---")
    policy.eval()
    auc_list, robustness_list, _ = validate(env_val, policy)
    auc_avg = sum(auc_list) / len(auc_list) if auc_list else 0.0
    print(f"  Validation: Avg AUC = {auc_avg:.4f} ({len(auc_list)} graphs)")
    with open(os.path.join(run_path, "val_metrics.json"), "w") as f:
        json.dump(
            {"auc_avg": auc_avg, "auc_list": auc_list, "n_graphs": len(auc_list)},
            f,
            indent=2,
        )
    print(f"  Results saved to {run_path}")


if __name__ == "__main__":
    main()
