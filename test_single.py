import os
import tyro
import torch
from utils import validate_one_graph, load_g
from dataclasses import dataclass

from networks.dismantle import SACPolicy


@dataclass
class Args:
    device: str='cuda:0'
    directory: str = 'graphs/test'
    ckpt_pth: str='saved/mind.ckpt'
    step_ratio: float = 0.0025
    """step ratio for batch node removal (e.g., 0.0025 means remove 0.25% of nodes per iteration)"""
    log_removals: bool = False
    """whether to log the removal sequence"""

args = tyro.cli(Args)
device = torch.device(args.device)

# Load policy
sac = SACPolicy(
    num_features=16,
    num_heads=4,
    num_mps=6,
).to(device)
sac.load_state_dict(torch.load(args.ckpt_pth)['policy_state_dict'])

# Load graph
name = 'test_single'
g = load_g(os.path.join(args.directory, 'simple_1.pkl'), name=f'single_1')

print(f"Testing on graph: {g['name']}")
print(f"Initial nodes: {g.vcount()}, Initial edges: {g.ecount()}")
print(f"Step ratio: {args.step_ratio}")

# Validate
if args.log_removals:
    auc, robustness, lcc_curve, removals = validate_one_graph(
        g, sac, 
        save_res=name, 
        step_ratio=args.step_ratio,
        log_removals=True
    )
    print(f"\nRemoval sequence (first 20): {removals[:20]}")
    print(f"Total nodes removed: {len(removals)}")
else:
    auc, robustness, lcc_curve = validate_one_graph(
        g, sac, 
        save_res=name, 
        step_ratio=args.step_ratio,
        log_removals=False
    )

print(f"\nFinal Results:")
print(f"  AUC: {auc:.6f}")
print(f"  Robustness: {robustness:.6f}")
print(f"  Dismantling steps: {len(lcc_curve)}")