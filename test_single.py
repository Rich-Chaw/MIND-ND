import os
import tyro
import torch
from utils import validate, load_g
from env import DismantleEnv
from dataclasses import dataclass

from networks.dismantle import SACPolicy


@dataclass
class Args:
    device: str='cuda:0'
    directory: str = 'graphs/real/FINDER'
    ckpt_pth: str='saved/mind.ckpt'

args = tyro.cli(Args)
device = torch.device(args.device)
sac = SACPolicy(
    num_features=16,
    num_heads=4,
    num_mps=6,
).to(device)
# sac.load_state_dict(torch.load(args.ckpt_pth, weights_only=True)['policy_state_dict'])
sac.load_state_dict(torch.load(args.ckpt_pth)['policy_state_dict'])

name = 'single_data'
g_list = []
g_name = 'Crime'
g_path = f"{directory}/{g_name}.pkl"
g_list.append(load_g(g_path,name=g_name))
env = DismantleEnv(graph_data=g_list, batch_size=1, is_val=True)
validate(env, sac, save_res=name)