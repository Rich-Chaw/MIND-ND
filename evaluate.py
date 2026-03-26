import os
import tyro
import torch
import igraph as ig
from utils import validate,validate_one_graph, load_g
from env import DismantleEnv
from dataclasses import dataclass


from networks.dismantle import load_dismantler
from copy import deepcopy
import baseline
from baseline import ensure_attribute
  
@dataclass
class Args:
    device: str='cuda:0'
    directory: str = 'graphs/test/1000_1000_BA_10'
    ckpt_pth: str='saved/mind/mind.ckpt'
    batch_size:int = 5


def mind_dismantling(
    graph: ig.Graph,
    ckpt_pth=None,
    device=torch.device('cpu'),
    step_ratio=None,
):
    graph = deepcopy(graph)
    ensure_attribute(graph)
    policy = load_dismantler(ckpt_pth, device)
    auc, robustness, _, removals = validate_one_graph(
        graph, policy=policy, step_ratio=step_ratio, log_removals=True
    )
    return auc, robustness, removals

def mind_dismantling_batch(graph_list:list,batch_size:int,ckpt_pth=None,device=torch.device('cpu')):
    policy = load_dismantler(ckpt_pth,device)
    env = DismantleEnv(graph_data=graph_list, batch_size=batch_size, is_val=True)
    auc, robustness, _, removals = validate(env,policy=policy,log_removals=True)
    return auc, robustness,removals


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device(args.device)
    policy = load_dismantler(args.ckpt_pth, device)

    g_list = []
    if args.directory == 'graphs/real':
        for t in ['bio', 'information', 'social', 'tech']:
            d = f'{args.directory}/{t}'
            g_list.extend(
                [
                    load_g(os.path.join(d, p), f'{t}_{os.path.splitext(os.path.basename(p))[0]}')
                    for p in sorted(os.listdir(d))
                ]
            )
    else:
        for p in sorted(os.listdir(args.directory)):
            if p.endswith('.pkl'):
                g_list.append(
                    load_g(
                        os.path.join(args.directory, p),
                        f'custom_{os.path.splitext(os.path.basename(p))[0]}',
                    )
                )

    mind_dismantling_batch(g_list, args.batch_size, args.ckpt_pth, device)
    from baseline import baseline_dismantling_batch

    baseline_dismantling_batch(g_list, args.batch_size, method='Degree')

# nohup python -u test.py --device cuda:0 --ckpt_pth saved/mind.ckpt --directory graphs/real > test.out 2>&1 &
