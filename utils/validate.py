import torch
import numpy as np
import csv
import os
from datetime import datetime
from copy import deepcopy
from scipy.integrate import simpson
from .graph_data import Batch, ig_to_data

def validate(env, policy, save_res=None,log_removals=False):
    try:
        device = next(policy.parameters()).device
    except:
        device = torch.device('cpu')

    policy.eval()

    csv_pth = None
    if save_res:
        os.makedirs('results', exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_pth = os.path.join('results', f'{save_res}_{time_str}.csv')
        with open(csv_pth, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Type", "Graph", "AUC", "Robustness"])

    auc_list = []
    robustness_list = []
    lcc_curve_list = []
    if log_removals:
        removals_list = []

    obs_list, _ = env.reset()
    finished = False

    while not finished:
        with torch.no_grad():
            batch_data = Batch(device, [ig_to_data(g) for g in obs_list])
            act_arr, *rest = policy.get_action(batch_data, val=True)
        act_arr = act_arr.cpu().numpy()

        obs_next_list, rew_arr, done_arr, info_list = env.step(act_arr)
        obs_next_list, _ = env.reset_async(done_arr)

        finished = (len(obs_next_list) == 0)
        obs_list = np.array(obs_next_list)

        for logger in info_list:
            print(f'{logger.name}: AUC={logger.auc:.6f}, Robustness={logger.robustness:.6f}')
            auc = logger.auc / logger.n_init
            auc_list.append(auc)
            robustness_list.append(logger.robustness)
            lcc_curve_list.append(logger.gcc_eps)
            if log_removals:
                removals_list.append(logger.removals)
            if csv_pth:
                t, g = logger.name.split("_", 1)
                with open(csv_pth, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([t, g, logger.auc, logger.robustness])

        if finished and csv_pth:
            with open(csv_pth, mode='r') as f:
                reader = list(csv.reader(f))
                header, rows = reader[0], reader[1:]

            rows.sort(key=lambda x: (x[0], x[1]))  # sort by Type, then Graph name

            with open(csv_pth, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

    policy.train()
    if log_removals:
        return auc_list, robustness_list, lcc_curve_list, removals_list
    else:
        return auc_list, robustness_list, lcc_curve_list


def validate_one_graph(graph, policy, save_res=None, step_ratio=None, log_removals=False):
    """
    Validate policy on a single graph with optional step_ratio for batch node removal.
    
    Args:
        graph: igraph.Graph object
        policy: trained policy network
        save_res: optional name for saving results to CSV
        step_ratio: if provided, removes step_ratio*n_nodes nodes per iteration
        log_removals: if True, return the removal sequence
    
    Returns:
        auc, robustness, lcc_curve, [removals if log_removals=True]
    """
    try:
        device = next(policy.parameters()).device
    except:
        device = torch.device('cpu')

    # Make a copy and initialize tracking
    g = deepcopy(graph)
    n_init = g.vcount()
    g.vs["i_init"] = list(range(n_init))
    
    policy.eval()

    gcc_eps = []  # LCC size at each step
    removals = []  # Removed node IDs (original)
    
    # Calculate step size
    if step_ratio is not None and step_ratio > 0:
        step_size = max(int(step_ratio * n_init), 1)
    else:
        step_size = 1

    # Main dismantling loop
    finished = False
    while not finished:
        # Get action probabilities from policy
        with torch.no_grad():
            batch_data = Batch(device, [ig_to_data(g)])
            _, log_probs = policy.get_action(batch_data, val=True)
        
        # Convert to probabilities and get top-k nodes
        probs = log_probs.exp().cpu().numpy()
        
        # Select top step_size nodes based on probabilities
        current_step = min(step_size, len(probs))
        top_k_indices = np.argsort(-probs)[:current_step]
        
        # Remove nodes sequentially
        for idx in sorted(top_k_indices, reverse=True):  # Remove in reverse order to avoid index shifts
            if idx < g.vcount():  # Safety check
                original_id = g.vs[idx]["i_init"]
                removals.append(original_id)
                g.delete_vertices(idx)
                # Check termination condition
                lcc_size = g.connected_components().giant().vcount() / n_init
                gcc_eps.append(lcc_size)
                if lcc_size < 0.1 or g.ecount() == 0:
                    finished = True
                    break

    # Calculate metrics
    auc = simpson(gcc_eps, dx=1)
    robustness = sum(gcc_eps[::-1][:-1]) / n_init
    
    # Save results if requested
    if save_res:
        os.makedirs('results', exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_str = f"_StepRatio_{step_ratio}" if step_ratio else ""
        csv_pth = os.path.join('results', f'{save_res}_{time_str}{step_str}.csv')
        with open(csv_pth, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Graph", "AUC", "Robustness", "Steps"])
            writer.writerow([graph['name'], auc, robustness, len(gcc_eps)])
        print(f"Results saved to {csv_pth}")
    
    print(f"Graph: {graph['name']}, AUC={auc:.6f}, Robustness={robustness:.6f}, Steps={len(gcc_eps)}")
    
    policy.train()
    
    if log_removals:
        return auc, robustness, gcc_eps, removals
    else:
        return auc, robustness, gcc_eps