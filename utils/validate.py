import torch
import numpy as np
import csv
import os
from datetime import datetime
from copy import deepcopy
from scipy.integrate import simpson
from torch_scatter import scatter_mean
from .graph_data import Batch, ig_to_data


def validate_with_context_task_encoder(
    env,
    policy,
    task_encoder,
    gnn_encoder,
    context_sequence_length,
    input_dim,
    state_emb_dim,
    latent_dim,
    device,
    save_res=None,
    log_removals=False,
):
    """
    Validate policy with task encoder: z evolves from zero (step 0) to matured z_n as
    context (s, a, r, s') accumulates.

    - Step 0: No (a, r, s') yet → use zero embedding z → take action a_0.
    - Step 1: GRU sees (s_0, a_0, r_0, s_1) → outputs z_1 → policy uses z_1 for next action.
    - Step 2: GRU sees first two transitions → outputs z_2 → policy uses z_2.
    - By step n: policy uses fully matured task representation z_n.
    """
    task_encoder.eval()
    gnn_encoder.eval()
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
    type_results = {}

    obs_list, _ = env.reset()
    # ongoing_trajectories length matches current obs_list (shrinks when envs finish in valid mode)
    ongoing_trajectories = [[] for _ in range(len(obs_list))]
    finished = False

    while not finished:
        num_envs = len(obs_list)
        with torch.no_grad():
            # Build per-env context from ongoing trajectories; empty → z = 0
            context_seqs_tensor = torch.zeros(
                num_envs, context_sequence_length, input_dim,
                device=device, dtype=torch.float32
            )
            flat_context_obs = []
            flat_context_obs_next = []
            flat_context_act = []
            flat_context_rew = []
            graph_to_env_row = []

            for env_idx, traj in enumerate(ongoing_trajectories):
                context_len = min(context_sequence_length, len(traj))
                if context_len == 0:
                    continue
                context_transitions = traj[-context_len:]
                start_row = context_sequence_length - context_len
                for t_idx, t in enumerate(context_transitions):
                    flat_context_obs.append(t['obs'])
                    flat_context_obs_next.append(t['obs_next'])
                    flat_context_act.append(t['act'])
                    flat_context_rew.append(t['rew'])
                    graph_to_env_row.append((env_idx, start_row + t_idx))

            if len(flat_context_obs) > 0:
                obs_ctx_b = Batch(device, [ig_to_data(g) for g in flat_context_obs])
                emb_ctx = gnn_encoder(obs_ctx_b)
                embed_dim = emb_ctx.shape[1] // 2
                graph_emb_ctx = emb_ctx[:, embed_dim:]
                state_ctx = scatter_mean(
                    graph_emb_ctx,
                    obs_ctx_b.batch_non_omni,
                    dim=0,
                    dim_size=obs_ctx_b.batch_size,
                )
                obs_next_ctx_b = Batch(device, [ig_to_data(g) for g in flat_context_obs_next])
                emb_next_ctx = gnn_encoder(obs_next_ctx_b)
                graph_emb_next_ctx = emb_next_ctx[:, embed_dim:]
                state_next_ctx = scatter_mean(
                    graph_emb_next_ctx,
                    obs_next_ctx_b.batch_non_omni,
                    dim=0,
                    dim_size=obs_next_ctx_b.batch_size,
                )
                act_ctx = torch.tensor(flat_context_act, device=device, dtype=torch.float32)
                rew_ctx = torch.tensor(flat_context_rew, device=device, dtype=torch.float32)
                for graph_idx, (env_idx, row_idx) in enumerate(graph_to_env_row):
                    context_seqs_tensor[env_idx, row_idx, :state_emb_dim] = state_ctx[graph_idx]
                    context_seqs_tensor[env_idx, row_idx, state_emb_dim] = act_ctx[graph_idx]
                    context_seqs_tensor[env_idx, row_idx, state_emb_dim + 1] = rew_ctx[graph_idx]
                    context_seqs_tensor[env_idx, row_idx, state_emb_dim + 2 : state_emb_dim * 2 + 2] = state_next_ctx[graph_idx]

            z_batch = task_encoder.get_z(context_seqs_tensor)
            z_batch_last = z_batch[:, -1, :]

            batch_data = Batch(device, [ig_to_data(g) for g in obs_list])
            act_arr, *rest = policy.get_action(batch_data, z=z_batch_last, val=True)
        act_arr = act_arr.cpu().numpy()

        obs_next_list, rew_arr, done_arr, info_list = env.step(act_arr)

        for env_idx in range(num_envs):
            ongoing_trajectories[env_idx].append({
                'obs': obs_list[env_idx],
                'act': act_arr[env_idx],
                'rew': rew_arr[env_idx],
                'obs_next': obs_next_list[env_idx],
                'done': done_arr[env_idx],
            })

        # In valid mode reset_async returns only still-running envs; keep trajectories in sync
        ongoing_trajectories = [ongoing_trajectories[i] for i in range(num_envs) if not done_arr[i]]
        obs_next_list, _ = env.reset_async(done_arr)
        finished = (len(obs_next_list) == 0)
        obs_list = np.array(obs_next_list)

        for logger in info_list:
            auc = logger.auc / logger.n_init
            auc_list.append(auc)
            robustness_list.append(logger.robustness)
            lcc_curve_list.append(logger.gcc_eps)
            if log_removals:
                removals_list.append(logger.removals)
            t = logger.name.split("_")[-1]
            g = logger.name.split("_", 1)[1]
            if t not in type_results:
                type_results[t] = []
            type_results[t].append(auc)
            if csv_pth:
                with open(csv_pth, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([t, g, logger.auc, logger.robustness])

        if finished and csv_pth:
            with open(csv_pth, mode='r') as f:
                reader = list(csv.reader(f))
                header, rows = reader[0], reader[1:]
            rows.sort(key=lambda x: (x[0], x[1]))
            with open(csv_pth, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

    for graph_type, aucs in type_results.items():
        avg_auc = sum(aucs) / len(aucs)
        print(f'{graph_type}: Avg AUC = {avg_auc:.6f} ({len(aucs)} graphs)')

    if log_removals:
        return auc_list, robustness_list, lcc_curve_list, removals_list
    return auc_list, robustness_list, lcc_curve_list

def validate_with_discriminator():
    pass

def validate(env, policy, save_res=None, log_removals=False, task_encoder=None):
    """
    Enhanced validate function that logs average AUC by graph type (BA, WS, ER, SBM,...).
    If task_encoder is provided (e.g. for hypernet-based policy), computes z from current
    obs and calls policy.get_action(batch_data, z=z, val=True).
    """
    try:
        device = next(policy.parameters()).device
    except:
        device = torch.device('cpu')

    policy.eval()
    if task_encoder is not None:
        task_encoder.eval()

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

    # Dictionary to group results by graph type
    type_results = {}

    obs_list, _ = env.reset()
    finished = False

    while not finished:
        with torch.no_grad():
            batch_data = Batch(device, [ig_to_data(g) for g in obs_list])
            if task_encoder is not None:
                z_batch = task_encoder.get_z(batch_data)
                act_arr, *rest = policy.get_action(batch_data, z=z_batch, val=True)
            else:
                act_arr, *rest = policy.get_action(batch_data, val=True)
        act_arr = act_arr.cpu().numpy()

        obs_next_list, rew_arr, done_arr, info_list = env.step(act_arr)
        obs_next_list, _ = env.reset_async(done_arr)

        finished = (len(obs_next_list) == 0)
        obs_list = np.array(obs_next_list)

        for logger in info_list:
            # print(f'{logger.name}: AUC={logger.auc:.6f}, Robustness={logger.robustness:.6f}')
            auc = logger.auc / logger.n_init
            auc_list.append(auc)
            robustness_list.append(logger.robustness)
            lcc_curve_list.append(logger.gcc_eps)
            if log_removals:
                removals_list.append(logger.removals)
            
            # Group by graph type
            t = logger.name.split("_")[-1]
            g = logger.name.split("_", 1)[1]
            if t not in type_results:
                type_results[t] = []
            type_results[t].append(auc)
            
            if csv_pth:
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

    # Log average AUC by graph type
    for graph_type, aucs in type_results.items():
        avg_auc = sum(aucs) / len(aucs)
        print(f'{graph_type}: Avg AUC = {avg_auc:.6f} ({len(aucs)} graphs)')
    

    if log_removals:
        return auc_list, robustness_list, lcc_curve_list, removals_list
    else:
        return auc_list, robustness_list, lcc_curve_list


def validate_step(obs_list,policy,device,step_size=1):
    with torch.no_grad():
        batch_data = Batch(device, [ig_to_data(g) for g in obs_list])
        _, log_probs = policy.get_action(batch_data, val=True)

    # Convert to probabilities and get top-k nodes
    probs = log_probs.exp().cpu().numpy()

    # Select top step_size nodes based on probabilities
    if step_size > 1:
        step_indices = np.argsort(-probs)[:min(step_size, len(probs))]
        step_indices = sorted(step_indices)[::-1] # Remove in reverse order to avoid index shifts
    else:
        step_indices = [np.argmax(probs)]
    return step_indices #[B,step_size]

def validate_one_graph(graph, policy, save_res=None, step_ratio=None, log_removals=False):
    """
    Validate policy on a single graph with optional step_ratio for batch node removal.
    """
    try:
        device = next(policy.parameters()).device
    except:
        device = torch.device('cpu')

    # Make a copy and initialize tracking
    g = deepcopy(graph)
    n_init = g.vcount()
    
    # Calculate step size
    if step_ratio is not None and step_ratio > 0:
        step_size = max(int(step_ratio * n_init), 1)
    else:
        step_size = 1

    policy.eval()
    from env.env import DismantleEnv
    env = DismantleEnv(graph_data=[graph], batch_size=1, is_val=True)
    obs_list, _ = env.reset()
    finished = False

    while not finished:
        step_indices = validate_step(obs_list,policy,device,step_size)
        for idx in step_indices:
            if idx < obs_list[0].vcount(): #safety check
                obs_next_list, rew_arr, done_arr, info_list = env.step(np.array([idx]))
                # obs_next_list, _ = env.reset_async(done_arr)
                if len(info_list) > 0:
                    removals = info_list[0].removals
                    auc = info_list[0].auc
                    robustness = info_list[0].robustness
                    gcc_eps = info_list[0].gcc_eps
                    finished = True
                    break
                obs_list = np.array(obs_next_list)
    
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