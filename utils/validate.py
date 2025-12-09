import torch
import numpy as np
import csv
import os
from datetime import datetime
from .graph_data import Batch, ig_to_data

#TODO:inplement step_ratio

def validate(env, policy, save_res=None, step_ratio=None,log_removals=False):
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
            act_arr, *rest = policy.get_action(
                Batch(device, [ig_to_data(g) for g in obs_list]), 
                val=True
            )
        act_arr = act_arr.cpu().numpy()

        obs_next_list, rew_arr, done_arr, info_list = env.step(act_arr)
        obs_next_list, _ = env.reset_async(done_arr)

        finished = (len(obs_next_list) == 0)

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

        obs_list = np.array(obs_next_list)

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