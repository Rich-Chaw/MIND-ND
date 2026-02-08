"""
NIRM wrapper for MIND-ND: call NIRM dismantling via subprocess and return (removals, score, MaxCCList).
Follows the same pattern as FINDER.py and AAA-NetDQN python_interface.
"""

import igraph as ig
import subprocess
import json
import tempfile
import pickle
import os
import sys

# Path to NIRM project and its python_interface.py
NIRM_ROOT = os.path.join("O:\\My_Codes\\GD2026\\NIRM")
NIRM_INTERFACE = os.path.join(NIRM_ROOT, "python_interface.py")

# Python executable: use current interpreter so NIRM env is used when available
NIRM_PYTHON = sys.executable


def NIRM_wrapper(graph: ig.Graph, model_path=None):
    """
    Use NIRM to dismantle a graph.

    Args:
        graph: igraph.Graph object
        model_path: Optional path to NIRM checkpoint .pkl (default: NIRM/checkpoints/NIRM_onepass.pkl)

    Returns:
        tuple: (removed_nodes_list, score, MaxCCList)
    """
    temp_graph_fd, temp_graph_file = tempfile.mkstemp(suffix='.pkl')
    temp_out_fd, temp_out_file = tempfile.mkstemp(suffix='.json')
    os.close(temp_graph_fd)
    os.close(temp_out_fd)

    try:
        with open(temp_graph_file, 'wb') as f:
            pickle.dump(graph, f)

        cmd = [NIRM_PYTHON, NIRM_INTERFACE, "--graph_file", temp_graph_file, "--out_file", temp_out_file]
        if model_path:
            cmd.extend(["--model_path", model_path])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(NIRM_INTERFACE),
        )

        if result.returncode == 0:
            with open(temp_out_file, 'r') as f:
                output_data = json.load(f)
            removals = output_data['removals']
            return removals
        else:
            print(f"Error in NIRM_wrapper returncode: {result.stderr}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Error in NIRM_wrapper: {e}", file=sys.stderr)
        return None
    finally:
        if os.path.exists(temp_graph_file):
            os.remove(temp_graph_file)
        if os.path.exists(temp_out_file):
            os.remove(temp_out_file)


if __name__ == "__main__":
    import time
    with open("../graphs/real/FINDER/Crime.pkl", 'rb') as f:
        graph = pickle.load(f)

    removals = NIRM_wrapper(graph)

    sys.path.append("..")
    from baseline import evaluate_sol, evaluate_sol_networkx
    start_time = time.time()
    auc, robustness = evaluate_sol(graph, removals)
    print(f"igraph Score: {robustness}, time: {time.time() - start_time}")

    start_time = time.time()
    robustness_networkx = evaluate_sol_networkx(graph, removals)
    print(f"networkx Score: {robustness_networkx}, time: {time.time() - start_time}")
