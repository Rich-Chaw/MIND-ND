from sqlite3 import Time
import igraph as ig
import networkx as nx
import subprocess
import json
import tempfile
import pickle
import os
import time
import sys

# Path to NIRM project and its python_interface.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINDER_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "FINDER"))
FINDER_INTERFACE = os.path.join(FINDER_ROOT, "python_interface.py")

# Python executable preference:
# 1) env var FINDER_PYTHON (if provided)
# 2) hard-coded tf_py37 path
# 3) current interpreter (sys.executable) as fallback
# FINDER_PYTHON = os.path.join("D:\\Anaconda3\\envs\\tf_py37\\python.exe")
FINDER_PYTHON = "~/autodl-tmp/miniconda3/envs/tf_py37/bin/python"


def _resolve_finder_python() -> str:
    candidate = os.environ.get("FINDER_PYTHON", FINDER_PYTHON)
    candidate = os.path.abspath(os.path.expanduser(candidate))
    if os.path.exists(candidate):
        return candidate
    return sys.executable


FINDER_PYTHON = _resolve_finder_python()

def FINDER_wrapper(graph:ig.Graph):
    """
    Use FINDER to dismantle a graph
    
    Args:
        graph: igraph.Graph object
        
    Returns:
        tuple: (removed_nodes_list, score, MaxCCList)
    """
    # Create temporary pickle file for the graph, absolute path
    temp_graph_fd, temp_graph_file = tempfile.mkstemp(suffix='.pkl')
    temp_out_fd, temp_out_file = tempfile.mkstemp(suffix='.json')
    os.close(temp_graph_fd)
    os.close(temp_out_fd)

    try:
        # Save graph to temporary pickle file
        with open(temp_graph_file, 'wb') as f:
            pickle.dump(graph, f)

        start_time = time.time()
        # Run the interface
        result = subprocess.run(
            [FINDER_PYTHON, FINDER_INTERFACE, "--graph_file", temp_graph_file, "--out_file", temp_out_file],
            capture_output=True, text=True, cwd=os.path.dirname(FINDER_INTERFACE)
        )
        
        if result.returncode == 0:
            with open(temp_out_file, 'r') as f:
                output_data = json.load(f)
            removals = output_data['removals']
            score = output_data['score']
            MaxCCList = output_data['MaxCCList']
            return removals, score, MaxCCList, time.time()-start_time
        else:
            print(f"Error in FINDER_wrapper returncode: {result.stderr}")
            return None, 0.0, None, None
    except Exception as e:
        print(f"Error in FINDER_wrapper: {e}")
        return None, 0.0, None, None

    finally:
        # Clean up temporary file
        if os.path.exists(temp_graph_file):
            os.remove(temp_graph_file)
        if os.path.exists(temp_out_file):
            os.remove(temp_out_file)
        

if __name__ == "__main__":
    import pickle
    import time
    with open("../graphs/real/FINDER/Digg.pkl",'rb') as f:
        graph = pickle.load(f)
    
    removals, score, _,_ = FINDER_wrapper(graph)
    print(f"finder Score: {score}")

    import sys
    sys.path.append("..")
    from baseline import evaluate_sol, evaluate_sol_networkx
    start_time = time.time()
    auc, robustness = evaluate_sol(graph, removals)
    print(f"igraph Score: {robustness}, time: {time.time() - start_time}")
    
    start_time = time.time()
    robustness_networkx = evaluate_sol_networkx(graph, removals)
    print(f"networkx Score: {robustness_networkx}, time: {time.time() - start_time}")
    