"""
GDM wrapper for MIND-ND: call CoreGDM/python_interface.py (GDM or CoreGDM heuristic)
via subprocess and return (removals, score, MaxCCList, runtime).
"""

import igraph as ig
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from typing import List, Tuple, Optional

# Path to CoreGDM project and its python_interface.py (relative to this file)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COREGDM_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "CoreGDM"))
COREGDM_INTERFACE = os.path.join(COREGDM_ROOT, "python_interface.py")

GDM_PYTHON = sys.executable


def GDM_wrapper(
    graph: ig.Graph,
    threshold: float = 0.01,
    heuristic: str = "GDM",
) -> Tuple[Optional[List[int]], float, List[float], Optional[float]]:
    """
    Use GDM or CoreGDM (from CoreGDM project) to dismantle a graph.

    Args:
        graph: igraph.Graph object.
        threshold: Dismantling LCC threshold in [0, 1] (default: 0.01).
        heuristic: "GDM" or "CoreGDM" (default: "GDM").

    Returns:
        (removals, score, MaxCCList, runtime).
        score is r_auc from the dismantler; MaxCCList is [] (not provided by interface).
        removals is None on failure; runtime is None on failure.
    """
    temp_graph_fd, temp_graph_file = tempfile.mkstemp(suffix=".pkl")
    temp_out_fd, temp_out_file = tempfile.mkstemp(suffix=".json")
    os.close(temp_graph_fd)
    os.close(temp_out_fd)

    try:
        with open(temp_graph_file, "wb") as f:
            pickle.dump(graph, f)

        cmd = [
            GDM_PYTHON,
            COREGDM_INTERFACE,
            "--graph_file",
            temp_graph_file,
            "--out_file",
            temp_out_file,
            "--threshold",
            str(threshold),
            "--heuristic",
            heuristic,
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=COREGDM_ROOT,
        )

        runtime = time.time() - start_time

        if result.returncode != 0:
            print(
                f"Error in GDM_wrapper returncode: {result.returncode} {result.stderr}",
                file=sys.stderr,
            )
            return None, 0.0, [], None

        with open(temp_out_file, "r") as f:
            output_data = json.load(f)

        removals = output_data.get("removals", [])
        score = float(output_data.get("r_auc", 0.0))
        # CoreGDM interface does not output MaxCCList
        MaxCCList = output_data.get("MaxCCList", [])
        return removals, score, MaxCCList, runtime

    except Exception as e:
        print(f"Error in GDM_wrapper: {e}", file=sys.stderr)
        return None, 0.0, [], None

    finally:
        if os.path.exists(temp_graph_file):
            try:
                os.remove(temp_graph_file)
            except OSError:
                pass
        if os.path.exists(temp_out_file):
            try:
                os.remove(temp_out_file)
            except OSError:
                pass


if __name__ == "__main__":
    graph = ig.Graph.Erdos_Renyi(n=100, p=0.05)

    removals, score, MaxCCList, runtime = GDM_wrapper(graph, threshold=0.01)

    sys.path.append("..")
    from baseline import evaluate_sol, evaluate_sol_networkx

    auc, robustness = evaluate_sol(graph, removals)
    print(
        f"GDM returned {len(removals) if removals else 0} removals, "
        f"score={score}, time={runtime:.3f}s",
    )

    removals, score, MaxCCList, runtime = GDM_wrapper(graph, threshold=0.01, heuristic='CoreGDM')

    sys.path.append("..")
    from baseline import evaluate_sol, evaluate_sol_networkx

    auc, robustness = evaluate_sol(graph, removals)
    print(
        f"GDM returned {len(removals) if removals else 0} removals, "
        f"score={score}, time={runtime:.3f}s",
    )

