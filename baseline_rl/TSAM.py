"""
TSAM wrapper for MIND-ND: call TSAM/python_interface.py via subprocess and
return (removals, score, MaxCCList). TSAM does not output score/MaxCCList; score is 0.0, MaxCCList [].
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

from networkx import Graph

# Path to TSAM project and its python_interface.py (relative to this file)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TSAM_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "TSAM"))
TSAM_INTERFACE = os.path.join(TSAM_ROOT, "python_interface.py")

TSAM_PYTHON = sys.executable


def TSAM_wrapper(
    graph: ig.Graph,
    threshold: float = 0.01,
) -> Tuple[Optional[List[int]], float, List[float]]:
    """
    Use TSAM to dismantle a graph.

    Args:
        graph: igraph.Graph object.
        threshold: Dismantling threshold (default: 0.01).

    Returns:
        (removals, score, MaxCCList). TSAM does not compute score/MaxCCList; score=0.0, MaxCCList=[].
        removals is None on failure.
    """
    temp_graph_fd, temp_graph_file = tempfile.mkstemp(suffix=".pkl")
    temp_out_fd, temp_out_file = tempfile.mkstemp(suffix=".json")
    os.close(temp_graph_fd)
    os.close(temp_out_fd)

    try:
        with open(temp_graph_file, "wb") as f:
            pickle.dump(graph, f)

        cmd = [
            TSAM_PYTHON,
            TSAM_INTERFACE,
            "--graph_file",
            temp_graph_file,
            "--out_file",
            temp_out_file,
            "--threshold",
            str(threshold),
        ]

        # Avoid "libgomp: Invalid value for environment variable OMP_NUM_THREADS"
        env = os.environ.copy()
        env.pop("OMP_NUM_THREADS", None)
        env["OMP_NUM_THREADS"] = "1"

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(TSAM_INTERFACE),
            env=env,
        )

        if result.returncode != 0:
            print(
                f"Error in TSAM_wrapper returncode: {result.returncode} {result.stderr}",
                file=sys.stderr,
            )
            return None, 0.0, [], None

        with open(temp_out_file, "r") as f:
            output_data = json.load(f)

        removals = output_data.get("removals", [])
        # TSAM does not output robustness score or MaxCCList; use placeholders
        score = float(output_data.get("score", 0.0))
        MaxCCList = output_data.get("MaxCCList", [])
        return removals, score, MaxCCList, time.time() - start_time

    except Exception as e:
        print(f"Error in TSAM_wrapper: {e}", file=sys.stderr)
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
    start = time.time()
    removals, score, MaxCCList, runtime = TSAM_wrapper(graph, threshold=0.01)
    sys.path.append("..")
    from baseline import evaluate_sol, evaluate_sol_networkx
    auc, robustness = evaluate_sol(graph, removals)
    print(
        f"TSAM returned {len(removals) if removals else 0} removals, "
        f"score={score}, time={runtime:.3f}s",
    )
