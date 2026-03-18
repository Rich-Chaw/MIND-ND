"""
GND wrapper for MIND-ND: call Generalized-Network-Dismantling via its
python_interface.py and return (removals, score, MaxCCList).
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

# Path to GND project and its python_interface.py (relative to this file: baseline_rl -> MIND-ND -> workspace -> Generalized-Network-Dismantling)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GND_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "Generalized-Network-Dismantling"))
GND_INTERFACE = os.path.join(GND_ROOT, "python_interface.py")

GND_PYTHON = sys.executable


def GND_wrapper(
    graph: ig.Graph,
    binary_path: Optional[str] = None,
    threshold: float = 0.01,
) -> Tuple[Optional[List[int]], float, List[float]]:
    """
    Use GND to dismantle a graph.

    Args:
        graph: igraph.Graph object.
        binary_path: Optional path to GND executable; forwarded to python_interface.
        threshold: Dismantling threshold; stop when GCC size < threshold * N (default: 0.01).

    Returns:
        (removals, score, MaxCCList). removals is None on failure.
    """
    temp_graph_fd, temp_graph_file = tempfile.mkstemp(suffix=".pkl")
    temp_out_fd, temp_out_file = tempfile.mkstemp(suffix=".json")
    os.close(temp_graph_fd)
    os.close(temp_out_fd)

    try:
        with open(temp_graph_file, "wb") as f:
            pickle.dump(graph, f)

        cmd = [
            GND_PYTHON,
            GND_INTERFACE,
            "--graph_file",
            temp_graph_file,
            "--out_file",
            temp_out_file,
            "--threshold",
            str(threshold),
        ]
        if binary_path:
            cmd.extend(["--binary_path", os.path.abspath(binary_path)])
    
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(GND_INTERFACE),
        )

        if result.returncode != 0:
            print(
                f"Error in GND_wrapper returncode: {result.returncode} {result.stderr}",
                file=sys.stderr,
            )
            return None, 0.0, []

        with open(temp_out_file, "r") as f:
            output_data = json.load(f)

        removals = output_data.get("removals", [])
        score = float(output_data.get("score", 0.0))
        MaxCCList = output_data.get("MaxCCList", [])
        return removals, score, MaxCCList, time.time() - start_time

    except Exception as e:
        print(f"Error in GND_wrapper: {e}", file=sys.stderr)
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
    import time

    graph = ig.Graph.Erdos_Renyi(n=100, p=0.05)

    removals, score, MaxCCList, runtime = GND_wrapper(graph)

    sys.path.append("..")
    from baseline import evaluate_sol, evaluate_sol_networkx

    auc, robustness = evaluate_sol(graph, removals)
    print(
        f"GND returned {len(removals) if removals else 0} removals, "
        f"score={score}, time={runtime}s",
    )
