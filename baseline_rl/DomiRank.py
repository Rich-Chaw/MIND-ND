"""
DomiRank wrapper for MIND-ND: call DomiRank/python_interface.py and
return (removals, score, MaxCCList).
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

# Path to DomiRank project and its python_interface.py (relative to this file)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOMIRANK_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "DomiRank"))
DOMIRANK_INTERFACE = os.path.join(DOMIRANK_ROOT, "python_interface.py")

DOMIRANK_PYTHON = sys.executable


def DomiRank_wrapper(
    graph: ig.Graph,
    analytical: bool = False,
    sampling: int = 0,
    threshold: float = 0.01,
) -> Tuple[Optional[List[int]], float, List[float]]:
    """
    Use DomiRank to dismantle a graph.

    Args:
        graph: igraph.Graph object.
        analytical: Whether to use analytical domirank solution.
        sampling: Sampling interval for robustness curve (0 = auto).
        threshold: Dismantling threshold; stop when normalized LCC < threshold (default: 0.01).

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
            DOMIRANK_PYTHON,
            DOMIRANK_INTERFACE,
            "--graph_file",
            temp_graph_file,
            "--out_file",
            temp_out_file,
            "--threshold",
            str(threshold),
        ]
        if analytical:
            cmd.append("--analytical")
        if sampling is not None and sampling > 0:
            cmd.extend(["--sampling", str(sampling)])
    
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(DOMIRANK_INTERFACE),
        )

        if result.returncode != 0:
            print(
                f"Error in DomiRank_wrapper returncode: {result.stderr}",
                file=sys.stderr,
            )
            return None, 0.0, [], None

        with open(temp_out_file, "r") as f:
            output_data = json.load(f)

        removals = output_data.get("removals", [])
        score = float(output_data.get("score", 0.0))
        MaxCCList = output_data.get("MaxCCList", [])
        return removals, score, MaxCCList, time.time() - start_time

    except Exception as e:
        print(f"Error in DomiRank_wrapper: {e}", file=sys.stderr)
        return None, 0.0, [], None

    finally:
        if os.path.exists(temp_graph_file):
            os.remove(temp_graph_file)
        if os.path.exists(temp_out_file):
            os.remove(temp_out_file)


if __name__ == "__main__":
    import time

    graph = ig.Graph.Erdos_Renyi(n=100, p=0.05)

    removals, score, MaxCCList, runtime = DomiRank_wrapper(graph, threshold=0.01)

    import sys
    sys.path.append("..")
    from baseline import evaluate_sol, evaluate_sol_networkx
    auc, robustness = evaluate_sol(graph, removals)
    print(
        f"DomiRank returned {len(removals) if removals else 0} removals, score={score}, "
        f"time={runtime:.3f}s",
    )
