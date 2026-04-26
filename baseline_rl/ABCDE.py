"""
ABCDE wrapper for MIND-ND.

Call abcde `python_interface.py` via subprocess and return removals/runtime.
"""

import igraph as ig
import subprocess
import json
import tempfile
import pickle
import os
import time
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABCDE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "abcde"))
ABCDE_INTERFACE = os.path.join(ABCDE_ROOT, "python_interface.py")

# Python executable preference:
# 1) env var ABCDE_PYTHON (if provided)
# 2) torch_py38 path on Windows
# 3) current interpreter (sys.executable)
_DEFAULT_PY = os.path.join("D:\\Anaconda3\\envs\\torch_py38\\python.exe")
ABCDE_PYTHON = os.environ.get("ABCDE_PYTHON", _DEFAULT_PY)
if not os.path.exists(ABCDE_PYTHON):
    ABCDE_PYTHON = sys.executable


def ABCDE_wrapper(graph: ig.Graph, model_path=None, adaptive=True, threshold=0.1, max_steps=None, device="cpu"):
    """
    Use ABCDE to dismantle a graph.

    Args:
        graph: igraph.Graph object
        model_path: Optional checkpoint path for ABCDE (.ckpt)
        adaptive: Whether to use adaptive mode (--adaptive)
        threshold: Terminal threshold passed through
        max_steps: Optional max steps passed through
        device: cpu/cuda passed to interface

    Returns:
        tuple: (removed_nodes_list, score, MaxCCList, runtime)
    """
    temp_graph_fd, temp_graph_file = tempfile.mkstemp(suffix=".pkl")
    temp_out_fd, temp_out_file = tempfile.mkstemp(suffix=".json")
    os.close(temp_graph_fd)
    os.close(temp_out_fd)

    try:
        with open(temp_graph_file, "wb") as f:
            pickle.dump(graph, f)

        cmd = [
            ABCDE_PYTHON,
            ABCDE_INTERFACE,
            "--graph_file",
            temp_graph_file,
            "--out_file",
            temp_out_file,
            "--threshold",
            str(threshold),
            "--device",
            str(device),
        ]
        if adaptive:
            cmd.append("--adaptive")
        if model_path:
            cmd.extend(["--model_path", model_path])
        if max_steps is not None:
            cmd.extend(["--max_steps", str(int(max_steps))])

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(ABCDE_INTERFACE),
        )
        runtime = time.time() - start_time

        if result.returncode == 0:
            with open(temp_out_file, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            removals = output_data.get("removals", [])
            score = output_data.get("robustness", 0.0)
            max_cc_list = output_data.get("MaxCCList", [])
            return removals, score, max_cc_list, runtime

        print(f"Error in ABCDE_wrapper returncode: {result.stderr}", file=sys.stderr)
        return None, 0.0, None, runtime
    except Exception as e:
        print(f"Error in ABCDE_wrapper: {e}", file=sys.stderr)
        return None, 0.0, None, None
    finally:
        if os.path.exists(temp_graph_file):
            os.remove(temp_graph_file)
        if os.path.exists(temp_out_file):
            os.remove(temp_out_file)

