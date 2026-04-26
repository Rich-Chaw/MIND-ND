import os
import pickle
import pickletools
import sys
from collections import deque
from typing import Any, Dict, List, Set, Tuple


TARGET_PATH = r"O:\My_Codes\GD2026\MIND-ND\graphs\real\bio\ppi.pkl"


def sizeof_deep(obj: Any, seen: Set[int] = None) -> int:
    """Rough recursive size in bytes, avoiding double counting."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += sizeof_deep(k, seen)
            size += sizeof_deep(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset, deque)):
        for item in obj:
            size += sizeof_deep(item, seen)
    elif hasattr(obj, "__dict__"):
        size += sizeof_deep(vars(obj), seen)
    elif hasattr(obj, "__slots__"):
        for slot in obj.__slots__:
            if hasattr(obj, slot):
                size += sizeof_deep(getattr(obj, slot), seen)
    return size


def summarize_dict(d: Dict[Any, Any], topk: int = 15) -> List[Tuple[str, str, int]]:
    rows = []
    for k, v in d.items():
        key_name = repr(k)
        if len(key_name) > 80:
            key_name = key_name[:77] + "..."
        rows.append((key_name, type(v).__name__, sizeof_deep(v)))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows[:topk]


def analyze_pickle(path: str) -> None:
    file_size = os.path.getsize(path)
    print("=" * 80)
    print(f"File: {path}")
    print(f"On-disk size: {file_size:,} bytes ({file_size / (1024 ** 2):.2f} MB)")
    print("=" * 80)

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except ModuleNotFoundError as e:
        print(f"Cannot fully unpickle due to missing dependency: {e}")
        print("-" * 80)
        print("Fallback: inspect pickle opcode payload distribution.")
        analyze_pickle_payload(path)
        return

    root_type = type(obj).__name__
    deep_size = sizeof_deep(obj)
    print(f"Root object type: {root_type}")
    print(f"Estimated in-memory size: {deep_size:,} bytes ({deep_size / (1024 ** 2):.2f} MB)")
    print(f"In-memory / on-disk ratio: {deep_size / max(file_size, 1):.2f}x")
    print("-" * 80)

    if isinstance(obj, dict):
        print(f"Top-level dict keys: {len(obj)}")
        print("Largest top-level values:")
        for key_name, tname, b in summarize_dict(obj):
            print(f"  - key={key_name:<30} type={tname:<20} size={b / (1024 ** 2):8.2f} MB")
    else:
        print("Top-level object is not a dict, skipping key-wise summary.")

    try:
        import networkx as nx  # noqa: F401
    except Exception:
        nx = None

    if nx is not None and hasattr(obj, "number_of_nodes") and hasattr(obj, "number_of_edges"):
        print("-" * 80)
        print("Looks like a NetworkX graph:")
        print(f"  nodes: {obj.number_of_nodes():,}")
        print(f"  edges: {obj.number_of_edges():,}")

        node_attrs = 0
        for _, attrs in obj.nodes(data=True):
            if attrs:
                node_attrs += len(attrs)
        edge_attrs = 0
        for _, _, attrs in obj.edges(data=True):
            if attrs:
                edge_attrs += len(attrs)

        print(f"  total node attributes: {node_attrs:,}")
        print(f"  total edge attributes: {edge_attrs:,}")
        print("  likely-heavy components in NetworkX are adjacency dicts and attribute dicts.")

    print("-" * 80)
    print("Interpretation tips:")
    print("1) Many small Python dict/list objects cause large overhead.")
    print("2) Dense edges increase adjacency metadata heavily (esp. in NetworkX).")
    print("3) Rich attributes on nodes/edges can dominate pickle size.")
    print("4) Pickle stores Python object structure, not a compact graph binary format.")


def analyze_pickle_payload(path: str) -> None:
    with open(path, "rb") as f:
        data = f.read()

    byte_payload = 0
    text_payload = 0
    global_refs: Dict[str, int] = {}
    payload_items = 0

    for op, arg, _ in pickletools.genops(data):
        name = op.name
        if name in {"BINBYTES", "SHORT_BINBYTES", "BINBYTES8"}:
            payload_items += 1
            byte_payload += len(arg)
        elif name in {"BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "UNICODE"}:
            payload_items += 1
            text_payload += len(arg.encode("utf-8", errors="ignore"))
        elif name in {"GLOBAL", "STACK_GLOBAL"}:
            key = str(arg)
            global_refs[key] = global_refs.get(key, 0) + 1

    print(f"Total pickle stream size: {len(data):,} bytes ({len(data) / (1024 ** 2):.2f} MB)")
    print(f"Detected payload items (bytes/unicode): {payload_items:,}")
    print(f"Embedded raw-bytes payload total: {byte_payload:,} bytes ({byte_payload / (1024 ** 2):.2f} MB)")
    print(f"Embedded unicode payload total: {text_payload:,} bytes ({text_payload / (1024 ** 2):.2f} MB)")
    print(f"Protocol/containers/metadata overhead: {len(data) - byte_payload - text_payload:,} bytes")

    if global_refs:
        print("Most frequent GLOBAL refs:")
        for k, v in sorted(global_refs.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {k}: {v}")

    print("-" * 80)
    print("Likely reason this file is large:")
    print("1) Graph serialization stores many nested Python containers.")
    print("2) Repeated topology/attributes produce lots of metadata records.")
    print("3) If generated by igraph/network tools, structural overhead is significant.")


if __name__ == "__main__":
    analyze_pickle(TARGET_PATH)
