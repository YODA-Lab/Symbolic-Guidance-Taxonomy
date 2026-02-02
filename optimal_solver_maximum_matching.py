# batch_maximum_matching_with_nodes.py
from __future__ import annotations
import json
import os
import re
from typing import Dict, Any, Iterable, List, Tuple, Union

# ---------- Core solver ----------

def _solve_maximum_matching(graph_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a maximum-cardinality matching for a graph in your JSON format."""
    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError("Please install networkx: pip install networkx") from e

    nodes = graph_obj["graph"]["nodes"]
    links = graph_obj["graph"]["links"]

    id_to_name = {n["id"]: n.get("name", str(n["id"])) for n in nodes}

    G = nx.Graph()
    G.add_nodes_from(id_to_name.keys())
    for e in links:
        u, v = e["source"], e["target"]
        if u == v:
            continue
        G.add_edge(u, v, weight=1)  # weight=1 so max_weight_matching with maxcardinality==True -> max cardinality

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    pairs_by_id = [tuple(sorted(edge)) for edge in matching]
    pairs_by_id.sort()
    pairs_by_name = [(id_to_name[u], id_to_name[v]) for (u, v) in pairs_by_id]

    partner = {nid: None for nid in id_to_name}
    for u, v in pairs_by_id:
        partner[u] = v
        partner[v] = u

    return {
        "size": len(pairs_by_id),
        "pairs_by_id": pairs_by_id,
        "pairs_by_name": pairs_by_name,
        "partner_by_id": partner,
    }

# ---------- Batch runner for pattern graph_<type>_<nodes>_<id>.json ----------

_FILENAME_RE = re.compile(r"^graph_(?P<type>[A-Za-z0-9\-]+)_(?P<nodes>\d+)_(?P<id>\d+)\.json$")

def _parse_filename(fname: str):
    """
    Parse 'graph_<type>_<nodes>_<id>.json' into (type:str, nodes:int, id:int).
    Return None if not matching.
    """
    m = _FILENAME_RE.match(fname)
    if not m:
        return None
    return m.group("type"), int(m.group("nodes")), int(m.group("id"))

def load_graph_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def batch_maximum_matchings_with_nodes(
    folder: str,
    graph_types: Iterable[str],
    node_counts: Union[int, Iterable[int]],
    id_start: int,
    id_end_inclusive: int,
    *,
    skip_missing: bool = True
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Scan `folder` for files named 'graph_<type>_<nodes>_<id>.json' and compute maximum matchings.

    Args:
      folder: directory path
      graph_types: list of types to include (e.g., ["ba", "ws"])
      node_counts: single int or list of ints (e.g., 4 or [4, 8, 16])
      id_start, id_end_inclusive: inclusive id range to include
      skip_missing: ignored here (we scan directory; nothing to miss by direct name), kept for API symmetry

    Returns:
      { "<type>": { <id>: {size, pairs_by_id, pairs_by_name, partner_by_id}, ... }, ... }
      (Only ids that matched BOTH the type and the requested node count(s) and id range are included.)

    Notes:
      - If multiple files with the same (type, id) but different node counts match your filters,
        the last one seen will win. If you want to keep them separate, change the return key
        to include nodes as well (e.g., results[type][(nodes, id)]).
    """
    # Normalize node_counts into a set
    if isinstance(node_counts, int):
        node_set = {node_counts}
    else:
        node_set = set(int(x) for x in node_counts)

    type_set = set(graph_types)
    results: Dict[str, Dict[int, Dict[str, Any]]] = {t: {} for t in type_set}

    # Scan directory once
    try:
        fnames = os.listdir(folder)
    except FileNotFoundError:
        if skip_missing:
            return results
        raise

    for fname in fnames:
        if not fname.endswith(".json"):
            continue
        parsed = _parse_filename(fname)
        if parsed is None:
            continue
        gtype, nnodes, gid = parsed

        if gtype not in type_set:
            continue
        if nnodes not in node_set:
            continue
        if not (id_start <= gid <= id_end_inclusive):
            continue

        fpath = os.path.join(folder, fname)
        try:
            gobj = load_graph_json(fpath)
            res = _solve_maximum_matching(gobj)
            results[gtype][gid] = res
        except Exception as e:
            results[gtype][gid] = {"error": str(e)}

    return results

# -----------------------
# Example usage:
# -----------------------
if __name__ == "__main__":
    folder = "./graphs"
    types = ["ba", "ws", "dt"]
    node_counts = [16]     # accept both 4-node and 8-node graphs
    id_start, id_end = 0, 20

    out = batch_maximum_matchings_with_nodes(folder, types, node_counts, id_start, id_end)
    for t, per_id in out.items():
        print(f"[{t}]")
        for i in sorted(per_id):
            r = per_id[i]
            if "error" in r:
                print(f"  id={i}: ERROR -> {r['error']}")
            else:
                print(f"  id={i}: size={r['size']} pairs={r['pairs_by_id']}")
