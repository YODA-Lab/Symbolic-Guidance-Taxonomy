#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import re
import argparse
from typing import Dict, Any, Iterable, Tuple, Union

# ==========================================================
# Optional: networkx import (only needed if you pass nx.Graph)
# ==========================================================
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None  # we'll guard usage

# ==========================================================
# Filename pattern: graph_<type>_<nodes>_<id>.json
# ==========================================================
_FILENAME_RE = re.compile(r"^graph_(?P<type>[A-Za-z0-9\-]+)_(?P<nodes>\d+)_(?P<id>\d+)\.json$")

def _parse_filename(fname: str):
    m = _FILENAME_RE.match(fname)
    if not m:
        return None
    return m.group("type"), int(m.group("nodes")), int(m.group("id"))

def load_graph_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==========================================================
# Converters / extractors
# ==========================================================
def _json_to_nx(graph_obj: Dict[str, Any]):
    """
    Convert your JSON format into a simple undirected nx.Graph (if networkx is available).
    """
    if nx is None:
        raise RuntimeError("networkx is required for nx.Graph input. Install with: pip install networkx")

    data = graph_obj["graph"]
    G = nx.Graph()

    for n in data.get("nodes", []):
        nid = n["id"]
        attrs = {k: v for k, v in n.items() if k != "id"}
        G.add_node(nid, **attrs)

    seen = set()
    for e in data.get("links", []):
        u, v = e["source"], e["target"]
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        edge_attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
        G.add_edge(a, b, **edge_attrs)

    return G

def _extract_nodes_edges(graph_input: Union[Dict[str, Any], "nx.Graph"]) -> Tuple[list, list]:
    """
    Accepts either:
      - dict in your JSON format (with 'graph' key)
      - networkx.Graph (undirected)

    Returns:
      nodes: list of node labels (as they appear in input)
      edges: list of (u, v) with u < v, deduped, no self-loops
    """
    # JSON path
    if isinstance(graph_input, dict) and "graph" in graph_input:
        nodes = [n["id"] for n in graph_input["graph"]["nodes"]]
        seen = set()
        edges = []
        for e in graph_input["graph"]["links"]:
            u, v = e["source"], e["target"]
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
        return nodes, edges

    # networkx path
    if nx is not None and isinstance(graph_input, nx.Graph):
        G = graph_input
        nodes = list(G.nodes())
        seen = set()
        edges = []
        for u, v in G.edges():
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
        return nodes, edges

    raise TypeError("graph_input must be a JSON dict with 'graph' key or a networkx.Graph")

# ==========================================================
# Exact solver via MILP (PuLP) or CP-SAT (OR-Tools)
# Objective: maximize #edges with different colors (k=3)
# ==========================================================
def _solve_max3cut_ilp(nodes: list, edges: list) -> Dict[str, Any]:
    try:
        import pulp  # type: ignore
    except ImportError as e:
        raise RuntimeError("PuLP not installed. Install with: pip install pulp") from e

    C = [0, 1, 2]
    model = pulp.LpProblem("Max3Cut_MinSameColorEdges", pulp.LpMinimize)

    x = {(v, c): pulp.LpVariable(f"x_{v}_{c}", 0, 1, cat="Binary") for v in nodes for c in C}
    y = {((u, v), c): pulp.LpVariable(f"y_{u}_{v}_{c}", 0, 1, cat="Binary") for (u, v) in edges for c in C}

    for v in nodes:
        model += pulp.lpSum(x[(v, c)] for c in C) == 1

    for (u, v) in edges:
        for c in C:
            model += y[((u, v), c)] >= x[(u, c)] + x[(v, c)] - 1
            model += y[((u, v), c)] <= x[(u, c)]
            model += y[((u, v), c)] <= x[(v, c)]

    model += pulp.lpSum(y[((u, v), c)] for (u, v) in edges for c in C)

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] not in ("Optimal", "Optimal Infeasible", "Integer Feasible"):
        raise RuntimeError(f"ILP solver did not succeed. Status={pulp.LpStatus[status]}")

    coloring = {}
    for v in nodes:
        chosen = 0
        for c in C:
            if pulp.value(x[(v, c)]) >= 0.5:
                chosen = c
                break
        coloring[v] = int(chosen)

    same_color_edges = 0
    for (u, v) in edges:
        if coloring[u] == coloring[v]:
            same_color_edges += 1
    satisfied = len(edges) - same_color_edges
    ratio = (satisfied / len(edges)) if edges else 0.0

    return {
        "num_edges": len(edges),
        "satisfied_edges": satisfied,
        "ratio": ratio,
        "coloring_by_id": coloring,
        "method": "pulp_cbc",
    }

def _solve_max3cut_cpsat(nodes: list, edges: list) -> Dict[str, Any]:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except ImportError as e:
        raise RuntimeError("OR-Tools not installed. Install with: pip install ortools") from e

    C = [0, 1, 2]
    model = cp_model.CpModel()

    x = {(v, c): model.NewBoolVar(f"x_{v}_{c}") for v in nodes for c in C}
    y = {((u, v), c): model.NewBoolVar(f"y_{u}_{v}_{c}") for (u, v) in edges for c in C}
    sat = {(u, v): model.NewIntVar(0, 1, f"sat_{u}_{v}") for (u, v) in edges}

    for v in nodes:
        model.Add(sum(x[(v, c)] for c in C) == 1)

    for (u, v) in edges:
        for c in C:
            model.Add(y[((u, v), c)] >= x[(u, c)] + x[(v, c)] - 1)
            model.Add(y[((u, v), c)] <= x[(u, c)])
            model.Add(y[((u, v), c)] <= x[(v, c)])

    for (u, v) in edges:
        tmp = model.NewIntVar(0, len(C), f"sum_same_{u}_{v}")
        model.Add(tmp == sum(y[((u, v), c)] for c in C))
        model.Add(sat[(u, v)] + tmp == 1)

    model.Maximize(sum(sat[(u, v)] for (u, v) in edges))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT failed with status={status}")

    coloring = {}
    for v in nodes:
        chosen = 0
        for c in C:
            if solver.Value(x[(v, c)]) == 1:
                chosen = c
                break
        coloring[v] = int(chosen)

    satisfied = sum(int(solver.Value(sat[(u, v)])) for (u, v) in edges)
    ratio = (satisfied / len(edges)) if edges else 0.0

    method = "ortools_cpsat_optimal" if status == cp_model.OPTIMAL else "ortools_cpsat_feasible"
    return {
        "num_edges": len(edges),
        "satisfied_edges": satisfied,
        "ratio": ratio,
        "coloring_by_id": coloring,
        "method": method,
    }

# ==========================================================
# Public entry: accepts JSON dict OR nx.Graph
# ==========================================================
def _solve_max_3color_edges(graph_input: Union[Dict[str, Any], "nx.Graph"]) -> Dict[str, Any]:
    """
    Solve optimal 3-color max-edge (Max 3-Cut with k=3).

    Parameters
    ----------
    graph_input : dict | nx.Graph
        - Your JSON dict (with 'graph' key), OR
        - a networkx.Graph

    Returns
    -------
    dict with keys:
        num_edges, satisfied_edges, ratio, coloring_by_id, method
    """
    nodes, edges = _extract_nodes_edges(graph_input)
    try:
        return _solve_max3cut_ilp(nodes, edges)
    except Exception as e_ilp:
        try:
            return _solve_max3cut_cpsat(nodes, edges)
        except Exception as e_cp:
            raise RuntimeError(
                "Neither PuLP nor OR-Tools are available/working.\n"
                f"PuLP error: {e_ilp}\nOR-Tools error: {e_cp}\n"
                "Please install one of them: pip install pulp OR pip install ortools."
            )

# ==========================================================
# Batch scan and JSON writer (unchanged API)
# ==========================================================
def build_nested_results(
    folder: str,
    graph_types: Iterable[str],
    node_counts: Union[int, Iterable[int]],
    id_start: int,
    id_end_inclusive: int,
) -> Dict[str, Dict[str, Dict[int, int]]]:
    """
    Returns nested dict:
      { "<nodes>": { "<type>": { <id:int>: <satisfied_edges:int> } } }
    """
    if isinstance(node_counts, int):
        node_set = {node_counts}
    else:
        node_set = set(int(x) for x in node_counts)
    type_set = set(graph_types)

    nested: Dict[str, Dict[str, Dict[int, int]]] = {}

    try:
        fnames = os.listdir(folder)
    except FileNotFoundError:
        return nested

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
            # We can pass the JSON dict directly; the solver handles both.
            res = _solve_max_3color_edges(gobj)
            satisfied = int(res["satisfied_edges"])
        except Exception:
            continue

        ns = str(nnodes)  # keep nodes-level as string for stable JSON grouping
        if ns not in nested:
            nested[ns] = {}
        if gtype not in nested[ns]:
            nested[ns][gtype] = {}
        nested[ns][gtype][gid] = satisfied  # id kept as INT

    return nested

def write_colorK3_json(
    output_folder: str,
    nested_results: Dict[str, Dict[str, Dict[int, int]]]
) -> str:
    """
    Writes optimal_solutions/colorK3.json with:
      {"task":"coloring", "results": { "<nodes>": { "<type>": { <id:int>: <satisfied_edges:int> }}}}
    """
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, "colorK3.json")

    payload = {
        "task": "coloring",
        "results": nested_results
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    return out_path

# ==========================================================
# CLI
# ==========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Build optimal_solutions/colorK3.json with Max 3-Cut (k=3) satisfied edges per graph.")
    p.add_argument("--graphs_folder", type=str, default="./graphs", help="Folder containing graph_*.json files.")
    p.add_argument("--types", type=str, nargs="+", default=["ba", "ws", "dt"], help="Graph types to include.")
    p.add_argument("--nodes", type=int, nargs="+", default=[16], help="Node counts to include (one or more).")
    p.add_argument("--id_start", type=int, default=0, help="Start id (inclusive).")
    p.add_argument("--id_end", type=int, default=20, help="End id (inclusive).")
    p.add_argument("--out_folder", type=str, default="./optimal_solutions", help="Output folder (will be created).")
    return p.parse_args()

def main():
    args = parse_args()
    nested = build_nested_results(
        folder=args.graphs_folder,
        graph_types=args.types,
        node_counts=args.nodes,
        id_start=args.id_start,
        id_end_inclusive=args.id_end,
    )
    out_path = write_colorK3_json(args.out_folder, nested)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
