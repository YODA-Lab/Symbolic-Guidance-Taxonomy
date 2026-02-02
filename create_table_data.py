# ======== SUMMARY PRINTERS (add this block) ========
import math
from create_figures_v2 import *
TASK_PRINT_LABEL = {
    "colors": "(Δ+1)-Coloring",          # used for local
    "matching": "Maximal Matching",      # used for local
    "vertex_cover": "Minimal Vertex Cover",  # used for local
    "colors_hard": "3-Coloring",
    "matching_hard": "Maximum Matching",
    "vertex_cover_hard": "Minimum Vertex Cover",
}

def _fmt_val(mean: float, sem: float) -> str:
    if not (math.isfinite(mean) and math.isfinite(sem)):
        return "—"
    return f"{mean:.2f} ± {sem:.2f}"

def _print_block_header(title: str):
    line = "=" * len(title)
    print(f"\n{title}\n{line}")

def _print_model_task_table(model_key: str, task_key: str, rows: list, *, is_hard: bool, algo_filter: Optional[str]):
    """
    rows: list of dicts produced earlier:
      EASY:  {"model_bucket","agent_type","n","avg","sem"}
      HARD:  {"model_bucket","algo_type","agent_type","n","avg","sem"}
    """
    # filter by model
    rows = [r for r in rows if r.get("model_bucket") == model_key]
    # hard: filter by algo
    if is_hard and algo_filter is not None:
        rows = [r for r in rows if r.get("algo_type") == algo_filter]

    if not rows:
        return  # nothing to print for this model/task

    # order agents and drop excluded
    agents = order_agents({r["agent_type"] for r in rows})

    # build lookup
    if is_hard:
        lut = {(r["agent_type"]): (r["avg"], r["sem"]) for r in rows}
    else:
        lut = {(r["agent_type"]): (r["avg"], r["sem"]) for r in rows}

    # title row
    tkey = f"{task_key}{'_hard' if is_hard else ''}"
    tlabel = TASK_PRINT_LABEL.get(tkey, task_key.replace("_", " ").title())
    print(f"\n[{MODEL_DISPLAY.get(model_key, model_key)}] {tlabel}")
    print("-" * (len(MODEL_DISPLAY.get(model_key, model_key)) + len(tlabel) + 3))

    # lines per agent
    for a in agents:
        m, s = lut.get(a, (float("nan"), float("nan")))
        print(f"  {display_agent(a):20s}  { _fmt_val(m, s) }")

def print_numeric_summaries(
    rows_easy_per_nodes: Dict[str, Dict[str, List[Dict[str, Any]]]],
    rows_hard_per_nodes: Dict[str, Dict[str, List[Dict[str, Any]]]],
    *,
    nodes_to_report: Optional[List[str]] = None
):
    """
    Prints:
      1) Local variants (EASY/MIS): colors, matching, vertex_cover
      2) Global variants (HARD/MIS): colors, matching, vertex_cover
      3) Global variants (HARD/Complete): colors, matching, vertex_cover
    For each: per model, prints ln(1+SSE) ± SEM per agent.
    """
    # choose nodes (defaults: all)
    all_nodes_easy = set(rows_easy_per_nodes.keys())
    all_nodes_hard = set(rows_hard_per_nodes.keys())
    all_nodes = sorted(all_nodes_easy.union(all_nodes_hard), key=lambda x: int(re.sub(r"\D", "", x) or "0"))
    nodes_sel = nodes_to_report or all_nodes
    if not nodes_sel:
        print("[WARN] No nodes found to summarize.")
        return

    for nodes in nodes_sel:
        # -------- 1) LOCAL (MIS) --------
        if nodes in rows_easy_per_nodes:
            _print_block_header(f"LOCAL (MIS) — num_nodes={nodes}")
            for task in ("colors", "matching", "vertex_cover"):
                rows = rows_easy_per_nodes[nodes].get(task, [])
                if not rows:
                    continue
                for model in MODEL_ORDER:
                    _print_model_task_table(model, task, rows, is_hard=False, algo_filter=None)
        else:
            print(f"\n[INFO] No LOCAL data for num_nodes={nodes}")

        # -------- 2) GLOBAL (MIS) --------
        if nodes in rows_hard_per_nodes:
            _print_block_header(f"GLOBAL (MIS) — num_nodes={nodes}")
            for task in ("colors", "matching", "vertex_cover"):
                rows = rows_hard_per_nodes[nodes].get(task, [])
                if not rows:
                    continue
                for model in MODEL_ORDER:
                    _print_model_task_table(model, task, rows, is_hard=True, algo_filter="incomplete")
        else:
            print(f"\n[INFO] No GLOBAL data (MIS) for num_nodes={nodes}")

        # -------- 3) GLOBAL (Complete Search) --------
        if nodes in rows_hard_per_nodes:
            _print_block_header(f"GLOBAL (Complete Search) — num_nodes={nodes}")
            for task in ("colors", "matching", "vertex_cover"):
                rows = rows_hard_per_nodes[nodes].get(task, [])
                if not rows:
                    continue
                for model in MODEL_ORDER:
                    _print_model_task_table(model, task, rows, is_hard=True, algo_filter="complete")
        else:
            print(f"\n[INFO] No GLOBAL data (Complete) for num_nodes={nodes}")
if __name__ == '__main__':
    print_numeric_summaries(rows_easy_per_nodes, rows_hard_per_nodes, nodes_to_report=None)
