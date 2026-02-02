#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(… original header unchanged …)
"""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Callable

from create_figures_global import build_nested_from_params

# --- global matplotlib spacing tweaks (very conservative) ---
import matplotlib as mpl
LEGEND_AXES_GAP = 0.1  # try 1.6–2.0 for roomy spacing

mpl.rcParams.update({
    "axes.labelpad": 2,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.titlepad": 6,
    # Note: this rcParam affects axes-anchored legends. We also pass borderaxespad explicitly
    # to figure-level legends below (the ones created with fig.legend) which is what fixes the overlap.

})
EXCLUDE_AGENTS = {"HeuristicHintGuided"}   # drop Heuristic Hints from all plots

# ================= Font configuration =================
@dataclass
class FontCfg:
    title: int = 14           # figure and subplot titles
    axis: int = 14            # axis label sizes
    tick: int = 11            # tick label sizes
    legend_title: int = 13
    legend_label: int = 12
    group_label: int = 11     # e.g., "Complete / MIS" labels above groups
    shared_label: int = 13    # shared x/y labels around grids

    # convenience accessors
    def legend_kwargs(self):
        return dict(title_fontsize=self.legend_title, prop={"size": self.legend_label})

def derive_success_cfg(global_cfg: FontCfg,
                       title: Optional[int], axis: Optional[int], tick: Optional[int],
                       legend_title: Optional[int], legend_label: Optional[int]) -> FontCfg:
    """Create a success-figure-only FontCfg from overrides or fall back to globals."""
    return FontCfg(
        title = title if title is not None else global_cfg.title,
        axis  = axis  if axis  is not None else global_cfg.axis,
        tick  = tick  if tick  is not None else global_cfg.tick,
        legend_title = legend_title if legend_title is not None else global_cfg.legend_title,
        legend_label = legend_label if legend_label is not None else global_cfg.legend_label,
        group_label  = global_cfg.group_label,
        shared_label = global_cfg.shared_label,
    )

# =============== GLOBALS (set in main) ===============
ALL_FIGS_DIR: Optional[Path] = None
FONT: FontCfg = FontCfg()
SUCCESS_FONT: FontCfg = FontCfg()   # will be derived from FONT unless overridden

def _savefig_both(fig, out_path: Path, *, dpi: int = 200, bbox_inches: str = "tight"):
    """Save figure as PDF to out_path (forced .pdf) and to the 'all_figures' mirror."""
    import matplotlib.pyplot as plt

    # Force .pdf regardless of what the caller passed
    out_pdf = out_path.with_suffix(".pdf")

    ensure_dir(out_pdf.parent)
    # dpi is ignored for vector backends like PDF; harmless to leave it
    fig.savefig(out_pdf, dpi=dpi, bbox_inches=bbox_inches, format="pdf")

    if ALL_FIGS_DIR is not None:
        ensure_dir(ALL_FIGS_DIR)
        fig.savefig((ALL_FIGS_DIR / out_pdf.name), dpi=dpi, bbox_inches=bbox_inches, format="pdf")

# ---------- config ----------
DEFAULT_KEYS = ["num_nodes", "task", "problem_level", "algorithm_type", "model_name", "agent_type"]

ALGO_ORDER   = ["complete", "incomplete"]
ALGO_DISPLAY = {"complete": "Backtracking", "incomplete": "MIS"}

# Hatching to indicate algorithm inside bars (color is the agent)
# circles for Complete Search; diagonal lines for MIS
ALGO_HATCH = {
    "complete": "ooo",   # circles (dense enough to read)
    "incomplete": "//",  # diagonal lines
}
# ======== SUMMARY PRINTERS (add this block) ========

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

TASK_Y_LIMITS = {
    "colors": (0.4, 1.0),
    "vertex_cover": (0.0, 1.0),
    "matching": (0.0, 1.0),
    "consensus": (0.0, 1.0),
    "leader_election": (0.0, 1.0),
}
DEFAULT_Y_LIMITS = (0.0, 1.0)

MODEL_ORDER   = ["llama3", "qwen2.5", "gemini"]
MODEL_DISPLAY = {"llama3": "Llama-3.1", "qwen2.5": "Qwen-2.5", "gemini": "Gemini-2.5"}

AGENT_ORDER = [
    "FreeForm",
    "HeuristicHintGuided",
    "FunctionDescriptionOnly",
    "FunctionPseudoCodeOnly",
    "MainPseudocodeOnly",
    "CompletePseudoCode",
    "Code"
]
AGENT_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]
AGENT_COLORS = {agent: AGENT_PALETTE[i % len(AGENT_PALETTE)] for i, agent in enumerate(AGENT_ORDER)}

# --- human-readable labels for agents (legend/display only) ---
AGENT_DISPLAY = {
    "Code": "Code",
    "CompletePseudoCode": "Full-\nPseudocode",
    "MainPseudocodeOnly": "Main-\nPseudocode",
    "FunctionPseudoCodeOnly": "Function-\nPseudocode",
    "FunctionDescriptionOnly": "Function-\nDescription",
    "HeuristicHintGuided": "Heuristic-\nHint",
    "FreeForm": "AgentsNet",
}

# For the Complete vs MIS figures, drop Code & FreeForm
EXCLUDE_AGENTS_FOR_GEMINI_ALGO_FIG = { "FreeForm"}

def display_agent(agent: str) -> str:
    return AGENT_DISPLAY.get(agent, str(agent))

def get_agent_color(agent: str) -> str:
    if agent in AGENT_COLORS:
        return AGENT_COLORS[agent]
    idx = (abs(hash(str(agent))) % len(AGENT_PALETTE))
    return AGENT_PALETTE[idx]

TASK_ALIASES = {
    "colors": {"colors", "coloring"},
    "coloring": {"colors", "coloring"},
    "consensus": {"consensus"},
    "matching": {"matching"},
    "vertex_cover": {"vertex_cover", "vertexcover", "vertex-cover", "vc"},
    "leader_election": {"leader_election", "leader-election", "leaderelection", "le"},
}

def normalize_task_name(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()
    for canon, aliases in TASK_ALIASES.items():
        if s in aliases:
            return canon
    return s

def get_y_limits(task: str):
    canon = normalize_task_name(task or "")
    return TASK_Y_LIMITS.get(canon, DEFAULT_Y_LIMITS)

# ---------- helpers ----------
def is_dictlike(x: Any) -> bool: return isinstance(x, dict)
def is_listlike(x: Any) -> bool: return isinstance(x, list)

def iter_records(payload: Any):
    if is_dictlike(payload):
        yield payload
    elif is_listlike(payload):
        for it in payload:
            if is_dictlike(it):
                yield it

# --- graph_index filter: include ONLY records with graph_index <= 4 ---
def _passes_graph_index(rec: Dict[str, Any], max_idx: int = 4) -> bool:
    try:
        gi = rec.get("graph_index", None)
        if gi is None:
            return False
        gi = float(gi)
        return gi <= float(max_idx)
    except Exception:
        return False

def load_moments(file_path: Path, score_key: str) -> Tuple[int, float, float]:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return (0, 0.0, 0.0)
    n = 0
    s = 0.0
    ss = 0.0
    for rec in iter_records(payload):
        if not _passes_graph_index(rec):
            continue
        try:
            x = float(rec.get(score_key, None))
        except Exception:
            continue
        n += 1
        s += x
        ss += x * x
    return (n, s, ss)

def load_moments_metric(file_path: Path, key: str, transform: Optional[Callable[[float], float]]) -> Tuple[int, float, float]:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return (0, 0.0, 0.0)
    n = 0
    s = 0.0
    ss = 0.0
    for rec in iter_records(payload):
        if not _passes_graph_index(rec):
            continue
        try:
            x = float(rec.get(key, None))
            if not math.isfinite(x):
                continue
            if transform is not None:
                x = transform(max(0.0, x))
        except Exception:
            continue
        n += 1
        s += x
        ss += x * x
    return (n, s, ss)

def load_success(file_path: Path, score_key: str) -> Tuple[int, int]:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return (0, 0)
    n = 0
    succ = 0
    for rec in iter_records(payload):
        if not _passes_graph_index(rec):
            continue
        try:
            x = float(rec.get(score_key, None))
        except Exception:
            continue
        n += 1
        if x == 1.0:
            succ += 1
    return (n, succ)

def mean_std_sem(n: int, s: float, ss: float) -> Tuple[float, float, float]:
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    m = s / n
    if n == 1:
        return (m, 0.0, 0.0)
    var = max((ss - n * m * m) / (n - 1), 0.0)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    return (m, std, sem)

def bernoulli_sem(n: int, successes: int) -> float:
    if n <= 0:
        return float("nan")
    p = successes / n
    return math.sqrt(p * (1.0 - p) / n)

def slugify(s: str) -> str:
    s = re.sub(r"\s+", "-", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9._-]", "", s)
    return s[:100] if len(s) > 100 else s

_LLAMA_PAT  = re.compile(r"(?:\bmeta-?\s*llama\b|\bllama[-_\s]*3\b)", re.IGNORECASE)
_GEMINI_PAT = re.compile(r"\bgemini\b", re.IGNORECASE)
_QWEN_PAT   = re.compile(r"\bqwen\s*2\.?5|\bqwen2\.?5|\bqwen\b", re.IGNORECASE)
_PYTHON_PAT = re.compile(r"\bpython\b", re.IGNORECASE)

def buckets_for(model_name: str, agent_type: str = None) -> List[str]:
    name = str(model_name or "").lower()
    at   = (agent_type or "").strip().lower()
    if _PYTHON_PAT.search(name) or at == "code":
        return ["llama3", "qwen2.5", "gemini"]
    if _LLAMA_PAT.search(name):
        return ["llama3"]
    if _QWEN_PAT.search(name):
        return ["qwen2.5"]
    if _GEMINI_PAT.search(name):
        return ["gemini"]
    return []

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def order_agents(agents_found):
    found = [a for a in agents_found if a not in EXCLUDE_AGENTS]
    ordered = [a for a in AGENT_ORDER if a in found]
    leftovers = sorted(a for a in found if a not in AGENT_ORDER)
    return ordered + leftovers

def sniff_problem_level(paths: List[str]) -> str:
    for fp in paths:
        try:
            data = json.loads(Path(fp).read_text(encoding="utf-8"))
            lvl = data.get("problem_level")
            if isinstance(lvl, str) and lvl.strip():
                return lvl.strip().lower()
        except Exception:
            pass
    joined = " ".join(paths).lower()
    if "hard" in joined: return "hard"
    if "poly" in joined: return "poly"
    return "poly"

def effective_algos_for_combo(original_algo: str, agent_type: str, problem_level: str) -> List[str]:
    agent = (str(agent_type) or "").strip().lower()
    lvl   = (str(problem_level) or "").strip().lower()
    if agent == "freeform":
        if lvl == "poly":
            return ["complete"]
        if lvl == "hard":
            return ["complete", "incomplete"]
    return [str(original_algo or "").strip().lower()] if original_algo else []

def y_limits_from_rows(rows: List[Dict[str, Any]], floor: float = 0.0, pad: float = 0.08) -> Tuple[float, float]:
    vals = []
    for r in rows:
        try:
            a = float(r.get("avg", float("nan")))
            s = float(r.get("sem", 0.0))
            if math.isfinite(a):
                vals.append(a + (s if math.isfinite(s) else 0.0))
        except Exception:
            pass
    if not vals:
        return (floor, floor + 1.0)
    ymax = max(vals)
    if ymax <= floor:
        ymax = floor + 1.0
    return (floor, ymax * (1.0 + pad))

def metric_config(context: str, mode: str, sse_transform: str) -> Tuple[str, str, Optional[Tuple[float,float]], Optional[Callable[[float], float]]]:
    mode = (mode or "score").lower()
    sse_transform = (sse_transform or "log1p").lower()

    def sse_label(which: str) -> str:
        if sse_transform == "log1p":
            return f"Mean ln(1 + SSE)"
        return f"Mean ln(1 + SSE)"

    def sse_transform_fn() -> Optional[Callable[[float], float]]:
        if sse_transform == "log1p":
            return math.log1p
        return None

    if context == "hard":
        if mode == "sse":
            return ("SSE_hard", sse_label("hard"), None, sse_transform_fn())
        if mode == "norm_sse":
            return ("norm_SSE_hard", "average norm_SSE_hard", (0.0, 1.0), None)
        return ("score_opt", "average score_opt", (0.0, 1.0), None)
    else:
        if mode == "sse":
            return ("SSE_easy", sse_label("easy"), None, sse_transform_fn())
        if mode == "norm_sse":
            return ("norm_SSE_easy", "average norm_SSE_easy", (0.0, 1.0), None)
        return ("score", "average score", (0.0, 1.0), None)

# ---------- LEGEND helpers (now use FONT config) ----------
def add_full_agent_legend(fig, *, loc="upper center", bbox=(0.5, 1.02), title="", font: FontCfg = None):
    from matplotlib.patches import Patch
    font = font or FONT
    agents = [a for a in AGENT_ORDER if a not in EXCLUDE_AGENTS]
    handles = [Patch(facecolor=get_agent_color(a)) for a in agents]
    labels  = [display_agent(a) for a in agents]
    fig.legend(handles, labels, loc=loc, bbox_to_anchor=bbox, ncol=len(labels), frameon=False,
               title=title, **font.legend_kwargs(), handlelength=1.8, columnspacing=1.2,
               borderpad=0.8, labelspacing=0.4, handletextpad=0.6, borderaxespad=LEGEND_AXES_GAP)


def add_agent_legend_subset(fig, agents: List[str], *, loc="upper left", bbox=(0.02, 1.03), title="", font: FontCfg = None):
    from matplotlib.patches import Patch
    font = font or FONT
    handles = [Patch(facecolor=get_agent_color(a)) for a in agents]
    labels  = [display_agent(a) for a in agents]
    fig.legend(
        handles, labels,
        loc=loc, bbox_to_anchor=bbox,
        ncol=len(labels), frameon=False,
        title=title, **font.legend_kwargs(),
        handlelength=1.8, columnspacing=1.2, borderpad=0.6, labelspacing=0.35, handletextpad=0.6,
        borderaxespad=LEGEND_AXES_GAP ,  # <-- add
    )

def add_algo_hatch_legend(fig, algos_present: List[str], *, loc="upper right", bbox=(0.98, 1.03), font: FontCfg = None):
    from matplotlib.patches import Patch
    font = font or FONT
    handles = [
        Patch(facecolor="white", edgecolor="black",
              hatch=ALGO_HATCH.get(a, ""), label=ALGO_DISPLAY.get(a, a.title()))
        for a in algos_present
    ]
    if handles:
        fig.legend(
            handles, [ALGO_DISPLAY.get(a, a.title()) for a in algos_present],
            loc=loc, bbox_to_anchor=bbox, frameon=False, ncol=len(handles),
            title="Algorithm", **font.legend_kwargs(),
            handlelength=2.0, borderpad=0.4, labelspacing=0.3, handletextpad=0.6,
            borderaxespad=LEGEND_AXES_GAP ,  # <-- add
        )

# ---------- plotting helpers for subplots ----------
def _plot_algo_model_grouped_on_ax(ax, rows: List[Dict[str, Any]], *, y_label: str,
                                   algo_display_map: Optional[Dict[str, str]] = None,
                                   y_limits: Optional[Tuple[float, float]] = None,
                                   font: FontCfg = None):
    import matplotlib.transforms as mtransforms
    font = font or FONT

    label_map = dict(ALGO_DISPLAY)
    if algo_display_map:
        label_map.update(algo_display_map)

    algos_present = [a for a in ALGO_ORDER if any(r["algo_type"] == a for r in rows)]
    single_algo = (len(algos_present) == 1)

    cats: List[Tuple[str, str]] = []
    group_sizes: Dict[str, int] = {}
    for algo in algos_present:
        models_present = [m for m in MODEL_ORDER
                          if any((r["algo_type"] == algo and r["model_bucket"] == m) for r in rows)]
        if models_present:
            for m in models_present:
                cats.append((algo, m))
            group_sizes[algo] = len(models_present)

    if not cats:
        return

    agents = order_agents({r["agent_type"] for r in rows})
    table: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
    for r in rows:
        table[(r["algo_type"], r["model_bucket"], r["agent_type"])] = (r["avg"], r["sem"])

    width = 0.8 / max(1, len(agents))
    idx = list(range(len(cats)))

    for i, agent in enumerate(agents):
        ys   = [table.get((a, m, agent), (float("nan"), float("nan")))[0] for (a, m) in cats]
        sems = [table.get((a, m, agent), (float("nan"), float("nan")))[1] for (a, m) in cats]
        offs = [k + (i - (len(agents) - 1) / 2) * width for k in idx]
        color = get_agent_color(agent)
        ax.bar(offs, ys, width=width, label=display_agent(agent), color=color)
        ax.errorbar(offs, ys, yerr=sems, fmt="none", capsize=4, linewidth=2, ecolor="black", zorder=3)

    ax.set_xticks(idx)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for (_algo, m) in cats], fontsize=font.tick)
    ax.tick_params(axis='y', labelsize=font.tick)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ymin, ymax = y_limits_from_rows(rows, floor=0.0, pad=0.08)
        ax.set_ylim(ymin, ymax)

    ax.grid(axis='y', linestyle='--', alpha=0.35)

    if not single_algo:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        pos = 0
        internal_boundaries = []
        for algo in ALGO_ORDER:
            size = group_sizes.get(algo, 0)
            if size:
                start = pos
                end   = pos + size - 1
                center = (start + end) / 2.0
                ax.text(center, 0.97, label_map.get(algo, algo),
                        transform=trans, ha="center", va="top",
                        fontsize=font.group_label, fontweight="bold", zorder=5)
                pos += size
                internal_boundaries.append(pos - 0.5)
        for x in internal_boundaries[:-1]:
            ax.axvline(x, linestyle="--", alpha=0.25, linewidth=1)

    ax.margins(x=0.02)

def _plot_algo_only_on_ax(ax, rows: List[Dict[str, Any]], *, y_label: str,
                          y_limits: Optional[Tuple[float, float]] = None,
                          font: FontCfg = None):
    font = font or FONT
    rows = [r for r in rows if r.get("model_bucket") == "gemini"]
    if not rows:
        return

    algos_present = [a for a in ALGO_ORDER if any(r["algo_type"] == a for r in rows)]
    if not algos_present:
        return

    agents = order_agents({r["agent_type"] for r in rows})
    table: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for r in rows:
        table[(r["algo_type"], r["agent_type"])] = (r["avg"], r["sem"])

    idx = list(range(len(algos_present)))
    width = 0.8 / max(1, len(agents))

    for i, agent in enumerate(agents):
        ys   = [table.get((a, agent), (float("nan"), float("nan")))[0] for a in algos_present]
        sems = [table.get((a, agent), (float("nan"), float("nan")))[1] for a in algos_present]
        offs = [k + (i - (len(agents) - 1) / 2) * width for k in idx]
        color = get_agent_color(agent)
        ax.bar(offs, ys, width=width, label=display_agent(agent), color=color)
        ax.errorbar(offs, ys, yerr=sems, fmt="none", capsize=4, linewidth=2, ecolor="black", zorder=3)

    ax.set_xticks(idx)
    ax.set_xticklabels([ALGO_DISPLAY.get(a, a.title()) for a in algos_present], fontsize=font.tick)
    ax.tick_params(axis='y', labelsize=font.tick)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        tmp = [{"avg": table.get((a, ag), (float("nan"), 0.0))[0],
                "sem": table.get((a, ag), (float("nan"), 0.0))[1]}
               for a in algos_present for ag in agents]
        ymin, ymax = y_limits_from_rows(tmp, floor=0.0, pad=0.08)
        ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

# --- helper to place shared labels tightly around the subplot grid
def _tight_shared_labels(fig, axes, xlabel: str, ylabel: str,
                         x_offset: float = 0.028, y_offset: float = 0.038,
                         font: FontCfg = None):
    font = font or FONT
    left   = min(ax.get_position().x0 for ax in axes)
    right  = max(ax.get_position().x1 for ax in axes)
    bottom = min(ax.get_position().y0 for ax in axes)
    top    = max(ax.get_position().y1 for ax in axes)

    fig.text((left + right) / 2.0, bottom - y_offset, xlabel,
             ha="center", va="top", fontsize=font.shared_label)
    fig.text(left - x_offset, (bottom + top) / 2.0, ylabel,
             ha="center", va="center", rotation=90, fontsize=font.shared_label)

# ---------- HARD 1×3 subfigs (all algos grouped, original) ----------
def plot_hard_subfigs_row(nodes: str, rows_by_task: Dict[str, List[Dict[str, Any]]],
                          out_path: Path, y_label: str, font: FontCfg = None,
                          y_limits: Optional[Tuple[float, float]] = None):
    import matplotlib.pyplot as plt
    font = font or FONT

    task_order = ["colors", "matching", "vertex_cover"]
    title_map = {"colors": "3-Coloring", "matching": "Maximum Matching", "vertex_cover": "Minimum Vertex Cover"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), squeeze=False)
    axes = axes[0]

    for i, t in enumerate(task_order):
        ax = axes[i]
        rows = rows_by_task.get(t, [])
        if rows:
            _plot_algo_model_grouped_on_ax(ax, rows, y_label=y_label, algo_display_map=None, y_limits=y_limits, font=font)
        else:
            ax.text(0.5, 0.5, "No HARD data", ha="center", va="center", fontsize=font.tick, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title_map.get(t, t.replace('_',' ').title())}", fontsize=font.title)

    add_full_agent_legend(fig, loc="upper center", bbox=(0.5, 1.03), title="", font=font)
    # Reserve extra headroom so axes don't creep under the legend
    plt.tight_layout(rect=[0.03, 0.12, 0.99, 0.88])
    _tight_shared_labels(fig, axes, "", y_label, x_offset=0.022, y_offset=0.045, font=font)

    _savefig_both(fig, out_path)
    print(f"[OK] Saved HARD triple subfigs -> {out_path}")
    plt.close(fig)

# ---------- HARD 1×3 subfigs (Gemini-only, algorithm on x) ----------
def plot_hard_subfigs_row_algo_only(nodes: str, rows_by_task: Dict[str, List[Dict[str, Any]]],
                                    out_path: Path, y_label: str, font: FontCfg = None,
                                    y_limits: Optional[Tuple[float, float]] = None):
    import matplotlib.pyplot as plt
    font = font or FONT

    task_order = ["colors", "matching", "vertex_cover"]
    title_map = {"colors": "3-Coloring", "matching": "Maximum Matching", "vertex_cover": "Minimum Vertex Cover"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), squeeze=False)
    axes = axes[0]

    for i, t in enumerate(task_order):
        ax = axes[i]
        rows = [r for r in rows_by_task.get(t, []) if r.get("model_bucket") == "gemini"]
        if rows:
            _plot_algo_only_on_ax(ax, rows, y_label=y_label, y_limits=y_limits, font=font)
        else:
            ax.text(0.5, 0.5, "No HARD data (Gemini)", ha="center", va="center", fontsize=font.tick, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title_map.get(t, t.replace('_',' ').title())}", fontsize=font.title)

    add_full_agent_legend(fig, loc="upper center", bbox=(0.5, 1.03), title="", font=font)
    plt.tight_layout(rect=[0.03, 0.12, 0.99, 0.88])
    _tight_shared_labels(fig, axes, "Algorithm Type", y_label, x_offset=0.020, y_offset=0.045, font=font)

    _savefig_both(fig, out_path)
    print(f"[OK] Saved GEMINI-only HARD triple (algo x-axis) -> {out_path}")
    plt.close(fig)

# ---------- plotting helpers for EASY / HARD with only models on x ----------
def _plot_models_on_x_ax(ax, rows, *, y_label: str, y_limits=None, font: FontCfg = None):
    font = font or FONT
    if not rows:
        return

    models_present = [m for m in MODEL_ORDER if any(r["model_bucket"] == m for r in rows)]
    if not models_present:
        return

    agents = order_agents({r["agent_type"] for r in rows})
    table = {(r["model_bucket"], r["agent_type"]): (r["avg"], r["sem"]) for r in rows}

    idx = list(range(len(models_present)))
    width = 0.8 / max(1, len(agents))

    for i, agent in enumerate(agents):
        ys   = [table.get((m, agent), (float("nan"), float("nan")))[0] for m in models_present]
        sems = [table.get((m, agent), (float("nan"), float("nan")))[1] for m in models_present]
        offs = [k + (i - (len(agents) - 1) / 2) * width for k in idx]
        color = get_agent_color(agent)
        ax.bar(offs, ys, width=width, label=display_agent(agent), color=color)
        ax.errorbar(offs, ys, yerr=sems, fmt="none", capsize=4, linewidth=2, ecolor="black", zorder=3)

    ax.set_xticks(idx)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models_present], fontsize=font.tick)
    ax.tick_params(axis='y', labelsize=font.tick)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ymin, ymax = y_limits_from_rows(rows, floor=0.0, pad=0.08)
        ax.set_ylim(ymin, ymax)

    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.margins(x=0.02)

def _plot_agents_on_x_ax(ax, rows, *, y_label: str, y_limits=None, font: FontCfg = None):
    font = font or FONT
    if not rows:
        return

    agents = order_agents({r["agent_type"] for r in rows})
    table  = {r["agent_type"]: (r["avg"], r["sem"]) for r in rows}

    idx = list(range(len(agents)))
    ys   = [table.get(a, (float("nan"), float("nan")))[0] for a in agents]
    sems = [table.get(a, (float("nan"), float("nan")))[1] for a in agents]

    ax.bar(idx, ys, width=0.8, color=[get_agent_color(a) for a in agents])
    ax.errorbar(idx, ys, yerr=sems, fmt="none", capsize=4, linewidth=2, ecolor="black", zorder=3)

    ax.set_xticks(idx)
    ax.set_xticklabels([])  # legend carries agent names
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', labelsize=font.tick)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ymin, ymax = y_limits_from_rows(rows, floor=0.0, pad=0.08)
        ax.set_ylim(ymin, ymax)

    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.margins(x=0.02)

# ---------- HARD split by algorithm into TWO figures (models on x) ----------
def _filter_rows_for_algo(rows: List[Dict[str, Any]], algo: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if str(r.get("algo_type")) == algo:
            out.append({
                "model_bucket": r.get("model_bucket"),
                "agent_type": r.get("agent_type"),
                "avg": r.get("avg"),
                "sem": r.get("sem"),
            })
    return out

def plot_hard_subfigs_row_models_by_algo(nodes: str,
                                         rows_by_task: Dict[str, List[Dict[str, Any]]],
                                         out_path: Path,
                                         y_label: str,
                                         algo: str,
                                         font: FontCfg = None,
                                         y_limits: Optional[Tuple[float, float]] = None):
    import matplotlib.pyplot as plt
    font = font or FONT
    title_map = {"colors": "3-Coloring", "matching": "Maximum Matching", "vertex_cover": "Minimum Vertex Cover"}
    task_order = ["colors", "matching", "vertex_cover"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), squeeze=False)
    axes = axes[0]

    for i, t in enumerate(task_order):
        ax = axes[i]
        rows = rows_by_task.get(t, [])
        rows = _filter_rows_for_algo(rows, algo)
        if rows:
            _plot_models_on_x_ax(ax, rows, y_label=y_label, y_limits=y_limits, font=font)
        else:
            ax.text(0.5, 0.5, f"No HARD data ({ALGO_DISPLAY.get(algo, algo)})", ha="center", va="center", fontsize=font.tick, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title_map.get(t, t.replace('_',' ').title())}", fontsize=font.title)

    add_full_agent_legend(fig, loc="upper center", bbox=(0.5, 1.03), title="", font=font)
    plt.tight_layout(rect=[0.03, 0.12, 0.99, 0.88])
    _tight_shared_labels(fig, axes, "", y_label, x_offset=0.022, y_offset=0.045, font=font)

    _savefig_both(fig, out_path)
    print(f"[OK] Saved HARD triple (models; {ALGO_DISPLAY.get(algo, algo)}) -> {out_path}")
    plt.close(fig)

# ---------- Gemini-only HARD (agents x; circle/diag hatching; no Code/FreeForm) ----------
def _plot_agents_with_algo_hatching(ax, rows, *, y_label: str, y_limits=None, font: FontCfg = None):
    font = font or FONT
    rows = [r for r in rows if str(r.get("model_bucket")) == "gemini"]
    if not rows:
        return [], []

    all_agents = order_agents({r["agent_type"] for r in rows})
    agents = [a for a in all_agents if a not in EXCLUDE_AGENTS_FOR_GEMINI_ALGO_FIG]
    algos  = [a for a in ALGO_ORDER if any(r["algo_type"] == a for r in rows)]
    if not agents or not algos:
        return agents, algos

    table = {}
    for r in rows:
        a = r["agent_type"]
        if a in EXCLUDE_AGENTS_FOR_GEMINI_ALGO_FIG:
            continue
        key = (a, r["algo_type"])
        table[key] = (float(r.get("avg", float("nan"))), float(r.get("sem", float("nan"))))

    n_algo = len(algos)
    width = 0.8 / max(1, n_algo)
    idx   = list(range(len(agents)))

    for j, algo in enumerate(algos):
        offs = [k + (j - (n_algo - 1) / 2) * width for k in idx]
        ys   = [table.get((a, algo), (float("nan"), float("nan")))[0] for a in agents]
        sems = [table.get((a, algo), (float("nan"), float("nan")))[1] for a in agents]
        colors = [get_agent_color(a) for a in agents]
        ax.bar(offs, ys, width=width, color=colors, edgecolor="black",
               hatch=ALGO_HATCH.get(algo, ""), linewidth=0.6, zorder=2)
        ax.errorbar(offs, ys, yerr=sems, fmt="none", capsize=4, linewidth=2,
                    ecolor="black", zorder=3)

    ax.set_xticks(idx)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', labelsize=font.tick)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        tmp_rows = []
        for a in agents:
            for algo in algos:
                avg, sem = table.get((a, algo), (float("nan"), 0.0))
                tmp_rows.append({"avg": avg, "sem": sem})
        ymin, ymax = y_limits_from_rows(tmp_rows, floor=0.0, pad=0.08)
        ax.set_ylim(ymin, ymax)

    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.margins(x=0.02)

    return agents, algos

def plot_hard_gemini_agents_by_algo(nodes: str,
                                    rows_by_task: Dict[str, List[Dict[str, Any]]],
                                    out_path: Path, y_label: str, font: FontCfg = None,
                                    y_limits: Optional[Tuple[float, float]] = None):
    import matplotlib.pyplot as plt
    font = font or FONT

    task_order = ["colors", "matching", "vertex_cover"]
    title_map  = {"colors": "3-Coloring", "matching": "Maximum Matching", "vertex_cover": "Minimum Vertex Cover"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), squeeze=False)
    axes = axes[0]

    algos_present_global = set()
    agents_union = []

    for i, t in enumerate(task_order):
        ax = axes[i]
        rows = rows_by_task.get(t, [])
        rows = [r for r in rows if str(r.get("model_bucket")) == "gemini"]
        if rows:
            agents_used, algos_used = _plot_agents_with_algo_hatching(ax, rows, y_label=y_label, y_limits=y_limits, font=font)
            for a in algos_used:
                algos_present_global.add(a)
            for ag in agents_used:
                if ag not in agents_union:
                    agents_union.append(ag)
        else:
            ax.text(0.5, 0.5, "No HARD data (Gemini)", ha="center", va="center", fontsize=font.tick, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])

        ax.set_title(f"{title_map.get(t, t.replace('_',' ').title())}", fontsize=font.title)

    if agents_union:
        add_agent_legend_subset(fig, agents_union, loc="upper left",  bbox=(0.02, 1.03), title="", font=font)
    add_algo_hatch_legend(fig, sorted(algos_present_global), loc="upper right", bbox=(0.98, 1.03), font=font)

    plt.tight_layout(rect=[0.03, 0.12, 0.99, 0.88])
    _tight_shared_labels(fig, axes, "", y_label, x_offset=0.020, y_offset=0.045, font=font)

    _savefig_both(fig, out_path)
    print(f"[OK] Saved GEMINI-only HARD triple (agents x; circle/diagonal algos) -> {out_path}")
    plt.close(fig)

# ---------- Llama3 + Qwen2.5 in one figure (2×3 grid; agents x; two bars per agent) ----------
def _plot_agents_with_algo_hatching_for_model(ax, rows, model_bucket: str, *, y_label: str, y_limits=None, font: FontCfg = None):
    font = font or FONT
    rows = [r for r in rows if str(r.get("model_bucket")) == model_bucket]
    if not rows:
        return [], []

    all_agents = order_agents({r["agent_type"] for r in rows})
    agents = [a for a in all_agents if a not in EXCLUDE_AGENTS_FOR_GEMINI_ALGO_FIG]
    algos  = [a for a in ALGO_ORDER if any(r["algo_type"] == a for r in rows)]
    if not agents or not algos:
        return agents, algos

    table = {}
    for r in rows:
        a = r["agent_type"]
        if a in EXCLUDE_AGENTS_FOR_GEMINI_ALGO_FIG:
            continue
        key = (a, r["algo_type"])
        table[key] = (float(r.get("avg", float("nan"))), float(r.get("sem", float("nan"))))

    n_algo = len(algos)  # expect 2
    width = 0.8 / max(1, n_algo)
    idx   = list(range(len(agents)))

    for j, algo in enumerate(algos):
        offs  = [k + (j - (n_algo - 1) / 2) * width for k in idx]
        ys    = [table.get((a, algo), (float("nan"), float("nan")))[0] for a in agents]
        sems  = [table.get((a, algo), (float("nan"), float("nan")))[1] for a in agents]
        colors = [get_agent_color(a) for a in agents]
        ax.bar(offs, ys, width=width, color=colors, edgecolor="black",
               hatch=ALGO_HATCH.get(algo, ""), linewidth=0.6, zorder=2)
        ax.errorbar(offs, ys, yerr=sems, fmt="none", capsize=4, linewidth=2,
                    ecolor="black", zorder=3)

    ax.set_xticks(idx)
    ax.set_xticklabels([])  # legend carries agent names
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', labelsize=font.tick)

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        tmp_rows = []
        for a in agents:
            for algo in algos:
                avg, sem = table.get((a, algo), (float("nan"), 0.0))
                tmp_rows.append({"avg": avg, "sem": sem})
        ymin, ymax = y_limits_from_rows(tmp_rows, floor=0.0, pad=0.08)
        ax.set_ylim(ymin, ymax)

    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.margins(x=0.02)
    return agents, algos

def plot_hard_llama_qwen_agents_by_algo(nodes: str, rows_by_task: Dict[str, List[Dict[str, Any]]],
                                        out_path: Path, y_label: str, font: FontCfg = None,
                                        y_limits: Optional[Tuple[float, float]] = None):
    import matplotlib.pyplot as plt
    font = font or FONT

    task_order = ["colors", "matching", "vertex_cover"]
    title_map  = {"colors": "3-Coloring", "matching": "Maximum Matching", "vertex_cover": "Minimum Vertex Cover"}
    model_rows = [("llama3", "Llama3"), ("qwen2.5", "Qwen2.5")]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.0), squeeze=False)

    algos_present_global = set()
    agents_union = []

    for rix, (model_key, model_disp) in enumerate(model_rows):
        for cix, task in enumerate(task_order):
            ax   = axes[rix, cix]
            rows = rows_by_task.get(task, [])
            if rows:
                agents_used, algos_used = _plot_agents_with_algo_hatching_for_model(
                    ax, rows, model_bucket=model_key, y_label=y_label, y_limits=y_limits, font=font
                )
                for a in algos_used:
                    algos_present_global.add(a)
                for ag in agents_used:
                    if ag not in agents_union:
                        agents_union.append(ag)
            else:
                ax.text(0.5, 0.5, f"No HARD data ({model_disp})", ha="center", va="center",
                        fontsize=font.tick, alpha=0.7)
                ax.set_xticks([]); ax.set_yticks([])

            ax.set_title(f"{title_map.get(task, task.replace('_',' ').title())} — {model_disp}", fontsize=font.title)

    if agents_union:
        add_agent_legend_subset(fig, agents_union, loc="upper left",  bbox=(0.02, 0.99), title="", font=font)
    add_algo_hatch_legend(fig, sorted(algos_present_global), loc="upper right", bbox=(0.98, 0.99), font=font)

    import matplotlib.pyplot as plt  # noqa
    plt.tight_layout(rect=[0.03, 0.06, 0.99, 0.90])
    all_axes = [ax for row in axes for ax in row]
    _tight_shared_labels(fig, all_axes, "", y_label, x_offset=0.020, y_offset=0.045, font=font)

    _savefig_both(fig, out_path)
    print(f"[OK] Saved LLAMA+QWEN HARD figure (agents x; circle/diagonal algos) -> {out_path}")
    plt.close(fig)

# ---------- EASY triple figures ----------
def plot_easy_subfigs_row_all_models(nodes: str, rows_by_task: Dict[str, List[Dict[str, Any]]],
                                     out_path: Path, y_label: str, font: FontCfg = None,
                                     y_limits: Optional[Tuple[float, float]] = None):
    import matplotlib.pyplot as plt
    font = font or FONT
    title_map = {"colors": "(Δ+1)-Coloring", "matching": "Maximal Matching", "vertex_cover": "Minimal Vertex Cover"}
    task_order = ["colors", "matching", "vertex_cover"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), squeeze=False)
    axes = axes[0]

    for i, t in enumerate(task_order):
        ax = axes[i]
        rows = rows_by_task.get(t, [])
        if rows:
            _plot_models_on_x_ax(ax, rows, y_label=y_label, y_limits=y_limits, font=font)
        else:
            ax.text(0.5, 0.5, "No EASY data", ha="center", va="center", fontsize=font.tick, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title_map.get(t, t.replace('_',' ').title())}", fontsize=font.title)

    add_full_agent_legend(fig, loc="upper center", bbox=(0.5, 1.02), title="", font=font)
    plt.tight_layout(rect=[0.03, 0.12, 0.99, 0.86])
    _tight_shared_labels(fig, axes, "", y_label, x_offset=0.022, y_offset=0.045, font=font)

    _savefig_both(fig, out_path)
    print(f"[OK] Saved EASY triple (all models) -> {out_path}")
    plt.close(fig)

def plot_easy_subfigs_row_gemini_only(nodes: str, rows_by_task: Dict[str, List[Dict[str, Any]]],
                                      out_path: Path, y_label: str, font: FontCfg = None,
                                      y_limits: Optional[Tuple[float, float]] = None):
    import matplotlib.pyplot as plt
    font = font or FONT
    title_map = {"colors": "(Δ+1)-Coloring", "matching": "Maximal Matching", "vertex_cover": "Minimal Vertex Cover"}
    task_order = ["colors", "matching", "vertex_cover"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), squeeze=False)
    axes = axes[0]

    for i, t in enumerate(task_order):
        ax = axes[i]
        rows = [r for r in rows_by_task.get(t, []) if r.get("model_bucket") == "gemini"]
        if rows:
            _plot_agents_on_x_ax(ax, rows, y_label=y_label, y_limits=y_limits, font=font)
        else:
            ax.text(0.5, 0.5, "No EASY data (Gemini)", ha="center", va="center", fontsize=font.tick, alpha=0.7)
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title_map.get(t, t.replace('_',' ').title())}", fontsize=font.title)

    add_full_agent_legend(fig, loc="upper center", bbox=(0.5, 1.02), title="", font=font)
    plt.tight_layout(rect=[0.03, 0.12, 0.99, 0.86])
    _tight_shared_labels(fig, axes, "", y_label, x_offset=0.020, y_offset=0.045, font=font)

    _savefig_both(fig, out_path)
    print(f"[OK] Saved EASY triple (Gemini-only) -> {out_path}")
    plt.close(fig)

# ---------- Single-task figure builder ----------
def plot_algo_model_grouped(nodes: str, task: str, rows: List[Dict[str, Any]],
                            out_path: Path, y_label: str,
                            algo_display_map: Optional[Dict[str, str]] = None,
                            y_limits: Optional[Tuple[float, float]] = None,
                            font: FontCfg = None):
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms  # noqa: F401
    font = font or FONT

    if not rows:
        print(f"[WARN] No rows to plot for {out_path}")
        return

    algos_present = [a for a in ALGO_ORDER if any(r["algo_type"] == a for r in rows)]
    cats = []
    for algo in algos_present:
        for m in MODEL_ORDER:
            if any((r["algo_type"] == algo and r["model_bucket"] == m) for r in rows):
                cats.append((algo, m))

    if not cats:
        print(f"[WARN] No categories to plot for {out_path}")
        return

    fig_width = max(14, 1.6 * len(cats))
    fig, ax = plt.subplots(figsize=(fig_width, 7.2))

    _plot_algo_model_grouped_on_ax(
        ax,
        rows,
        y_label=y_label,
        algo_display_map=algo_display_map,
        y_limits=y_limits,
        font=font
    )

    ax.set_title(f"{y_label} (num_nodes={nodes}, task={task})", fontsize=font.title + 2)  # slightly larger
    single_algo = (len(algos_present) == 1)
    ax.set_xlabel("" if single_algo else "algorithm / model", fontsize=font.axis, labelpad=2)
    ax.set_ylabel(y_label, fontsize=font.axis, labelpad=2)
    ax.margins(x=0.02)

    add_full_agent_legend(fig, loc="lower center", bbox=(0.5, -0.08), title="", font=font)

    fig.subplots_adjust(bottom=0.30)
    import matplotlib.pyplot as plt  # noqa
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    _savefig_both(fig, out_path)
    print(f"[OK] Saved figure -> {out_path}")
    plt.close(fig)

# ---------- NEW: pooled Gemini EASY success-rate (3 tasks in one figure) ----------
def plot_gemini_success_tasks_triple_easy(
    nodes: str,
    rows: List[Dict[str, Any]],
    out_path: Path,
    *,
    task_order: Optional[List[str]] = None,
    task_label_map: Optional[Dict[str, str]] = None,
    font: FontCfg = None,   # <-- use SUCCESS_FONT when calling
    legend_rows: int = 2,   # <--- force multi-row legend
):
    """
    rows: [{task, agent_type, p, sem}], pooled across runs and (allowed) algos, Gemini-only.
    Uses AGENT_DISPLAY via display_agent() for legend labels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    font = font or SUCCESS_FONT

    if not rows:
        print(f"[WARN] No rows to plot for EASY triple success -> {out_path}")
        return

    if task_order is None:
        task_order = ["colors", "matching", "vertex_cover"]
    if task_label_map is None:
        task_label_map = {
            "colors": "(Δ+1)-Coloring",
            "matching": "Maximal Matching",
            "vertex_cover": "Minimal Vertex Cover",
        }

    tasks_present = [t for t in task_order if any(r["task"] == t for r in rows)]
    agents = order_agents({r["agent_type"] for r in rows})

    # table[(task, agent)] -> (p, sem)
    table: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for r in rows:
        table[(r["task"], r["agent_type"])] = (float(r["p"]), float(r["sem"]))

    idx = list(range(len(tasks_present)))
    width = 0.8 / max(1, len(agents))

    fig, ax = plt.subplots(figsize=(max(11, 1.25 * len(tasks_present)), 7.0))
    legend_handles: List[Patch] = []
    legend_labels: List[str] = []

    for i, agent in enumerate(agents):
        ys   = [table.get((t, agent), (float("nan"), float("nan")))[0] for t in tasks_present]
        sems = [table.get((t, agent), (float("nan"), float("nan")))[1] for t in tasks_present]
        offs = [k + (i - (len(agents) - 1) / 2) * width for k in idx]
        color = get_agent_color(agent)
        ax.bar(offs, ys, width=width, label=display_agent(agent), color=color)
        ax.errorbar(offs, ys, yerr=sems, fmt="none", capsize=4, linewidth=2, ecolor="black", zorder=3)
        legend_handles.append(Patch(facecolor=color))
        legend_labels.append(display_agent(agent))

    ax.set_xticks(idx)
    ax.set_xticklabels([task_label_map.get(t, t.replace("_", " ").title()) for t in tasks_present], fontsize=font.tick)

    ax.set_ylabel("Success rate", fontsize=font.axis)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    if legend_labels:
        import math as _math
        legend_rows = max(1, int(legend_rows))
        ncol = _math.ceil(len(legend_labels) / legend_rows)
        top_rect = 0.86 if legend_rows >= 2 else 0.90

        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.02),
            ncol=ncol, frameon=False,
            title="",
            title_fontsize=font.legend_title,
            prop={"size": font.legend_label},
            handlelength=1.8, columnspacing=1.2, borderpad=0.6,
            borderaxespad=LEGEND_AXES_GAP ,  # <-- add
        )
        plt.tight_layout(rect=[0.03, 0.12, 0.99, top_rect])
    else:
        plt.tight_layout(rect=[0.03, 0.12, 0.99, 0.90])
    _savefig_both(fig, out_path)
    print(f"[OK] Saved EASY success-rate (Gemini pooled) -> {out_path}")
    plt.close(fig)

# ====== NEW: helpers for unified y-scale on error figures ======
def _update_global_max_from_rows(rows: List[Dict[str, Any]], current_max: float) -> float:
    m = current_max
    for r in rows or []:
        try:
            a = float(r.get("avg", float("nan")))
            s = float(r.get("sem", 0.0))
            if not math.isfinite(a):
                continue
            if not math.isfinite(s):
                s = 0.0
            m = max(m, a + s)
        except Exception:
            pass
    return m

def _limits_from_max(max_val: float, *, floor: float = 0.0, pad: float = 0.08) -> Tuple[float, float]:
    if not math.isfinite(max_val) or max_val <= floor:
        return (floor, floor + 1.0)
    return (floor, max_val * (1.0 + pad))

# ---------- main ----------
def main():
    global ALL_FIGS_DIR, FONT, SUCCESS_FONT

    ap = argparse.ArgumentParser(
        description="Per-task figures + EASY figures + HARD 1×3 subfigs (including per-algorithm splits, plus agents-by-algo variants)."
    )
    # ----- metrics and data selection -----
    ap.add_argument("--metric", type=str, default="sse",
                    choices=["sse", "norm_sse", "score"],
                    help="Which metric to plot in other figures. 'score' uses EASY=score, HARD=score_opt.")
    ap.add_argument("--sse-transform", type=str, default="log1p",
                    choices=["none", "log1p"],
                    help="Per-record transform applied when --metric sse.")
    ap.add_argument("--task-filter", type=str, default="all", help="Task to include for other figs: canonical name or 'all'.")
    ap.add_argument("--roots", nargs="+", default=None, help="One or more result roots (e.g., results).")
    ap.add_argument("--root", type=str, default=None, help="[Deprecated] Single root.")
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS, help="Grouping keys passed to build_nested_from_params.")
    ap.add_argument("--out-dir", type=str, default="figs_scoreopt_hard_by_model", help="Output dir")

    # ----- global font sizes -----
    ap.add_argument("--font-title", type=int, default=18)
    ap.add_argument("--font-axis", type=int, default=16)
    ap.add_argument("--font-tick", type=int, default=14)
    ap.add_argument("--font-legend-title", type=int, default=16)
    ap.add_argument("--font-legend-label", type=int, default=16)
    ap.add_argument("--font-group-label", type=int, default=14)
    ap.add_argument("--font-shared-label", type=int, default=14)

    # ----- success-figure-only font sizes (optional overrides) -----
    ap.add_argument("--success-font-title", type=int, default=18)
    ap.add_argument("--success-font-axis", type=int, default=16)
    ap.add_argument("--success-font-tick", type=int, default=14)
    ap.add_argument("--success-font-legend-title", type=int, default=13)
    ap.add_argument("--success-font-legend-label", type=int, default=14)
    ap.add_argument("--success-legend-rows", type=int, default=1,
                    help="Number of rows for the legend in the success-rate figure (default: 2).")

    # ----- unified y-scale controls for error figures -----
    ap.add_argument("--errors-share-y", action="store_true", default=True,
                    help="If true, all error figures (SSE/norm_SSE) share the same y scale per 'nodes'.")
    ap.add_argument("--errors-ymin", type=float, default=0.0,
                    help="Lower y bound for error figures (used if --errors-share-y).")
    ap.add_argument("--errors-ymax", type=float, default=None,
                    help="Upper y bound for error figures (if omitted, computed from data).")
    ap.add_argument("--errors-pad", type=float, default=0.08,
                    help="Top padding fraction used when computing --errors-ymax automatically.")

    args = ap.parse_args()

    # Build font configs
    FONT = FontCfg(
        title=args.font_title,
        axis=args.font_axis,
        tick=args.font_tick,
        legend_title=args.font_legend_title,
        legend_label=args.font_legend_label,
        group_label=args.font_group_label,
        shared_label=args.font_shared_label,
    )
    SUCCESS_FONT = derive_success_cfg(
        FONT,
        title=args.success_font_title,
        axis=args.success_font_axis,
        tick=args.success_font_tick,
        legend_title=args.success_font_legend_title,
        legend_label=args.success_font_legend_label,
    )

    if not args.roots:
        args.roots = [args.root] if args.root else ["results","results_extra"]

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # ALL FIGURES flat mirror
    ALL_FIGS_DIR = out_dir / "all_figures"
    ensure_dir(ALL_FIGS_DIR)

    # Build nested
    nested_all: Dict[str, Any] = {}
    for rp in args.roots:
        this_nested = build_nested_from_params(root_path=rp, keys=args.keys)
        def _merge_nested(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                out = dict(a or {})
                for k, v in (b or {}).items():
                    out[k] = _merge_nested(out.get(k), v)
                return out
            return b if a is None else a
        nested_all = _merge_nested(nested_all, this_nested)

    def iter_leaves(node: dict, depth: int, keys: List[str], acc: Dict[str, str]):
        if depth == len(keys):
            files = node if isinstance(node, list) else []
            files = sorted(set(files))
            yield (dict(acc), files)
            return
        for val, child in node.items():
            acc[keys[depth]] = val
            yield from iter_leaves(child, depth + 1, keys, acc)
        if depth > 0:
            acc.pop(keys[depth-1], None)

    # Track global maxima for EASY/HARD error plots, per nodes
    hard_max_by_nodes: Dict[str, float] = {}
    easy_max_by_nodes: Dict[str, float] = {}

    # ================= HARD =================
    metric_key_hard, y_label_hard, fixed_ylim_hard, transform_hard = metric_config("hard", args.metric, args.sse_transform)
    rows_hard_per_nodes: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    agg_hard: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]] = {}
    for combo, files in iter_leaves(nested_all, 0, args.keys, {}):
        raw_task   = combo.get("task")
        if not raw_task: continue
        task = normalize_task_name(raw_task)
        filt = (args.task_filter or "all").strip().lower()
        if filt != "all":
            canon = normalize_task_name(filt)
            accepted = TASK_ALIASES.get(canon, {canon})
            if task not in accepted: continue
        level = sniff_problem_level(files)
        if level != "hard": continue

        raw_model   = combo.get("model_name")
        agent_type  = combo.get("agent_type")
        algo_orig   = combo.get("algorithm_type")
        nodes       = combo.get("num_nodes")

        eff_algos = effective_algos_for_combo(algo_orig, agent_type, level)
        if not eff_algos or nodes is None or agent_type is None: continue

        n_total = s_total = ss_total = 0.0
        for fp in files:
            if transform_hard is None:
                n, s, ss = load_moments(Path(fp), metric_key_hard)
            else:
                n, s, ss = load_moments_metric(Path(fp), metric_key_hard, transform_hard)
            n_total += n; s_total += s; ss_total += ss
        if n_total == 0: continue

        buckets = buckets_for(raw_model, agent_type)
        if not buckets: continue

        for bucket in buckets:
            for algo in eff_algos:
                agg_hard.setdefault(str(nodes), {}) \
                        .setdefault(str(task), {}) \
                        .setdefault(bucket, {}) \
                        .setdefault(str(algo), {}) \
                        .setdefault(str(agent_type), {"n": 0.0, "sum": 0.0, "sumsq": 0.0})
                cell = agg_hard[str(nodes)][str(task)][bucket][str(algo)][str(agent_type)]
                cell["n"] += n_total; cell["sum"] += s_total; cell["sumsq"] += ss_total

    for nodes, task_map in agg_hard.items():
        for task, model_map in task_map.items():
            rows: List[Dict[str, Any]] = []
            for model in MODEL_ORDER:
                if model not in model_map: continue
                for algo in ALGO_ORDER:
                    if algo not in model_map[model]: continue
                    for agent, d in model_map[model][algo].items():
                        n = int(d["n"]); mean, _, sem = mean_std_sem(n, d["sum"], d["sumsq"])
                        rows.append({"model_bucket": model, "algo_type": algo, "agent_type": agent, "n": n, "avg": mean, "sem": sem})

            rows_hard_per_nodes.setdefault(str(nodes), {}).setdefault(str(task), [])
            if rows:
                rows_hard_per_nodes[str(nodes)][str(task)] = list(rows)

            if not rows:
                continue

            safe_task = slugify(task)
            task_dir = Path(args.out_dir) / safe_task / str(nodes)
            ensure_dir(task_dir)

            csv_path = task_dir / f"avg_scoreopt_hard_by_algo_model_nodes_{nodes}_task_{safe_task}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["model_bucket", "algo_type", "agent_type", "n", "avg", "sem"])
                w.writeheader(); [w.writerow(r) for r in rows]

            png_path = task_dir / f"bar_scoreopt_hard_by_algo_model_nodes_{nodes}_task_{safe_task}.pdf"

            # ---- unified limits for error metrics ----
            if args.errors_share_y and args.metric in {"sse", "norm_sse"}:
                hard_max_by_nodes[str(nodes)] = _update_global_max_from_rows(
                    rows, hard_max_by_nodes.get(str(nodes), 0.0)
                )
                if args.errors_ymax is not None:
                    ylims = (args.errors_ymin, args.errors_ymax)
                else:
                    ylims = _limits_from_max(hard_max_by_nodes.get(str(nodes), 0.0),
                                             floor=args.errors_ymin, pad=args.errors_pad)
            else:
                ylims = fixed_ylim_hard if fixed_ylim_hard is not None else y_limits_from_rows(rows, floor=0.0, pad=0.08)

            plot_algo_model_grouped(
                str(nodes), str(task), rows, png_path,
                y_label=y_label_hard,
                algo_display_map=None,
                y_limits=ylims,
                font=FONT
            )

    # helper to compute fixed limits for a block (HARD)
    def _fixed_limits_for_nodes_block(nodes_key: str, tmap: Dict[str, List[Dict[str, Any]]], metric: str):
        if not (args.errors_share_y and metric in {"sse", "norm_sse"}):
            return None
        m = hard_max_by_nodes.get(nodes_key, 0.0)
        for _rows in tmap.values():
            m = _update_global_max_from_rows(_rows, m)
        hard_max_by_nodes[nodes_key] = m
        if args.errors_ymax is not None:
            return (args.errors_ymin, args.errors_ymax)
        return _limits_from_max(m, floor=args.errors_ymin, pad=args.errors_pad)

    # HARD: 1×3 grouped (both algos) per nodes + variants
    for nodes, tmap in rows_hard_per_nodes.items():
        for t in ("colors", "matching", "vertex_cover"):
            tmap.setdefault(t, [])
        ylims_block = _fixed_limits_for_nodes_block(str(nodes), tmap, args.metric)

        out_trip = Path(args.out_dir) / "hard_triple" / str(nodes)
        ensure_dir(out_trip)
        out_png = out_trip / f"bar_HARD_triple_by_task_nodes_{nodes}.pdf"
        plot_hard_subfigs_row(str(nodes), rows_by_task=tmap, out_path=out_png, y_label=y_label_hard, font=FONT, y_limits=ylims_block)

        out_trip_gem = Path(args.out_dir) / "hard_triple_gemini_algo" / str(nodes)
        ensure_dir(out_trip_gem)
        out_png_gem = out_trip_gem / f"bar_HARD_triple_GEMINI_algo_nodes_{nodes}.pdf"
        plot_hard_subfigs_row_algo_only(str(nodes),
                                        rows_by_task={k: v for k, v in tmap.items() if k in {"colors", "matching", "vertex_cover"}},
                                        out_path=out_png_gem,
                                        y_label=y_label_hard,
                                        font=FONT,
                                        y_limits=ylims_block)

        # Split by algorithm (Complete vs MIS)
        out_split_base = Path(args.out_dir) / "hard_triple_models_by_algo" / str(nodes)
        ensure_dir(out_split_base)
        out_complete = out_split_base / f"bar_HARD_triple_MODELS_complete_nodes_{nodes}.pdf"
        plot_hard_subfigs_row_models_by_algo(str(nodes), tmap, out_complete, y_label_hard, algo="complete", font=FONT, y_limits=ylims_block)
        out_mis = out_split_base / f"bar_HARD_triple_MODELS_mis_nodes_{nodes}.pdf"
        plot_hard_subfigs_row_models_by_algo(str(nodes), tmap, out_mis, y_label_hard, algo="incomplete", font=FONT, y_limits=ylims_block)

        # Gemini-only, x = agents, two bars per agent (circle/diagonal hatches), NO Code/FreeForm
        out_gem_agents_algo = Path(args.out_dir) / "hard_triple_gemini_agents_by_algo" / str(nodes)
        ensure_dir(out_gem_agents_algo)
        out_png_gem_agents = out_gem_agents_algo / f"bar_HARD_triple_GEMINI_agentsByAlgo_nodes_{nodes}.pdf"
        plot_hard_gemini_agents_by_algo(str(nodes), tmap, out_png_gem_agents, y_label_hard, font=FONT, y_limits=(0,5))

        # ONE figure for Llama3 + Qwen2.5 (2×3), x = agents, circle/diagonal hatches
        out_two_models = Path(args.out_dir) / "hard_triple_LLAMA_QWEN_agents_by_algo" / str(nodes)
        ensure_dir(out_two_models)
        out_png_two = out_two_models / f"bar_HARD_triple_LLAMA_QWEN_agentsByAlgo_nodes_{nodes}.pdf"
        plot_hard_llama_qwen_agents_by_algo(str(nodes), tmap, out_png_two, y_label_hard, font=FONT, y_limits=ylims_block)

    # ================= EASY =================
    metric_key_easy, y_label_easy, fixed_ylim_easy, transform_easy = metric_config("easy", args.metric, args.sse_transform)
    agg_easy: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]] = {}
    for combo, files in iter_leaves(nested_all, 0, args.keys, {}):
        raw_task = combo.get("task")
        if not raw_task: continue
        task = normalize_task_name(raw_task)
        filt = (args.task_filter or "all").strip().lower()
        if filt != "all":
            canon = normalize_task_name(filt)
            accepted = TASK_ALIASES.get(canon, {canon})
            if task not in accepted: continue

        level = sniff_problem_level(files)
        is_colors = task in TASK_ALIASES["colors"]
        if is_colors and level != "poly": continue

        raw_model   = combo.get("model_name")
        agent_type  = combo.get("agent_type")
        algo_orig   = combo.get("algorithm_type")
        nodes       = combo.get("num_nodes")

        eff_algos = effective_algos_for_combo(algo_orig, agent_type, level)
        if not eff_algos or nodes is None or agent_type is None: continue

        n_total = s_total = ss_total = 0.0
        for fp in files:
            if transform_easy is None:
                n, s, ss = load_moments(Path(fp), metric_key_easy)
            else:
                n, s, ss = load_moments_metric(Path(fp), metric_key_easy, transform_easy)
            n_total += n; s_total += s; ss_total += ss
        if n_total == 0: continue

        buckets = buckets_for(raw_model, agent_type)
        if not buckets: continue

        if not is_colors:
            eff_algos = [a for a in eff_algos if a == "incomplete"]
            if not eff_algos:
                continue

        for bucket in buckets:
            for algo in eff_algos:
                agg_easy.setdefault(str(nodes), {}) \
                        .setdefault(str(task), {}) \
                        .setdefault(bucket, {}) \
                        .setdefault(str(algo), {}) \
                        .setdefault(str(agent_type), {"n": 0.0, "sum": 0.0, "sumsq": 0.0})
                cell = agg_easy[str(nodes)][str(task)][bucket][str(algo)][str(agent_type)]
                cell["n"] += n_total; cell["sum"] += s_total; cell["sumsq"] += ss_total

    for nodes, task_map in agg_easy.items():
        for task, model_map in task_map.items():
            is_colors = task in TASK_ALIASES["colors"]
            rows: List[Dict[str, Any]] = []
            for model in MODEL_ORDER:
                if model not in model_map: continue
                algos_to_use = (["incomplete"] if not is_colors else ALGO_ORDER)
                for algo in algos_to_use:
                    if algo not in model_map[model]: continue
                    for agent, d in model_map[model][algo].items():
                        n = int(d["n"]); mean, _, sem = mean_std_sem(n, d["sum"], d["sumsq"])
                        rows.append({"model_bucket": model, "algo_type": algo, "agent_type": agent, "n": n, "avg": mean, "sem": sem})

            if not rows: continue
            safe_task = slugify(task)
            task_dir = Path(args.out_dir) / safe_task / str(nodes)
            ensure_dir(task_dir)

            csv_path = task_dir / f"avg_score_easy_by_algo_model_nodes_{nodes}_task_{safe_task}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["model_bucket", "algo_type", "agent_type", "n", "avg", "sem"])
                w.writeheader(); [w.writerow(r) for r in rows]

            png_path = task_dir / f"bar_score_easy_by_algo_model_nodes_{nodes}_task_{safe_task}.pdf"

            # ---- unified limits for error metrics ----
            if args.errors_share_y and args.metric in {"sse", "norm_sse"}:
                easy_max_by_nodes[str(nodes)] = _update_global_max_from_rows(
                    rows, easy_max_by_nodes.get(str(nodes), 0.0)
                )
                if args.errors_ymax is not None:
                    ylims = (args.errors_ymin, args.errors_ymax)
                else:
                    ylims = _limits_from_max(easy_max_by_nodes.get(str(nodes), 0.0),
                                             floor=args.errors_ymin, pad=args.errors_pad)
            else:
                ylims = fixed_ylim_easy if fixed_ylim_easy is not None else y_limits_from_rows(rows, floor=0.0, pad=0.08)

            plot_algo_model_grouped(
                str(nodes), str(task), rows, png_path,
                y_label=y_label_easy,
                algo_display_map=({"complete": "MIS"} if is_colors else None),
                y_limits=ylims,
                font=FONT
            )

    # helper to compute fixed limits for a block (EASY)
    def _fixed_limits_for_nodes_block_easy(nodes_key: str, tmap: Dict[str, List[Dict[str, Any]]], metric: str):
        if not (args.errors_share_y and metric in {"sse", "norm_sse"}):
            return None
        m = easy_max_by_nodes.get(nodes_key, 0.0)
        for _rows in tmap.values():
            m = _update_global_max_from_rows(_rows, m)
        easy_max_by_nodes[nodes_key] = m
        if args.errors_ymax is not None:
            return (args.errors_ymin, args.errors_ymax)
        return _limits_from_max(m, floor=args.errors_ymin, pad=args.errors_pad)

    # ---------- EASY merged triple figures ----------
    agg_easy_collapsed: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
    def _easy_allowed_algos(task_is_colors: bool) -> List[str]:
        return ["complete"] if task_is_colors else ["incomplete"]

    for combo, files in iter_leaves(nested_all, 0, args.keys, {}):
        raw_task = combo.get("task")
        if not raw_task:
            continue
        task = normalize_task_name(raw_task)

        if task not in {"colors", "matching", "vertex_cover"}:
            continue

        filt = (args.task_filter or "all").strip().lower()
        if filt != "all":
            canon = normalize_task_name(filt)
            accepted = TASK_ALIASES.get(canon, {canon})
            if task not in accepted:
                continue

        level = sniff_problem_level(files)
        is_colors = task in TASK_ALIASES["colors"]
        if is_colors and level != "poly":
            continue

        raw_model  = combo.get("model_name")
        agent_type = combo.get("agent_type")
        nodes      = combo.get("num_nodes")
        if nodes is None or agent_type is None:
            continue

        algo_orig  = combo.get("algorithm_type")
        allowed_algos = _easy_allowed_algos(is_colors)
        if algo_orig and algo_orig not in allowed_algos:
            continue

        n_total = s_total = ss_total = 0.0
        for fp in files:
            if transform_easy is None:
                n, s, ss = load_moments(Path(fp), metric_key_easy)
            else:
                n, s, ss = load_moments_metric(Path(fp), metric_key_easy, transform_easy)
            n_total += n; s_total += s; ss_total += ss
        if n_total == 0:
            continue

        buckets = buckets_for(raw_model, agent_type)
        if not buckets:
            continue

        for bucket in buckets:
            agg_easy_collapsed.setdefault(str(nodes), {}) \
                              .setdefault(str(task), {}) \
                              .setdefault(bucket, {}) \
                              .setdefault(str(agent_type), {"n": 0.0, "sum": 0.0, "sumsq": 0.0})
            cell = agg_easy_collapsed[str(nodes)][str(task)][bucket][str(agent_type)]
            cell["n"]    += n_total
            cell["sum"]  += s_total
            cell["sumsq"]+= ss_total

    rows_easy_per_nodes: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for nodes, task_map in agg_easy_collapsed.items():
        for task, model_map in task_map.items():
            rows: List[Dict[str, Any]] = []
            for model in MODEL_ORDER:
                if model not in model_map:
                    continue
                for agent, d in model_map[model].items():
                    n = int(d["n"]); mean, _, sem = mean_std_sem(n, d["sum"], d["sumsq"])
                    rows.append({"model_bucket": model, "agent_type": agent, "n": n, "avg": mean, "sem": sem})
            if rows:
                rows_easy_per_nodes.setdefault(str(nodes), {})[str(task)] = rows

    for nodes, tmap in rows_easy_per_nodes.items():
        for t in ("colors", "matching", "vertex_cover"):
            tmap.setdefault(t, [])

        ylims_easy_block = _fixed_limits_for_nodes_block_easy(str(nodes), tmap, args.metric)

        out_trip = Path(args.out_dir) / "easy_triple_models" / str(nodes)
        ensure_dir(out_trip)
        out_png = out_trip / f"bar_EASY_triple_by_task_models_nodes_{nodes}.pdf"
        plot_easy_subfigs_row_all_models(str(nodes), rows_by_task=tmap, out_path=out_png, y_label=y_label_easy, font=FONT, y_limits=ylims_easy_block)

        out_trip_gem = Path(args.out_dir) / "easy_triple_gemini_agents" / str(nodes)
        ensure_dir(out_trip_gem)
        out_png_gem = out_trip_gem / f"bar_EASY_triple_GEMINI_agents_nodes_{nodes}.pdf"
        plot_easy_subfigs_row_gemini_only(str(nodes), rows_by_task=tmap, out_path=out_png_gem, y_label=y_label_easy, font=FONT, y_limits=ylims_easy_block)

    # ========== NEW: ONE pooled Gemini EASY success-rate figure ==========
    # Complete for colors (Δ+1), Incomplete (MIS) for matching & vertex_cover
    easy_by_task: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, int]]]]] = {}

    for combo, files in iter_leaves(nested_all, 0, args.keys, {}):
        raw_task = combo.get("task")
        if not raw_task:
            continue
        task = normalize_task_name(raw_task)
        if task not in {"colors", "matching", "vertex_cover"}:
            continue

        level = sniff_problem_level(files)
        is_colors = task in TASK_ALIASES["colors"]
        if is_colors and level != "poly":
            continue

        raw_model  = combo.get("model_name")
        agent_type = combo.get("agent_type")
        algo_orig  = combo.get("algorithm_type")
        nodes      = combo.get("num_nodes")
        if nodes is None or agent_type is None:
            continue

        # Gemini-only buckets
        if "gemini" not in buckets_for(raw_model, agent_type):
            continue

        # Allowed algos per task (colors→complete; others→incomplete)
        eff_algos = effective_algos_for_combo(algo_orig, agent_type, level)
        if not eff_algos:
            continue
        if not is_colors:
            eff_algos = [a for a in eff_algos if a == "incomplete"]
            if not eff_algos:
                continue

        # Count successes (score==1.0), filtered to graph_index<=4 by load_success()
        n_total = succ_total = 0
        for fp in files:
            n, succ = load_success(Path(fp), "score")
            n_total += n; succ_total += succ
        if n_total == 0:
            continue

        for algo in eff_algos:
            easy_by_task.setdefault(str(nodes), {}) \
                        .setdefault(str(algo), {}) \
                        .setdefault(str(task), {}) \
                        .setdefault(str(agent_type), {"n": 0, "succ": 0})
            cell = easy_by_task[str(nodes)][str(algo)][str(task)][str(agent_type)]
            cell["n"] += n_total
            cell["succ"] += succ_total

    # Aggregate across allowed algos and plot
    for nodes, algo_map in easy_by_task.items():
        desired_tasks_order = ["colors", "matching", "vertex_cover"]
        task_label_map = {
            "colors": "(Δ+1)-Coloring",
            "matching": "Maximal Matching",
            "vertex_cover": "Minimal Vertex Cover",
        }

        pooled: Dict[str, Dict[str, Dict[str, int]]] = {}
        for algo in ("complete", "incomplete"):
            if algo not in algo_map:
                continue
            for task_key in desired_tasks_order:
                if task_key not in algo_map[algo]:
                    continue
                for agent, d in algo_map[algo][task_key].items():
                    pooled.setdefault(task_key, {}).setdefault(agent, {"n": 0, "succ": 0})
                    pooled[task_key][agent]["n"]    += int(d["n"])
                    pooled[task_key][agent]["succ"] += int(d["succ"])

        rows_tasks_only: List[Dict[str, Any]] = []
        for task_key in desired_tasks_order:
            if task_key not in pooled:
                continue
            for agent, d in pooled[task_key].items():
                n = int(d["n"]); s = int(d["succ"])
                if n <= 0:
                    continue
                p = s / n
                sem = bernoulli_sem(n, s)
                rows_tasks_only.append({"task": task_key, "agent_type": agent, "p": p, "sem": sem})

        if not rows_tasks_only:
            continue

        # CSV (for reproducibility / paper tables)
        task_dir = Path(args.out_dir) / "gemini_by_task"
        ensure_dir(task_dir)
        csv_path = task_dir / f"success_rate_easy_gemini_colors_matching_vertexcover_TASKSONLY_nodes_{nodes}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["task", "agent_type", "n", "successes", "success_rate", "sem"])
            w.writeheader()
            for task_key in desired_tasks_order:
                if task_key not in pooled:
                    continue
                for agent, d in pooled[task_key].items():
                    n = int(d["n"]); s = int(d["succ"])
                    if n <= 0:
                        continue
                    w.writerow({
                        "task": task_key,
                        "agent_type": agent,
                        "n": n,
                        "successes": s,
                        "success_rate": (s / n),
                        "sem": bernoulli_sem(n, s),
                    })

        # Figure (saved both to out_dir and to ALL_FIGS_DIR mirror)
        png_path = task_dir / f"bar_success_rate_easy_gemini_colors_matching_vertexcover_TASKSONLY_nodes_{nodes}.pdf"
        plot_gemini_success_tasks_triple_easy(
            str(nodes),
            rows_tasks_only,
            png_path,
            task_order=desired_tasks_order,
            task_label_map=task_label_map,
            font=SUCCESS_FONT,
            legend_rows=args.success_legend_rows,  # force multi-row legend
        )
    print_numeric_summaries(rows_easy_per_nodes, rows_hard_per_nodes, nodes_to_report=None)

if __name__ == "__main__":
    main()
