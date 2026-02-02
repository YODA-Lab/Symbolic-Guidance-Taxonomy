#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot log(1 + SSE_hard) vs rounds_multiplier_k with a curve per agent_type.

Scans: results_rounds/ (recursively)
JSON expects keys:
  - agent_type (str)
  - SSE_hard (number)
  - rounds_multiplier_k or k (optional if k can be parsed from folder/file name)

Heuristics to infer k if missing in JSON:
  - Parent folder like ".../k=3", ".../k_2.5", ".../k-4", or just ".../2.5"
  - Filename containing "k3", "k=3", "k2.5", etc.

Outputs:
  - PNG plot
  - Optional CSV summary of aggregated means/SEM (if --write-csv)

Usage:
  python plot_rounds_sse.py \
    --base results_rounds \
    --out sse_hard_vs_rounds_multiplier.png \
    --aggregate mean \
    --write-csv summary.csv
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results_rounds", help="Base folder to scan")
    ap.add_argument("--out", default="sse_hard_vs_rounds_multiplier.png", help="Output PNG path")
    ap.add_argument("--title", default="SSE_hard vs Rounds Multiplier by Agent Type", help="Figure title")
    ap.add_argument("--aggregate", choices=["none", "mean"], default="mean",
                    help="If multiple points share the same (agent_type, k), aggregate them")
    ap.add_argument("--write-csv", default="", help="Optional CSV summary path to write aggregated data")
    ap.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    return ap.parse_args()


# Float-friendly regex patterns for k in folder/file names
k_folder_patterns = [
    re.compile(r"k[=\-_\s]*([0-9]*\.?[0-9]+)", re.IGNORECASE),  # k=3, k_2.5, k-4
    re.compile(r"([0-9]*\.?[0-9]+)$"),                           # last number in folder name
]
k_file_patterns = [
    re.compile(r"k[=\-_\s]*([0-9]*\.?[0-9]+)", re.IGNORECASE),  # in filename
]
float_token = re.compile(r"^[0-9]*\.?[0-9]+$")


def infer_k_from_path(root: str, filename: str) -> Optional[float]:
    # Try parent folders (from closest outward)
    parts = root.split(os.sep)
    for p in reversed(parts):
        for pat in k_folder_patterns:
            m = pat.search(p)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
        # handle folders that are exactly a number, e.g. ".../2.5"
        if float_token.match(p or ""):
            try:
                return float(p)
            except Exception:
                pass

    # Try filename patterns like "results_k3.json", "run_k=5.json", "out_k2.5.json"
    for pat in k_file_patterns:
        m = pat.search(filename)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass

    return None


def safe_get_k(content: dict, root: str, fname: str) -> Optional[float]:
    # Prefer explicit JSON fields if present
    for key in ("rounds_multiplier_k", "k", "round_multiplier_k", "rounds_k"):
        if key in content:
            try:
                return float(content[key])
            except Exception:
                pass
    # Otherwise infer from path
    return infer_k_from_path(root, fname)


def load_points(base: str) -> List[Tuple[str, float, float]]:
    """
    Returns list of (agent_type, k (float), log1p_sse_hard (float))
    """
    points: List[Tuple[str, float, float]] = []
    for root, _, files in os.walk(base):
        for f in files:
            if not f.endswith(".json"):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    content = json.load(fh)
            except Exception:
                continue

            agent_type = content.get("agent_type")
            sse_hard = content.get("SSE_hard")

            if agent_type is None or sse_hard is None:
                continue

            k = safe_get_k(content, root, f)
            if k is None:
                continue

            try:
                sse_val = float(sse_hard)
            except Exception:
                continue

            log1p_sse = math.log1p(sse_val)
            points.append((str(agent_type), float(k), float(log1p_sse)))
    return points


def aggregate_points(points: List[Tuple[str, float, float]]) -> Dict[str, Dict[float, List[float]]]:
    """
    Returns dict[agent_type][k] = list of log1p_sse values
    """
    grouped: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for agent_type, k, val in points:
        grouped[agent_type][k].append(val)
    return grouped


def sem(values: List[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return (var ** 0.5) / (n ** 0.5)


def plot(grouped: Dict[str, Dict[float, List[float]]],
         out_path: str,
         title: str,
         aggregate: str,
         dpi: int = 160):
    plt.figure(figsize=(8, 6))

    has_any = False
    for agent_type, by_k in grouped.items():
        if not by_k:
            continue

        ks_sorted = sorted(by_k.keys())
        if aggregate == "mean":
            y_means = [mean(by_k[k]) for k in ks_sorted]
            y_sems = [sem(by_k[k]) for k in ks_sorted]
            plt.errorbar(ks_sorted, y_means, yerr=y_sems, marker="o", capsize=3, label=agent_type)
        else:
            # plot all points without aggregation
            x_vals, y_vals = [], []
            for k in ks_sorted:
                for v in by_k[k]:
                    x_vals.append(k)
                    y_vals.append(v)
            plt.plot(x_vals, y_vals, marker="o", linestyle="-", label=agent_type)
        has_any = True

    plt.xlabel("Rounds Multiplier (k)")
    plt.ylabel("log(1 + SSE_hard)")
    plt.title(title)
    if has_any:
        plt.legend(title="Agent Type")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    print(f"Saved figure to: {out_path}")


def maybe_write_csv(grouped: Dict[str, Dict[float, List[float]]], csv_path: str, aggregate: str):
    if not csv_path:
        return
    try:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if aggregate == "mean":
                w.writerow(["agent_type", "k", "mean_log1p_SSE_hard", "sem", "n"])
                for agent_type, by_k in grouped.items():
                    for k in sorted(by_k.keys()):
                        vals = by_k[k]
                        w.writerow([agent_type, k, mean(vals), sem(vals), len(vals)])
            else:
                w.writerow(["agent_type", "k", "log1p_SSE_hard"])
                for agent_type, by_k in grouped.items():
                    for k in sorted(by_k.keys()):
                        for v in by_k[k]:
                            w.writerow([agent_type, k, v])
        print(f"Wrote CSV summary to: {csv_path}")
    except Exception as e:
        print(f"Failed to write CSV: {e}")


def main():
    args = parse_args()
    points = load_points(args.base)
    if not points:
        print("No data points found. Make sure your JSON files include 'agent_type' and 'SSE_hard', and that k is present or inferable.")
        return

    grouped = aggregate_points(points)
    plot(grouped, args.out, args.title, args.aggregate, dpi=args.dpi)
    if args.write_csv:
        maybe_write_csv(grouped, args.write_csv, args.aggregate)


if __name__ == "__main__":
    main()
