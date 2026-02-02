#!/usr/bin/env python3
"""
Group JSON files into nested dict by exact keys you provide.

Special case:
  If `independent_key` is in `keys` and a record has that key equal to `independent_value`
  (case-insensitive), that record/file is grouped under BOTH 'complete' and 'incomplete',
  and never under the raw 'independent' value.

Leaves contain lists of JSON file paths.

Examples (CLI):
  python group_exact.py --root results --keys num_nodes task algorithm_type
  python group_exact.py --root results --keys num_nodes task --out grouped.json

Examples (import as a module):
  from group_exact import build_nested_from_params
  nested = build_nested_from_params("results", ["num_nodes","task","algorithm_type"])
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

# ---------------------- internal helpers ----------------------

def iter_records(payload: Any):
    """Yield dict-like records from a JSON payload (dict or list of dicts)."""
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item

def add_path(nested: dict, key_values: List[str], path_str: str):
    """Insert file path under nested[key1][key2]...[list], deduping paths."""
    cur = nested
    for v in key_values[:-1]:
        cur = cur.setdefault(v, {})
    cur.setdefault(key_values[-1], [])
    leaf = cur[key_values[-1]]
    if not leaf or leaf[-1] != path_str:
        if path_str not in leaf:
            leaf.append(path_str)

def _build_nested(root: Path, keys: List[str], independent_key: Optional[str], independent_value: str) -> dict:
    nested: dict = {}
    total_files = 0
    used_files = 0

    algo_idx = keys.index(independent_key) if (independent_key in keys) else None
    independent_value_lc = independent_value.strip().lower()

    for p in root.rglob("*.json"):
        if p.suffix.lower() != ".json" or p.name.endswith(".json.bak"):
            continue

        total_files += 1
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Skip unreadable JSON: {p} ({e})", file=sys.stderr)
            continue

        file_had_match = False
        for rec in iter_records(data):
            # collect values for required keys; if any missing, skip this record
            vals: List[str] = []
            for k in keys:
                if k not in rec:
                    vals = []
                    break
                vals.append(str(rec[k]))
            if not vals:
                continue

            # Special handling for algorithm_type == 'independent' (or your provided value)
            if algo_idx is not None:
                v = vals[algo_idx].strip().lower()
                if v == independent_value_lc:
                    for mapped in ("complete", "incomplete"):
                        vals2 = list(vals)
                        vals2[algo_idx] = mapped
                        add_path(nested, vals2, str(p))
                        file_had_match = True
                    continue  # don't add an 'independent' branch

            # Normal case
            add_path(nested, vals, str(p))
            file_had_match = True

        if file_had_match:
            used_files += 1

    print(f"[INFO] Scanned files: {total_files}, grouped: {used_files}")
    return nested

# ---------------------- public API ----------------------

def build_nested_from_params(
    root_path: str = "results",
    keys: List[str] = ["num_nodes", "task", "algorithm_type", "model_name", "agent_type"],
    independent_key: Optional[str] = "algorithm_type",
    independent_value: str = "algo_independent",
) -> dict:
    """
    Programmatic API: return the nested dictionary.

    Args:
        root_path: path to the root folder (e.g., 'results')
        keys: exact key names to group by, in order (e.g., ['num_nodes','task','algorithm_type'])
        independent_key: name of the key that triggers duplication when value == independent_value.
                         Set to None to disable special handling.
        independent_value: string value (case-insensitive) that should duplicate into
                           'complete' and 'incomplete'. Default 'independent'.

    Returns:
        dict: nested mapping keyed by your keys, leaves = list of file paths.
    """
    root = Path(root_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    return _build_nested(root, keys, independent_key, independent_value)

# ---------------------- CLI (optional) ----------------------

def print_summary(nested: dict, keys: List[str]):
    """Pretty summary: counts at leaves."""
    def _walk(node, depth, prefix):
        if depth == len(keys):
            print(f"{prefix} -> {len(node)} files")
            return
        for k in sorted(node, key=lambda x: (str(type(x)), str(x))):
            _walk(node[k], depth + 1, f"{prefix}{keys[depth]}={k} ")
    _walk(nested, 0, "")

def main():
    ap = argparse.ArgumentParser(description="Group JSON files by exact keys into a nested dictionary.")
    ap.add_argument("--root", type=str, default="results", help="Root folder containing subfolders with JSON files")
    ap.add_argument("--keys", nargs="+", default=["num_nodes", "task", "algorithm_type", "model_name", "agent_type"],
                    help="Exact key names to group by (order matters)")
    ap.add_argument("--independent-key", type=str, default="algorithm_type",
                    help="Key that, when equal to `independent-value`, duplicates into 'complete' and 'incomplete'. "
                         "Set to '' to disable.")
    ap.add_argument("--independent-value", type=str, default="independent",
                    help="Value that triggers duplication (case-insensitive). Default: 'independent'")
    ap.add_argument("--out", type=str, default=None, help="Optional path to save the nested dict as JSON")
    args = ap.parse_args()

    indep_key = args.independent_key if args.independent_key.strip() else None

    nested = _build_nested(
        root=Path(args.root).resolve(),
        keys=args.keys,
        independent_key=indep_key,
        independent_value=args.independent_value,
    )

    print("\nSummary:")
    print_summary(nested, args.keys)

    if args.out:
        out_path = Path(args.out).resolve()
        out_path.write_text(json.dumps(nested, indent=2), encoding="utf-8")
        print(f"\n[OK] Wrote nested mapping to {out_path}")

if __name__ == "__main__":
    main()
