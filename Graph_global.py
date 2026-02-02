from typing import Any, Dict, List, Optional, Tuple

# =========================
# YOUR AGENT ORDER & ALIASES
# (Canonicalize agent names first; colors are keyed by these canonical names.)
# =========================
AGENT_ALIAS = {
    "FreeFormAgent": "FreeForm",
    "HeuristicHintGuidedAgent": "HeuristicHintGuided",
    "AlgorithmDescriptionNonSymbolic": "FunctionDescriptionOnly",
    "PseudocodeOnlyAgent": "FunctionPseudoCodeOnly",
    "MainPseudocodeOnlyAgent": "MainPseudocodeOnly",
    "CompletePseudoCodeAgent": "CompletePseudoCode",
    # add more aliases as needed...
}

AGENT_ORDER = [
    "Code",
    "FreeForm",
    "HeuristicHintGuided",
    "FunctionDescriptionOnly",
    "FunctionPseudoCodeOnly",
    "MainPseudocodeOnly",
    "CompletePseudoCode",
]

# =========================
# PREDETERMINED COLORS (Okabe–Ito palette + a couple of extras)
# Keyed by the CANON agent names above.
# Reuse this dict across scripts for consistent colors everywhere.
# =========================
AGENT_COLORS: Dict[str, str] = {
    "FreeForm":               "#56B4E9",  # sky blue
    "HeuristicHintGuided":    "#E69F00",  # orange
    "FunctionDescriptionOnly":" #009E73".strip(),  # bluish green
    "FunctionPseudoCodeOnly": "#CC79A7",  # reddish purple
    "MainPseudocodeOnly":     "#0072B2",  # blue
    "CompletePseudoCode":     "#D55E00",  # vermillion
}
# Fallback color cycle for unknown agents (stable via hashing)

AGENT_ALIAS.update({
  "code": "Code", "main": "MainPseudocodeOnly",
  "complete": "CompletePseudoCode", "CompletePsuedoCode": "CompletePseudoCode",
  "abt": "Optima", "dsa": "Suboptimal"
})

# ---- Extra aliases for the coloring experiment ----
AGENT_ALIAS.update({
    # LLM variants
    "code": "Code",
    "main": "MainPseudocodeOnly",
    "mainpseudocodeonly": "MainPseudocodeOnly",
    "complete": "CompletePseudoCode",
    "completepseudocode": "CompletePseudoCode",
    "completepsuedocode": "CompletePseudoCode",   # common typo

    # Kinds (so keys like 'abt_main' / 'dsa_complete' map cleanly)
    "abt": "Optima",
    "dsa": "Suboptimal",
})

# ---- Fixed colors for these canonical names ----
AGENT_COLORS.update({
    "Code":                "#6A3D9A",  # violet
    "MainPseudocodeOnly":  "#0072B2",  # blue (matches prior)
    "CompletePseudoCode":  "#D55E00",  # vermillion (matches prior)
    # If you ever color the kinds directly:
    "Optima":              "#F0E442",  # yellow
    "Suboptimal":          "#000000",  # black
})

# (Optional) If you want legend order to start with the LLM variants:
# AGENT_ORDER = ["Code", "MainPseudocodeOnly", "CompletePseudoCode"] + AGENT_ORDER

# ---- Helpers used by plotting scripts ----
import hashlib

def canon_name(name: str) -> str:
    """Canonicalize any raw label (case/typos handled via AGENT_ALIAS)."""
    if not isinstance(name, str):
        return name
    key = name.strip()
    low = key.lower()
    return AGENT_ALIAS.get(low, AGENT_ALIAS.get(key, key))

_FALLBACK_CYCLE = ["#56B4E9", "#009E73", "#CC79A7", "#1B9E77", "#E31A1C", "#A6761D"]

def name_color(name: str) -> str:
    """Fixed color for the canonical name; stable hashed fallback otherwise."""
    n = canon_name(name)
    if n in AGENT_COLORS:
        return AGENT_COLORS[n]
    h = int(hashlib.md5(n.encode("utf-8")).hexdigest(), 16)
    return _FALLBACK_CYCLE[h % len(_FALLBACK_CYCLE)]
