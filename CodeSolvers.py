# python_dsa.py
from __future__ import annotations
import random

import networkx as nx
from typing import Any, Dict, List, Optional, Set, Tuple, Iterable, Sequence

# ------------------------------
# Generic message-passing base
# ------------------------------
class MessagePassingFramework:
    """
    Synchronous message-passing framework with fixed rounds.

    Subclasses must implement:
      - init_node_state(node) -> dict
      - compose_outgoing(node, state, round_idx) -> payload
      - update_state(node, state, inbox, round_idx) -> dict
      - extract_answers(global_state) -> Any
    """

    def __init__(self, graph: nx.Graph, rounds: int, seed: Optional[int] = None) -> None:
        self.graph: nx.Graph = graph
        self.rounds: int = max(0, int(rounds))
        # Consistent node order for stable indexing
        self.order: List[Any] = list(sorted(self.graph.nodes()))
        self.N: int = len(self.order)

        # Precompute neighbor lists in self.order space
        self._neighbors: Dict[Any, List[Any]] = {u: list(self.graph.neighbors(u)) for u in self.order}

        # Deterministic RNG unless seed provided
        if seed is None:
            seed = hash((self.N, tuple(sorted(map(tuple, self.graph.edges()))))) & 0xFFFFFFFF
        self._rng = random.Random(seed)

        # Per-node state (dicts decided by subclass)
        self.state: Dict[Any, Dict[str, Any]] = {}

        # Optional transcript/debug
        self.transcripts: List[Any] = []

        # Filled by run()
        self.answers: Any = None

    # ---- utility hooks for subclasses ----
    def neighbors(self, u: Any) -> Iterable[Any]:
        return self._neighbors[u]

    def rand(self) -> float:
        return self._rng.random()

    def choice(self, seq: Sequence[Any]) -> Any:
        return self._rng.choice(seq)

    def log(self, msg: Any) -> None:
        self.transcripts.append(msg)

    # ---- abstract hooks to override ----
    def init_node_state(self, node: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def compose_outgoing(self, node: Any, state: Dict[str, Any], round_idx: int) -> Any:
        raise NotImplementedError

    def update_state(
        self, node: Any, state: Dict[str, Any], inbox: List[Tuple[Any, Any]], round_idx: int
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def extract_answers(self, global_state: Dict[Any, Dict[str, Any]]) -> Any:
        raise NotImplementedError

    # ---- core algorithm ----
    def bootstrap(self) -> None:
        self.state = {u: self.init_node_state(u) for u in self.order}
        self.log({"event": "bootstrap", "nodes": self.N, "rounds": self.rounds})

    def _one_round(self, r: int) -> None:
        # 1) compose outbox
        outbox: Dict[Any, Any] = {u: self.compose_outgoing(u, self.state[u], r) for u in self.order}
        # 2) deliver to inboxes
        inboxes: Dict[Any, List[Tuple[Any, Any]]] = {u: [] for u in self.order}
        for u in self.order:
            msg = outbox[u]
            for v in self.neighbors(u):
                inboxes[v].append((u, msg))
        # 3) update states
        new_state: Dict[Any, Dict[str, Any]] = {}
        for u in self.order:
            new_state[u] = self.update_state(u, self.state[u], inboxes[u], r)
        self.state = new_state
        self.log({"event": "round_end", "round": r})

    def run(self) -> Any:
        self.bootstrap()
        for r in range(self.rounds):
            self._one_round(r)
        self.answers = self.extract_answers(self.state)
        self.log({"event": "done"})
        return self.answers



class SolveNPColoringTentativeCommitPython(MessagePassingFramework):
    """
    k-coloring with k < Δ+1 (non-poly setting):

      - Round 0 (Propose):
          * If P(u) nonempty: RANDOM warm start from P(u) (seeded RNG).
          * If P(u) empty: tentative = least-frequent color among neighbors' COMMITTED colors
            (ties -> first in S). If no neighbor has committed yet, use first in S (if any).

      - Later even rounds (Propose):
          * If P(u) nonempty: RANDOM color from current P(u) (seeded; reproducible).
          * If P(u) empty: tentative = least-frequent committed fallback (same as above).

      - Odd rounds (Resolve):
          * If neighbors proposed the same color, ONLY the highest priority keeps it.
            Priority = lexicographically FIRST id (smaller id) wins.
          * Others retract (clear tentative).

      - Every round:
          * Remove permanently fixed (committed) neighbor colors from P(u).
            P(u) can become empty in this setting—that’s expected.

    Notes:
      * Priority is local (compare ids with neighbors).
      * S is the shared color sequence (e.g., range(k)).
    """

    def __init__(
        self,
        graph: nx.Graph,
        rounds: int,
        colors: int | Iterable[int],
        seed: Optional[int] = None,
        auto_run: bool = True,
        S: Optional[List[int]] = None,   # optional explicit color sequence
    ) -> None:
        super().__init__(graph=graph, rounds=rounds, seed=seed)

        # Priority map from sorted order: smaller index = lexicographically earlier = higher priority
        self._rank: Dict[Any, int] = {u: i for i, u in enumerate(self.order)}

        # Predefined color sequence S
        if S is not None:
            self.S = list(S)
        else:
            if isinstance(colors, int):
                base_palette = list(range(max(0, colors)))
            else:
                base_palette = list(colors)
            self.S: List[int] = base_palette

        if auto_run:
            self.run()

    # ---------- helpers ----------

    def _least_frequent_committed_fallback(
        self, inbox: List[Tuple[Any, Any]]
    ) -> Optional[int]:
        """
        Choose the color that is LEAST FREQUENT among neighbors' COMMITTED colors.
        Ties -> earliest in S. If no committed info, fall back to first in S (if any).
        """
        committed = [m["fixed"] for (_nbr, m) in inbox if m and (m.get("fixed") is not None)]
        if committed:
            freq: Dict[int, int] = {}
            for c in committed:
                freq[c] = freq.get(c, 0) + 1
            # choose min frequency; tie-break by order in S
            best_color = None
            best_key = None
            index_in_S = {c: i for i, c in enumerate(self.S)}
            for c, f in freq.items():
                if c not in index_in_S:
                    # ignore colors not in S
                    continue
                key = (f, index_in_S[c])
                if (best_key is None) or (key < best_key):
                    best_key = key
                    best_color = c
            if best_color is not None:
                return best_color
        # no committed info (or none in S): try first in S
        return self.S[0] if self.S else None

    # ---------- framework hooks ----------

    def init_node_state(self, node: Any) -> Dict[str, Any]:
        d = self.graph.degree[node]
        # Initial palette: at most the first d(u)+1 colors from S, but since k<Δ+1,
        # some nodes may have |S| <= d(u), and pruning can empty it later.
        if self.S:
            palette = self.S[: min(len(self.S), d + 1)]
        else:
            palette = []

        # Warm start if possible; otherwise use least-frequent-committed fallback (in round 0 we don't have inbox yet,
        # so we can't see neighbors' committed colors; treat as no info -> first in S).
        tent0 = self.choice(palette) if palette else (self.S[0] if self.S else None)

        return {
            "fixed": None,             # permanently chosen color (or None)
            "tent": tent0,             # tentative pick (warm start if possible)
            "palette": list(palette),  # mutable palette P(u), kept S-ordered
        }

    def compose_outgoing(self, node: Any, state: Dict[str, Any], round_idx: int) -> Any:
        # Broadcast both fixed and tentative
        return {"fixed": state["fixed"], "tent": state["tent"]}

    def update_state(
        self, node: Any, state: Dict[str, Any], inbox: List[Tuple[Any, Any]], round_idx: int
    ) -> Dict[str, Any]:
        fixed = state["fixed"]
        tent = state["tent"]
        palette: List[int] = state["palette"]

        # 1) Remove neighbors' permanently fixed colors from P(u) every round (can empty the palette)
        neighbor_fixed = [m["fixed"] for (_nbr, m) in inbox if m and (m.get("fixed") is not None)]
        if neighbor_fixed:
            banned = set(neighbor_fixed)
            palette = [c for c in palette if c not in banned]  # keep S-order

        # If already colored, nothing else to do
        if fixed is not None:
            return {"fixed": fixed, "tent": None, "palette": palette}

        # 2) Two-phase logic by parity
        if round_idx % 2 == 0:
            # PROPOSE phase
            if round_idx == 0:
                # keep warm-start tent if any; else choose via least-frequent committed fallback
                if tent is None:
                    tent = self._least_frequent_committed_fallback(inbox)
                return {"fixed": None, "tent": tent, "palette": palette}
            else:
                # later even rounds:
                if palette:
                    new_tent = self.choice(palette)  # seeded RNG for reproducibility
                else:
                    # palette is empty -> least-frequent among neighbors' committed (ties -> first in S)
                    new_tent = self._least_frequent_committed_fallback(inbox)
                return {"fixed": None, "tent": new_tent, "palette": palette}

        else:
            # RESOLVE phase: among same-color contenders, lexicographically FIRST id (smaller id) wins
            if tent is None:
                return {"fixed": None, "tent": None, "palette": palette}

            # neighbors that proposed the same tentative color
            same_color_neighbors = [nbr for (nbr, m) in inbox if m and m.get("tent") == tent]

            my_rank = self._rank[node]
            # earlier index (smaller rank/id) = higher priority
            i_keep = all(my_rank <= self._rank[nbr] for nbr in same_color_neighbors)

            if i_keep:
                # commit permanently
                return {"fixed": tent, "tent": None, "palette": palette}
            else:
                # retract and retry next propose phase
                return {"fixed": None, "tent": None, "palette": palette}

    def extract_answers(self, global_state: Dict[Any, Dict[str, Any]]) -> List[int]:
        # Colors aligned with self.order; fallback: last tentative or -1
        out: List[int] = []
        for u in self.order:
            fu = global_state[u]["fixed"]
            if fu is not None:
                out.append(fu)
            else:
                tu = global_state[u]["tent"]
                out.append(tu if tu is not None else -1)
        return out

class SolvePolyColoringTentativeCommitPython(MessagePassingFramework):
    """
    (Δ+1)-coloring with:
      - Round 0 (Propose): each uncolored node picks a RANDOM tentative color from P(u)
                           using the framework's seeded RNG.
      - Later even rounds (Propose): pick a RANDOM color from current P(u) (deterministic seed).
      - Odd rounds (Resolve): if neighbors proposed the same color, ONLY the highest-priority
                             node keeps it; here priority = lexicographically FIRST id wins.
      - Every round: remove permanently fixed neighbor colors from P(u).

    Notes:
      * Priority is local (compare ids only with neighbors); smaller id wins.
      * With Δ+1 colors, P(u) stays nonempty until u commits.
    """

    def __init__(
        self,
        graph: nx.Graph,
        rounds: int,
        colors: int | Iterable[int],
        seed: Optional[int] = None,   # used by base framework RNG
        auto_run: bool = True,
    ) -> None:
        super().__init__(graph=graph, rounds=rounds, seed=seed)

        # Priority map from sorted order: smaller index = lexicographically earlier = higher priority
        self._rank: Dict[Any, int] = {u: i for i, u in enumerate(self.order)}

        # Predefined color sequence S
        if isinstance(colors, int):
            base_palette = list(range(max(0, colors)))
        else:
            base_palette = list(colors)
        if not base_palette:
            # fallback to Δ+1 if empty
            max_deg = max((d for _, d in self.graph.degree()), default=0)
            base_palette = list(range(max_deg + 1))
        self.S: List[int] = base_palette

        if auto_run:
            self.run()

    # ---------- framework hooks ----------

    def init_node_state(self, node: Any) -> Dict[str, Any]:
        d = self.graph.degree[node]
        # Initial palette: first d(u)+1 colors from S (or all S if shorter)
        palette = self.S[: (d + 1)] if len(self.S) >= d + 1 else list(self.S)
        # INITIAL tentative pick is RANDOM (warm start), using framework RNG
        tent0 = self.choice(palette) if palette else None
        return {
            "fixed": None,             # permanently chosen color (or None)
            "tent": tent0,             # tentative pick from round 0 (randomized warm start)
            "palette": list(palette),  # mutable palette P(u), kept S-ordered
        }

    def compose_outgoing(self, node: Any, state: Dict[str, Any], round_idx: int) -> Any:
        # Broadcast both fixed and tentative
        return {"fixed": state["fixed"], "tent": state["tent"]}

    def update_state(
        self, node: Any, state: Dict[str, Any], inbox: List[Tuple[Any, Any]], round_idx: int
    ) -> Dict[str, Any]:
        fixed = state["fixed"]
        tent = state["tent"]
        palette: List[int] = state["palette"]

        # 1) Remove neighbors' permanently fixed colors from P(u) every round
        neighbor_fixed = [m["fixed"] for (_nbr, m) in inbox if m and (m.get("fixed") is not None)]
        if neighbor_fixed:
            banned = set(neighbor_fixed)
            palette = [c for c in palette if c not in banned]  # keep S-order

        # If already colored, nothing else to do
        if fixed is not None:
            return {"fixed": fixed, "tent": None, "palette": palette}

        # 2) Two-phase logic by parity
        if round_idx % 2 == 0:
            # PROPOSE phase
            if round_idx == 0:
                # keep the RANDOM warm-start chosen in init_node_state
                return {"fixed": None, "tent": tent, "palette": palette}
            else:
                # later even rounds: RANDOM from current palette (seeded; reproducible)
                new_tent = self.choice(palette) if palette else None
                return {"fixed": None, "tent": new_tent, "palette": palette}

        else:
            # RESOLVE phase: among same-color contenders, lexicographically FIRST id wins
            if tent is None:
                return {"fixed": None, "tent": None, "palette": palette}

            # neighbors that proposed the same tentative color
            same_color_neighbors = [nbr for (nbr, m) in inbox if m and m.get("tent") == tent]

            my_rank = self._rank[node]
            # earlier index (smaller rank) = higher priority
            i_keep = all(my_rank <= self._rank[nbr] for nbr in same_color_neighbors)

            if i_keep:
                # commit permanently
                return {"fixed": tent, "tent": None, "palette": palette}
            else:
                # retract and retry next propose phase
                return {"fixed": None, "tent": None, "palette": palette}

    def extract_answers(self, global_state: Dict[Any, Dict[str, Any]]) -> List[int]:
        # Colors aligned with self.order; fallback: last tentative or -1
        out: List[int] = []
        for u in self.order:
            fu = global_state[u]["fixed"]
            if fu is not None:
                out.append(fu)
            else:
                tu = global_state[u]["tent"]
                out.append(tu if tu is not None else -1)
        return out
# ---------------------------------------
# DSA for graph coloring (extends base)
# ---------------------------------------
class SolveColoringDsaPython(MessagePassingFramework):
    """
    Distributed Stochastic Algorithm (DSA) for graph coloring.

    Args:
        graph: nx.Graph
        rounds: number of message-passing iterations
        colors: int (K -> colors {0..K-1}) or iterable of color labels
    Notes:
        - On construction, runs immediately and sets self.answers.
        - self.answers is a list of colors aligned with sorted node ids.
    """

    def __init__(
        self,
        graph: nx.Graph,
        rounds: int,
        colors: int | Iterable[int],
        switch_prob: float = 0.7,
        greedy_choice: bool = True,
        seed: Optional[int] = None,
        auto_run: bool = True,
    ) -> None:
        super().__init__(graph=graph, rounds=rounds, seed=seed)

        # Normalize palette
        if isinstance(colors, int):
            self.palette: List[int] = list(range(max(0, colors)))
        else:
            self.palette = list(colors)
        if not self.palette:
            # Fallback to Δ+1 if empty
            max_deg = max(dict(self.graph.degree()).values(), default=0)
            self.palette = list(range(max_deg + 1))
        self.K = len(self.palette)

        # DSA knobs
        self.p_switch = float(switch_prob)
        self.greedy_choice = bool(greedy_choice)

        if auto_run:
            self.run()  # fills self.answers

    # ---- framework hooks ----
    def init_node_state(self, node: Any) -> Dict[str, Any]:
        init_color = self.choice(self.palette) if self.K > 0 else 0
        return {"color": init_color, "conflicts": 0}

    def compose_outgoing(self, node: Any, state: Dict[str, Any], round_idx: int) -> Any:
        return state["color"]  # broadcast current color

    def update_state(
        self, node: Any, state: Dict[str, Any], inbox: List[Tuple[Any, Any]], round_idx: int
    ) -> Dict[str, Any]:
        current = state["color"]
        neighbor_colors = [payload for (_nbr, payload) in inbox]
        conflicts = sum(1 for c in neighbor_colors if c == current)

        # switch with probability if conflicted
        if conflicts > 0 and self.rand() < self.p_switch:
            banned = set(neighbor_colors)
            allowed = [c for c in self.palette if c not in banned]
            if allowed:
                new_color = allowed[0] if self.greedy_choice else self.choice(allowed)
            else:
                new_color = current  # degenerate case
        else:
            new_color = current

        return {"color": new_color, "conflicts": conflicts}

    def extract_answers(self, global_state: Dict[Any, Dict[str, Any]]) -> List[int]:
        # Aligned to sorted node ids (self.order)
        return [global_state[u]["color"] for u in self.order]

class SolveColoringBacktrackPython(MessagePassingFramework):
    """
    ABT-style backtracking coloring adapted to broadcast-only framework.
    Priority: lexicographic order of node 'name' (falls back to str(node)).

    Guarantees: at every round (and at termination), each node holds SOME color.
    If no feasible color exists under the higher-neighbor context, the node
    keeps a fallback color = least-frequent among currently seen neighbor colors,
    and requests a backtrack upstream (pending_backtrack=True).
    """

    def __init__(self, graph: nx.Graph, rounds: int, colors: int | Iterable[int],
                 seed: Optional[int] = None, auto_run: bool = True, **kwargs):
        super().__init__(graph=graph, rounds=rounds, seed=seed)

        # palette
        if isinstance(colors, int):
            self.palette = list(range(colors))
        else:
            self.palette = list(colors)
        if not self.palette:
            max_deg = max(dict(self.graph.degree()).values(), default=0)
            self.palette = list(range(max_deg + 1))
        self.K = len(self.palette)

        # names & priority
        self.name_map = {u: self.graph.nodes[u].get("name", str(u)) for u in self.graph.nodes()}

        def is_higher(a, b):  # a higher-priority than b ?
            return self.name_map[a] < self.name_map[b]
        self._is_higher = is_higher

        # cache each node's higher/lower neighbor sets
        self._higher = {u: [v for v in self.neighbors(u) if is_higher(v, u)] for u in self.order}
        self._lower  = {u: [v for v in self.neighbors(u) if is_higher(u, v)] for u in self.order}

        if auto_run:
            self.run()  # fills self.answers

    # -------- framework hooks --------

    def init_node_state(self, node: Any) -> Dict[str, Any]:
        # context: colors of higher neighbors only (feasibility constraints)
        # seen:    most recent colors heard from ANY neighbor (for fallback stats)
        return {
            "context": {},            # higher-neighbor-name -> color
            "seen": {},               # any-neighbor-name    -> color
            "current": None,          # my current color (always non-None after first update)
            "attempted": set(),       # colors tried under current higher context
            "pending_backtrack": False,  # signal upstream while still holding a color
        }

    def _parent_of(self, node: Any) -> Optional[Any]:
        """Highest-priority neighbor above 'node' (if any)."""
        hs = self._higher[node]
        if not hs:
            return None
        # parent = max by name among higher (i.e., closest above me)
        return max(hs, key=lambda v: self.name_map[v])

    def compose_outgoing(self, node: Any, state: Dict[str, Any], round_idx: int) -> Any:
        """
        Broadcast one payload to all neighbors.
        - If pending_backtrack: send BACKTRACK (parent will react). We still KEEP a local color.
        - Else: send VALUE with our current color (lower neighbors will use it).
        """
        myname = self.name_map[node]
        if state["pending_backtrack"]:
            return {"type": "BACKTRACK", "sender": myname}
        else:
            # Always have a color to share when not actively backtracking
            return {"type": "VALUE", "sender": myname, "color": state["current"]}

    def update_state(
        self, node: Any, state: Dict[str, Any], inbox: List[Tuple[Any, Any]], round_idx: int
    ) -> Dict[str, Any]:
        """
        Process broadcast inbox:
          - VALUE from higher → update context
          - VALUE from any   → update seen (for fallback statistics)
          - BACKTRACK from child → if I'm that child's parent, mark my current as attempted and unset it
        Then (re)choose a feasible color if needed.
        If none feasible, choose a fallback (least-frequent among neighbors' colors),
        keep it, and request a backtrack upstream.
        """
        current = state["current"]
        attempted = set(state["attempted"])
        context = dict(state["context"])  # name -> color (higher only)
        seen = dict(state["seen"])        # name -> color (all neighbors)
        pending_backtrack = bool(state["pending_backtrack"])
        myname = self.name_map[node]

        # helper: is 'src' my parent from their perspective?
        def i_am_parent_of(src: Any) -> bool:
            parent = self._parent_of(src)
            return parent == node

        # 1) incorporate inbox
        context_changed = False
        child_backtrack_arrived = False

        for src, payload in inbox:
            ptype = payload.get("type")
            sname = self.name_map[src]
            if ptype == "VALUE":
                # record what we see from ANY neighbor for fallback stats
                if "color" in payload:
                    seen[sname] = payload["color"]
                # and if it's a higher neighbor, this constrains feasibility
                if self._is_higher(src, node):
                    newc = payload.get("color")
                    if context.get(sname) != newc:
                        context[sname] = newc
                        context_changed = True

            elif ptype == "BACKTRACK":
                # Only honor BACKTRACK if I am this sender's parent
                if i_am_parent_of(src):
                    if current is not None:
                        attempted.add(current)
                    current = None
                    child_backtrack_arrived = True
            else:
                # ignore unknown/none
                pass

        # 2) decide if we must (re)choose
        conflict = current is not None and (current in context.values())
        need_choice = (current is None) or conflict or child_backtrack_arrived

        if need_choice:
            # ABT standard: reset attempted if higher context changed (search space changed)
            if context_changed:
                attempted = set()

            # try to pick a feasible color: not used by higher and not retried under this context
            picked = None
            for c in self.palette:
                if c not in context.values() and c not in attempted:
                    picked = c
                    break

            if picked is not None:
                current = picked
                attempted.add(picked)
                pending_backtrack = False  # we found a feasible choice; no need to ask parent now
            else:
                # --- Fallback path: still hold SOME color (least-frequent among neighbors) ---
                # Build frequency over palette from what we have seen so far.
                freq = {c: 0 for c in self.palette}
                for col in seen.values():
                    if col in freq:
                        freq[col] += 1
                # choose color with minimal frequency; tie-break by smallest color id
                least = min(self.palette, key=lambda c: (freq[c], c))
                current = least
                # We remain in a "stuck" state relative to higher constraints, so request backtrack.
                pending_backtrack = True
                # Do NOT add fallback to attempted: it's not a feasible attempt under current constraints.

        # Also clear pending_backtrack if there is no longer any conflict and we weren't forced to re-choose
        if not need_choice and not conflict:
            pending_backtrack = False

        return {
            "context": context,
            "seen": seen,
            "current": current,
            "attempted": attempted,
            "pending_backtrack": pending_backtrack,
        }

    def extract_answers(self, global_state: Dict[Any, Dict[str, Any]]) -> List[Optional[int]]:
        # Always non-None after first round due to fallback logic
        return [global_state[u]["current"] for u in self.order]



# assumes your MessagePassingFramework is already defined/imported
# from python_dsa import MessagePassingFramework

class SolveMatchingGreedyMaximal(MessagePassingFramework):
    """
    Greedy maximal matching via synchronous message passing.

    Messages:
      - {"type": "PROPOSE", "sender": myname, "to": nbrname}
      - {"type": "ACCEPT",  "sender": myname, "to": nbrname}
      - {"type": "REJECT",  "sender": myname, "to": nbrname}
      - {"type": "MATCHED", "sender": myname, "partner": partnername}  # broadcast

    Tie-breaks:
      - When multiple proposers arrive, pick the one with the lexicographically HIGHEST name.
      - When proposing, choose the lexicographically HIGHEST candidate not in (unavailable ∪ tried).
    """

    def __init__(self, graph: nx.Graph, rounds: int, seed: Optional[int] = None, auto_run: bool = True):
        super().__init__(graph=graph, rounds=rounds, seed=seed)

        # stable names for tie-breaking; fallback to str(u) if no 'name' attribute
        self.name_map = {u: self.graph.nodes[u].get("name", str(u)) for u in self.graph.nodes()}

        if auto_run:
            self.run()  # fills self.answers

    # ---------------- Framework hooks ----------------

    def init_node_state(self, node: Any) -> Dict[str, Any]:
        return {
            "name": self.name_map[node],
            "matched": False,
            "partner": None,             # partner name (string) or None
            "unavailable": set(),        # neighbor names known to be matched
            "tried": set(),              # neighbor names we proposed to & got REJECT
            "pending_proposal_to": None, # neighbor name we proposed to and await reply from
            "proposers": set(),          # proposers collected from inbox this round
            "outgoing": [],              # list of msgs to broadcast next round
        }

    def compose_outgoing(self, node: Any, state: Dict[str, Any], round_idx: int) -> Any:
        """
        Broadcast the messages prepared in the PREVIOUS update_state.
        Receivers will filter by 'to' (or accept broadcast MATCHED).
        """
        return list(state["outgoing"])  # sent to all neighbors; receivers filter

    def update_state(
        self,
        node: Any,
        state: Dict[str, Any],
        inbox: List[Tuple[Any, Any]],
        round_idx: int
    ) -> Dict[str, Any]:
        """
        Process messages in two conceptual phases per round (staged over rounds):
          1) Read announcements & proposals, handle accepts/rejects to ME.
          2) If unmatched:
               - Choose among proposers greedily (highest name), ACCEPT one, REJECT others; set partner.
               - Otherwise, propose to best candidate (highest name) not in unavailable ∪ tried.
          3) If matched: broadcast MATCHED.
        The actual sending happens next round via compose_outgoing().
        """
        name = state["name"]
        matched = state["matched"]
        partner = state["partner"]
        unavailable = set(state["unavailable"])
        tried = set(state["tried"])
        pending_to = state["pending_proposal_to"]
        proposers = set()  # collected this round
        outgoing: List[Dict[str, Any]] = []  # messages for NEXT round

        # -------- 1) Parse inbox (broadcast payloads may be a list or a single dict) --------
        def iter_msgs(payload):
            if isinstance(payload, list):
                for m in payload:
                    if isinstance(m, dict):
                        yield m
            elif isinstance(payload, dict):
                yield payload

        for src, payload in inbox:
            for msg in iter_msgs(payload):
                mtype = msg.get("type")
                sender = msg.get("sender")
                to = msg.get("to")

                # Ignore messages not addressed to me (except MATCHED which is broadcast)
                if mtype != "MATCHED" and to != name:
                    continue

                if mtype == "MATCHED":
                    # Mark sender (and optionally its partner) as unavailable
                    unavailable.add(sender)
                    partner_s = msg.get("partner")
                    if partner_s is not None:
                        unavailable.add(partner_s)

                elif mtype == "PROPOSE":
                    # Proposal to me
                    if not matched:
                        proposers.add(sender)
                    else:
                        # I'm already matched -> will send REJECT back next round
                        outgoing.append({"type": "REJECT", "sender": name, "to": sender})

                elif mtype == "ACCEPT":
                    # My outgoing proposal was accepted
                    if pending_to == sender and not matched:
                        matched = True
                        partner = sender
                        pending_to = None
                        # announce match next round
                        outgoing.append({"type": "MATCHED", "sender": name, "partner": partner})

                elif mtype == "REJECT":
                    # My proposal got rejected
                    if pending_to == sender and not matched:
                        tried.add(sender)
                        pending_to = None

        # -------- 2) If I'm unmatched, either accept a proposer or make a new proposal --------
        if not matched:
            if proposers:
                # Deterministic greedy: pick highest-name proposer, reject others
                chosen = max(proposers)  # lexicographically highest name
                for p in proposers:
                    if p == chosen:
                        outgoing.append({"type": "ACCEPT", "sender": name, "to": p})
                    else:
                        outgoing.append({"type": "REJECT", "sender": name, "to": p})
                matched = True
                partner = chosen
                pending_to = None
                # announce next round so neighbors mark us unavailable
                outgoing.append({"type": "MATCHED", "sender": name, "partner": partner})
            else:
                # No proposers -> consider making a proposal
                # Candidates = neighbors not in unavailable ∪ tried ∪ {partner}
                neighbor_names = {self.name_map[v] for v in self.neighbors(node)}
                candidates = sorted(
                    list(neighbor_names - unavailable - tried - ({partner} if partner else set()))
                )
                if candidates:
                    # Greedy: highest lexicographic name
                    target = candidates[-1]
                    outgoing.append({"type": "PROPOSE", "sender": name, "to": target})
                    pending_to = target
                # else: no one to propose to this round

        # -------- 3) If matched and no messages queued, ensure we periodically announce MATCHED --------
        if matched:
            # Announce MATCHED occasionally to propagate availability quickly
            # (optional; comment out if too chatty)
            if not any(m.get("type") == "MATCHED" for m in outgoing):
                outgoing.append({"type": "MATCHED", "sender": name, "partner": partner})

        # Build and return next state
        return {
            "name": name,
            "matched": matched,
            "partner": partner,
            "unavailable": unavailable,
            "tried": tried,
            "pending_proposal_to": pending_to,
            "proposers": set(),   # reset per round
            "outgoing": outgoing, # will be broadcast next round
        }

    def extract_answers(self, global_state: Dict[Any, Dict[str, Any]]) -> List[str]:
        """
        Return partner names aligned with self.order.
        If unmatched, return 'None' string per your spec.
        """
        ans: List[str] = []
        for u in self.order:
            partner = global_state[u]["partner"]
            ans.append(partner if partner is not None else "None")
        return ans

Edge = Tuple[Any, Any]
UEdge = frozenset  # undirected edge as frozenset({u,v})

class SolveMatchingEdmonds:
    """
    Round-limited incremental Edmonds (via blossom target + controlled augmentations).

    Args:
        graph: nx.Graph with node attribute 'name' (fallback: str(node))
        rounds: maximum number of outer iterations to perform
        max_augment_per_round: cap on augmenting-path flips per round (>=1)
        weight_attr: optional edge attribute to use as weight (for tie-breaking)

    Behavior:
        - On each round, compute a target optimal matching M* via blossom.
        - Move current matching M toward M* by flipping up to 'max_augment_per_round'
          alternating augmenting paths found in the symmetric difference.
        - Converges to M* in at most (# of augmentations) rounds.
    """

    def __init__(
        self,
        graph: nx.Graph,
        rounds: int = 1,
        max_augment_per_round: int = 1,
        weight_attr: Optional[str] = None,
    ) -> None:
        self.G = graph
        self.rounds = max(0, int(rounds))
        self.max_aug = max(1, int(max_augment_per_round))
        self.weight_attr = weight_attr

        # stable node order & names for answers
        self.order: List[Any] = list(sorted(self.G.nodes()))
        self.name_of: Dict[Any, str] = {u: self.G.nodes[u].get("name", str(u)) for u in self.G.nodes()}

        # current matching M as a set of undirected edges (frozenset({u,v}))
        self.M: Set[UEdge] = set()

        # compute incremental rounds
        self._run_rounds()

        # final answers aligned with self.order
        self.answers: List[str] = self._make_answers()

    # ---------------- internal helpers ----------------

    def _as_uedge(self, u: Any, v: Any) -> UEdge:
        return frozenset((u, v))

    def _matching_to_pairs(self, M: Set[UEdge]) -> List[Tuple[Any, Any]]:
        return [tuple(e) for e in M]

    def _compute_target_optimal(self) -> Set[UEdge]:
        """Compute target M* using blossom (max-cardinality; optional weight for tie-breaks)."""
        Mstar = nx.algorithms.matching.max_weight_matching(
            self.G,
            maxcardinality=True,
            weight=(self.weight_attr if self.weight_attr else None),
        )
        # NetworkX returns set of (u,v) with u<v sometimes; normalize to undirected frozenset
        return {self._as_uedge(u, v) for (u, v) in Mstar}

    def _symmetric_difference_components(
        self, M: Set[UEdge], Mstar: Set[UEdge]
    ) -> List[Set[UEdge]]:
        """
        Return connected components of the symmetric difference graph (edges = M Δ M*).
        Each component is an alternating path or an even cycle.
        """
        diff = (M ^ Mstar)
        if not diff:
            return []

        H = nx.Graph()
        # add vertices that are touched by diff edges
        for e in diff:
            u, v = tuple(e)
            H.add_edge(u, v, e=e, in_M=(e in M), in_Mstar=(e in Mstar))
        # components as edge-sets
        comps: List[Set[UEdge]] = []
        for nodes in nx.connected_components(H):
            E = set()
            for u, v, d in H.subgraph(nodes).edges(data=True):
                E.add(d["e"])
            comps.append(E)
        return comps

    def _is_augmenting_path_component(
        self, comp: Set[UEdge], M: Set[UEdge], Mstar: Set[UEdge]
    ) -> bool:
        """
        An alternating component is augmenting (relative to M) iff it is a path whose
        number of edges from Mstar exceeds those from M by exactly 1.
        Equivalently, the path starts and ends with M* edges (free endpoints w.r.t. M).
        """
        # Count membership
        m_count = sum(1 for e in comp if e in M)
        ms_count = len(comp) - m_count  # those in M*
        if ms_count != m_count + 1:
            return False

        # Ensure it's a path (not a cycle): endpoints have degree 1 in the diff subgraph
        # Build small subgraph for the component
        H = nx.Graph()
        for e in comp:
            u, v = tuple(e)
            H.add_edge(u, v)
        deg1 = [n for n, d in H.degree() if d == 1]
        return len(deg1) == 2  # exactly two endpoints

    def _flip_component(self, comp: Set[UEdge], M: Set[UEdge]) -> None:
        """
        Flip along the alternating path: edges in M are removed; edges in comp-M are added.
        Assumes 'comp' is an augmenting path component.
        """
        for e in comp:
            if e in M:
                M.remove(e)
            else:
                M.add(e)

    def _one_round(self) -> bool:
        """
        Perform one incremental round toward M*.
        Returns True if at least one augmenting path was applied; False otherwise.
        """
        Mstar = self._compute_target_optimal()
        if self.M == Mstar:
            return False  # already optimal

        comps = self._symmetric_difference_components(self.M, Mstar)
        if not comps:
            return False

        applied = 0
        for comp in comps:
            if applied >= self.max_aug:
                break
            if self._is_augmenting_path_component(comp, self.M, Mstar):
                self._flip_component(comp, self.M)
                applied += 1

        return applied > 0

    def _run_rounds(self) -> None:
        for _ in range(self.rounds if self.rounds > 0 else 1):
            progressed = self._one_round()
            if not progressed:
                break  # reached optimal before exhausting rounds

    def _make_answers(self) -> List[str]:
        # Build partner map from self.M
        partner: Dict[Any, Any] = {}
        for e in self.M:
            u, v = tuple(e)
            partner[u] = v
            partner[v] = u
        return [self.name_of.get(partner.get(u), "None") if u in partner else "None" for u in self.order]



# ------------------------------------------------------------
# Solver implementing your STEP 0–7 algorithm via message passing
# ------------------------------------------------------------
import random
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple

class SolveMatchingPriorityStatus(MessagePassingFramework):
    """
    Priority+Status matching (your MIS-style rules):

    Each round, every node broadcasts STATUS:
      {type:"STATUS", name: myname, committed: bool, partner: partner_name_or_None, tentative: tentative_name_or_None}

    Logic per round:
      - Prune neighbors that are committed; clear tentative if invalid.
      - If free & no tentative: propose to a random available neighbor (STEP 4).
      - If a higher-priority neighbor is tentatively pointing to me and I'm free: accept by setting tentative = that neighbor (STEP 4).
      - Resolve conflicts (STEP 5): if mutual with v and no earlier-name agent is proposing to me or to v,
        then commit; else back off.
      - Continue until stable or rounds exhausted (STEP 6/7).
    """

    def __init__(self, graph: nx.Graph, rounds: int, seed: Optional[int] = None, auto_run: bool = True):
        super().__init__(graph=graph, rounds=rounds, seed=seed)
        # use our framework RNG for determinism unless you want a separate one
        self.rng = self._rng if seed is None else random.Random(seed)
        # Canonical names for priority (earlier = higher priority)
        self.name_map = {u: self.graph.nodes[u].get("name", str(u)) for u in self.graph.nodes()}
        if auto_run:
            self.run()  # fills self.answers

    # --------------- Framework hooks ---------------

    def init_node_state(self, node: Any) -> Dict[str, Any]:
        name = self.name_map[node]
        P = {self.name_map[v] for v in self.neighbors(node)}  # option set by neighbor NAME
        return {
            "name": name,
            "committed": False,
            "partner": None,          # committed partner name or None
            "tentative": None,        # tentative target name or None
            "P": set(P),              # remaining options (neighbor names)
            "neighbors_latest": {},   # name -> dict( committed:bool, partner:str|None, tentative:str|None )
            "outgoing": [],           # messages to send next round
        }

    def compose_outgoing(self, node: Any, state: Dict[str, Any], round_idx: int):
        """
        Broadcast last prepared messages (STATUS only).
        Receivers will merge by name.
        """
        return list(state["outgoing"])

    def update_state(
        self,
        node: Any,
        state: Dict[str, Any],
        inbox: List[Tuple[Any, Any]],
        round_idx: int
    ) -> Dict[str, Any]:
        # Unpack
        name = state["name"]
        committed = state["committed"]
        partner = state["partner"]
        tentative = state["tentative"]
        P = set(state["P"])
        neighbors_latest: Dict[str, Dict[str, Any]] = dict(state["neighbors_latest"])
        outgoing: List[Dict[str, Any]] = []

        # Helper: compare priority (earlier name = higher priority)
        def higher_priority(a: str, b: str) -> bool:
            return a < b  # lexicographic

        # Helper: my neighbors by NAME
        neighbor_names = {self.name_map[v] for v in self.neighbors(node)}

        # -------------------- STEP 2: consume STATUS --------------------
        def iter_msgs(payload):
            if isinstance(payload, list):
                for m in payload:
                    if isinstance(m, dict):
                        yield m
            elif isinstance(payload, dict):
                yield payload

        for _src, payload in inbox:
            for msg in iter_msgs(payload):
                if msg.get("type") != "STATUS":
                    continue
                n = msg.get("name")
                if n in neighbor_names:
                    neighbors_latest[n] = {
                        "committed": bool(msg.get("committed", False)),
                        "partner": msg.get("partner"),
                        "tentative": msg.get("tentative"),
                    }

        # -------------------- STEP 3: prune P, clear invalid tentative --------------------
        # Remove neighbors that are committed
        for n, st in list(neighbors_latest.items()):
            if st.get("committed", False):
                P.discard(n)
        if tentative is not None and tentative not in P:
            tentative = None

        # -------------------- STEP 4: propose_or_accept --------------------
        if not committed and tentative is None and P:
            # pick a random available neighbor from P
            tentative = self.rng.choice(list(P))

        # Offer handling: if a higher-priority neighbor points to me and I'm free, accept by setting tentative to them
        if not committed:
            # offers = neighbors with Tentative == me and not committed
            offers = [
                n for n, st in neighbors_latest.items()
                if (st.get("tentative") == name) and (not st.get("committed", False)) and (n in neighbor_names)
            ]
            if offers:
                # among offers, if any higher-priority than me exists, accept the highest-priority (earliest name)
                better = [n for n in offers if higher_priority(n, name)]
                if better:
                    candidate = sorted(better)[0]  # earliest name wins
                    if candidate in P:
                        tentative = candidate

        # -------------------- STEP 5: resolve_conflicts_by_priority --------------------
        # Commit iff mutual and no earlier-name agent is proposing to me or to my tentative neighbor
        if not committed and tentative is not None:
            v = tentative
            v_state = neighbors_latest.get(v, {"tentative": None, "committed": False})
            if v_state.get("tentative") == name and not v_state.get("committed", False):
                # mutual this round
                # collect proposers to me (A) and to v (B)
                A = {n for n, st in neighbors_latest.items() if st.get("tentative") == name}
                B = {n for n, st in neighbors_latest.items() if st.get("tentative") == v}
                # include self and v in the competition set
                contenders = A | B | {name, v}
                winner = sorted(contenders)[0]  # earliest name overall
                if winner in {name, v}:
                    # we (the pair) win -> commit
                    committed = True
                    partner = v
                    tentative = None
                else:
                    # beaten by earlier-edge contender; back off
                    tentative = None

        # -------------------- STATUS to broadcast for next round --------------------
        outgoing.append({
            "type": "STATUS",
            "name": name,
            "committed": committed,
            "partner": partner,
            "tentative": tentative,
        })

        # Build next state
        return {
            "name": name,
            "committed": committed,
            "partner": partner,
            "tentative": tentative,
            "P": P,
            "neighbors_latest": neighbors_latest,
            "outgoing": outgoing,
        }

    # --------------- Outputs ---------------

    def extract_answers(self, global_state: Dict[Any, Dict[str, Any]]) -> List[str]:
        """
        Return partner names aligned with the framework's stable node order (self.order).
        If unmatched, return 'None' string (per your spec).
        """
        return [global_state[u]["partner"] or "None" for u in self.order]





import networkx as nx
from typing import Any, Dict, List, Optional, Tuple

from typing import Any, Dict, List, Tuple, Optional
import networkx as nx

class SolveNaiveMatching(MessagePassingFramework):
    """
    Deterministic greedy matching with quality boosts.

    Pipeline:
      1) Bootstrap: Karp–Sipser leaf peeling (locks degree-1 matches).
      2) Synchronous message passing until stable:
           - Propose only to LOWER-priority neighbors; prefer FREE ones (smallest degree first).
           - Accept any HIGHER-priority proposal (preempt even from a lower commit).
           - One tentative at a time; commit on mutual.
           - Commit is 'sticky' unless partner clearly points away.
      3) End repair:
           - Upgrade mutual tentatives to commits.
           - Greedy fill to ensure maximality.
           - (Optional) Exact polish to maximum with Blossom.

    Output: answers = list of partner names (or "None") in self.order.
    """

    def __init__(
        self,
        graph: nx.Graph,
        rounds: int = 0,
        seed: Optional[int] = None,
        debug: bool = False,
        polish_optimal: bool = True,     # set False to see pure distributed result
    ) -> None:
        super().__init__(graph, rounds, seed)
        self.debug = debug
        self.polish_optimal = polish_optimal

    # ---------- helpers ----------
    def _rank(self, u: Any) -> int:
        return self.order.index(u)

    def _is_higher(self, a: Any, b: Any) -> bool:
        """a outranks b iff a comes earlier in self.order."""
        return self._rank(a) < self._rank(b)

    def _view(self, s: Dict[str, Any], v: Any) -> Dict[str, Any]:
        return s["inbox_view"].get(v, {})

    def _status_of(self, s: Dict[str, Any], v: Any) -> str:
        return self._view(s, v).get("status", "free")

    def _partner_of(self, s: Dict[str, Any], v: Any):
        return self._view(s, v).get("partner", None)

    def _available_neighbors(self, u: Any, state: Dict[str, Any]) -> List[Any]:
        """Neighbors not hard-committed to someone else (last round view)."""
        avail = []
        for v in self.neighbors(u):
            v_status = self._status_of(state, v)
            v_partner = self._partner_of(state, v)
            if not (v_status == "committed" and v_partner is not None and v_partner != u):
                avail.append(v)
        avail.sort(key=self._rank)
        return avail

    # ---------- bootstrap with Karp–Sipser leaf peeling ----------
    def bootstrap(self) -> None:
        # init clean state
        self.state = {u: self.init_node_state(u) for u in self.order}

        unmatched: set = set(self.order)
        deg = {u: self.graph.degree(u) for u in self.order}

        def residual_neighbors(u):
            return [v for v in self.graph.neighbors(u) if v in unmatched]

        changed = True
        while changed:
            changed = False
            # pick leaves deterministically (lowest rank first)
            leaves = [u for u in sorted(unmatched, key=self._rank) if deg[u] == 1]
            for u in leaves:
                if u not in unmatched:
                    continue
                nbrs = residual_neighbors(u)
                if not nbrs:
                    # isolated leaf became isolated; just drop from unmatched
                    unmatched.discard(u)
                    continue
                v = min(nbrs, key=self._rank)
                # lock u-v
                self.state[u]["status"] = "committed"
                self.state[u]["partner"] = v
                self.state[v]["status"] = "committed"
                self.state[v]["partner"] = u
                unmatched.discard(u)
                unmatched.discard(v)
                changed = True
                # update residual degrees
                for w in self.graph.neighbors(u):
                    if w in unmatched:
                        deg[w] -= 1
                for w in self.graph.neighbors(v):
                    if w in unmatched:
                        deg[w] -= 1

        # mark degree-0 residuals as unmatched but present
        # (they will remain "free" throughout)
        if self.debug:
            self.log({"event": "bootstrap_kss_done",
                      "locked_pairs": [(u, self.state[u]["partner"]) for u in self.order
                                       if self.state[u]["partner"] is not None]})

    # ---------- abstract hooks ----------
    def init_node_state(self, node: Any) -> Dict[str, Any]:
        return {
            "status": "free",              # 'free' | 'tentative' | 'committed'
            "partner": None,               # committed partner
            "tentative": None,             # neighbor we're currently trying
            "epoch": 0,                    # bump when pivoting decisions happen
            "inbox_view": {},              # last round snapshot per-neighbor
            "intent_propose_to": None,
            "intent_accept_to": None,
            "intent_withdraw_to": None,
            "intent_broadcast_matched": False,
        }

    def compose_outgoing(self, node: Any, s: Dict[str, Any], round_idx: int) -> Dict[str, Any]:
        return {
            "sender": node,
            "epoch": s["epoch"],
            "status": s["status"],
            "partner": s["partner"],
            "tentative": s["tentative"],
            "propose_to": s["intent_propose_to"],
            "accept_to": s["intent_accept_to"],
            "withdraw_to": s["intent_withdraw_to"],
            "broadcast_matched": s["intent_broadcast_matched"],
        }

    def update_state(
        self,
        node: Any,
        s: Dict[str, Any],
        inbox: List[Tuple[Any, Any]],
        round_idx: int
    ) -> Dict[str, Any]:
        # ---- ingest inbox (latest epoch per neighbor) ----
        view: Dict[Any, Dict[str, Any]] = {}
        for (nbr, msg) in inbox:
            prev = view.get(nbr)
            if prev is None or msg.get("epoch", 0) >= prev.get("epoch", 0):
                view[nbr] = msg
        s["inbox_view"] = view

        me = node
        status = s["status"]
        partner = s["partner"]
        tentative = s["tentative"]

        # clear intents
        s["intent_propose_to"] = None
        s["intent_accept_to"] = None
        s["intent_withdraw_to"] = None
        s["intent_broadcast_matched"] = False

        # committed nodes only broadcast status; no transitions
        if status == "committed":
            return s

        # collect signals
        proposals_to_me = sorted([v for v, m in view.items() if m.get("propose_to") == me], key=self._rank)
        withdraws_from = sorted([v for v, m in view.items() if m.get("withdraw_to") == me], key=self._rank)

        # availability
        available = self._available_neighbors(me, s)
        def is_available(v: Any) -> bool:
            return v in available

        def is_free(v: Any) -> bool:
            return self._status_of(s, v) == "free"

        def is_tentative_only(v: Any) -> bool:
            # usable if tentative with no partner, or partner == me
            return self._status_of(s, v) == "tentative" and self._partner_of(s, v) in (None, me)

        # ---- withdraw cleanup ----
        if tentative in withdraws_from:
            tentative = None
            status = "free"

        # ---- accept highest-priority superior proposer (preempt if needed) ----
        superiors = [v for v in proposals_to_me if self._is_higher(v, me)]
        if superiors:
            chosen = superiors[0]
            if tentative is not None and tentative != chosen:
                s["intent_withdraw_to"] = tentative
            tentative = chosen
            partner = None
            status = "tentative"
            s["intent_accept_to"] = chosen
            s["epoch"] += 1

        # ---- commit if mutual with tentative ----
        if status != "committed" and tentative is not None:
            tv = tentative
            tv_view = view.get(tv, {})
            mutual = (tv_view.get("propose_to", None) == me) or (tv_view.get("accept_to", None) == me)
            if mutual and is_available(tv):
                status = "committed"
                partner = tv
                tentative = None
                s["intent_broadcast_matched"] = True
            elif not is_available(tv):
                s["intent_withdraw_to"] = tv
                tentative = None
                status = "free"

        # ---- propose downward when free; prefer FREE neighbors of smallest degree ----
        if status == "free" and tentative is None:
            lowers_free = [v for v in self.neighbors(me) if self._is_higher(me, v) and is_free(v)]
            lowers_tent = [v for v in self.neighbors(me) if self._is_higher(me, v) and is_tentative_only(v)]
            def key_fn(v): return (self.graph.degree(v), self._rank(v))  # Karp–Sipser flavored
            lowers_free.sort(key=key_fn)
            lowers_tent.sort(key=key_fn)
            candidates = lowers_free or lowers_tent
            if candidates:
                chosen = candidates[0]
                tentative = chosen
                status = "tentative"
                s["intent_propose_to"] = chosen

        # finalize
        s["status"] = status
        s["partner"] = partner
        s["tentative"] = tentative
        return s

    # ---------- run-until-stable ----------
    def _snapshot(self) -> Dict[Any, tuple]:
        return {
            u: (self.state[u]["status"], self.state[u]["partner"], self.state[u]["tentative"])
            for u in self.order
        }

    def run(self, max_steps: int = 10000) -> List[str]:
        self.bootstrap()
        prev = self._snapshot()
        for r in range(max_steps):
            self._one_round(r)
            now = self._snapshot()
            if now == prev:
                self.answers = self.extract_answers(self.state)
                self.log({"event": "done", "rounds_run": r})
                return self.answers
            prev = now
        self.answers = self.extract_answers(self.state)
        self.log({"event": "done_max_steps", "rounds_run": max_steps})
        return self.answers

    # ---------- answers & end repair ----------
    def extract_answers(self, S: Dict[Any, Dict[str, Any]]) -> List[str]:
        # 1) upgrade mutual tentatives → committed
        for u in self.order:
            tu = S[u]["tentative"]
            if tu is not None and S.get(tu, {}).get("tentative") == u:
                S[u]["partner"] = tu
                S[u]["status"] = "committed"
                S[u]["tentative"] = None

        # 2) greedy fill to ensure maximal (deterministic order)
        edges_sorted = sorted(
            (tuple(sorted((u, v), key=self._rank)) for u, v in self.graph.edges()),
            key=lambda e: (self._rank(e[0]), self._rank(e[1]))
        )
        for u, v in edges_sorted:
            if S[u]["partner"] is None and S[v]["partner"] is None:
                S[u]["partner"] = v
                S[v]["partner"] = u
                S[u]["status"] = S[v]["status"] = "committed"

        # 3) optional exact polish to maximum with Blossom
        if self.polish_optimal:
            M = nx.max_weight_matching(self.graph, maxcardinality=True)
            mate = {u: None for u in self.order}
            for a, b in M:
                mate[a] = b
                mate[b] = a
            # overwrite partners to exact maximum
            for u in self.order:
                S[u]["partner"] = mate[u]
                S[u]["status"] = "committed" if mate[u] is not None else "free"
                S[u]["tentative"] = None

        return [str(S[u]["partner"]) if S[u]["partner"] is not None else "None" for u in self.order]


import networkx as nx
from typing import Any, Dict, List, Optional, Tuple, Set

class SolveCentralMISMatching:
    """
    Centralized MIS-on-line-graph solver that GUARANTEES a maximal matching.

    Idea:
      • Treat edges of G as vertices of the line graph L(G) implicitly.
      • Give each edge e a unique, deterministic priority rank (lex over node names).
      • In phases, select every edge that is a STRICT local minimum of its conflict
        neighborhood (edges sharing an endpoint with it), commit all simultaneously,
        delete all incident edges, repeat.
      • With unique ranks, each nonempty active set has at least one strict local minimum,
        so convergence yields an MIS of L(G) ⇒ a maximal matching of G.

    Modes:
      • run_until_converge=True  → iterate until no active edges remain (typical).
      • run_until_converge=False → do exactly `rounds` phases, then greedy-fill to ensure maximality.

    Output of run(): list[str] partner names (or "None") aligned with sorted node order.
    """

    def __init__(
        self,
        graph: nx.Graph,
        rounds: int = 0,
        run_until_converge: bool = True,
        debug: bool = False,
        node_order: Optional[List[Any]] = None,
    ):
        self.G = graph
        self.rounds = max(0, int(rounds))
        self.run_until_converge = run_until_converge
        self.debug = debug
        self.node_order: List[Any] = node_order or sorted(self.G.nodes(), key=lambda x: str(x))

    # ---------- helpers ----------
    @staticmethod
    def _canon(u: Any, v: Any) -> Tuple[Any, Any]:
        return (u, v) if str(u) <= str(v) else (v, u)

    def _active_edges_init(self) -> List[Tuple[Any, Any]]:
        E = { self._canon(u, v) for u, v in self.G.edges() }
        return sorted(E, key=lambda e: (str(e[0]), str(e[1])))

    def _build_rank(self, edges: List[Tuple[Any, Any]]) -> Dict[Tuple[Any, Any], int]:
        # Unique total order → unique integer rank
        return { e: i for i, e in enumerate(edges) }

    def _phase_select_strict_local_min(
        self,
        active_edges: Set[Tuple[Any, Any]],
        rank: Dict[Tuple[Any, Any], int],
        by_node: Dict[Any, List[Tuple[Any, Any]]],
    ) -> List[Tuple[Any, Any]]:
        """Pick all edges e that are strict local minima among edges incident to their endpoints."""
        selected: List[Tuple[Any, Any]] = []
        for e in active_edges:
            u, v = e
            nbrs = []
            if u in by_node: nbrs.extend(by_node[u])
            if v in by_node: nbrs.extend(by_node[v])
            # EXCLUDE self from the neighborhood test
            nbrs = [x for x in nbrs if x != e]
            r_e = rank[e]
            if not nbrs or all(r_e < rank[n] for n in nbrs):
                selected.append(e)
        selected.sort(key=lambda e: (str(e[0]), str(e[1])))
        return selected

    # ---------- core ----------
    def run(self) -> List[str]:
        partner: Dict[Any, Optional[Any]] = {n: None for n in self.G.nodes()}
        initial_edges = self._active_edges_init()
        rank = self._build_rank(initial_edges)
        active_edges: Set[Tuple[Any, Any]] = set(initial_edges)

        def one_phase(phase_idx: int) -> int:
            # Build by-node index over current active edges
            by_node: Dict[Any, List[Tuple[Any, Any]]] = {}
            for (u, v) in active_edges:
                by_node.setdefault(u, []).append((u, v))
                by_node.setdefault(v, []).append((u, v))

            chosen = self._phase_select_strict_local_min(active_edges, rank, by_node)
            if not chosen:
                if self.debug:
                    print(f"[phase {phase_idx}] no strict local minima; stopping.")
                return 0

            # Commit all chosen edges simultaneously (non-conflicting by construction)
            matched_nodes: Set[Any] = set()
            for (u, v) in chosen:
                if partner[u] is None and partner[v] is None:
                    partner[u] = v
                    partner[v] = u
                    matched_nodes.add(u); matched_nodes.add(v)

            # Remove all edges incident to newly matched nodes
            if matched_nodes:
                active_edges.difference_update(
                    {e for e in list(active_edges) if e[0] in matched_nodes or e[1] in matched_nodes}
                )

            if self.debug:
                print(f"[phase {phase_idx}] +{len(chosen)} edges; remaining={len(active_edges)}")
            return len(chosen)

        if self.run_until_converge:
            phase = 0
            while active_edges:
                phase += 1
                added = one_phase(phase)
                if added == 0:
                    # Shouldn’t happen with unique ranks, but guard + greedy-fill to ensure maximality.
                    break
            # After convergence, active_edges is empty ⇒ already maximal.
        else:
            # Do exactly K phases, then greedy fill to guarantee maximality.
            for phase in range(1, self.rounds + 1):
                if not active_edges:
                    break
                one_phase(phase)

            if active_edges:
                # Greedy fill on the remaining edges (deterministic order).
                rest = sorted(active_edges, key=lambda e: (str(e[0]), str(e[1])))
                used: Set[Any] = {n for n, p in partner.items() if p is not None}
                for (u, v) in rest:
                    if u not in used and v not in used:
                        partner[u] = v
                        partner[v] = u
                        used.add(u); used.add(v)
                active_edges.clear()

        # Safety: maximality (no free–free edge remains)
        for (u, v) in self.G.edges():
            if partner[u] is None and partner[v] is None:
                raise AssertionError(f"Maximality violated: uncovered edge ({u},{v}).")

        return [str(partner[n]) if partner[n] is not None else "None" for n in self.node_order]

##########################


import networkx as nx
from typing import List, Literal

Mode = Literal["maximal", "maximum"]

import networkx as nx
from typing import List, Optional, Literal

Mode = Literal["maximal", "maximum"]

def matching_assignment(G: nx.Graph, mode: Mode = "maximal") -> List[Optional[int]]:
    """
    Return a partner list for a matching on G (nodes must be 0..n-1).

    partner[i] = j if i matched with j, else None.

    mode:
      - "maximal": fast greedy maximal matching (no augmenting path checks)
      - "maximum": exact maximum-cardinality matching (Blossom algorithm)
    """
    n = G.number_of_nodes()
    if set(G.nodes()) != set(range(n)):
        raise ValueError("Graph nodes must be labeled 0..n-1 for this function.")

    if mode == "maximal":
        M = nx.maximal_matching(G)  # set of frozenset({u,v})
    elif mode == "maximum":
        M = nx.max_weight_matching(G, maxcardinality=True)  # set of pairs
    else:
        raise ValueError("mode must be 'maximal' or 'maximum'")

    partner: List[Optional[int]] = [None] * n
    for u, v in M:
        partner[u] = v
        partner[v] = u
    return partner



from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

class SolveMatchingBacktrack(MessagePassingFramework):
    """
    Distributed, priority-disciplined backtracking for Maximum Matching,
    following the provided 9-step pseudocode.

    Messaging (queued): STATUS (broadcast), PROPOSE, ACCEPT, REJECT, COMMIT, BACKTRACK.
    Only proposes to Lower neighbors; each node has at most one parent (closest Higher).
    One deterministic upgrade per round on incoming proposals.
    """

    # --------- STEP 0: helpers for global order, neighbors, parent ----------
    def _disp_name(self, u: Any) -> str:
        return self.graph.nodes[u].get("name", str(u))

    def _alpha_order(self) -> List[Any]:
        return sorted(self.graph.nodes(), key=lambda x: self._disp_name(x))

    def _neighbors_ordered(self, u: Any) -> List[Any]:
        return sorted(self.neighbors(u), key=lambda x: self._disp_name(x))

    def _split_higher_lower(self, u: Any) -> Tuple[List[Any], List[Any]]:
        name = self._disp_name(u)
        higher = [v for v in self.neighbors(u) if self._disp_name(v) < name]
        lower  = [v for v in self.neighbors(u) if self._disp_name(v) > name]
        higher.sort(key=lambda x: self._disp_name(x))
        lower.sort(key=lambda x: self._disp_name(x))
        return higher, lower

    def _parent(self, higher_sorted: List[Any]) -> Optional[Any]:
        return higher_sorted[-1] if higher_sorted else None

    # --------- bootstrap: rebind self.order to global alphabetical ----------
    def bootstrap(self) -> None:
        # Use alphabetical order of display names for the whole run
        self.order = self._alpha_order()
        super().bootstrap()

    # --------- STEP 1: node state ----------
    def init_node_state(self, node: Any) -> Dict[str, Any]:
        higher, lower = self._split_higher_lower(node)
        parent = self._parent(higher)
        return {
            # status machine
            "status": "free",          # free | tentative | committed
            "partner": None,           # node id of current tentative/committed partner
            "pending": None,           # the Lower neighbor we’re proposing to
            "phase_id": 0,             # bump on self-initiated choice changes
            "tried_in_phase": set(),   # neighbors tried in current phase

            # topology
            "higher": higher,
            "lower": lower,
            "parent": parent,
            "neighbors": self._neighbors_ordered(node),

            # knowledge of neighbors (latest STATUS by phase)
            "view": {},                # v -> {"status": str, "partner": node|None, "phase": int}

            # control flags
            "dirty": True,             # if True, broadcast STATUS this round
            "pause_for_backtrack": False,

            # queued messages to send *next* round (flush in compose_outgoing)
            "outbox": [],              # list of (type, target, payload)
        }

    # --------- message constructors ----------
    def _msg_status(self, u, st) -> Tuple[str, Optional[Any], Dict[str, Any]]:
        return ("STATUS", None, {"from": u, "status": st["status"], "partner": st["partner"], "phase": st["phase_id"]})

    def _msg_propose(self, u, v, st) -> Tuple[str, Any, Dict[str, Any]]:
        return ("PROPOSE", v, {"from": u, "phase": st["phase_id"]})

    def _msg_accept(self, u, v, st) -> Tuple[str, Any, Dict[str, Any]]:
        return ("ACCEPT", v, {"from": u, "phase": st["phase_id"]})

    def _msg_reject(self, u, v, st) -> Tuple[str, Any, Dict[str, Any]]:
        return ("REJECT", v, {"from": u, "phase": st["phase_id"]})

    def _msg_commit(self, u, v, st) -> Tuple[str, Any, Dict[str, Any]]:
        return ("COMMIT", v, {"from": u, "phase": st["phase_id"]})

    def _msg_backtrack(self, u, v, st, child=None) -> Tuple[str, Any, Dict[str, Any]]:
        return ("BACKTRACK", v, {"from": u, "phase": st["phase_id"], "child": child})

    # --------- feasibility predicate ----------
    def _feasible(self, u, v, st) -> bool:
        # Only propose to Lower neighbors
        if v not in st["lower"]:
            return False

        view = st["view"].get(v, {})
        v_status = view.get("status", "free")
        v_partner = view.get("partner", None)

        if v_status == "committed" and v_partner != u:
            return False
        if v_status == "tentative" and v_partner is not None:
            # can preempt only if u is earlier than v's current tentative partner
            return self._disp_name(u) < self._disp_name(v_partner)
        return True

    # --------- STEP 2: choose_first_candidate ----------
    def _choose_first(self, u, st):
        higher = st["higher"]
        # require at least one STATUS seen from each Higher before acting
        if higher and not all(h in st["view"] for h in higher):
            return None

        for v in st["lower"]:
            if v in st["tried_in_phase"]:
                continue
            if self._feasible(u, v, st):
                st["pending"] = v
                st["partner"] = v          # keep partner in sync for tentative state
                st["phase_id"] += 1
                st["status"] = "tentative"
                st["dirty"] = True
                st["tried_in_phase"] = set({v})
                # send actual PROPOSE next round (via compose_outgoing)
                return v
        return None

    # --------- STEP 3: proposals to me (one upgrade per round) ----------
    def _handle_proposals(self, u, st, proposals: List[Dict[str, Any]]) -> bool:
        """
        Process incoming PROPOSEs. Appends ACCEPT/REJECT to st['outbox'].
        Returns True if we performed an upgrade (accept or switch) this round.
        """
        if not proposals:
            return False

        outbox = st["outbox"]
        proposers = sorted([p["from"] for p in proposals], key=lambda x: self._disp_name(x))
        winner = proposers[0]

        if st["status"] == "committed":
            for p in proposers:
                outbox.append(self._msg_reject(u, p, st))
            return False

        # If we somehow are 'tentative' but lost partner, treat as free
        if st["status"] == "tentative" and st.get("partner") is None:
            st["status"] = "free"

        if st["status"] == "free":
            st["status"] = "tentative"
            st["partner"] = winner
            st["phase_id"] += 1
            st["dirty"] = True
            outbox.append(self._msg_accept(u, winner, st))
            for p in proposers[1:]:
                outbox.append(self._msg_reject(u, p, st))
            return True

        if st["status"] == "tentative":
            cur = st["partner"]
            # guard against None
            if cur is None:
                st["partner"] = winner
                st["phase_id"] += 1
                st["dirty"] = True
                outbox.append(self._msg_accept(u, winner, st))
                for p in proposers[1:]:
                    outbox.append(self._msg_reject(u, p, st))
                return True

            # switch only if winner is earlier than current tentative
            if self._disp_name(winner) < self._disp_name(cur):
                outbox.append(self._msg_reject(u, cur, st))
                st["partner"] = winner
                st["phase_id"] += 1
                st["dirty"] = True
                outbox.append(self._msg_accept(u, winner, st))
                for p in proposers:
                    if p not in {winner, cur}:
                        outbox.append(self._msg_reject(u, p, st))
                return True
            else:
                for p in proposers:
                    outbox.append(self._msg_reject(u, p, st))
                return False

        return False

    # --------- STEP 4: responses to my own pending ----------
    def _handle_responses_to_own(self, u, st, accepts: List[Dict], rejects: List[Dict]) -> None:
        outbox = st["outbox"]
        pend = st["pending"]
        if pend is None:
            return

        accepted_by_pending = any(a["from"] == pend for a in accepts)
        rejected_by_pending = any(r["from"] == pend for r in rejects)

        if accepted_by_pending:
            st["status"] = "committed"
            st["partner"] = pend
            st["pending"] = None
            st["phase_id"] += 1
            st["dirty"] = True
            outbox.append(self._msg_commit(u, st["partner"], st))
            return

        if rejected_by_pending:
            st["tried_in_phase"].add(pend)
            st["pending"] = None
            st["partner"] = None
            st["status"] = "free"
            st["dirty"] = True

    # --------- STEP 5 & 6: try next or backtrack ----------
    def _propose_next_or_backtrack(self, u, st) -> None:
        outbox = st["outbox"]
        # try next feasible Lower not tried in phase
        for v in st["lower"]:
            if v in st["tried_in_phase"]:
                continue
            if self._feasible(u, v, st):
                st["pending"] = v
                st["partner"] = v
                st["phase_id"] += 1
                st["status"] = "tentative"
                st["dirty"] = True
                st["tried_in_phase"].add(v)
                outbox.append(self._msg_propose(u, v, st))
                return

        # none found → backtrack
        if st["parent"] is None:
            st["pause_for_backtrack"] = False
            return
        st["pause_for_backtrack"] = True
        outbox.append(self._msg_backtrack(u, st["parent"], st))

    # --------- STEP 7: handle BACKTRACK from child ----------
    def _handle_backtrack_from_child(self, u, st) -> None:
        outbox = st["outbox"]

        if st["status"] == "tentative":
            cur = st["partner"]
            for v in st["lower"]:
                if v == cur:
                    continue
                if v in st["tried_in_phase"]:
                    continue
                if self._feasible(u, v, st):
                    # switch tentative target proactively
                    st["partner"] = v
                    st["pending"] = v
                    st["phase_id"] += 1
                    st["dirty"] = True
                    st["tried_in_phase"].add(v)
                    outbox.append(self._msg_propose(u, v, st))
                    return
            # escalate up if possible
            if st["parent"] is not None:
                outbox.append(self._msg_backtrack(u, st["parent"], st, child=u))
            return

        if st["status"] == "committed":
            # release only if immediate safe alternative exists
            for v in st["lower"]:
                if v == st["partner"]:
                    continue
                if v in st["tried_in_phase"]:
                    continue
                if self._feasible(u, v, st):
                    st["partner"] = v
                    st["pending"] = v
                    st["phase_id"] += 1
                    st["dirty"] = True
                    st["tried_in_phase"].add(v)
                    outbox.append(self._msg_propose(u, v, st))
                    return
            if st["parent"] is not None:
                outbox.append(self._msg_backtrack(u, st["parent"], st, child=u))
            return

        # free: forward up if possible
        if st["parent"] is not None:
            outbox.append(self._msg_backtrack(u, st["parent"], st, child=u))

    # --------- STEP 8: update views from STATUS ----------
    def _refresh_views_from_status(self, st, statuses: List[Dict[str, Any]]) -> None:
        for s in statuses:
            v = s["from"]
            prev = st["view"].get(v)
            if prev is None or int(s.get("phase", 0)) >= int(prev.get("phase", -1)):
                st["view"][v] = {
                    "status": s.get("status", "free"),
                    "partner": s.get("partner", None),
                    "phase": s.get("phase", 0),
                }

    # --------- STEP 9: finalize_now (last round) ----------
    def _finalize_now(self, u, st) -> None:
        outbox = st["outbox"]

        # Close unanswered pending
        if st["pending"] is not None:
            st["tried_in_phase"].add(st["pending"])
            st["pending"] = None
            st["dirty"] = True

        # Upgrade mutual tentatives to committed when safe
        if st["status"] == "tentative" and st["partner"] is not None:
            v = st["partner"]
            v_view = st["view"].get(v, {})
            if v_view.get("status") in {"tentative", "free"} and v_view.get("partner") in {None, u}:
                st["status"] = "committed"
                st["phase_id"] += 1
                st["dirty"] = True
                outbox.append(self._msg_commit(u, v, st))
                return

        # Conservative forced-commit (anytime mode)
        if st["status"] == "free":
            for v in st["lower"]:
                v_view = st["view"].get(v, {})
                v_stat, v_part = v_view.get("status", "free"), v_view.get("partner", None)
                if v_stat == "free" or (v_stat == "tentative" and v_part == u):
                    st["status"] = "committed"
                    st["partner"] = v
                    st["phase_id"] += 1
                    st["dirty"] = True
                    outbox.append(self._msg_commit(u, v, st))
                    break

    # --------- compose_outgoing (flush queued + routine sends) ----------
    def compose_outgoing(self, node: Any, st: Dict[str, Any], round_idx: int) -> List[Tuple[str, Optional[Any], Dict[str, Any]]]:
        out: List[Tuple[str, Optional[Any], Dict[str, Any]]] = []

        # Broadcast STATUS only when something changed
        if st.get("dirty", False):
            out.append(self._msg_status(node, st))
            st["dirty"] = False

        # Keep sending current proposal (idempotent w.r.t. phase_id)
        if st["pending"] is not None:
            out.append(self._msg_propose(node, st["pending"], st))

        # Re-issue backtrack request while paused
        if st.get("pause_for_backtrack", False) and st["parent"] is not None:
            out.append(self._msg_backtrack(node, st["parent"], st))

        # Flush queued point-to-point messages from previous update
        if st["outbox"]:
            out.extend(st["outbox"])
            st["outbox"] = []

        return out

    # --------- update_state (receive & decide; queue messages for next round) ----------
    def update_state(
        self, u: Any, st: Dict[str, Any], inbox: List[Tuple[Any, Any]], round_idx: int
    ) -> Dict[str, Any]:
        # Collect neighbor messages
        messages: List[Tuple[Any, Tuple[str, Optional[Any], Dict[str, Any]]]] = []
        for (src, payload) in inbox:
            if isinstance(payload, list):
                for m in payload:
                    messages.append((src, m))
            elif isinstance(payload, tuple):
                messages.append((src, payload))
            elif payload is not None:
                messages.append((src, payload))

        # Partition
        broadcasts: List[Dict] = []
        proposals: List[Dict] = []
        accepts: List[Dict] = []
        rejects: List[Dict] = []
        commits: List[Dict] = []
        backtracks: List[Dict] = []

        for _, m in messages:
            if not isinstance(m, tuple) or len(m) != 3:
                continue
            mtype, target, data = m
            if mtype == "STATUS" and target is None:
                broadcasts.append(data)
            elif target == u:
                if mtype == "PROPOSE":
                    proposals.append(data)
                elif mtype == "ACCEPT":
                    accepts.append(data)
                elif mtype == "REJECT":
                    rejects.append(data)
                elif mtype == "COMMIT":
                    commits.append(data)
                elif mtype == "BACKTRACK":
                    backtracks.append(data)

        # STEP 8: integrate STATUS
        self._refresh_views_from_status(st, broadcasts)

        # React to commits: if they point to us, stabilize locally
        for a in commits:
            v = a["from"]
            if st["status"] in {"tentative", "free"} and st["partner"] in {None, v}:
                st["status"] = "committed"
                st["partner"] = v
                st["pending"] = None
                st["dirty"] = True

        # STEP 3: handle proposals (limit to one upgrade)
        _ = self._handle_proposals(u, st, proposals)

        # STEP 4: my pending proposal outcomes
        self._handle_responses_to_own(u, st, accepts, rejects)

        # Keep pending aligned if we’re tentative without pending
        if st["status"] == "tentative" and st["partner"] is not None and st["pending"] is None:
            st["pending"] = st["partner"]

        # STEP 2/5: if free and not waiting on backtrack, choose or continue
        if st["status"] == "free" and not st.get("pause_for_backtrack", False):
            if st["pending"] is None:
                self._choose_first(u, st)
            # else: compose_outgoing will re-send proposal

        # STEP 5/6: if blocked (free or tentative) and no pending, try next or backtrack
        if st["pending"] is None and st["status"] in {"free", "tentative"}:
            self._propose_next_or_backtrack(u, st)

        # STEP 7: backtrack requests from children
        if backtracks:
            self._handle_backtrack_from_child(u, st)

        # STEP 9: finalization on last round
        if round_idx == self.rounds - 1:
            self._finalize_now(u, st)
            # Mark dirty so a final STATUS would be sent next round if it existed.
            # Not needed for extract_answers.

        return st

    # --------- extract answers (display names aligned with self.order) ----------
    def extract_answers(self, global_state: Dict[Any, Dict[str, Any]]) -> List[Optional[str]]:
        answers: List[Optional[str]] = []
        for u in self.order:
            partner = global_state[u].get("partner", None)
            answers.append(self._disp_name(partner) if partner is not None else None)
        return answers



import networkx as nx
from typing import Any, Dict, List, Optional, Tuple

class SolveVertexCoverPriority(MessagePassingFramework):
    """
    Priority-based distributed vertex cover.
    Converges to a minimal vertex cover after pruning.
    """

    def _disp_name(self, u):
        return self.graph.nodes[u].get("name", str(u))

    def _alpha_order(self):
        return sorted(self.graph.nodes(), key=lambda x: self._disp_name(x))

    def _neighbors_ordered(self, u):
        return sorted(self.neighbors(u), key=lambda x: self._disp_name(x))

    def bootstrap(self):
        self.order = self._alpha_order()
        super().bootstrap()

    def init_node_state(self, node):
        return {
            "cover": "YES",                   # initial state: all in cover
            "neighbors": self._neighbors_ordered(node),
            "view": {},                       # neighbor states
        }

    def _msg_status(self, u, st):
        return ("STATUS", None, {"from": u, "state": st["cover"]})

    # ✅ fix: return single tuple, not list
    def compose_outgoing(self, node, st, round_idx):
        return self._msg_status(node, st)

    def _integrate_statuses(self, st, statuses):
        for s in statuses:
            v, sv = s["from"], s.get("state")
            if sv in {"YES", "NO"}:
                st["view"][v] = sv

    # -------- minimality check --------
    def _can_drop_without_breaking(self, u, st):
        """Check if this YES node can safely drop to NO and still cover all edges."""
        if st["cover"] != "YES":
            return False
        # Only safe if all neighbors are YES
        for v in st["neighbors"]:
            if st["view"].get(v) != "YES":
                return False
        return True

    def update_state(self, u, st, inbox, round_idx):
        # absorb neighbor statuses
        statuses = []
        for (src, payload) in inbox:
            if isinstance(payload, tuple) and payload[0] == "STATUS":
                statuses.append(payload[2])
        self._integrate_statuses(st, statuses)

        # ---- Phase 1: repair uncovered edges ----
        if st["cover"] == "NO":
            for v in st["neighbors"]:
                if st["view"].get(v) == "NO":
                    # both NO → higher priority takes responsibility
                    if self._disp_name(u) < self._disp_name(v):
                        st["cover"] = "YES"
                    break

        # ---- Phase 2: prune redundant YES ----
        elif self._can_drop_without_breaking(u, st):
            st["cover"] = "NO"

        return st

    def extract_answers(self, global_state):
        return [global_state[u]["cover"] for u in self.order]


class SolveVertexCoverBacktrack(MessagePassingFramework):
    """
    Distributed, priority-disciplined backtracking for Minimum Vertex Cover.
    Implements the 10-step pseudocode with Step 4 pruning.

    Local label ∈ {UNDECIDED, YES, NO}.
    Messages: STATUS, COVER_REQ, COVER_LOCK, REJECT_BRANCH, BACKTRACK, INCUMBENT.
    """

    # ---------- STEP 0: helpers ----------
    def _disp_name(self, u):
        return self.graph.nodes[u].get("name", str(u))

    def _alpha_order(self):
        return sorted(self.graph.nodes(), key=lambda x: self._disp_name(x))

    def _neighbors_ordered(self, u):
        return sorted(self.neighbors(u), key=lambda x: self._disp_name(x))

    def _split_higher_lower(self, u):
        name = self._disp_name(u)
        higher = [v for v in self.neighbors(u) if self._disp_name(v) < name]
        lower  = [v for v in self.neighbors(u) if self._disp_name(v) > name]
        higher.sort(key=lambda x: self._disp_name(x))
        lower.sort(key=lambda x: self._disp_name(x))
        return higher, lower

    def _parent(self, higher_sorted):
        return higher_sorted[-1] if higher_sorted else None

    # ---------- bootstrap ----------
    def bootstrap(self):
        self.order = self._alpha_order()
        super().bootstrap()

    # ---------- STEP 1: init state ----------
    def init_node_state(self, node):
        higher, lower = self._split_higher_lower(node)
        parent = self._parent(higher)
        return {
            # local decision
            "label": "UNDECIDED",     # UNDECIDED | YES | NO
            "phase_id": 0,

            # topology
            "higher": higher,
            "lower": lower,
            "parent": parent,
            "neighbors": self._neighbors_ordered(node),

            # knowledge
            "view": {},               # neighbor -> {label, phase}
            "incumbent": None,        # (size, signature) best known globally

            # branching context (captain-oriented edge tuple (captain, other))
            "edge_captain": None,     # (u,v) with u is earlier endpoint; set when I'm captain
            "pending_branch": None,   # (choice, edge), choice ∈ {"YOU=YES", "OTHER=YES"}
            "tried_here": set(),      # {(edge, choice)} already tried locally (completeness)

            # flow control
            "pause_for_backtrack": False,
            "outbox": [],
            "dirty": True,
        }

    # ---------- messages ----------
    def _msg_status(self, u, st):
        return ("STATUS", None, {"from": u, "label": st["label"], "phase": st["phase_id"]})

    def _msg_cover_req(self, u, v, edge, st):
        return ("COVER_REQ", v, {"from": u, "edge": edge, "phase": st["phase_id"]})

    def _msg_cover_lock(self, u, v, edge, st):
        return ("COVER_LOCK", v, {"from": u, "edge": edge, "phase": st["phase_id"]})

    def _msg_reject_branch(self, u, v, edge, st):
        return ("REJECT_BRANCH", v, {"from": u, "edge": edge, "phase": st["phase_id"]})

    def _msg_backtrack(self, u, v, st, child=None):
        return ("BACKTRACK", v, {"from": u, "phase": st["phase_id"], "child": child})

    def _msg_incumbent(self, u, best, sig, st):
        return ("INCUMBENT", None, {"from": u, "size": best, "sig": sig, "phase": st["phase_id"]})

    # ---------- visibility utilities (for Step 4) ----------
    def _visible_label(self, u, st, v):
        """Return my current belief about v's label (neighbors or self)."""
        if v == u:
            return st["label"]
        vv = st["view"].get(v, {})
        return vv.get("label", "UNDECIDED")

    def _uncovered_incident_edges(self, u, st):
        """
        Locally visible *incident* edges (u,v) that appear uncovered:
        i.e., neither endpoint currently believed YES.
        Admissible: may underestimate uncovered edges; safe for pruning.
        Returns oriented edges as (captain, other) with captain earlier by name.
        """
        edges = []
        for v in st["neighbors"]:
            lu = self._visible_label(u, st, u)
            lv = self._visible_label(u, st, v)
            if lu != "YES" and lv != "YES":
                # Orient by alphabet order so captain is first
                if self._disp_name(u) < self._disp_name(v):
                    edges.append((u, v))
                else:
                    edges.append((v, u))
        # deduplicate
        seen = set()
        uniq = []
        for e in edges:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        return uniq

    def _greedy_matching_LB(self, edges):
        """
        Maximal greedy matching size over locally visible uncovered incident edges.
        Lower bound on min vertex cover size (admissible).
        """
        matched = set()
        msize = 0
        for (a, b) in edges:
            if a in matched or b in matched:
                continue
            matched.add(a); matched.add(b)
            msize += 1
        return msize

    def _current_yes_visible(self, u, st):
        """
        My current visible count of YES among (self ∪ neighbors).
        Admissible undercount of global YES; safe in current_yes + LB ≥ B* test.
        """
        cnt = 0
        if st["label"] == "YES":
            cnt += 1
        for v in st["neighbors"]:
            if self._visible_label(u, st, v) == "YES":
                cnt += 1
        return cnt

    # ---------- STEP 2: choose witness and start ----------
    def _choose_witness_and_start(self, u, st):
        # Wait for all higher STATUS (or skip if none)
        if st["higher"] and not all(h in st["view"] for h in st["higher"]):
            return

        # Find a locally uncovered incident edge and be its captain if earlier
        for v in st["neighbors"]:
            lu = self._visible_label(u, st, u)
            lv = self._visible_label(u, st, v)
            if lu != "YES" and lv != "YES":
                if self._disp_name(u) < self._disp_name(v):
                    edge = (u, v)  # captain is u
                    # Try in fixed order: YOU=YES first, then OTHER=YES
                    if (edge, "YOU=YES") not in st["tried_here"]:
                        st["tried_here"].add((edge, "YOU=YES"))
                        st["edge_captain"] = edge
                        st["pending_branch"] = ("YOU=YES", edge)
                        st["label"] = "YES"
                        st["phase_id"] += 1
                        st["dirty"] = True
                        return
                    elif (edge, "OTHER=YES") not in st["tried_here"]:
                        st["tried_here"].add((edge, "OTHER=YES"))
                        st["edge_captain"] = edge
                        st["pending_branch"] = ("OTHER=YES", edge)
                        st["label"] = "NO"
                        st["phase_id"] += 1
                        st["dirty"] = True
                        st["outbox"].append(self._msg_cover_req(u, v, edge, st))
                        return
        # If no uncovered edge locally, we might already be consistent under current context.

    # ---------- STEP 3: apply branch choice (idempotent) ----------
    def _apply_branch_choice(self, u, st):
        if st["pending_branch"] is None:
            return
        choice, edge = st["pending_branch"]
        if choice == "YOU=YES":
            # Ensure label is YES (already set in chooser)
            if st["label"] != "YES":
                st["label"] = "YES"
                st["phase_id"] += 1
                st["dirty"] = True
        elif choice == "OTHER=YES":
            # Ensure label = NO and COVER_REQ sent (chooser already did)
            if st["label"] != "NO":
                st["label"] = "NO"
                st["phase_id"] += 1
                st["dirty"] = True

    # ---------- STEP 4: share status and prune (branch-and-bound) ----------
    def _share_status_and_prune(self, u, st):
        # Always broadcast STATUS on changes via compose_outgoing (st["dirty"])
        # Compute admissible lower bound from locally uncovered incident edges
        edges = self._uncovered_incident_edges(u, st)
        LB = self._greedy_matching_LB(edges)
        current_yes = self._current_yes_visible(u, st)

        inc = st["incumbent"]
        if inc is None:
            return  # nothing to prune against yet
        B_star = int(inc[0])

        if current_yes + LB >= B_star:
            # This branch cannot beat incumbent → reject current branch locally
            pb = st["pending_branch"]
            if pb is None:
                # No active branch here; just escalate backtrack if possible
                if st["parent"] is not None:
                    st["pause_for_backtrack"] = True
                    st["outbox"].append(self._msg_backtrack(u, st["parent"], st))
                return

            choice, edge = pb
            # Mark this choice tried
            st["tried_here"].add((edge, choice))

            # If OTHER=YES was requested, inform the other endpoint to release/abort
            if choice == "OTHER=YES":
                captain, other = edge
                # Since edge is (captain, other) and I'm captain if I'm the earlier endpoint:
                # If I'm not captain (defensive), figure out the "other" endpoint relative to me
                if u == captain:
                    v_to_notify = other
                else:
                    v_to_notify = captain
                st["outbox"].append(self._msg_reject_branch(u, v_to_notify, edge, st))

            # Try sibling if untried; otherwise BACKTRACK
            sibling = "OTHER=YES" if choice == "YOU=YES" else "YOU=YES"
            if (edge, sibling) not in st["tried_here"]:
                st["pending_branch"] = (sibling, edge)
                if sibling == "YOU=YES":
                    st["label"] = "YES"
                    st["phase_id"] += 1
                    st["dirty"] = True
                else:
                    st["label"] = "NO"
                    st["phase_id"] += 1
                    st["dirty"] = True
                    # Send a COVER_REQ for sibling immediately if I'm captain
                    captain, other = edge
                    if u == captain:
                        st["outbox"].append(self._msg_cover_req(u, other, edge, st))
                return
            else:
                # Both choices tried here → backtrack upward
                if st["parent"] is not None:
                    st["pause_for_backtrack"] = True
                    st["outbox"].append(self._msg_backtrack(u, st["parent"], st))

    # ---------- STEP 5: descend/escalate handled implicitly by branching + backtrack ----------

    # ---------- STEP 6: handle COVER_REQ ----------
    def _on_cover_req(self, u, st, msg):
        edge = msg["edge"]
        # If I can hold YES without contradicting my branch, lock it
        if st["label"] in {"UNDECIDED", "YES"}:
            st["label"] = "YES"
            st["phase_id"] += 1
            st["dirty"] = True
            st["outbox"].append(self._msg_cover_lock(u, msg["from"], edge, st))
        else:
            st["outbox"].append(self._msg_reject_branch(u, msg["from"], edge, st))

    # ---------- STEP 7: on REJECT_BRANCH feedback ----------
    def _on_reject_branch(self, u, st, msg):
        # If our current branch got rejected, try the sibling or backtrack
        if st["pending_branch"] is None:
            return
        choice, edge = st["pending_branch"]
        if edge != msg.get("edge"):
            return

        # mark tried and switch to sibling if available
        st["tried_here"].add((edge, choice))
        sibling = "OTHER=YES" if choice == "YOU=YES" else "YOU=YES"
        if (edge, sibling) not in st["tried_here"]:
            st["pending_branch"] = (sibling, edge)
            if sibling == "YOU=YES":
                st["label"] = "YES"
                st["phase_id"] += 1
                st["dirty"] = True
            else:
                st["label"] = "NO"
                st["phase_id"] += 1
                st["dirty"] = True
                captain, other = edge
                if u == captain:
                    st["outbox"].append(self._msg_cover_req(u, other, edge, st))
        else:
            # both tried → escalate
            if st["parent"] is not None:
                st["pause_for_backtrack"] = True
                st["outbox"].append(self._msg_backtrack(u, st["parent"], st))

    # ---------- STEP 8: incumbent updates ----------
    def _on_incumbent(self, st, msg):
        size, sig = msg["size"], msg["sig"]
        if st["incumbent"] is None or size < st["incumbent"][0]:
            st["incumbent"] = (int(size), sig)

    # ---------- STEP 9: on BACKTRACK from child ----------
    def _on_backtrack(self, u, st, msg):
        edge = st["edge_captain"]
        if edge:
            # try any remaining sibling choice here (enforces completeness)
            if (edge, "YOU=YES") not in st["tried_here"]:
                st["tried_here"].add((edge, "YOU=YES"))
                st["pending_branch"] = ("YOU=YES", edge)
                st["label"] = "YES"
                st["phase_id"] += 1
                st["dirty"] = True
                return
            if (edge, "OTHER=YES") not in st["tried_here"]:
                st["tried_here"].add((edge, "OTHER=YES"))
                st["pending_branch"] = ("OTHER=YES", edge)
                st["label"] = "NO"
                st["phase_id"] += 1
                st["dirty"] = True
                captain, other = edge
                if u == captain:
                    st["outbox"].append(self._msg_cover_req(u, other, edge, st))
                return
        # otherwise forward up
        if st["parent"] is not None:
            st["pause_for_backtrack"] = True
            st["outbox"].append(self._msg_backtrack(u, st["parent"], st, child=u))

    # ---------- STEP 10: finalize ----------
    def _finalize_now(self, u, st):
        # conservative: if undecided at the end, prefer NO (keeps cover small)
        if st["label"] == "UNDECIDED":
            st["label"] = "NO"
            st["dirty"] = True

    # ---------- compose_outgoing ----------
    def compose_outgoing(self, node, st, round_idx):
        out = []
        if st.get("dirty", False):
            out.append(self._msg_status(node, st))
            st["dirty"] = False
        if st["outbox"]:
            out.extend(st["outbox"])
            st["outbox"] = []
        return out

    # ---------- update_state ----------
    def update_state(self, u, st, inbox, round_idx):
        # Collect & partition messages
        statuses, cover_reqs, rejects, backtracks, incumbents = [], [], [], [], []
        for (src, payload) in inbox:
            if isinstance(payload, list):
                for m in payload:
                    self._partition(u, m, statuses, cover_reqs, rejects, backtracks, incumbents)
            elif isinstance(payload, tuple):
                self._partition(u, payload, statuses, cover_reqs, rejects, backtracks, incumbents)
            elif payload is not None:
                self._partition(u, payload, statuses, cover_reqs, rejects, backtracks, incumbents)

        # Integrate STATUS (latest phase wins)
        for s in statuses:
            v = s["from"]
            prev = st["view"].get(v)
            if prev is None or int(s.get("phase", 0)) >= int(prev.get("phase", -1)):
                st["view"][v] = {"label": s.get("label", "UNDECIDED"), "phase": int(s.get("phase", 0))}

        # React to control messages
        for msg in cover_reqs:
            self._on_cover_req(u, st, msg)
        for msg in incumbents:
            self._on_incumbent(st, msg)
        for msg in rejects:
            self._on_reject_branch(u, st, msg)
        for msg in backtracks:
            self._on_backtrack(u, st, msg)

        # Step 3: apply current branch (idempotent)
        self._apply_branch_choice(u, st)

        # Step 2: if free to choose, start a branch
        if st["pending_branch"] is None and st["label"] == "UNDECIDED" and not st.get("pause_for_backtrack", False):
            self._choose_witness_and_start(u, st)

        # Step 4: prune against incumbent (if any)
        self._share_status_and_prune(u, st)

        # Step 10: finalize on last round
        if round_idx == self.rounds - 1:
            self._finalize_now(u, st)

        return st

    def _partition(self, u, m, statuses, cover_reqs, rejects, backtracks, incumbents):
        if not isinstance(m, tuple) or len(m) != 3:
            return
        mtype, target, data = m
        if mtype == "STATUS" and target is None:
            statuses.append(data)
        elif target == u:
            if mtype == "COVER_REQ":
                cover_reqs.append(data)
            elif mtype == "REJECT_BRANCH":
                rejects.append(data)
            elif mtype == "BACKTRACK":
                backtracks.append(data)
            elif mtype == "COVER_LOCK":
                # COVER_LOCK is informative; no special queue needed here
                pass
        if mtype == "INCUMBENT":
            incumbents.append(data)

    # ---------- extract answers ----------
    def extract_answers(self, global_state):
        # Return each node's final label in order (UNDECIDED/YES/NO)
        return [global_state[u].get("label", "UNDECIDED") for u in self.order]
