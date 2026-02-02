class MatchingBackTrackFunctionPseudoCodeOnlyV2(Matching, FunctionPseudoCodeOnly):
    def __init__(
        self,
        graph: nx.Graph,
        rounds: int,
        problem_level,
        model_name: str = "gpt-4o-mini",
        model_provider: str = "openai",
        chain_of_thought: bool = True,
        is_must_send: bool = True,
    ):
        Matching.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        FunctionPseudoCodeOnly.__init__(self)
        self.problem_level = problem_level

    def get_symbolic_functions(self):
        main_function = """
    1) establish_order_and_parent
    • Sort all agent names alphabetically to define a strict total order ≺.
    • Higher = {v : v ≺ me}; Lower = {v : me ≺ v}.
    • parent = the immediate predecessor in Higher (the ≺-max element of Higher), or None if Higher is empty.
    • Rule of thumb: you ONLY ever PROPOSE to Lower (never to Higher).

    2) init_state
    • state ∈ {free, tentative(x), committed(x)} (exactly one at a time).
    • pending_proposal = None.
    • candidate_list = neighbors in preferred order (default: alphabetical, i.e., by ≺).
    • context = {nei: {status: "unknown", partner: None, phase_id: -1} for nei in neighbors}.
    • phase_id = 0. tried_set = ∅, where entries are pairs (phase_id, candidate).
    • Bump phase_id ONLY when your own state (free/tentative/committed) or pending_proposal changes; clear tried_set when phase_id bumps.

    3) choose_first_candidate
    • Wait rule: if Higher is empty, act immediately; else wait until you have received STATUS from every Higher (or a timeout).
    • Define FEASIBLE(cand) for cand ∈ Lower:
      – cand is not known committed to someone else.
      – If cand is tentative with x ≠ me, then me ≺ x (I outrank cand’s current tentative).
      – (phase_id, cand) ∉ tried_set.
    • Scan candidate_list in ≺ order; pick the first FEASIBLE(cand).
    • If found:
      – pending_proposal = cand
      – SEND PROPOSE(me) → cand
      – phase_id += 1; SEND STATUS(mode="tentative", partner=cand, phase_id) to all Lower
    • If none found:
      – Call backtrack_if_needed (STEP 8).

    4) on_propose_from_neighbor(nei)
    • Collect all proposers this round; let p* be the ≺-minimum (earliest alphabetically).
    • If state = free:
      – state = tentative(p*); REJECT all others; phase_id += 1; SEND STATUS("tentative", p*, phase_id) to Lower.
    • If state = tentative(x):
      – If p* ≺ x: switch to tentative(p*); REJECT x and all other proposers; phase_id += 1; STATUS.
      – Else: REJECT all proposers.
    • If state = committed(x):
      – REJECT all proposers (commits are NOT broken by proposals).
    • One-upgrade-per-round: do at most one tentative upgrade due to proposals in the same round.

    5) on_accept_from(candidate) 
      – state = committed(candidate); pending_proposal = None
      – SEND COMMIT(me↔candidate)
      – phase_id += 1; SEND STATUS("committed", candidate, phase_id) to all Lower

    6) on_reject_from(candidate):
      – tried_set.add((phase_id, candidate))
      – Call propose_next_or_backtrack (STEP 7).

    7) propose_next_or_backtrack
    • Re-scan Lower in ≺ order for FEASIBLE(cand) (as in STEP 3).
    • If found:
      – pending_proposal = cand; SEND PROPOSE(me→cand); phase_id += 1; SEND STATUS("tentative", cand, phase_id) to Lower; RETURN.
    • If none found: call backtrack_if_needed (STEP 8).

    8) backtrack_if_needed
    • If parent is None:
      – SEND FAIL to all Lower (optional); remain free; SEND STATUS("free", None, phase_id) once; stop proposing further.
    • Else:
      – SEND BACKTRACK → parent
      – SEND STATUS(mode="requesting_help", partner=None, phase_id) to all Lower
      – Pause initiating new proposals until any Higher’s STATUS updates your view (which will trigger a phase bump).

    9) on_backtrack_from_child(child)
    • If state = tentative(x):
      – Attempt propose_next_or_backtrack() to shift pressure (may change your tentative); STATUS on change.
    • If state = committed(x):
      – Atomic switch allowed ONLY if you can immediately propose to some FEASIBLE y ∈ Lower:
        * SEND UNDO_COMMIT → x; set state = free (ephemeral)
        * Immediately SEND PROPOSE → y; set pending_proposal = y; phase_id += 1; STATUS
      – If no immediate FEASIBLE y exists: forward BACKTRACK to parent (or FAIL if parent=None).

    10) share_status
    • Send STATUS only on change; include {mode ∈ {"free","tentative","committed","requesting_help"}, partner, phase_id}.
    • Receivers update context[sender] and ignore any STATUS whose phase_id is < their stored phase for that sender.
    • Do not spam periodic STATUS; send exactly once per change.

    11) finalize_now
    • Close: if pending_proposal is unanswered, add (phase_id, pending_proposal) to tried_set and clear pending_proposal.
    • Mutual-tentative first: if state=tentative(p) and context[p].status=="tentative" and context[p].partner==me → upgrade to committed(p); SEND COMMIT; phase_id+=1; STATUS.
    • Opportunistic one-shot (if still free): pick the earliest clean q ∈ Lower (q is free or tentative only with me) for which no known contender r with r ≺ me exists; SEND a single COMMIT(me↔q).
    • If multiple COMMITs are received, accept the earliest sender by ≺ and reject the rest (at most one final commit).
    • Send one final STATUS reflecting your terminal state; stop.
    """
        return main_function


class MatchingBackTracFunctionDescriptionOnlyV2(Matching, FunctionDescriptionOnly):
    def __init__(
            self,
            graph: nx.Graph,
            rounds: int,
            problem_level,
            model_name: str = "gpt-4o-mini",
            model_provider: str = "openai",
            chain_of_thought: bool = True,
            is_must_send: bool = True,
    ):
        Matching.__init__(
            self,
            graph=graph,
            rounds=rounds,
            problem_level=problem_level,
            model_name=model_name,
            model_provider=model_provider,
            chain_of_thought=chain_of_thought,
            is_must_send=is_must_send,
        )
        FunctionDescriptionOnly.__init__(self)
        self.problem_level = problem_level

    def get_symbolic_functions(self):
        main_function = """
        1) establish_priority_and_parent
        • Agree on a total alphabetical order so everyone knows who they defer to (Higher) and who defers to them (Lower).
        • Choose your parent as the immediate predecessor in that order (closest Higher); if none, you have no parent.
        • You only initiate proposals toward Lower neighbors (never to Higher).

        2) init_state
        • Start clean: exactly one state at a time (free, tentative(x), or committed(x)); clear any pending proposal.
        • Keep a light notebook of what you hear and a small version counter so neighbors can ignore stale info.
        • Update the version only when your own state or pending choice changes.

        3) choose_first_candidate
        • First make sure you’ve heard from Higher (or a sensible timeout) so you avoid clashing with them.
        • Pick a compatible Lower neighbor (not known committed; if they’re tentative, you must outrank their current tentative).
        • Announce a single tentative choice and reach out once—don’t spam multiple offers.

        4) on_propose_from_neighbor(nei)
        • Apply the shared tie-break consistently; accept at most one proposer and decline the rest.
        • If you already hold a tentative, switch only for a strictly better (higher-priority) proposer—and at most once this round.
        • Never break a commitment just because proposals arrived.

        5) on_accept_from(candidate)
        • Turn your tentative into a firm commitment, tell neighbors you’re taken, and drop other pending offers.
        • Keep the message minimal and consistent so others can recompute quickly.
        • Proceed with normal listening thereafter.

        6) on_reject_from(candidate)
        • Record that this path didn’t work under your current view and move on.
        • Try a different compatible option; don’t repeat the same failed attempt until your view changes.
        • If none remain, prepare to backtrack (see below).

        7) propose_next_or_backtrack
        • Continue down your ordered list, attempting the next compatible Lower neighbor.
        • If no compatible option exists right now, escalate politely to your parent for help.
        • While waiting on escalation, avoid creating new conflicts.

        8) backtrack_if_needed
        • Ask your parent (closest Higher) to reconsider so capacity can free up downstream.
        • Pause initiating fresh tentatives until something changes upstream and your view updates.
        • If you have no parent, stop proposing further and announce that you remain free.

        9) on_backtrack_from_child(child)
        • First try a local shuffle: if you’re tentative, attempt a different compatible Lower to relieve pressure.
        • If you’re committed, only release if you can immediately switch to a safe alternative in one atomic step; otherwise forward the backtrack to your parent.
        • Keep the flow one-directional upward to avoid circular backtracking.

        10) share_status
        • Send concise updates only on real changes (free/tentative/committed/requesting_help) with who you’re paired with if any.
        • Neighbors refresh their view and ignore older versions; no periodic repeats.
        • This keeps feasibility checks cheap and messages sparse.

        11) finalize_now
        • Close any dangling offers, then prefer mutual, uncontested tentatives and upgrade them to commitments.
        • If still free, at most one clean, deterministic commit to a Lower neighbor that has no better known contender.
        • Resolve simultaneous commits by the shared tie-break, announce your final state once, and stop.
        """
        return main_function
