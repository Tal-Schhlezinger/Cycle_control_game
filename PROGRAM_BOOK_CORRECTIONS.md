# Corrections to `cycle_control_program_book.md`

This document lists specific corrections to apply to the original program book. The original is overall accurate and high-quality; these are surgical fixes for items that are either overstated, missing nuance, or have been addressed by the time you read this.

For each correction, I show the original text, the suggested replacement, and why.

---

## Correction 1: Section 3.6, item 6 — soften "lazy no-legal-moves" risk language

### Current text
> 6. **`end_on_no_legal_moves` is lazily checked only near the end of the board.** This is a performance optimization, but if a ruleset can produce early mutual lockout, the game may fail to terminate when it logically should.

### Replace with
> 6. **`end_on_no_legal_moves` is lazily checked only when ≤12.5% of cells remain empty.** This is a performance optimization. In all observed games (1000+ across multiple radii) mutual lockout has only occurred near full board fill, so the optimization has been correct in practice. However, it has not been *proven* safe: under strict adjacency + neutrality, a position with both players locked out at 30% empty is theoretically constructible. The proper fix is incremental legal-move tracking (see optimization book Section 4.4), which makes the check exact AND fast. Until that is implemented, treat the lazy check as a known correctness risk to be retired in Phase 2.

### Why
The original wording overstates the risk. We've never observed the failure mode in practice, but it's still a real correctness concern that needs fixing eventually. Better to characterize it accurately.

---

## Correction 2: Section 4.5, "Critical notes" — clarify what `_count_legal_moves_for` actually does

### Current text (third bullet)
> The lazy `end_on_no_legal_moves` check is unsafe if early lockout is possible. Either prove early mutual lockout cannot happen under committed rules, or check it whenever legal-move absence is suspected.

### Replace with
> The lazy `end_on_no_legal_moves` check is correct in all observed cases but has not been proven safe under strict adjacency + neutrality. The retiring fix is incremental legal-move counts maintained in state, which would make the check `O(1)` and exact. Until then, the threshold (≤12.5% empty cells) is a heuristic, not a proof.

### Why
Same as Correction 1.

---

## Correction 3: Section 5.6, "Bot classes" → `GreedyBot.choose_action` complexity

### Current text
> `GreedyBot.choose_action(...)` ... `O(L·(clone + apply + eval))`. Worst can be `O(L·V²)` or worse if expensive territory weights enabled.

### Replace with
> `GreedyBot.choose_action(...)` ... `O(L·(clone + apply + eval))`. With current default weights this is `O(L·V)` per move (`L` candidates × `V` for clone). The `O(L·V²)` worst case only applies if `exclusive_territory_diff` or `territory_diff` weights are enabled (they are not, by default). Greedy_2 specifically uses frontier-only territory which is `O(V)` per evaluation.

### Why
The original phrasing implies typical Greedy_2 behavior is `O(L·V²)`; it isn't, because the expensive flood-fill weights are disabled. The pessimistic case only applies if weights are explicitly re-enabled.

---

## Correction 4: Section 5.4, "Notes" → mention frontier drift risk

### Current text
> This is the right direction for search performance. However, because it still calls `engine.apply_placement`, it inherits engine legality and end-check costs. A truly high-performance search engine would need a lighter internal transition function or cached legality data.

### Append
> Additional concern: `SearchState.apply` computes frontier deltas around the placed node by checking neighbors' adjacency to existing stones. The implementation handles the placing player's frontier correctly, but the opponent's frontier delta is computed less carefully — specifically, when `node` becomes occupied, neighbors that previously had `node` as their only opponent-color reference do not have their frontier status updated. In the current implementation this is masked because `frontier_diff` is computed from running counts that are reset at each `SearchState` construction (once per move at the root), so drift cannot accumulate across moves. But within a single deep search, the counts may drift from ground truth. A correct bitset-based frontier (per optimization book Section 4.2) would eliminate this concern entirely.

### Why
This is a real subtle issue that the original author may not have observed. Worth flagging now so it's accounted for when the bitset state is implemented.

---

## Correction 5: Section 8 — clarify which docs are stale

### Current text
> ### Important documentation drift
> 
> There is drift between docs and code:
> 
> - `cycle_control/ai/__init__.py` says SearchBot is not implemented, but it is.

### Replace with
> ### Documentation drift to fix
>
> - `cycle_control/ai/__init__.py` docstring claims sections of the AI checklist are "not yet implemented" but the AI layer (action space, bot interface, siege, search, tournament) IS implemented. Only the RL training stack (PettingZoo env, GNN, CNN, PPO) is unimplemented. Update the docstring to reflect this.
> - README/UI help disagree about scoring marker colors. Reconcile.
> - The scripts use "committed" language but do not use `RulesConfig.committed()`. Fix the scripts (see action checklist item 1).

### Why
More precise and points at the action checklist for the actual fix.

---

## Correction 6: Section 11 priority list — restructure

### Replace section "## Highest priority" entirely

The original list has 5 highest-priority items. Reorder to reflect dependencies and how impactful each is:

```
## Highest priority — fix BEFORE any further bot tournaments

1. **Fix committed-rules construction in analysis scripts.**
   The most important fix: tournament data published to date may not reflect the actual committed ruleset. Without this fix, all bot strength conclusions are suspect. Replace manual `RulesConfig(...)` construction in `greedy_validation.py` and `search_analysis.py` with `RulesConfig.committed(radius)`.

2. **Fix `FrontierRandomBot.__init__` to store topology.**
   Currently silently degrades to uniform random. Add `self._topology = topology` to constructor.

3. **Fix `ActionSpace.build_mask` to use the precomputed dict.**
   Eliminates an `O(L·V)` factor in every bot move. One-line change.

4. **Fix `_on_apply_modes` in `ui.py`.**
   Add `end_on_no_legal_moves=self.rules.end_on_no_legal_moves` to the `RulesConfig(...)` constructor call.

5. **Remove `default_territory_eval` from `cycle_control/ai/bots/__init__.py:__all__`.**
   It's exported but no longer defined. Will fail at import time on any code that does `from cycle_control.ai.bots import default_territory_eval`.

## Medium priority — clean up before adding RL

6. **Reconcile SearchBot docstring vs. implementation.**
   Either rename to `_minimax`/"minimax" everywhere (recommended — it IS minimax) OR convert to negamax (more code change for no real benefit).

7. **Update `cycle_control/ai/__init__.py` docstring.**
   Currently states sections are "not yet implemented" that have been implemented.

8. **Remove unused imports.**
   `sys` in `scoring.py`, `MoveError` in `search_utils.py`, etc.

## Low priority — defer until Phase 2 of optimization roadmap

9. **Replace lazy `end_on_no_legal_moves` with exact incremental tracking.**
   Required for correctness guarantees, but not blocking RL development.

10. **Persistence: validate board matches topology on load.**
    Defensive only.
```

### Why
The original list mixes urgent correctness issues with lower-priority cleanup. Reorganizing makes clear what blocks the next RL work vs. what can be deferred.

---

## Correction 7: Section 14 "Bottom-line assessment" — make experimental-validity bottom line explicit

### Current ends with
> The biggest performance issue is repeated full-board scanning during legality/mask/greedy/search operations. The next serious engineering step should be cached legality/frontier state, not more AI complexity.

### Replace with
> The biggest immediate risk is **experimental validity**. Some scripts construct rule configurations manually instead of using `RulesConfig.committed()`, so tournament results may have been measured under slightly different rules than intended. Fix the scripts BEFORE running any further analyses or claiming bot strength comparisons.
> 
> The biggest performance issue is repeated full-board scanning. Indexed topology + bitset state would give the largest speedups but is a multi-week investment. For the current research phase (validating the game design and getting a basic RL pipeline working), the small fixes in items 1-5 of section 11 are enough. Defer the bigger optimization work until after Phase 0 RL training has produced its first checkpoints.

### Why
Bottom-line should explicitly distinguish "fix immediately" from "fix eventually."

---

## What does NOT need correcting

The following sections are accurate as written:

- Section 1 (executive summary)
- Section 2 (complexity notation)
- Section 3.1-3.5 (architecture diagrams, scoring model, turn model)
- Section 4.1-4.10 (file-by-file, with the exceptions above)
- Section 5.1-5.7 (AI files, with the exceptions above)
- Section 6 (UI)
- Section 7 (tests)
- Section 9 (runtime scenarios)
- Section 10 (performance hotspots — accurate, just timing-of-fixes question)
- Section 12 (suggested architecture improvements — sound)
- Section 13 (file responsibility index)
