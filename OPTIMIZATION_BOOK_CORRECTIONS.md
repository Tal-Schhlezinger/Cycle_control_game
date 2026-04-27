# Corrections to `cycle_control_optimization_book.md`

The optimization book is technically sound. The directions and priorities are correct. Main corrections needed are about **timing** (when to do which optimization) and a few specific claims to refine.

---

## Correction 1: Section 0 — add scope/timing note

### Add new subsection at end of Section 0

```markdown
### 0.1 Scope of this document vs. immediate fixes

This document describes a multi-phase optimization roadmap that, if fully implemented, would likely make the codebase 10-100× faster on tactical search and 5-20× faster on greedy/tournament play.

**Important: do not interpret this roadmap as "implement everything before going to RL."**

Phase 1 (small safe wins) should be implemented immediately because it includes correctness fixes. Phases 2-6 are real performance work that should be deferred until after Phase 0 RL training has produced its first checkpoints. The reasoning:

1. Most Phase 2-6 optimizations matter for *deep search* and *long tournaments*. The current research phase needs neither — it needs to verify the game design via RL training.
2. The biggest absolute time savings come from RL infrastructure (PettingZoo, batched inference, GPU utilization) which is orthogonal to the optimizations listed here.
3. Some optimizations (Zobrist hashing, transposition tables) are most valuable when search depth >= 4. Current SearchBot is depth=2 by default.

So: implement Phase 1 to leave the codebase sound, then resume RL development. Return to Phases 2-6 as specific bottlenecks emerge during training.
```

### Why
The roadmap is sound but the document doesn't make clear which parts are "do now" vs. "do later." Without this note, a reader might spend weeks on optimizations that don't unblock the next research milestone.

---

## Correction 2: Section 3.1 — clarify the lazy-end-condition fix isn't a P0 blocker

### Current text
> #### 3.1 Replace lazy `end_on_no_legal_moves`
> Current `MoveEngine._check_end_conditions` only checks no-legal-moves when the board is nearly full. That is a speed shortcut, but it is not logically safe under strict adjacency + neutrality. The game can theoretically become stuck while many cells remain empty.

### Replace with
> #### 3.1 Replace lazy `end_on_no_legal_moves` (priority: medium, not P0)
> Current `MoveEngine._check_end_conditions` only checks no-legal-moves when ≤12.5% of cells remain empty. This is a speed shortcut. In all observed games (1000+ games at R=2..5), mutual lockout has only occurred near full board fill, so the optimization has been correct in practice.
> 
> However, it has not been *proven* safe under strict adjacency + neutrality. A position with both players locked out at 30% empty is theoretically constructible. This is a real correctness concern, but it has not been observed and is not blocking the next milestone. The proper fix is the incremental legal-move tracking from Section 4.4, which makes the check both exact and `O(1)`.
> 
> **Recommendation:** retire the lazy check as part of Phase 2 (exact legal cache), not as an immediate fix.

### Why
The original phrasing suggests this is a P0 (top priority correctness fix). It is a correctness concern, but in practice has not bitten us, and the proper fix requires the larger incremental-tracking infrastructure that's already planned for Phase 2.

---

## Correction 3: Section 4.5 — flag the existing frontier drift

### Add to the end of Section 4.5 (or create new subsection 4.5.1)

```markdown
### 4.5.1 Known issue: SearchState frontier delta is incomplete

The current `SearchState.apply()` computes frontier deltas for the placing player but does NOT fully handle how the placement affects the opponent's frontier counts. Specifically: when `node` becomes occupied, neighbors of `node` that previously had `node` as their only opponent-color reference do not have their frontier status updated.

This is masked in the current code because `frontier_diff` uses running counts that are reset at each `SearchState` construction (once per move at the search root), so drift cannot accumulate across moves. But within a single deep search, the counts may drift from ground truth, which means SearchBot's leaf evaluations are not exact.

**Fix:** the bitset-based frontier representation proposed in Section 4.2 (`frontier_bits: dict[Player, int]`) sidesteps this entirely because the frontier is recomputed via bit operations from current occupancy bits. There is no incremental delta to get wrong.
```

### Why
This is a real but subtle issue not mentioned in either document. Adding it now ensures it's not lost.

---

## Correction 4: Phase 1 list (Section 6) — split into "blocking" vs. "should-fix-soon"

### Current text
```
## Phase 1 — small safe wins
1. Fix `ActionSpace.build_mask()` to use precomputed node mapping.
2. Fix `FrontierRandomBot.__init__` to store topology and action space.
3. Replace repeated `any(s == own_state for s in state.board.values())` with maintained stone counts or a helper count.
4. Add `empty_count` to state or engine-derived fast state.
5. Use `RulesConfig.committed()` in scripts.
6. Add tests for committed-rules equivalence.
```

### Replace with
```
## Phase 1 — small safe wins

### Phase 1a — BLOCKING (fix before next bot tournament or RL work)

1. Use `RulesConfig.committed()` in `greedy_validation.py` and `search_analysis.py`.
   Without this, tournament data is suspect.
2. Fix `FrontierRandomBot.__init__` to store topology.
   Without this, the bot is silently uniform random.
3. Fix `ActionSpace.build_mask()` to use precomputed dict.
   `O(V²)` → `O(V)` on every bot move.
4. Fix `_on_apply_modes` in `ui.py` to preserve `end_on_no_legal_moves`.
   Without this, manual playtest uses wrong rules.
5. Remove `default_territory_eval` from `cycle_control/ai/bots/__init__.py:__all__`.
   Currently breaks `from cycle_control.ai.bots import default_territory_eval`.
6. Add tests asserting committed-rules equivalence in scripts.
   Prevent regression of #1.

### Phase 1b — should fix soon, not blocking

7. Replace repeated `any(s == own_state for s in state.board.values())` 
   with a per-player stone count maintained in state. `O(V)` → `O(1)`.
8. Add `empty_count` to state. `O(V)` → `O(1)` for the empty-fraction check
   that gates the lazy `end_on_no_legal_moves` test.
9. Reconcile SearchBot docstring vs. implementation.
   Either rename to `_minimax` everywhere or convert to true negamax.
10. Update `cycle_control/ai/__init__.py` docstring (says sections are not 
    implemented when they are).
11. Remove unused imports across the codebase.

Expected impact: 1a fixes correctness blockers. 1b is small wins, low risk, 
and clears stale items from the audit.
```

### Why
The existing list has 6 items but doesn't distinguish blocking from non-blocking. The action checklist for Claude Code needs this distinction explicit.

---

## Correction 5: Section 7 — exact-vs-approximate table — refine "auto-fill" entry

### Current row
| Auto-fill when stuck | Changes game process | May be intended experiment rule, but it is not neutral search optimization. |

### Replace with
| Auto-fill when stuck | Intended rule, not optimization | This is part of the committed ruleset (no passing — fill all reachable cells). It is a rules feature, not a performance optimization. List as "rules" rather than "optimization" in this context. |

### Why
The original implies auto-fill is something to consider as a tradeoff. It isn't — it's a designed feature of the committed rules (introduced in Phase 4 of implementation notes). The optimization book should not present it as an optimization choice.

---

## Correction 6: Phase 4 list (Section 6) — caveat the SearchBot upgrades

### Current text
```
## Phase 4 — SearchBot serious upgrade
1. Add Zobrist hash.
2. Add transposition table.
3. Add PV/hash move ordering.
4. Add killer/history heuristics.
5. Use exact cached legal counts in `_leaf_eval`.
6. Throttle time checks.
```

### Add caveat at end of Phase 4
```
**Caveat on Phase 4 timing:**

Phase 4 (full SearchBot upgrade) is most useful when search depth >= 4 and 
when SearchBot is used as a primary research tool. At current usage 
(depth=2, used as a Phase 0 training opponent), the speedup matters less.

If after Phase 0 the trained agents need to be evaluated against a strong 
search-based opponent (e.g. for a published Elo benchmark), implement 
Phase 4 then. Otherwise consider the simpler approach of using past 
checkpoints of the trained agent as the strong baseline (the AlphaZero 
style — the network with shallow rollouts is the strong opponent).
```

### Why
Avoid the trap of "implement Phase 4 because it's on the list." Make explicit when it pays off.

---

## What does NOT need correcting

These sections are accurate and well-thought-out:

- Section 1 (online research summary)
- Section 2 (current bottleneck profile)
- Section 4.1, 4.2, 4.3, 4.4, 4.6 (architecture-level optimization strategy)
- Section 5 (file-by-file optimizations)
- Section 8 (suggested new internal APIs)
- Section 9 (blunt verdict)
- Section 10 (references)
