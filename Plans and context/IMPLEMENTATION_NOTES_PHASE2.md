# Cycle Control AI — Implementation Notes (Phase 2)

**Status:** SearchBot + search-before-learning analysis per checklist Sections 4.3, 4.5, and 6.
**Covers:** adds to Phase 1 delivery.
**Not yet covered:** Sections 7-12 (networks, PPO, training phases).

---

## What was built in Phase 2

```
cycle_control/ai/bots/search_bot.py   # SearchBot with negamax + alpha-beta
search_analysis.py                     # search-before-learning analysis script
```

Tests in `tests_ai.py` grew from 32 to 40. All 40 pass. All 87 engine tests still pass.

### SearchBot design choices

- **Algorithm:** alpha-beta minimax (not negamax despite function name — cleaner handling of search_side fixed while color_to_move changes within multi-placement turns)
- **Move ordering:** by leaf eval, descending for maximizer. Significant pruning speedup.
- **Leaf evaluator:** defaults to territory-focused evaluation matching Greedy_2's weights. Rationale: Phase 1 found territory dominates under the committed ruleset, so the leaf eval should reflect that — a cycle-focused leaf eval would misjudge positions at every non-terminal leaf.
- **Iterative deepening:** supported via `time_budget_s`. Useful for tournament play where depth-per-move varies with branching factor.
- **Terminal detection:** full `+1_000_000 / -1_000_000` terminal values, so search prefers winning sequences over territorial advantages.
- **Not implemented:** transposition tables (requires state hashing, checklist Section 7.2.D), quiescence search (not obviously valuable for this game), null-move pruning.

---

## Checklist Section 4.5 phase gate — now PASSES

Before Phase 2, Section 4.5 could not be satisfied: one-ply greedy lost to Random on the committed ruleset, so "heuristics meaningfully stronger than Random" was not true.

After Phase 2, SearchBot at depth 2 beats Random **100%** on R=2 committed, and at depth 3 beats Greedy_2 (previously the strongest bot) **90%**. The phase gate is satisfied.

Actual bot skill ladder on the committed ruleset at R=2 (by Elo after round-robin):

| Bot | Elo | Notes |
|---|---|---|
| Search_d2 | 1449 | clear top, +289 over Greedy_2 |
| Greedy_2 | 1160 | territory-focused one-ply |
| Random | 1116 | |
| Greedy_1 | 1075 | cycle-focused one-ply — worse than Random! |

Deeper search (d=3) is available but ~5-8x slower; useful for research analysis but not required for Phase 0 bootstrap.

---

## Search-before-learning analysis (checklist Section 6)

Run `python search_analysis.py --radius 2 --games 10`. Findings at R=2:

### Finding 1: Non-transitive Elo ordering

Expected order by design: Search > Greedy_2 > Greedy_1 > Random.
Actual order by Elo: Search > Greedy_2 > Random > Greedy_1.

**Greedy_1 has lower Elo than Random.** This is empirical confirmation that pure cycle-greedy play is actively worse than random choice on the committed ruleset. The analysis script flags this as "a signal of non-transitivity or evaluation mismatch" — both are true here.

### Finding 2: Search provides substantial skill improvement (+289 Elo over Greedy_2)

The analysis script reports "Search substantially improves on Greedy — game has real tactical depth." This is important evidence for RL: the game rewards lookahead, meaning a trained neural policy with MCTS or just deeper decision-making can meaningfully improve over one-ply heuristics. If search had given only marginal improvement, the game would be strategically shallow and not worth RL research.

### Finding 3: Healthy draw rate (18.3%)

Draw rate is moderate across all pairings, not dominated by defender-favoring dynamics. The rules appear playable — neither trivially decisive nor deadlock-prone.

### Finding 4: Greedy_2 vs Search_d2 draws 50%

Between the top two bots, half of games end in draws. This is a strong signal that the game has meaningful strategic equilibrium at higher skill levels — not "whoever plays first wins" and not "whoever builds more stones wins."

### What this means for Phase 0 (base training)

The opponent mix recommended for Phase 0 (per checklist Section 10) should be:
- SearchBot as the main skill anchor (depth=2 for speed, depth=3 for quality checks)
- Greedy_2 as a secondary skill anchor (reasonable strength, much faster than search)
- Greedy_1 as a *diversity* anchor (plays differently — cycle-greed — even though it's weak)
- Random for baseline

The original checklist had "~20% heuristic anchors (G1 + G2 + SearchBot)". This should be rebalanced: the anchors are NOT equally skilled, and training should reflect that.

---

## Revised recommendation for Phase 0 opponent mix

Per Phase 1 and Phase 2 findings, the Phase 0 opponent mix (Section 10.1 of checklist) should be:

| Opponent | Share | Purpose |
|---|---|---|
| Own snapshots (recent, last 8-16) | ~50% | standard self-play |
| Older own snapshots | ~20% | anti-forgetting |
| SearchBot d=2 | ~15% | skill anchor |
| Greedy_2 | ~5% | cheap skill anchor |
| Greedy_1 | ~5% | strategic diversity anchor |
| Random | ~5% | baseline sanity |

This replaces the checklist's "~20% heuristic anchors" bucket with a weighted mix that reflects actual bot skill. Document this deviation from the checklist when running Phase 0.

---

## Performance notes

Timings on the development box (R=2, no serious optimization):

| Bot | Games vs Random (10 games) |
|---|---|
| Greedy_1 / Greedy_2 | ~0.5s / ~0.7s |
| SearchBot d=1 | ~0.5s |
| SearchBot d=2 | ~1.5s |
| SearchBot d=3 | ~8s |

For R=3 the numbers roughly triple. For R=10, SearchBot at depth 3 is likely >1 minute per game and should only be used for sparse evaluation, not as a live training opponent. In the Phase 0 mix above, use d=2 as the training anchor and d=3 only for periodic evaluation.

Transposition tables would likely give 3-10x speedup. Worth adding if search becomes a bottleneck in Phase 0 — not needed now.

---

## Updated "what to do next"

In order:

1. **Phase 0 base training scaffolding** (checklist Section 10) — PettingZoo AEC environment wrapper, observation encoder (Section 9.1), training harness with the opponent mix above
2. **GNN architecture** (Section 7) — PyG-based, board-size transferable per Ben-Assayag & El-Yaniv
3. **CNN architecture** (Section 8) — fully convolutional per the size-transfer constraint
4. **PPO training loop** (Section 9.2-9.5) with MaskablePPO
5. **Phase 0 training** on R=4 first, then curriculum to R=7, R=10
6. **Phase 1 cloning + fine-tuning** to produce the 4-agent starter pool

Realistic schedule estimate: each of items 1-4 is a session's work; 5-6 run for hours on hardware.

---

## Confidence statements

- Implementation correctness: **~95%** (40/40 AI tests pass, 87/87 engine tests pass)
- SearchBot strength claims: **~90%** (results reproducible across seeds; skill ordering clear and consistent)
- Validity of the "search d=2 is a good Phase 0 anchor" recommendation: **~80%** (based on R=2 data; need to validate at R=3+ before Phase 0, and SearchBot d=2 may be too slow at R=7+)
- Finding that committed rules have "real tactical depth": **~75%** (Search_d2 +289 Elo over Greedy_2 is strong evidence, but single-board data; validate at R=3+ before drawing firm conclusions)
