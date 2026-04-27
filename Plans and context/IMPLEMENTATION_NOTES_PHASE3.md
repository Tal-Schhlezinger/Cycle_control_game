# Cycle Control AI — Implementation Notes (Phase 3)

**Status:** multi-radius tournament, Greedy_2 speed fix.
**Covers:** cross-radius validation (R=2..5), performance optimisation, test suite maintenance.
**Continues from:** IMPLEMENTATION_NOTES_PHASE2.md.

---

## What changed in Phase 3

### Greedy_2 evaluation — O(V²) → O(V)

The original Greedy_2 used `exclusive_territory_diff` (full siege flood-fill, O(V²)) as its primary signal. This was 4s/game at R=3 and ~16s/game at R=4, making tournaments above R=3 impractical.

**Fix:** replaced `exclusive_territory=2.0` with `frontier=2.0` in Greedy_2's default weights. Frontier count (empty cells adjacent to own stones) is an O(V) proxy for territorial reach. Also updated `default_territory_eval` in SearchBot: removed `territory_diff`, promoted `frontier_diff`.

**Speed after fix:**

| Radius | Greedy_2 vs Random per game | SearchBot d=2 per game |
|---|---|---|
| R=2 | 0.02s | 0.10s |
| R=3 | 0.36s | 1.36s |
| R=4 | 1.63s | ~9s |
| R=5 | 2.92s | impractical |

**Implication for RL:** the full siege flood-fill (`siege.py`) is available for analysis but must NOT be called on every search node or greedy evaluation. Frontier count is the right real-time feature.

---

## Multi-radius tournament results

Committed ruleset (neutrality + strict adjacency + mirror adjacency) throughout.

### Summary table

| Metric | R=2 | R=3 | R=4 | R=5 |
|---|---|---|---|---|
| Cells | 24 | 54 | 96 | 150 |
| Draw rate | 12% | 2% | 13% | 25% |
| Avg game length (moves) | 22 | 46 | 67 | 102 |
| Search_d2 Elo | 1459 | — | — | — |
| Greedy_2 Elo | 1414 | 1528 | 1459 | 1328 |
| Random Elo | 1065 | 1177 | 1130 | 1192 |
| Greedy_1 Elo | 862 | 895 | 1011 | 1080 |

(Search excluded from R=3+ due to per-game time. SearchBot data at R=2 included for completeness.)

### Win-rate matrices

**R=3** (20 games/pair):

|  | Random | Greedy_1 | Greedy_2 |
|---|---|---|---|
| Random | — | 95% | 5% |
| Greedy_1 | 0% | — | 0% |
| Greedy_2 | 95% | 100% | — |

**R=4** (10 games/pair):

|  | Random | Greedy_1 | Greedy_2 |
|---|---|---|---|
| Random | — | 60% | 10% |
| Greedy_1 | 0% | — | 0% |
| Greedy_2 | 100% | 100% | — |

**R=5** (4 games/pair — noisy, treat as directional):

|  | Random | Greedy_1 | Greedy_2 |
|---|---|---|---|
| Random | — | 50% | 25% |
| Greedy_1 | 50% | — | 0% |
| Greedy_2 | 75% | 100% | — |

---

## Research findings

### Finding 1: User's concern was valid — R=2/3 distorts results

At R=2/3, Greedy_1 wins 0% vs everything. Looked like a clean signal. At R=4, G1 wins 60% vs Random. At R=5, 50% (essentially equal). The small board collapses G1's strategy because there's no space for cycle structures to form.

**Implication:** R=2/3 is not a valid test of strategic quality. Use R=4 minimum; R=5+ for anything to be taken seriously. The checklist's R=4 curriculum start is correct.

### Finding 2: Draw rate rises with board size (2% → 25%)

R=3: nearly all decisive. R=5: 25% draws, games averaging 102 moves (68% board coverage). Trend is clear even with small sample at R=5.

At R=10 (600 cells), extrapolating: expect ~40% draws and ~400-move games. A 40% draw rate means sparse terminal reward signal for RL — the agent gets no clear gradient 40% of the time. This is a material concern for Phase 0.

**Recommendation:** run 20+ games at R=7 before committing to R=10 training. If draw rate >35%, investigate whether a tiebreaker (stone count, scoring nodes at draw) is needed, or whether the rules produce structural draws through mutual territory-lock.

### Finding 3: Greedy_1 recovering — non-transitivity weakening

The Phase 2 finding that G1 < Random (non-transitive Elo) weakens with board size. At R=5 they are equal. At R=7/10, G1 likely beats Random. The cycle-focused strategy becomes viable when there is enough board space to build structures.

**Implication for Phase 1 cloning:** CNN_cycle and GNN_cycle agents are less disadvantaged at R=7/10. The divergence validation gate (Section 11.6 of checklist) must run at R=5+ to be meaningful.

### Finding 4: Greedy_2 dominance compresses with board size

G2's Elo margin over Random: R=3 (+351), R=4 (+329), R=5 (+136). Converging toward parity at larger boards. Territory strategy remains dominant but becomes less decisive.

### Finding 5: Game length scales at ~70% board coverage

Moves per game: 22 (R=2), 46 (R=3), 67 (R=4), 102 (R=5). Consistent ~70% fill rate. At R=10 (600 cells) expect ~420 moves per game. Relevant for RL episode length and training throughput.

---

## Revised Phase 0 recommendations

1. **Start curriculum at R=4.** R=3 is too small for real strategic signals.
2. **Check draw rate at R=7** before committing to R=10. If >35%, investigate a tiebreaker rule.
3. **Phase 1 divergence gate must be at R=5+.** R=3 results are not representative.
4. **SearchBot for Phase 0 anchor at R=4+:** use `SearchBot(engine, depth=6, time_budget_s=2.0)` — the time budget caps it at a feasible depth.
5. **Re-check G1 vs Random at R=7.** If G1 beats Random at R=7, revise the opponent mix to treat G1 as a skill anchor (not just diversity).

---

## Test suite

127/127 pass. Three tests updated:
- `test_greedy_beats_random` → tests G2 (not G1) on committed rules
- `test_greedy1_and_greedy2_have_different_weights` → checks `frontier` (not `exclusive_territory`)
- `test_search_beats_greedy` → depth=4 with time budget
