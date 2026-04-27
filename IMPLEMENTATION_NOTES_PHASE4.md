# Cycle Control AI — Implementation Notes (Phase 4)

**Status:** critical bug fix, corrected tournament results.
**Covers:** pass-selection bug in GreedyBot and SearchBot, corrected R=4 findings.
**Continues from:** IMPLEMENTATION_NOTES_PHASE3.md.

---

## Critical bug fix: bots were passing mid-game

### Root cause

Both `GreedyBot` and `SearchBot` were treating the pass action as having
evaluation score = 0 (state unchanged). On an empty or mid-game board
where no piece structure exists yet, placing a stone often scores negative
(it costs frontier/mobility relative to the opponent in the short term),
so 0 > -N made pass look like the best move. Both bots were passing
whenever the board was in a symmetric or unfavorable local state.

This affected:
- The opening (both bots pass → 2 moves → 0-0 draw)
- Mid-game transitions between structures (one bot passes → opponent
  gets a free turn → cascading advantage)
- Any position where the eval function found all placements negative

### Fix

`GreedyBot.choose_action`: filter pass out of candidates whenever legal
placements exist. Pass is only considered when there are literally no
legal placements available.

`SearchBot._search_root` and `SearchBot._negamax`: same filter.

This is also theoretically correct: passing when moves exist hands the
opponent a free turn. No rational strategy should voluntarily pass when
placements are legal.

### Impact on prior results

Every result prior to Phase 4 that involved Greedy_1 or SearchBot is
suspect. Greedy_2 was less affected (its frontier/mobility terms meant
placements often scored positively), but was still affected in some
mid-game positions.

**Prior results that must be discarded:**
- Phase 1 finding: "Greedy_1 < Random" — was a bug artifact
- Phase 2 finding: "SearchBot d=2 barely beats Greedy_2" — bug artifact
- Phase 3 multi-radius tournament (R=2..5) — all data is suspect
- All draw-rate measurements from Phase 3 — draws were mostly pass-induced

---

## Corrected R=4 tournament results (20 games/pair, committed ruleset)

Run by user on their machine after the Phase 4 fix. This is the first
fully clean data.

### Win-rate matrix

|  | Random | Greedy_1 | Greedy_2 | Search_d2 |
|---|---|---|---|---|
| Random | — | 0% | 0% | 0% |
| Greedy_1 | 90% | — | 0% | 0% |
| Greedy_2 | 100% | 100% | — | 5% |
| Search_d2 | 100% | 100% | 95% | — |

### Elo ratings

| Bot | Elo |
|---|---|
| Search_d2 | 1667 |
| Greedy_2 | 1290 |
| Greedy_1 | 983 |
| Random | 861 |

### Key metrics

- **Draw rate: 0.0%** across all 240 games. Games are fully decisive at R=4.
- **Avg game length: ~110 moves** (115% of 96 cells). Games end via
  consecutive passes after the board is effectively partitioned by the
  neutrality + strict adjacency rules, not by filling every cell.
- **Elo order: exactly as expected** — Search_d2 > Greedy_2 > Greedy_1 > Random.
- **Search_d2 Elo gap over Greedy_2: +377** → "game has real tactical depth."

---

## Corrected research findings

### Finding 1: Phase 1/2 "Greedy_1 < Random" was entirely a bug

Greedy_1 now wins 90% vs Random at R=4. Cycle-focused play is a
viable strategy. The earlier finding that cycle-greed was worse than
random was caused by G1 passing when it shouldn't — random was "winning"
by default because G1 handed it free turns.

**Implication for Phase 0:** both Greedy_1 and Greedy_2 are valid skill
anchors. The previously recommended downgrade of G1 to "diversity anchor
only" was based on bad data. G1 is a genuine baseline opponent.

### Finding 2: Zero draws at R=4

The Phase 3 draw-rate trend (2% → 25% as radius grows) was dominated
by the pass bug. Clean R=4 data shows 0% draws. The concerned
recommendation to "check draw rate at R=7 before committing to R=10"
may still be worth doing, but the alarm level is much lower.

Games being decisive at R=4 is a positive signal for RL: terminal
reward is clean (+1/-1), every game produces a clear training signal.

### Finding 3: Game length is 110-118 moves at R=4

~115% board coverage means strict adjacency + neutrality causes games
to end slightly after apparent board fill (both players pass once they
run out of reachable cells). This is structurally healthy — it means
the rules produce natural game termination without pathological endings.

At R=10 (600 cells), expect ~690 moves per game. This is long but not
unusual for abstract strategy games on large boards; Go games on 19×19
average ~200-250 moves on 361 cells (~65% coverage, much lower than
our ~115%). The higher coverage in our game is because neutrality and
strict adjacency constrain players to contiguous territories that must
be fully filled before passing.

**Implication for RL throughput:** 700 moves/game × however many games
per training iteration. Episode length is long. Use smaller boards (R=4,
R=5) during early training, curriculum to R=10.

### Finding 4: Search_d2 dominates at R=4

Search_d2 wins 95% vs Greedy_2 and 100% vs everything else. This is
stronger than expected — the +377 Elo gap suggests the game rewards
lookahead heavily at R=4. This is evidence for RL: the neural net needs
to learn something beyond one-ply heuristics, and there is genuine
strategic depth to exploit.

### Finding 5: Greedy_2 still clearly dominates Greedy_1

G2 wins 100% vs G1 at R=4. The territory-focused strategy is stronger
than the cycle-focused strategy at this board size under these rules.
Both are valid sparring partners, but they produce clearly different
game styles and outcomes.

---

## Revised Phase 0 recommendations (replacing Phase 3 version)

1. **Both G1 and G2 are valid skill anchors.** Use both in Phase 0
   opponent mix. The recommended mix:

   | Opponent | Share | Purpose |
   |---|---|---|
   | Own snapshots (recent) | ~45% | self-play |
   | Own older snapshots | ~15% | anti-forgetting |
   | Search_d2 | ~15% | primary skill anchor |
   | Greedy_2 | ~10% | secondary skill anchor |
   | Greedy_1 | ~10% | strategic diversity (cycle style) |
   | Random | ~5% | baseline sanity |

2. **Start curriculum at R=4.** Clean data confirms R=4 is the minimum
   for meaningful strategic signals.

3. **Draw rate concern is reduced.** Zero draws at R=4. Still worth
   checking at R=7 before R=10 training, but no alarm.

4. **Phase 1 divergence gate can be at R=4.** No need to push to R=5+
   since R=4 already shows clear G1/G2 divergence (G2 wins 100% vs G1).

5. **SearchBot as Phase 0 anchor:** `SearchBot(engine, depth=2,
   time_budget_s=2.0)` at R=4. The 10s/game timing means it's feasible
   as a training opponent at ~5-10% of games (not the majority).

---

## Test suite status

127/127 tests pass. No new tests added in Phase 4 (the fix was
behavioral, not structural). The existing tests cover legality of
returned actions but not the "never pass when placements exist" logic
explicitly — consider adding a dedicated test in the next pass:

```python
def test_greedy_never_passes_when_placements_exist():
    # At opening, all 96 cells legal; bot must not choose pass
    engine = _make_committed_engine(radius=4)
    state = engine.initial_state()
    mask = build_legal_mask(engine, state)
    for bot in [Greedy1(engine), Greedy2(engine)]:
        a = bot.choose_action(state, mask, Player.BLACK)
        assert a != engine.topology.node_count(), \
            f"{bot.name} chose pass when placements were available"
```

---

## Open questions for Phase 0

1. **Draw rate at R=7?** Still worth one quick check before R=10.
2. **SearchBot at R=5+?** 10s/game at R=4 with d=2. At R=5 this is
   likely 20-30s/game. Use time_budget_s=2.0 to cap it.
3. **Does the G2 > G1 dominance hold at R=7/10?** At R=4 it's 100%;
   the gap may narrow at larger boards as cycle strategies gain space.
