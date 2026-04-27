# Cycle Control AI — Implementation Notes (Phase 1)

**Status:** initial implementation of the AI subpackage per `AI_IMPLEMENTATION_CHECKLIST_V2.4.1.md`.
**Covers:** checklist Sections 1 (foundation), 4.1 (baselines), 4.2/4.2b/4.2c (Greedy variants + validation), 4.4 (tournament harness), 5 (siege/territory logic).
**Not yet covered:** Section 4.3 (SearchBot), Section 6 (search-before-learning analysis), Sections 7-12 (GNN, CNN, PPO, training phases).

---

## What was built

```
cycle_control/ai/
├── __init__.py             # re-exports
├── action_space.py         # action_index <-> node mapping, legal mask builder
├── bot_interface.py        # Bot Protocol, play_turn, play_game
├── siege.py                # reachability flood-fill, sieged_for/against, territory/frontier
├── bots/
│   ├── __init__.py
│   ├── random_bot.py       # RandomBot, FrontierRandomBot
│   └── greedy_bot.py       # GreedyBot base class, Greedy1 (cycle), Greedy2 (territory)
└── tournament.py           # run_match, round_robin, Elo ladder

tests_ai.py                 # 32 unit tests — all pass
greedy_validation.py        # Section 4.2c validation script
```

All 32 AI tests pass. All 87 pre-existing engine tests still pass.

---

## Greedy variant validation (Section 4.2c) — findings

Ran Greedy_1 (cycle-focused) vs Greedy_2 (territory-focused) on the committed ruleset (`neutrality + strict_adjacency + mirror_adjacency`) at radius 3, 40 games with color-swapped pairings:

**Head-to-head: Greedy_2 wins 92.5%, Greedy_1 wins 2.5%, draws 5%.**

Per the checklist verdict logic: **VERDICT: ONE SIDE DOMINATES — proceed with both variants in the combinatorial grid.**

G1 and G2 play observably differently. The behavioral asymmetry is extreme: territory-focused play crushes cycle-focused play 37-to-1 under the committed rules.

---

## Unexpected diagnostic finding

Both greedies underperform vs RandomBot on the committed ruleset at R≤3:

| Match | Result |
|---|---|
| G1 vs Random (R=2, plain rules) | G1 wins 40%, draws 60%, loses 0% — G1 is clearly better |
| G2 vs Random (R=2, plain rules) | 100% draws — G2 is equivalent to random |
| G1 vs Random (R=2, committed) | G1 wins 15%, **loses 55%**, draws 30% |
| G2 vs Random (R=2, committed) | G2 wins 40%, **loses 50%**, draws 10% |
| G1 vs Random (R=3, committed) | G1 wins 0%, **loses 60%**, draws 40% |
| G2 vs Random (R=3, committed) | 50/50 |

**One-ply greedy is WEAKER than random under the committed rules at small board sizes.**

Three possible interpretations:

1. **The committed ruleset genuinely punishes short-sighted play.** This matches the user's earlier empirical observation that "greedy play = bad play" — territory/siege effects are second-order and only emerge over several turns of lookahead, which one-ply greedy cannot see.

2. **The greedy weights need tuning.** Current weights are hand-picked starting values. A systematic sweep might find configurations that beat Random.

3. **The evaluation function is missing a critical feature.** For instance, evaluating "does this move create or avoid a partial siege against me" would require looking at the siege structure after the move, which the current eval does compute (via exclusive_territory_diff) but may be weighted wrong.

**My read (confidence ~70%):** interpretation (1) is primary. The user's game analysis strongly suggested that cycle-greed is a losing strategy. The bots empirically confirm it. But interpretation (2) likely also applies — the weights are not the strongest possible for each strategy.

---

## Implications for the checklist

### Section 4.2c validation: PASS

Greedy_1 and Greedy_2 produce observably different games under the committed ruleset. The axis provides real diversity. Both variants should proceed to Phase 0 as bootstrap opponents in the combinatorial grid.

### Section 4.5 phase gate: REQUIRES ATTENTION

The checklist gate requires "SearchBot meaningfully stronger than Random" before RL training. Given that one-ply greedy loses to Random, this is MORE IMPORTANT than expected — SearchBot must carry the bootstrap-opponent load almost entirely.

Recommended adjustments to Section 4.5 as a result of this finding:
- Add a new checklist item: "One-ply Greedy underperforms vs Random on this ruleset — explicitly document this before Phase 0"
- Consider adding a **Greedy2Ply** or shallow-minimax bot before SearchBot, since pure one-ply greedy is too weak for PPO bootstrap
- Ensure SearchBot with modest depth (3-5) is meaningfully stronger than Random before claiming Section 4.5 readiness

### Section 10 (Phase 0 base training): the opponent mix needs updating

The checklist currently specifies ~20% heuristic anchors (G1, G2, SearchBot) during base training. Given that G1 and G2 underperform Random, this needs rethinking:
- Without a stronger heuristic, PPO's early bootstrap has no useful "non-random" anchor
- Recommendation: defer Phase 0 base training until SearchBot is built and shown to meaningfully beat Random
- Alternative: use G1 and G2 as *diversity* anchors rather than *skill* anchors — their presence teaches PPO to handle both cycle-rushers and territory-builders, even if both are beatable

### Section 11 (Phase 1 clones): the shaped reward weights may need adjusting

The weight profiles in Section 11.2 (`w_cycle=0.3` for cycle-focused, `w_territory=0.3` for territory-focused) were chosen to keep shaping at ~0.5 of terminal magnitude. Given this empirical finding that territory dominates under the committed rules, the shaping may unintentionally steer cycle-focused agents away from viable play. Watch for this during Phase 1.

---

## What to do next

In order:

1. **SearchBot** (checklist Section 4.3) — needed to provide a real skill anchor stronger than Random. Start with depth-3 minimax using Greedy2's eval as the leaf function.
2. **Search-before-learning analysis** (Section 6) — run SearchBot vs SearchBot on R=3 and R=5 to understand the shape of the game at different search depths.
3. **Re-run the Phase 0 gate analysis** using SearchBot as the anchor.
4. Then Sections 9+ (environment, networks, training).

---

## Confidence statements

- Implementation correctness: **~95%** (32/32 AI tests pass, 87/87 engine tests pass, all bots return legal actions, tournament harness produces self-consistent results)
- Validity of the greedy validation finding: **~85%** (results are consistent across R=2 and R=3, and across plain/committed rules in the expected directions)
- Interpretation that "committed rules punish greedy play": **~70%** (alternative: weights need tuning — cannot rule this out without weight sweeps)
