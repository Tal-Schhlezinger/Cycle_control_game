# Cycle Control — AI Implementation Checklist V2.3

**Status:** operational checklist derived from `AI_DESIGN_V2.3`  
**Purpose:** convert the V2.1.1 design into a concrete implementation and validation queue, with explicit phase gates and a built-in coverage audit against the design.

---

## 0. How to use this file

This file is **not** the design document.  
It is the execution document.

Use it to answer:
- what must be built,
- in what order,
- what blocks the next phase,
- what is optional,
- what must be measured before declaring success.

### 0.1 Checkbox meaning

- `[ ]` = not started
- `[~]` = in progress / partially complete
- `[x]` = completed and verified
- `BLOCKER` = must be done before the next phase
- `OPTIONAL` = useful, but not required for initial compliance

### 0.2 Rules for using the checklist

1. Do not skip phase gates.
2. Do not treat training progress as proof that the game is good.
3. Do not add major complexity early just because it sounds advanced.
4. If a phase fails its gate, fix the phase instead of moving forward anyway.
5. Keep this file aligned with `AI_DESIGN_V2.3`.


## 0.3 Practical tooling recommendations from the polishing pass

These are not random preferences. They reflect what current tooling most directly supports for this kind of project.

### Recommended default stack
- Use **PettingZoo AEC** if the main public environment is multi-agent and sequential.
- Use a **single-agent training wrapper** only where it genuinely simplifies the learner side.
- If PPO is the first RL baseline, prefer a stack with **explicit invalid-action masking support** rather than inventing custom masking late.

### Recommended concrete choices
- For environment validation, run:
  - PettingZoo `api_test` on the multi-agent environment
  - Gymnasium / SB3 environment checks on any single-agent wrapper
- For masked PPO, prefer:
  - `sb3-contrib` `MaskablePPO`, or
  - a custom implementation only if there is a strong reason
- For GNN implementation, prefer:
  - **PyTorch Geometric (PyG)** as the default graph library
- For later search-guided learning experiments, treat:
  - **OpenSpiel** as a reference implementation source for MCTS / AlphaZero-style patterns, even if the final project does not adopt OpenSpiel directly

### Anti-drift rule
- [ ] Record the exact chosen stack in the repo before large implementation starts
- [ ] Record package versions used for the first stable run

---

## 1. Scope and non-negotiables

### 1.1 Committed ruleset
- [ ] Confirm the committed ruleset is the default AI research target:
  - `mirror_adjacency = True`
  - `strict_adjacency_rule = True`
  - `neutrality_rule = True`
  - `partial_credit_k = 0`

### 1.2 Board-size curriculum
- [ ] Support `R=4` or `R=5` for debugging / small-board analysis
- [ ] Support `R=6` or `R=7` for first learning runs
- [ ] Support `R=10` only after earlier gates are passed

### 1.3 Core non-goals
- [ ] Do **not** train across many rule variants before one variant is understood
- [ ] Do **not** treat RL as a substitute for game analysis
- [ ] Do **not** optimize only for agent strength while ignoring game-quality evidence
- [ ] Do **not** treat CNN as automatically the main serious model

---

## 2. Shared AI foundation

### 2.1 Unified bot protocol
- [ ] Define one shared bot interface usable by:
  - Random bots
  - Heuristic bots
  - Search bots
  - Learned agents
  - Future MCTS / AlphaZero-style agents
- [ ] The interface accepts:
  - game state
  - legal action mask
  - player color / perspective
- [ ] The interface returns:
  - one action index only
- [ ] Include `reset()` or equivalent per-game reset hook

### 2.2 Action-space conventions
- [ ] Define a deterministic node ordering
- [ ] Encode action space as:
  - `0..N-1 = place at node i`
  - `N = pass`
- [ ] Keep a dedicated pass action
- [ ] Ensure the policy head always outputs size `N+1`

### 2.3 Turn handling
- [ ] Keep one action = one placement or pass
- [ ] Keep engine-driven multi-placement turn handling
- [ ] Do **not** move to a full-turn `(node1, node2_or_pass)` action design
- [ ] Verify opening / normal / truncated turn phases are handled correctly

### 2.4 Action masking
- [ ] Build legal action mask generation
- [ ] Guarantee masks align with engine legality
- [ ] Verify masked policies never intentionally sample illegal actions
- [ ] Define illegal-action handling as a safety net, not as a normal path

### 2.5 Determinism and reproducibility
- [ ] Seed bot randomness
- [ ] Seed search randomness if used
- [ ] Seed learning runs
- [ ] Record package versions for reproducible baselines
- [ ] Record the exact environment / training stack choice
- [ ] Ensure evaluation can be rerun reproducibly

---

## 3. Engine preparation for AI

This phase is a **BLOCKER** for serious search or large-scale training.

### 3.1 Incremental legal frontier
- [ ] Maintain per-player “has any stones” info
- [ ] Maintain per-player frontier candidates
- [ ] Handle opening-stage legality separately where needed
- [ ] Stop relying on full-board legal scans as the primary search/training path
- [ ] Benchmark old vs new move generation

### 3.2 Cached local neighbor counts
- [ ] Maintain black-neighbor counts per node
- [ ] Maintain white-neighbor counts per node
- [ ] Update counts incrementally after each move
- [ ] Use these counts in legality checks
- [ ] Use these counts in heuristic features where appropriate
- [ ] Benchmark old vs new local legality / eval speed

### 3.3 Cheap simulation state
- [ ] Define a lightweight simulation state separate from UI/history state
- [ ] Ensure simulation state clones fast
- [ ] Ensure simulation state contains all legality/terminal information needed by bots
- [ ] Verify search/self-play never depends on replay-only baggage

### 3.4 State hashing
- [ ] Hash board occupancy
- [ ] Hash player-to-move
- [ ] Hash turn phase
- [ ] Hash any pass/truncation state that affects legality or terminal logic
- [ ] Verify hash stability
- [ ] Verify identical states hash identically
- [ ] Verify meaningfully different states do not collide in normal testing

### 3.5 Symmetry utilities
- [ ] Implement rotations
- [ ] Implement reflections
- [ ] Implement node-index transforms
- [ ] Implement policy-target transforms
- [ ] Add tests for symmetry correctness
- [ ] Add helper(s) for augmentation and analysis use

### 3.6 Optional engine extras
- [ ] OPTIONAL: transposition table support
- [ ] OPTIONAL: incremental feature extraction hooks
- [ ] OPTIONAL: canonical opening position library
- [ ] OPTIONAL: performance benchmark scripts beyond the basics

### 3.7 Phase gate — engine ready for AI
Do **not** proceed until all are true:

- [ ] Frontier legality is implemented and tested
- [ ] Neighbor counts are cached and tested
- [ ] Simulation state exists and clones cheaply
- [ ] Hashing exists and is validated
- [ ] Symmetry utilities exist and are validated
- [ ] Performance is measurably better for search/self-play than the naive path

---

## 4. Baseline bots

### 4.1 Sanity bots
- [ ] Implement `RandomBot`
- [ ] Implement `LegalFirstBot`
- [ ] Implement `FrontierRandomBot`
- [ ] Verify all sanity bots always return legal actions
- [ ] Verify they respect pass handling and turn phases

### 4.2 Greedy_1 — cycle and structure focused
- [ ] Implement one-ply Greedy_1
- [ ] Evaluation: own cycle/scoring nodes minus opponent (heavy weight)
- [ ] Evaluation: largest connected component size (moderate weight)
- [ ] Evaluation: mobility / legal options count (light weight)
- [ ] Evaluation: territory/siege estimate (zero or minimal weight)
- [ ] Add deterministic tiebreaking
- [ ] Document weights explicitly
- [ ] Add unit / behavior tests on hand-crafted positions

### 4.2b Greedy_2 — territory and frontier focused
- [ ] Implement one-ply Greedy_2
- [ ] Evaluation: territory / siege estimate — own reachable cells minus opponent (heavy weight)
- [ ] Evaluation: frontier size — own expandable border cells (moderate weight)
- [ ] Evaluation: opponent mobility penalty (negative weight on opponent legal moves)
- [ ] Evaluation: cycle/scoring nodes (light tiebreaker weight only)
- [ ] Add deterministic tiebreaking
- [ ] Document weights explicitly
- [ ] Add unit / behavior tests on hand-crafted positions

### 4.2c Greedy variant validation — required before population grid is committed
- [ ] Run Greedy_1 vs Greedy_2 head-to-head (≥100 games, color-swapped, fixed seed)
- [ ] Inspect game trajectories: opening patterns, mid-game shape, score distributions
- [ ] Confirm Greedy_1 and Greedy_2 play observably differently
- [ ] If games are indistinguishable: collapse to one variant, remove the greedy axis, reduce population from 16 to 8 agents and document the decision
- [ ] If games are observably different: proceed with both variants in the combinatorial grid

### 4.3 SearchBot
- [ ] Choose one initial search baseline:
  - beam search, or
  - depth-limited search, or
  - lightweight MCTS with heuristic leaves
- [ ] Prefer the simplest search that is strong enough to serve as a real benchmark before moving to more elaborate search
- [ ] Implement SearchBot
- [ ] Ensure SearchBot uses the same action encoding / legality conventions as all other bots
- [ ] Ensure SearchBot works on the lightweight simulation state
- [ ] Add search-specific tests / regression positions
- [ ] Store a small fixed set of benchmark positions for SearchBot regression

### 4.4 Tournament harness
- [ ] Implement head-to-head match runner
- [ ] Support multiple seeds / repeated games
- [ ] Support color swapping
- [ ] Support fixed openings if needed
- [ ] Output win rate, score margin, draw rate, game length

### 4.5 Phase gate — classical baselines ready
Do **not** proceed until all are true:

- [ ] Random / LegalFirst / FrontierRandom exist and are stable
- [ ] Greedy_1 and Greedy_2 both exist and are documented
- [ ] Greedy_1 vs Greedy_2 validation complete with documented result
- [ ] SearchBot exists and is usable
- [ ] Tournament harness works
- [ ] SearchBot is meaningfully stronger than Random
- [ ] Greedy and Search behavior is interpretable enough to debug

---

## 5. Territory / siege logic

### 5.1 Heuristic territory module
- [ ] Implement a territory / siege / future-reach estimator if useful
- [ ] Keep it inside heuristic/search tooling first
- [ ] Ensure it is not silently treated as “ground truth”

### 5.2 Correct role of territory logic
- [ ] Use it for:
  - Greedy evaluation
  - Search evaluation
  - offline analysis
  - rule metrics
- [ ] Do **not** make siege/interior planes mandatory first-model inputs
- [ ] Do **not** make the learned agent’s success depend on handcrafted territory features from day 1

### 5.3 Optional later uses
- [ ] OPTIONAL: ablation with territory-derived input planes
- [ ] OPTIONAL: auxiliary prediction head for territory concepts

### 5.4 Phase gate — territory logic in the right place
- [ ] Territory logic exists only as heuristic / analysis support in the first iteration
- [ ] First learned models do not require handcrafted siege planes

---

## 6. Search-before-learning analysis

This phase is a **BLOCKER** before strong claims about the game.

### 6.1 Small-board study
- [ ] Run exact or near-exact analysis on `R=3`
- [ ] Run exact or near-exact analysis on `R=4` if feasible
- [ ] Consider budgeted search on `R=5`
- [ ] Measure:
  - first-player win rate
  - draw rate
  - score margins
  - opening sensitivity
  - forced-win / forced-draw signals
  - parity / pathology indicators
- [ ] Store results in a reproducible format

### 6.2 Mid-board study
- [ ] Run `Greedy vs Greedy` on `R=5..7`
- [ ] Run `Search vs Search` on `R=5..7`
- [ ] Run `Greedy vs Search` on `R=5..7`
- [ ] Use randomized openings where useful
- [ ] Observe whether different strategic styles appear
- [ ] Test whether territory/siege intuition looks real or overstated
- [ ] Produce benchmark positions for later agent evaluation

### 6.3 Analysis scripts and outputs
- [ ] Implement small-board analysis scripts
- [ ] Implement opening-stat scripts where needed
- [ ] Store benchmark positions
- [ ] Store summary metrics in a reusable format

### 6.4 Phase gate — search analysis complete
- [ ] Small-board study completed
- [ ] Mid-board search study completed
- [ ] Benchmark positions produced
- [ ] There is at least a preliminary answer to whether the rules look promising or suspicious before RL

---

## 7. Model family A — GNN

This is the **main serious candidate**.

### 7.1 Graph representation
- [ ] Build graph representation over the real board topology
- [ ] Include standard adjacency
- [ ] Include mirror adjacency under the committed ruleset
- [ ] Verify graph construction correctness

### 7.2 Node features
- [ ] own stone
- [ ] opponent stone
- [ ] empty
- [ ] on-board / valid-node flag
- [ ] local own-neighbor count
- [ ] local opponent-neighbor count
- [ ] move-number or game-progress signal
- [ ] turn-phase indicators
- [ ] active-player indicator

### 7.3 Policy/value outputs
- [ ] one node logit per board node
- [ ] one pass logit
- [ ] one scalar value head

### 7.4 GNN implementation quality checks
- [ ] Use PyTorch Geometric (PyG) by default unless there is a strong reason not to
- [ ] Represent states cleanly as graph data objects
- [ ] Use proper batched graph loading rather than ad-hoc manual batching
- [ ] Forward pass works on batched states
- [ ] Output dimensions match `N+1` policy + scalar value
- [ ] Mask integration works
- [ ] Symmetry augmentation path is compatible if used
- [ ] Model can train without handcrafted siege planes

### 7.5 Phase gate — GNN model ready
- [ ] Graph construction validated
- [ ] Node features validated
- [ ] Outputs validated
- [ ] Training input/output path is stable

---

## 8. Model family B — CNN baseline

This is the **comparison baseline**, not the presumed main model.

### 8.1 Board tensor representation
- [ ] Build padded board tensor representation
- [ ] Include at minimum:
  - own stones
  - opponent stones
  - empty/on-board indicator
  - optional neighbor-count channels
  - phase / active-player channels

### 8.2 CNN outputs
- [ ] one action logit per node plus pass
- [ ] one scalar value head

### 8.3 CNN design constraints
- [ ] Do **not** require handcrafted siege/interior channels in the first version
- [ ] Keep the same training/eval harness as the GNN where possible
- [ ] Make CNN a fair comparison, not a different experimental universe

### 8.4 Phase gate — CNN model ready
- [ ] Tensor representation validated
- [ ] Output head validated
- [ ] Compatible with the same mask / self-play / evaluation framework

---

## 9. Environment and training harness

### 9.1 Environment design
- [ ] Environment emits observation + legal mask
- [ ] Environment supports multi-placement turns correctly
- [ ] Environment supports pass action correctly
- [ ] Environment is reproducible under seeding
- [ ] Environment is easy to wrap for self-play
- [ ] Run PettingZoo `api_test` on the multi-agent environment
- [ ] Run Gymnasium / SB3 environment checks on any single-agent wrapper used for training

### 9.2 Framework choice
- [ ] Choose concrete environment / RL stack
- [ ] Confirm that masked action handling actually works in the chosen stack
- [ ] Document any framework-specific limitations
- [ ] If using `sb3-contrib` `MaskablePPO`, use:
  - `MaskableEvalCallback`
  - mask-aware evaluation helpers
- [ ] If using subprocess vectorized environments with maskable PPO, ensure the mask function is implemented in the environment path that SB3 actually expects
- [ ] If **not** using `sb3-contrib` for masked PPO, document why and where masking is implemented instead

### 9.3 Self-play support
- [ ] Build self-play wrapper / orchestration
- [ ] Support snapshot pools
- [ ] Support fixed heuristic/search anchors
- [ ] Support color randomization
- [ ] Support curriculum by board size

### 9.4 Logging / tracking
- [ ] Log training config
- [ ] Log seeds
- [ ] Log checkpoint IDs
- [ ] Log opponent sampling composition
- [ ] Log evaluation results separately from training rewards

### 9.5 Phase gate — training harness ready
- [ ] Env is correct
- [ ] Masking is correct
- [ ] Self-play works
- [ ] Checkpointing works
- [ ] Logging is sufficient to reproduce runs

---

## 9b. Combinatorial population — ceiling vs. starter pool

The full 16-agent grid is the **long-term ceiling**, not the implementation target.

### 9b.1 Actual implementation target: 4-agent starter pool
- [ ] Define the 4 starter agents: `CNN_cycle`, `CNN_territory`, `GNN_cycle`, `GNN_territory`
- [ ] Confirm architecture axis: CNN and GNN both available
- [ ] Confirm style axis: cycle-focused and territory-focused profiles defined (see Section 10.2)

### 9b.2 Expansion path (committed but not yet built)
- [ ] Phase 2 expansion: add `CNN_mid`, `GNN_mid` → 6 agents — only after Phase 1 diversity confirmed
- [ ] Phase 4 ceiling: full 16-agent grid — optional, only if 6-agent pool is stable and more coverage is needed

### 9b.3 Full grid reference table (ceiling, not current target)

| Agent | Architecture | Greedy bootstrap | Reward shaping |
|---|---|---|---|
| A00–A07 | CNN | G1 or G2 | R0–R3 |
| A08–A15 | GNN | G1 or G2 | R0–R3 |

Do not build this table until Phases 0–2 are validated.

---

## 10. Phase 0 — shared base training

Train one neutral base per architecture before any style variants.

### 10.1 CNN_base
- [ ] Train CNN_base on terminal win/loss only (no reward shaping)
- [ ] Opponent pool: ~50% own snapshots, ~30% older own, ~20% heuristic anchors (G1 + G2 + SearchBot)
- [ ] Monitor win rate vs Greedy
- [ ] Monitor policy entropy — must stay meaningfully above minimum before split
- [ ] Assess game trajectory variation — base should still show style variation across opponents
- [ ] Save checkpoint at split point: win rate 65–80% vs Greedy, not converged
- [ ] Save CNN_base as a permanent frozen benchmark anchor

### 10.2 GNN_base
- [ ] Same procedure as CNN_base
- [ ] Save GNN_base as a permanent frozen benchmark anchor

### 10.3 Phase 0 gate — BLOCKER for Phase 1
- [ ] Both bases train stably
- [ ] Both beat Greedy reliably (≥65% win rate)
- [ ] Both assessed as "competent but not converged"
- [ ] Both checkpoints saved and documented
- [ ] Split timing decision recorded with rationale

---

## 11. Phase 1 — clone and fine-tune → 4 agents

### 11.1 Cloning procedure
- [ ] Clone CNN_base → CNN_cycle and CNN_territory (exact weight copies, different agent IDs)
- [ ] Clone GNN_base → GNN_cycle and GNN_territory
- [ ] Resume PPO training from cloned checkpoint (no re-initialization)

### 11.2 Reward shaping profiles

**Cycle-focused (CNN_cycle, GNN_cycle)**
- [ ] `w_terminal = 1.0` (primary)
- [ ] `w_cycle` = high (own cycle/scoring nodes minus opponent)
- [ ] `w_component` = moderate
- [ ] `w_territory` = low
- [ ] `w_mobility` = low
- [ ] Document exact weights

**Territory-focused (CNN_territory, GNN_territory)**
- [ ] `w_terminal = 1.0` (primary)
- [ ] `w_territory` = high (reachable area/siege estimate minus opponent)
- [ ] `w_frontier` = moderate
- [ ] `w_mobility` = moderate (opponent penalty)
- [ ] `w_cycle` = low
- [ ] Document exact weights

### 11.3 Opponent pool during fine-tuning per agent
- [ ] ~40% own recent snapshots
- [ ] ~30% cross-style snapshots (other 3 agents in the starter pool)
- [ ] ~10% own older snapshots
- [ ] ~20% heuristic anchors + both base models (CNN_base, GNN_base)

### 11.4 Phase 1 evaluation
- [ ] All 4 agents train stably from base checkpoint
- [ ] All 4 beat Greedy reliably
- [ ] Cross-style win rates tracked (each vs each)
- [ ] Game trajectories inspected for visual style differences
- [ ] Base models used as neutral reference in evaluation

### 11.5 Divergence validation gate — BLOCKER for Phase 2
- [ ] CNN_cycle and CNN_territory play observably differently
- [ ] GNN_cycle and GNN_territory play observably differently
- [ ] CNN and GNN families show architecture-level differences
- [ ] If any two agents are indistinguishable: diagnose before expanding, document finding

---

## 12. Phase 2 — midpoint expansion → 6 agents (conditional)

Only after Phase 1 divergence gate passes.

### 12.1 Midpoint cloning
- [ ] Clone CNN_base → CNN_mid
- [ ] Clone GNN_base → GNN_mid
- [ ] Fine-tune on balanced profile: `w_cycle ≈ w_territory`, both medium; `w_mobility` low/moderate

### 12.2 Midpoint validation
- [ ] CNN_mid and GNN_mid train stably
- [ ] Both play observably differently from the cycle and territory extremes
- [ ] If midpoint agents are indistinguishable from either extreme: document and skip

### 12.3 Phase 2 gate
- [ ] Midpoint divergence confirmed, or midpoint skip documented with reason

---

## 12b. Phase 3 — cross-architecture mixing

After 4 or 6 agents are stable and diverse.

- [ ] Add cross-architecture opponents to each agent's pool
- [ ] GNN variants face CNN snapshots and vice versa
- [ ] All agents still face own-family snapshots and heuristic anchors
- [ ] Evaluate whether cross-mixing improves robustness or destabilizes training
- [ ] Document result

---

## 12c. Phase 4 — optional advanced methods (ceiling)

Only if Phases 0–3 are complete and diversity is still insufficient.

### 12c.1 Optional PBT Step 1 — hyperparameter variation
- [ ] OPTIONAL: trigger only if population plateaus despite Phase 1–3 diversity
- [ ] OPTIONAL: hyperparameter variation only (lr, entropy, clip, GAE lambda)
- [ ] OPTIONAL: keep shaping weights consistent within each agent
- [ ] OPTIONAL: keep terminal objective fixed
- [ ] OPTIONAL: document why PBT is needed before implementing

### 12c.2 Optional PBT Step 2 — reward-shaping mutation
- [ ] OPTIONAL: trigger only if Step 1 runs and population still collapses
- [ ] OPTIONAL: diagnose first — collapse likely indicates training bug or degenerate game dynamics
- [ ] OPTIONAL: allow small shaping weight perturbations only if no other explanation found

### 12c.3 Optional full 16-agent grid
- [ ] OPTIONAL: only if 6-agent pool is validated and more coverage is needed
- [ ] OPTIONAL: add G1/G2 bootstrap axis and additional R variants
- [ ] OPTIONAL: document population size decision

---

## 13. Evaluation and reporting

### 13.1 Agent metrics
- [ ] Elo / league rating
- [ ] head-to-head win rate
- [ ] win rate by color
- [ ] score margin
- [ ] game length
- [ ] draw rate
- [ ] benchmark-position move quality

### 13.2 Game research metrics
- [ ] first-player edge
- [ ] draw frequency
- [ ] score-margin profile
- [ ] strategic diversity
- [ ] cross-opponent robustness
- [ ] trajectory diversity
- [ ] shaping-profile survival: which reward-shaping agents develop strong play — this is a direct game-design signal about which strategic priorities correspond to winning under the committed ruleset

### 13.3 Benchmark pool
- [ ] Maintain RandomBot
- [ ] Maintain LegalFirstBot / FrontierRandomBot
- [ ] Maintain GreedyBot
- [ ] Maintain SearchBot
- [ ] Maintain frozen GNN checkpoints (one per shaping profile)
- [ ] Maintain frozen CNN checkpoints (one per shaping profile)
- [ ] Maintain curated benchmark positions

### 13.4 Reporting discipline
- [ ] Separate “agent improved” from “game is good”
- [ ] Report failures and suspicious results honestly
- [ ] Include confidence / uncertainty notes where appropriate
- [ ] Where practical, report repeated-run variability rather than a single lucky number
- [ ] Store results in a reproducible format

### 13.5 Phase gate — evaluation quality acceptable
- [ ] Benchmark pool exists
- [ ] Agent metrics exist
- [ ] Game metrics exist
- [ ] Reports separate model strength from game-quality conclusions

---

## 14. Risks and sanity checks

### 14.1 Rule risks
- [ ] Monitor first-player edge
- [ ] Monitor draw degeneracy
- [ ] Monitor small-board vs large-board mismatch
- [ ] Monitor mirror-adjacency pathologies

### 14.2 Heuristic risks
- [ ] Do not treat Greedy/Search evaluation as truth
- [ ] Do not assume territory intuition is correct without evidence

### 14.3 Model risks
- [ ] Check whether GNN underperforms despite theory
- [ ] Check whether CNN overfits local motifs
- [ ] Check whether either family only looks good due to poor opponent diversity
- [ ] If reward-shaping diversity is used: check whether surviving shaping profiles reflect real game structure or are artifacts of noisy self-play

### 14.4 Training risks
- [ ] Check for self-play co-adaptation
- [ ] Check for policy collapse
- [ ] Check for weird internal metagames
- [ ] Check for masking bugs
- [ ] Check whether curriculum transfers
- [ ] Check for search-depth sensitivity: SearchBot conclusions may not generalize to deeper search or stronger RL — verify conclusions hold across search budget levels

### 14.5 Research-validity sanity check
- [ ] Ask explicitly: does stronger play make the game look better, worse, or merely stranger?
- [ ] Do not answer the game-design question using only one training curve

---

## 15. File/module targets

Use this as the practical target structure, adapting only if there is a clear reason.

```text
cycle_control/ai/
    __init__.py

    core/
        action_space.py
        bot_interface.py
        symmetry.py
        hashing.py
        simulation_state.py

    heuristics/
        eval.py
        territory.py
        greedy_bot.py
        search_bot.py

    training/
        env.py
        opponent_pool.py
        selfplay.py
        league.py
        evaluate.py

    models/
        common.py
        gnn.py
        cnn.py

    analysis/
        small_board_search.py
        opening_stats.py
        benchmark_positions.py
        rule_metrics.py

    cli/
        train_gnn.py
        train_cnn.py
        run_league.py
        eval_model.py

tests_ai/
    test_action_space.py
    test_symmetry.py
    test_hashing.py
    test_env.py
    test_greedy_bot.py
    test_search_bot.py
    test_gnn_smoke.py
    test_cnn_smoke.py
```

### 15.1 File-structure checklist
- [ ] `core/` exists or equivalent responsibilities are clearly assigned
- [ ] `heuristics/` exists or equivalent responsibilities are clearly assigned
- [ ] `training/` exists or equivalent responsibilities are clearly assigned
- [ ] `models/` supports both GNN and CNN
- [ ] `analysis/` exists for rule research scripts
- [ ] `cli/` or equivalent scripts exist for repeatable runs
- [ ] `tests_ai/` or equivalent targeted AI test coverage exists

---

## 16. Final acceptance criteria

Do **not** call the V2 AI plan “working” until all of the following are true:

### 16.1 Foundations
- [ ] committed ruleset is fixed
- [ ] board-size curriculum exists
- [ ] unified bot interface exists
- [ ] action encoding and masking are stable

### 16.2 Engine readiness
- [ ] frontier legality exists
- [ ] cached neighbor counts exist
- [ ] simulation state exists
- [ ] hashing exists
- [ ] symmetry support exists

### 16.3 Classical baselines
- [ ] Random exists
- [ ] Greedy exists
- [ ] Search exists
- [ ] tournament harness exists

### 16.4 Search-before-learning evidence
- [ ] small-board analysis complete
- [ ] mid-board search analysis complete
- [ ] benchmark positions generated

### 16.5 Learned agents
- [ ] CNN_base and GNN_base train stably and are saved as frozen benchmarks
- [ ] All 4 starter agents (CNN_cycle, CNN_territory, GNN_cycle, GNN_territory) train stably from base checkpoints
- [ ] All 4 beat Greedy reliably
- [ ] At least one agent is meaningfully competitive with Search or explains why not
- [ ] Cycle-focused and territory-focused agents play observably differently within each architecture family
- [ ] CNN and GNN families show architecture-level differences

### 16.6 Cross-architecture evaluation
- [ ] GNN vs CNN comparison exists
- [ ] mixed-opponent training has been tested or deliberately deferred with reason

### 16.7 Evaluation quality
- [ ] agent metrics are tracked
- [ ] game metrics are tracked
- [ ] benchmark pool exists
- [ ] reports separate game conclusions from model conclusions

---

## 17. Coverage audit against AI_DESIGN_V2.3

This section confirms the checklist covers the V2.1.1 design.

### 17.1 Coverage map

- [x] **V2.1.1 Section 1 — Core research goal**  
  Covered by Sections 1, 13, 14, 16 of this checklist.

- [x] **V2.1.1 Section 2 — Committed ruleset and board-size curriculum**  
  Covered by Sections 1 and 16.

- [x] **V2.1.1 Section 3 — Philosophy shift**  
  Reflected throughout, especially Sections 3, 4, 6, 7, 8, 10, 11, 12, 13.

- [x] **V2.1.1 Sections 4–6 — Preserved V1/V2 ideas**  
  Covered by Sections 2, 9, 10, 13, 15, 16.

- [x] **V2.1.1 Section 7 — Engine work before serious AI**  
  Covered by Section 3 and acceptance checks in Section 16.

- [x] **V2.1.1 Section 8 — Bot stack and implementation order**  
  Covered by Section 4 and phase dependencies.

- [x] **V2.1.1 Section 9 — Territory/siege heuristics role**  
  Covered by Section 5.

- [x] **V2.1.1 Section 10 — Search before learning**  
  Covered by Section 6.

- [x] **V2.1.1 Section 11 — Learning architectures (GNN / CNN)**  
  Covered by Sections 7 and 8.

- [x] **V2.1.1 Section 12 — Environment and action space**  
  Covered by Sections 2 and 9.

- [x] **V2.3 Section 13 — Training phases (shared-base-then-split → 4-agent starter pool)**  
  Covered by Sections 9b, 10, 11, 12, 12b, 12c. Base training is in Section 10. Clone+fine-tune → 4 agents in Section 11. Midpoint expansion in Section 12. Cross-architecture mixing in Section 12b. Optional PBT and full grid ceiling in Section 12c.

- [x] **V2.1.1 Section 14 — RL algorithm choices**  
  Covered by Sections 10 and 12.

- [x] **V2.1.1 Section 15 — Evaluation framework**  
  Covered by Section 13, including shaping-profile survival metric in 13.2.

- [x] **V2.1.1 Section 16 — Risks and uncertainties**  
  Covered by Section 14, including the two new V2.1.1 risks (shaping convergence validity, search-depth sensitivity).

- [x] **V2.1.1 Section 17 — Implementation roadmap**  
  Covered by the structure and ordering of Sections 3 through 16.

- [x] **V2.1.1 Section 18 — File structure**  
  Covered by Section 15.

- [x] **V2.1.1 Sections 19–21 — Recommendations and summary**  
  Reflected by the overall execution ordering and Section 18.

### 17.2 Coverage audit result
- [x] Every major V2.1.1 section has a corresponding checklist block.
- [x] The checklist covers build tasks, research-validation tasks, and risk controls.
- [x] The V2.1.1 structural change — reward-shaping diversity in Phase 1 — is reflected in Sections 10.2, 10.3, 10.6, 11.1, 12.1, 12.2, 13.2, 13.3, 14.3, 16.5.
- [x] The V2.2 structural addition — combinatorial population design — is reflected in Sections 4.2, 4.2b, 4.2c, 4.5, 9b, 10.2, 10.3, 11.1, 16.5.
- [x] The two V2.1.1 new risks are reflected in Section 14.3 and 14.4.
- [x] The checklist preserves what V2 got right, and adds what V2 and V2.1 were still missing.

### 17.3 Final freeze condition for this checklist
- [x] Covers design, implementation, evaluation, and risk control.
- [x] Covers both content and file-structure requirements from V2.1.1.
- [x] Includes explicit phase gates.
- [x] Includes final acceptance criteria.
- [x] Includes a self-audit against `AI_DESIGN_V2.3`.

---

## 18. Short version

If everything here feels too large, the irreducible core is:

1. prepare the engine for AI,
2. build Random, Greedy_1, Greedy_2, Search — validate G1 ≠ G2 behaviorally,
3. analyze small and mid boards before trusting RL,
4. train CNN_base and GNN_base on neutral reward — save as frozen anchors when competent but not converged,
5. clone each base into cycle-focused and territory-focused variants — fine-tune each on shaped reward,
6. validate that the 4 agents play observably differently,
7. optionally expand to 6 agents by adding midpoint variants from the same bases,
8. then cross-architecture mixing, then optional PBT/full grid if needed,
9. evaluate the **game** separately from the **agent**.

That is the minimum disciplined path consistent with V2.3.
