# Cycle Control — AI Implementation Checklist V2.2

**Status:** operational checklist derived from `AI_DESIGN_V2.2`  
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
5. Keep this file aligned with `AI_DESIGN_V2.2`.


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

## 9b. Combinatorial population definition

This section must be completed before Phase 1 training starts. It defines the population that will train throughout Phases 1 and 2.

### 9b.1 Axis definitions
- [ ] Architecture axis: `{CNN, GNN}` — confirmed two implementations available
- [ ] Greedy bootstrap axis: `{Greedy_1, Greedy_2}` — validated per Section 4.2c
- [ ] Reward shaping axis: `{R0, R1, R2, R3}` — defined per Section 10.2
- [ ] Hyperparameter axis: random sample per agent (NOT a discrete grid axis)

### 9b.2 Full 16-agent grid (when all axes validated)

| Agent | Architecture | Greedy bootstrap | Reward shaping |
|---|---|---|---|
| A00 | CNN | G1 | R0 (cycle-focused) |
| A01 | CNN | G1 | R1 (territory-focused) |
| A02 | CNN | G1 | R2 (balanced) |
| A03 | CNN | G1 | R3 (mobility-focused) |
| A04 | CNN | G2 | R0 |
| A05 | CNN | G2 | R1 |
| A06 | CNN | G2 | R2 |
| A07 | CNN | G2 | R3 |
| A08 | GNN | G1 | R0 |
| A09 | GNN | G1 | R1 |
| A10 | GNN | G1 | R2 |
| A11 | GNN | G1 | R3 |
| A12 | GNN | G2 | R0 |
| A13 | GNN | G2 | R1 |
| A14 | GNN | G2 | R2 |
| A15 | GNN | G2 | R3 |

### 9b.3 Axis collapse rules
- [ ] If Greedy validation (4.2c) fails: remove greedy axis → 8 agents
- [ ] If any reward profile is indistinguishable from another: merge → fewer R variants
- [ ] If CNN fails to train stably: run GNN-only → 8 agents, document
- [ ] Record final committed population size and composition before Phase 1 starts

### 9b.4 Compute-scaled population (4-GPU minimum)
- [ ] For 4 GPUs: select 4 agents maximizing axis coverage, e.g. `(CNN,G1,R0)`, `(CNN,G2,R2)`, `(GNN,G1,R3)`, `(GNN,G2,R1)`
- [ ] Do not run 4 agents with identical architecture
- [ ] Record the 4-agent subset chosen and the reason

### 9b.5 Population composition gate
Do **not** start Phase 1 training until:
- [ ] Population size and composition are documented
- [ ] Each agent's (architecture, bootstrap, shaping, hp_sample) is recorded
- [ ] Axis collapse decisions (if any) are documented with reasons

---

## 10. Phase 1 learning — one architecture at a time, with reward-shaping diversity

### 10.1 Training order
- [ ] Train GNN first
- [ ] Train CNN second

### 10.2 Reward-shaping diversity — built in from Phase 1

This is a structural feature of Phase 1, not an optional later escalation.

Each agent in the population trains on a **consistent but agent-specific** internal reward mix throughout its lifetime. There is no weight copying, no tournament selection, no perturbation between agents. Each agent trains independently on its own objective.

- [ ] Define 3–4 agents per architecture family, each with a fixed shaping vector `(w_cycle, w_territory, w_mobility)`
- [ ] Suggested starting population:
  - `agent_0`: cycle-focused — `w_cycle=3.0, w_territory=0.5, w_mobility=0.05`
  - `agent_1`: territory-focused — `w_cycle=1.0, w_territory=2.5, w_mobility=0.1`
  - `agent_2`: balanced — `w_cycle=2.0, w_territory=1.5, w_mobility=0.1`
  - `agent_3`: mobility-focused — `w_cycle=1.5, w_territory=1.0, w_mobility=0.5`
- [ ] Terminal win/loss is always the primary training signal and the fitness metric
- [ ] Shaping weights are fixed per agent — they are not mutated or perturbed
- [ ] Document each agent's shaping vector and keep it stable across its full training run

### 10.3 Phase 1 opponent mix per agent

Each agent's opponent pool is drawn from the full population, not just its own snapshots:
- [ ] ~40% recent snapshots from the same agent
- [ ] ~30% snapshots from other agents in the population (cross-shaping exposure)
- [ ] ~10% older own snapshots (anti-forgetting)
- [ ] ~20% fixed heuristic/search anchors

### 10.4 PPO-first baseline
- [ ] Implement masked PPO baseline
- [ ] Verify PPO can learn legally and stably
- [ ] Do not overclaim PPO as the final architecture

### 10.5 Evaluation during training
- [ ] Evaluate each agent against GreedyBot periodically
- [ ] Evaluate each agent against SearchBot periodically
- [ ] Evaluate agents against each other (cross-shaping win rates)
- [ ] Record Elo / league rating per agent
- [ ] Record game metrics, not just reward
- [ ] Track which shaping profiles are producing stronger agents

### 10.6 Phase gate — single-family learning successful
For each architecture family:

- [ ] All agents train stably
- [ ] At least one agent beats Random reliably
- [ ] At least one agent beats Greedy reliably
- [ ] At least one agent is meaningfully competitive with SearchBot or reveals why not
- [ ] Agents with different shaping profiles play observably differently
- [ ] Outputs are reproducible enough to analyze
- [ ] There is evidence agents learned real strategy, not only legality or trivial local hacks

---

## 11. Phase 2 learning — heterogeneous opponent pool

This is the disciplined version of "teach each other."

### 11.1 Preconditions
- [ ] GNN independently works (all shaping variants)
- [ ] CNN independently works (all shaping variants)
- [ ] Agents with different shaping profiles play observably differently
- [ ] Fixed heuristic/search anchors still exist

### 11.2 Mixed-opponent training
- [ ] Train GNN agents partly against CNN agent snapshots
- [ ] Train CNN agents partly against GNN agent snapshots
- [ ] Keep same-family snapshots in the pool
- [ ] Keep cross-shaping snapshots from all agents in the pool
- [ ] Keep heuristic/search anchors in the pool

### 11.3 Explicit anti-confusion constraints
- [ ] Do **not** use direct policy imitation between CNN and GNN as the default method
- [ ] Do **not** remove stable benchmark opponents
- [ ] Do **not** rely on only one changing opponent family

### 11.4 Heterogeneous training evaluation
- [ ] Compare pre-mixing vs post-mixing robustness
- [ ] Check whether strategy diversity increases
- [ ] Check whether exploitability decreases
- [ ] Check whether one architecture only learned to beat its own family before mixing

### 11.5 Phase gate — mixed-opponent phase justified
- [ ] Mixed-opponent training adds measurable value, or
- [ ] It is dropped with a documented reason

---

## 12. Optional advanced methods

Only enter this phase if earlier phases work and there is a reason.

### 12.1 Optional PBT — Step 1: hyperparameter variation
- [ ] OPTIONAL: trigger only if both families plateau and strategy diversity is insufficient despite Phase 1 reward-shaping diversity
- [ ] OPTIONAL: use hyperparameter variation only — learning rate, entropy coefficient, clip range, GAE lambda
- [ ] OPTIONAL: use exploit/explore cycles: copy weights from stronger agents, perturb hyperparameters
- [ ] OPTIONAL: keep reward shaping weights consistent within each agent — do not perturb them in Step 1
- [ ] OPTIONAL: keep the true terminal win/loss objective fixed
- [ ] OPTIONAL: document why PBT is needed before implementing it

### 12.2 Optional PBT — Step 2: reward-shaping mutation (late diagnostic only)
- [ ] OPTIONAL: trigger only if Step 1 has run and population still collapses to a single strategic family despite Phase 1 shaping diversity
- [ ] OPTIONAL: if triggered, diagnose first — collapse after Phase 1 diversity likely indicates training instability, masking bugs, or degenerate game dynamics
- [ ] OPTIONAL: only allow small shaping weight perturbations within the PBT exploit/explore cycle if the above diagnosis finds no other explanation

### 12.3 Optional search-guided learning
- [ ] OPTIONAL: evaluate network-guided search
- [ ] OPTIONAL: evaluate search-improved targets
- [ ] OPTIONAL: use OpenSpiel as a reference source for MCTS / AlphaZero-style design patterns if useful
- [ ] OPTIONAL: consider AlphaZero-style path only after earlier baselines are solid

### 12.4 Optional ablations
- [ ] OPTIONAL: territory-plane ablation
- [ ] OPTIONAL: curriculum ablation
- [ ] OPTIONAL: opponent-pool composition ablation
- [ ] OPTIONAL: architecture ablation details beyond GNN vs CNN
- [ ] OPTIONAL: shaping-weight ablation (compare agents trained with vs without shaping diversity)

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
- [ ] GNN learner exists and trains (all shaping variants)
- [ ] CNN learner exists and trains (all shaping variants)
- [ ] at least one learner reliably beats Greedy
- [ ] at least one learner is meaningfully competitive with Search or explains why not
- [ ] agents with different shaping profiles play observably differently

### 16.6 Cross-architecture evaluation
- [ ] GNN vs CNN comparison exists
- [ ] mixed-opponent training has been tested or deliberately deferred with reason

### 16.7 Evaluation quality
- [ ] agent metrics are tracked
- [ ] game metrics are tracked
- [ ] benchmark pool exists
- [ ] reports separate game conclusions from model conclusions

---

## 17. Coverage audit against AI_DESIGN_V2.2

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

- [x] **V2.2 Section 13 — Training phases (combinatorial population + reward-shaping diversity)**  
  Covered by Sections 9b, 10, 11, 12. The combinatorial population definition is in Section 9b. Phase 1 reward-shaping diversity is in Section 10.2 and 10.3. Phase 3 two-step PBT is in Sections 12.1 and 12.2.

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
- [x] Includes a self-audit against `AI_DESIGN_V2.2`.

---

## 18. Short version

If everything here feels too large, the irreducible core is:

1. prepare the engine for AI,
2. build Random, Greedy_1, Greedy_2, Search — validate Greedy_1 ≠ Greedy_2 behaviorally,
3. analyze small and mid boards before trusting RL,
4. define the combinatorial population: `{CNN,GNN} × {G1,G2} × {R0..R3}` — collapse any axis that does not produce observable diversity,
5. train GNN family first, CNN family second, each with reward-shaping diversity built in from the start,
6. then use mixed-opponent training across both architecture families and all shaping profiles,
7. only then consider PBT or search-guided upgrades,
8. evaluate the **game** separately from the **agent**.

That is the minimum disciplined path consistent with V2.2.
