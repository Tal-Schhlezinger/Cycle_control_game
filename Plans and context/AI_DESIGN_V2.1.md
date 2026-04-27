# Cycle Control — AI Design V2.1

**Status:** design, pre-implementation.  
**Purpose:** define the AI research and implementation plan for evaluating whether the committed Cycle Control ruleset produces a strategically good two-player game under competent play.

---

## Changelog: V2 → V2.1

V2.1 makes no structural changes to V2. It makes the following targeted corrections:

1. **Section 0** — updated to reflect V2.1 lineage.
2. **Section 3.2** — PBT bullet clarified: "strictly optional and late" → correct framing with conditional escalation path.
3. **Section 7.2 priority note** — added explicit priority ordering within required engine upgrades (A+B before C-E).
4. **Section 13.3** — primary change: reward-shaping diversity restored as a conditional Phase 3 escalation option. V2 ruled it out; V2.1 rules it in as a late conditional, after hyperparameter-only PBT has been tried.
5. **Section 16** — minor risk additions: reward-shaping diversity risk and search-depth sensitivity.
6. **Section 17** — Phase F updated to match the corrected Phase 3 description.

---

## 0. Why this V2.1 exists

V2.1 is a correction pass on V2, not a replacement.

V2 was a strong structural improvement over V1 in every major area:
- search-before-learning
- GNN-first architecture
- removing siege from mandatory observation
- board size curriculum
- cleaner agent/game-metrics separation
- better file structure

V2 had one substantive overcorrection:

**Section 13.3 ruled out reward-shaping diversity as a PBT mechanism.** The reasoning was sound in one direction — shaping diversity should not be the default Phase 3 mechanism. But V2 went further and said "avoid making reward-shaping diversity the main training mechanism at first," which is correct, followed by leaving no path for it later, which is wrong. If standard hyperparameter-PBT is tried and the population still collapses to a single strategic style, reward-shaping diversity is the natural next escalation. V2 closed that door without reason.

V2.1 reopens it as a conditional, correctly positioned after standard PBT has been attempted.

Everything else in V2 is kept as-is.

---

## 1. Core research goal

Determine whether the committed ruleset produces a strategically good two-player game under strong play.

### 1.1 Primary research questions

1. Does the committed ruleset support balanced outcomes under competent play?
2. Does it support more than one viable strategic style?
3. Do stronger agents make the game look richer or more degenerate?
4. Which AI architecture is a better fit for this game:
   - graph-native (GNN)
   - grid-like baseline (CNN)

### 1.2 Non-goals

- building a superhuman agent for its own sake
- maximizing benchmark score at any cost
- using RL as a substitute for understanding the game
- training across many rule variants at once before one variant is understood

---

## 2. Committed ruleset

Train and evaluate on one committed configuration:

```text
mirror_adjacency      = True
strict_adjacency_rule = True
neutrality_rule       = True
partial_credit_k      = 0
```

### 2.1 Why commit to one ruleset

This preserves a good idea from V1.

Reasons:
1. Not all rule combinations are likely playable or interesting.
2. Learned value functions are ruleset-specific.
3. Early multi-variant training muddies the research question.
4. The chosen ruleset is already the strongest candidate based on current reasoning.

### 2.2 But do not commit to one board size initially

V1 leaned too quickly toward radius 10 as the training target.

V2 uses a size curriculum:

- `R=4` or `R=5` for debugging and early search analysis
- `R=6` or `R=7` for first learning runs
- `R=10` only after the pipeline is proven

---

## 3. High-level philosophy shift from V1

### 3.1 V1 emphasis
- define RL env early
- use CNN-style observation as the default
- include siege/interior channels directly
- proceed from Greedy -> PPO -> optional PBT

### 3.2 V2.1 emphasis
- optimize engine for AI first
- build search baselines before neural learning
- treat GNN as a serious main candidate, not an afterthought
- treat CNN as a baseline comparison model
- use heterogeneous opponent pools later for robustness
- keep PBT optional and late; use hyperparameter variation first, reward-shaping diversity only if the population still collapses afterward
- separate **game evaluation** from **agent training**

This is the central structural difference between V1 and V2.

---

## 4. What V1 got right and is kept in V2

The following V1 decisions remain valid and are kept.

### 4.1 Unified bot interface

Keep a single bot protocol usable by:
- Random bots
- Greedy bots
- Search bots
- PPO agents
- future MCTS / AlphaZero-style agents

Recommended interface:

```python
class Bot(Protocol):
    def choose_action(
        self,
        state: GameState,
        legal_mask: np.ndarray,
        color: Player,
    ) -> int:
        ...

    def reset(self, seed: int | None = None) -> None:
        ...
```

### 4.2 Single placement per action call

Keep V1's decomposition.

Do **not** use full-turn actions of the form `(node1, node2_or_pass)` as the main action representation.  
That would inflate the action space unnecessarily.

The engine or environment should continue to drive the multi-placement turn loop.

### 4.3 Dedicated pass action

Keep a dedicated pass index in the action space.

### 4.4 Action masking

Keep invalid action masking.  
This remains necessary because the legal action set is a small subset of the full node set for much of the game.

### 4.5 Greedy baseline before RL

Keep this requirement.

### 4.6 Snapshot opponent pool

Keep the idea of training against:
- recent self snapshots
- older self snapshots
- fixed heuristic/search anchors

This remains correct and important.

---

## 5. What was missing from V1 and is added in V2

These were the major structural omissions in V1.

### 5.1 Engine-for-AI optimization phase

V1 treated the engine mostly as already-good-enough.  
That is false for search and large-scale training.

V2 adds a dedicated engine-preparation phase:
- frontier-based legality
- cached neighbor counts
- cheap simulation state
- state hashing
- symmetry transforms

### 5.2 Search-before-learning stage

V1 went from heuristic baseline to PPO too quickly.

V2 adds a mandatory search stage:
- shallow search / beam / MCTS-like baseline
- small-board analysis before RL conclusions

### 5.3 Explicit GNN path

V1 structurally centered CNN-style observations.  
V2 explicitly treats GNN as a first-class model family.

### 5.4 Cleaner distinction between agent metrics and game metrics

V1 mixed:
- "the model got stronger"
with
- "the game is balanced"

V2 separates:
- agent benchmarking
- research conclusions about the rules

### 5.5 Heterogeneous-opponent training

V1 had solo self-play and optional PBT.  
V2 adds a clearer intermediate phase:
- CNN and GNN do not directly imitate each other
- they appear in each other's opponent pools
- this is treated as opponent diversity, not magical cross-teaching

---

## 6. What V1 had that V2 must preserve explicitly

These V1 elements should not be lost.

### 6.1 Clear engine-driven multi-placement turn loop

V1 described this well. Keep the idea even if the exact code changes.

The bot should choose one action at a time.  
The environment should decide whether another placement remains.

### 6.2 Rule commitment rationale

V1 made a good case for training on one ruleset instead of all combinations.  
V2 preserves that logic.

### 6.3 Elo / ladder evaluation

Keep a ladder or league rating system.  
Do not rely only on training reward curves.

### 6.4 Open research risks section

V1 explicitly listed uncertainties.  
V2 should keep an uncertainty/risk section rather than pretending confidence is perfect.

### 6.5 Implementation order discipline

V1 had good instincts here.  
V2 keeps staged implementation gates, but changes their order.

---

## 7. Engine work required before serious AI

This is the first major implementation block.

### 7.1 Goals

The engine currently appears suitable for correctness, testing, and gameplay.  
It is not yet optimized enough for:
- heavy bot rollouts
- large search trees
- efficient self-play

### 7.2 Required upgrades

**Priority note:** A and B are immediate wins — cheap to implement, high impact on every downstream stage. Implement them first. C is needed before serious RL training. D and E can follow once C is in place; they are needed for transposition tables, augmentation, and search caching but not for first learning runs.

#### A. Incremental legal frontier

Do not compute legal moves by scanning every node every time.

Maintain, per player:
- whether the player has any stones
- frontier candidate empty nodes adjacent to own stones
- opening-state special handling

Under strict adjacency, legal moves after the opening are fundamentally frontier-based.

#### B. Cached local neighbor counts

Maintain per-node counts:
- black neighbors
- white neighbors

Update them incrementally after each move.

This speeds up:
- neutrality checks
- legality checks
- heuristic evaluation

#### C. Cheap simulation state

Separate:
- UI / replay / history state
from
- lightweight bot simulation state

Simulation state should clone fast and avoid unnecessary baggage.

#### D. State hashing

Add deterministic state hashing over:
- board occupancy
- player-to-move
- turn phase
- any pass / truncation state that matters to legality or terminal logic

Use cases:
- transposition tables
- repeated-state detection
- dataset deduplication
- search caching
- benchmark position sets

#### E. Symmetry utilities

Implement board symmetries once:
- rotations
- reflections
- node-index transforms
- policy-target transforms
- optional canonicalization helpers

This is needed for:
- augmentation
- analysis
- opening grouping
- debugging equivalent states

### 7.3 Optional but desirable engine additions

- transposition table support
- incremental feature extraction hooks
- canonical opening position library
- performance benchmark scripts

---

## 8. Bot stack and implementation order

Bots should be built in increasing sophistication, with each stage producing usable evaluation tools.

### 8.1 Stage 0 — trivial sanity bots

Implement:
- `RandomBot`
- `LegalFirstBot`
- `FrontierRandomBot`

Purpose:
- validate legality
- validate action encoding
- validate turn-phase handling
- provide smoke-test opponents

### 8.2 Stage 1 — heuristic bots

#### 8.2.1 GreedyBot

One-ply greedy evaluation over legal actions.

Suggested evaluation components:
- own cycle/scoring nodes minus opponent cycle/scoring nodes
- territory / future reach estimate
- largest connected component
- mobility / future legal options

GreedyBot remains important for:
- early benchmark anchor
- sanity-check training target
- easily interpretable behavior

#### 8.2.2 SearchBot

Before RL, add a stronger classical baseline:
- beam search, or
- depth-limited search, or
- lightweight MCTS with heuristic leaf eval

This is a structural addition absent from V1 and is mandatory in V2.

Reason:
If the game already reveals rich tactics under shallow search, that is evidence about the game itself and may reduce the need for early RL complexity.

---

## 9. Territory / siege heuristics in V2

V1 made siege/interior analysis central very early.  
That needs correction.

### 9.1 Keep territory logic as a heuristic tool

A territory / siege estimator is still valuable for:
- GreedyBot eval
- SearchBot eval
- post-game analysis
- research metrics

### 9.2 Do not make siege channels a required first observation feature

First learned models should **not** depend on handcrafted siege planes.

Reason:
- this bakes in one theory of the game too early
- this risks making the model imitate the heuristic rather than discover structure
- this weakens the interpretability of conclusions about what the model really learned

### 9.3 Recommended use of siege logic

Use it in this order:
1. heuristic/search evaluation
2. offline analysis
3. optional ablation features later
4. optional auxiliary prediction target later

Do not make it a default mandatory input in the first neural models.

---

## 10. Search before learning

This is one of the main V2 changes.

### 10.1 Small-board exact / near-exact study

Before drawing conclusions from RL, study small boards where search is more feasible.

Recommended sizes:
- `R=3`
- `R=4`
- maybe `R=5` with budgeted search

Measure:
- first-player win rate
- draw rate
- final score margins
- opening sensitivity
- forced-win / forced-draw signs
- parity/pathology indicators

If the rules are structurally broken on small boards, large-board RL may merely hide the problem.

### 10.2 Mid-board search study

For `R=5` to `R=7`, run many games:
- Greedy vs Greedy
- Search vs Search
- Greedy vs Search
- randomized openings if useful

Goal:
- observe whether multiple strategic styles emerge
- test whether siege pressure is real or overstated
- construct benchmark position sets for later learned-agent evaluation

---

## 11. Learning architectures

V2 supports two learner families, but not simultaneously at the start.

### 11.1 Learner A — GNN (main serious candidate)

This is the architecture V2 considers the more natural fit.

#### 11.1.1 Why GNN is the serious model

The game is fundamentally a graph:
- cycle structure
- connectivity
- local-global relational effects
- changing tactical importance of edges and neighborhoods

A graph-native model matches the object more naturally than a padded image representation.

#### 11.1.2 Node features

Suggested node-level features:
- own stone
- opponent stone
- empty
- on-board / valid node flag
- local own-neighbor count
- local opponent-neighbor count
- normalized move number or game-progress signal
- turn-phase indicators
- active-player indicator

Additional optional features may be added later only if clearly justified.

#### 11.1.3 Edge set

Use the real game adjacency graph, including:
- standard adjacency
- mirror adjacency if enabled by the committed ruleset

#### 11.1.4 Outputs

- one policy logit per node
- one extra pass logit
- one scalar value head

### 11.2 Learner B — CNN baseline

Keep a CNN path, but treat it as a comparison baseline rather than the presumed best model.

#### 11.2.1 Purpose
- test whether graph inductive bias matters materially
- measure how far local pattern learning alone gets
- provide a simpler baseline for tooling and debugging

#### 11.2.2 Input
Use a padded board representation with channels such as:
- own stones
- opponent stones
- empty/on-board
- optional neighbor count channels
- phase / active-player broadcast planes

But do **not** hardcode siege/interior channels as required inputs in the first version.

### 11.3 Architecture order

Implement in this order:
1. GNN first
2. CNN second

Reason:
- GNN is the architecture more likely to match the game
- CNN remains useful as an empirical comparison and debugging baseline

---

## 12. Environment and action space design

This section preserves the strongest structural choices from V1.

### 12.1 Action space

Let `N = topology.node_count()`.

Use:

```text
0 .. N-1   = place at node i
N          = pass
```

Policy output size = `N + 1`.

### 12.2 Action selection granularity

One action corresponds to one placement or pass.

The environment/engine drives the sequence of placements in a turn.

### 12.3 Masked legality

The environment must produce:
- observation
- legal action mask

Masked policies must not sample illegal moves.

### 12.4 Environment style

AEC-style turn stepping remains reasonable because the game is sequential and multi-phase.  
However, V2 is less dogmatic about framework than V1.

The important constraint is not PettingZoo specifically.  
The important constraint is:
- correct multi-placement stepping
- clean legal masks
- easy self-play orchestration
- evaluation reproducibility

PettingZoo AEC remains acceptable if implementation friction stays low.

---

## 13. Training phases

### 13.1 Phase 1 — single-architecture training

Train one learner family at a time.

Recommended order:
1. GNN
2. CNN

Each learner uses:
- masked action space
- self-play
- snapshot pool
- fixed heuristic/search anchors in the opponent mix

#### Suggested opponent sampling
- 50% recent self snapshots
- 30% older self snapshots
- 20% fixed heuristic/search baseline

This preserves the strongest V1 self-play idea while grounding it more firmly.

### 13.2 Phase 2 — heterogeneous opponent pool

Only after both architectures independently work.

Then train with cross-architecture opponent diversity:
- GNN sometimes faces CNN snapshots
- CNN sometimes faces GNN snapshots
- both still face same-family snapshots
- both still face search/heuristic anchors

This is the disciplined version of “teach each other.”

Interpretation:
- not direct policy imitation
- not magical mutual instruction
- simply richer opponent diversity

### 13.3 Phase 3 — optional population methods

Only consider PBT or more complex league mechanics if:
- both families plateau
- strategy diversity remains narrow
- standard mixed-opponent training fails to expose multiple viable styles

#### Step 1: hyperparameter-only PBT

If Phase 3 is triggered, start with hyperparameter variation only:
- learning rate, entropy coefficient, clip range, GAE lambda
- keep the true objective fixed (terminal win/loss)
- keep reward shaping weights fixed across the population

This is the minimum-complexity population method and should be tried first.

#### Step 2: reward-shaping diversity (conditional escalation)

If hyperparameter-only PBT runs and the population still collapses to a single strategic family, add reward-shaping diversity as the next escalation:

- give different agents different internal shaping weights (e.g. how much they internally value current cycles vs. territory vs. mobility)
- fitness remains terminal win rate only — shaping affects value-head training targets, not the final objective
- over generations, exploit/explore cycles will converge toward shapings that produce winning strategies
- surviving shaping distributions tell us which internal priorities correspond to winning play under this ruleset

This is a meaningful research output for Cycle Control specifically, not just a training trick. The game has multiple plausible strategies (cycle-rush, siege-build, mixed) and no prior theory to rank them. If the population naturally selects for one shaping profile, that is evidence about the game. If multiple profiles survive, that is evidence of genuine strategic diversity.

**Why this was ruled out in V2 and restored in V2.1:**  
V2 correctly said "do not make shaping diversity the default." V2.1 agrees. But V2 then implicitly closed the door entirely by leaving no path for it. The correct position is: hyperparameter PBT first, shaping diversity only if the population collapses and a richer diversity mechanism is needed. That is the conditional path V2.1 restores.

Do not proceed to Step 2 unless Step 1 has been genuinely attempted and found insufficient.

---

## 14. RL algorithm choices

### 14.1 First practical learner

Masked PPO is acceptable as the first learning baseline.

Why:
- straightforward
- good enough to test whether the game can be learned at all
- works well with a fixed action head and masks

### 14.2 But PPO is not sacred

This is important.

For deterministic, perfect-information board games, stronger long-term directions may include:
- search-guided training
- AlphaZero-style self-play
- MCTS + network hybrids

V2 therefore treats PPO as:
- useful first learner
- not necessarily final learner

### 14.3 Search-guided future path

If learning works but plateaus or looks strategically shallow, the next serious upgrade path is:
- network-guided search
- or search-improved targets

This path is more natural for the domain than immediately escalating to PBT.

---

## 15. Evaluation framework

V2 strengthens this area.

### 15.1 Agent metrics

Track:
- Elo / league rating
- head-to-head win rates
- win rates by color
- score margin
- game length
- draw rate
- benchmark-position move quality

### 15.2 Game research metrics

The main outputs about the rules should be:

1. **First-player edge**
   - Is Black close to 50-55%, or much stronger/weaker?

2. **Draw frequency**
   - Are draws rare, moderate, or dominant?

3. **Score margin profile**
   - Are games knife-edge, moderate, or runaway?

4. **Strategic diversity**
   - Do multiple strategic styles remain viable under strong play?

5. **Cross-opponent robustness**
   - Does one dominant style crush everything, or do different approaches remain competitive?

6. **Trajectory diversity**
   - Are strong self-play games varied, or repetitive and degenerate?

### 15.3 Fixed benchmark pool

Maintain a persistent benchmark pool including:
- RandomBot
- LegalFirstBot / FrontierRandomBot
- GreedyBot
- SearchBot
- frozen GNN checkpoints
- frozen CNN checkpoints
- curated benchmark positions

Without this, training curves become hard to trust.

---

## 16. Open research risks and uncertainties

V1 was right to include a risks section.  
V2 keeps one, but updates the contents.

### 16.1 Rule risks
- mirror adjacency may create hidden structural pathologies
- first-player edge may become too large
- small-board behavior may differ materially from large-board behavior

### 16.2 Heuristic risks
- territory/siege intuition may be overstated
- Greedy/SearchBot eval may bias conclusions if treated as truth

### 16.3 Model risks
- GNN may underperform in practice despite being conceptually better-matched
- CNN may overfit local motifs and miss deeper graph structure
- either architecture may look stronger only because opponent diversity is poor
- if reward-shaping diversity is used in Phase 3, surviving shaping weights reflect the game's strategic structure only if the training signal is clean — noisy self-play may produce misleading shaping convergence

### 16.4 Training risks
- co-adaptation in self-play
- policy collapse into one strategic family
- learning a weird internal metagame that does not reflect the real game
- masking bugs producing misleading results
- small-board curriculum not transferring as well as expected
- search-depth sensitivity: SearchBot eval quality depends heavily on depth budget; conclusions about the game from shallow search may not generalize to deeper search or RL

### 16.5 Research-validity risk
A stronger agent is not automatically evidence of a better game.  
The real question is what strong-vs-strong play looks like.

---

## 17. Implementation roadmap

This is the implementation order V2 recommends.

### Phase A — engine and AI foundations
1. frontier-based legality
2. cached neighbor counts
3. cheap simulation state
4. state hashing
5. symmetry utilities

### Phase B — classical bots and analysis
6. RandomBot
7. LegalFirstBot / FrontierRandomBot
8. GreedyBot
9. SearchBot
10. tournament harness
11. small-board analysis scripts
12. benchmark position generation

### Phase C — first learner
13. GNN feature encoder
14. GNN policy/value model
15. masked training loop
16. self-play with snapshot pool
17. evaluation pipeline

### Phase D — second learner
18. CNN baseline encoder/model
19. same training/eval harness
20. GNN vs CNN comparison

### Phase E — heterogeneous training
21. mixed-opponent pool using both architectures
22. cross-architecture robustness evaluation

### Phase F — optional advanced methods
23. PBT Step 1: hyperparameter variation only, if justified by Phase E plateau
24. PBT Step 2: reward-shaping diversity, only if Step 1 runs and population still collapses to single style
25. search-guided learning / AlphaZero-style upgrade if justified by learning plateau

Do not skip the evaluation gates between phases.

---

## 18. File structure comparison: V1 vs V2

This section explicitly compares the design **as file/module structure**, not only as content.

### 18.1 V1 proposed structure

V1 centered the project around the RL stack early:

```text
cycle_control/ai/
    __init__.py
    action_space.py
    bot_interface.py
    bots/
        random_bot.py
        greedy_bot.py
    siege.py
    features.py
    env.py
    network.py
    ppo_selfplay.py
    elo.py
    tournament.py
    cli.py
```

### 18.2 Problems with the V1 structure

1. It was too flat.
2. It implicitly prioritized RL over search and analysis.
3. It gave no first-class home to:
   - symmetry
   - hashing
   - simulation state
   - search bots
   - rule-analysis scripts
   - multi-model support
4. It centered the observation/network around a single model path too early.
5. It treated siege logic like a near-core dependency rather than a heuristic subsystem.

### 18.3 V2 structure

V2 reorganizes the project by responsibility:

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
```

### 18.4 Why the V2 structure is better

#### A. `core/`
V1 lacked a clean place for cross-cutting AI primitives.  
V2 gives a proper home to:
- action encoding
- bot interface
- symmetry logic
- hashing
- simulation state

These are neither “just env” nor “just model” concerns.

#### B. `heuristics/`
V1 had GreedyBot but structurally underplayed heuristic/search tooling.  
V2 elevates it.

This makes room for:
- evaluation functions
- territory estimators
- greedy bots
- search bots

#### C. `training/`
V1 mixed env/network/training concerns too aggressively.  
V2 separates orchestration from models.

#### D. `models/`
V1 implicitly assumed one main network path.  
V2 explicitly supports model-family comparison.

#### E. `analysis/`
This is a major structural addition absent from V1.

Research about the game itself needs a first-class place for:
- small-board search studies
- opening analysis
- rule-balance metrics
- curated benchmark states

#### F. `cli/`
V1 had a CLI mention but not enough structure.  
V2 makes it explicit and task-oriented.

### 18.5 What V1 had structurally that V2 initially risked losing

These are now explicitly preserved:
- action-space helper module
- unified bot interface
- tournament/evaluation utility
- clear env module
- ladder/league evaluation role

### 18.6 Optional future V2 additions if the project grows

If the project becomes large enough, add:

```text
    data/
        selfplay_dataset.py
        replay_buffer.py

    search/
        mcts.py
        transposition.py
        rollout_policy.py

    experiments/
        ablations.py
        curriculum.py
        pbt.py
```

Do **not** add these until they are justified.

---

## 19. Final V2 recommendation in one sentence

Build search baselines first, treat GNN as the serious main learner, keep CNN as a comparison baseline, and use mixed-opponent training later for robustness rather than starting with direct mutual teaching or early PBT.

---

## 20. Practical acceptance criteria before calling V2 “working”

The V2 pipeline is not considered ready until all of the following hold:

1. Engine-side AI utilities exist:
   - frontier legality
   - cached neighbor counts
   - simulation state
   - symmetry support
   - hashing

2. Classical baselines exist:
   - Random
   - Greedy
   - Search

3. Small-board analysis exists and produces interpretable results.

4. At least one neural learner beats GreedyBot reliably.

5. Cross-evaluation exists:
   - learned agent vs search baseline
   - GNN vs CNN
   - mixed-opponent robustness tests

6. Research outputs about the game are reported separately from training curves.

---

## 21. Summary

V2.1 is a correction pass on V2, not a replacement.

It keeps everything V2 got right:
- unified bot interface
- single-placement action decomposition
- masked action space
- Greedy baseline
- snapshot opponent pool
- staged implementation mindset
- search-before-learning
- GNN-first architecture
- siege features demoted from mandatory observation
- board size curriculum
- cleaner agent/game-metrics separation
- better file structure

It keeps the V2 corrections to V1:
- CNN as comparison baseline, not default serious model
- RL-after-search, not RL-first
- PBT late and conditional
- no mandatory siege observation channels in first models

The one V2 overcorrection V2.1 fixes:
- **Reward-shaping diversity** is restored as a conditional Phase 3 escalation step, positioned correctly after hyperparameter-only PBT has been tried and found insufficient. V2 closed this door without reason. V2.1 reopens it as a conditional, not a default.

That is the only substantive change. Everything else in V2 stands.
