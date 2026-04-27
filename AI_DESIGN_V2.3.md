# Cycle Control — AI Design V2.3

**Status:** design, pre-implementation.  
**Purpose:** define the AI research and implementation plan for evaluating whether the committed Cycle Control ruleset produces a strategically good two-player game under competent play.

---

## Changelog: V2 → V2.1 → V2.1.1 → V2.2

V2.1 made the following targeted corrections to V2:

1. Section 0 updated to reflect V2.1 lineage.
2. Section 3.2 PBT bullet clarified.
3. Section 7.2 priority note added for engine upgrade ordering.
4. Section 13.3 restored reward-shaping diversity as a conditional Phase 3 escalation. V2 ruled it out; V2.1 ruled it in as a late conditional after hyperparameter-only PBT.
5. Sections 16.3, 16.4 added two missing risks.
6. Section 17 Phase F updated.

**V2.1.1 makes one additional structural change to V2.1:**

7. **Section 13 — reward-shaping diversity promoted from Phase 3 conditional to Phase 1 structural feature.**

   V2.1 positioned reward-shaping diversity as a late escalation within PBT. That was still too conservative. The correct framing: each agent in the population trains on a *consistent but agent-specific* reward mix from the very start of Phase 1. This is not a PBT mechanism — there is no weight copying, no tournament selection, no perturbation. Each agent develops genuine strategy around its own objective. The diversity comes from population composition, not from within-agent mutation. When reward-diverse agents face each other in the opponent pool, they encounter tactically different opponents, which forces broader learning than pure self-play produces. Phase 3 is updated accordingly: shaping diversity is already built in, so Phase 3 Step 1 is hyperparameter-only PBT, and Step 2 (shaping mutation) is a late diagnostic tool for collapse that should not occur given Phase 1 diversity.

---

## 0. Why this V2.3 exists

V2.3 restructures the training approach in two ways.

**First:** the 16-agent combinatorial grid from V2.2 is demoted from "target population" to "optional long-term ceiling." The actual implementation target is a **4-agent starter pool**: `{CNN, GNN} × {cycle-focused, territory-focused}`. This is enough to give real strategic pressure from different directions without turning the project into an experiment before the pipeline works.

**Second:** the 4 agents are not trained independently from scratch. Instead, V2.3 introduces a **shared-base-then-split** training structure:
1. Train one neutral base per architecture family (CNN_base, GNN_base) on terminal win/loss only — no reward shaping, neutral opponent pool
2. Once the base beats Greedy reliably but has not converged to a dominant style, clone it into two variants
3. Fine-tune each clone on its shaped reward (cycle-focused or territory-focused) continuing self-play from the base checkpoint
4. The base models become permanent neutral benchmark anchors in the tournament pool

This avoids redundant learning of basic competence, makes style divergence from a stable foundation rather than from noise, and gives a clean recovery path (re-clone from base) if a variant collapses.

The midpoint expansion (CNN_mid, GNN_mid → 6 agents total) is a Phase 2 target, only after the 4-agent pool shows visible strategic diversity. The full 16-agent grid remains as an optional ceiling if the project reaches that scale.

Everything established in V2 through V2.2 that is not mentioned here stands unchanged.

**V2.2 adds one structural extension:**

8. **Section 13 and new Section 13.0 — Combinatorial population design (16-agent ceiling).**
   Two greedy variants (G1/G2), full `{CNN,GNN} × {G1,G2} × {R0..R3}` grid as the long-term ceiling.

**V2.3 restructures the training approach:**

9. **Shared-base-then-split training structure.** Rather than training 4 agents independently from scratch, V2.3 establishes a two-phase training approach: first train one neutral base model per architecture family (CNN_base, GNN_base) on terminal reward only, then clone each base into two style variants (cycle-focused, territory-focused) and fine-tune each clone on its shaped reward. This avoids redundant learning of basic competence, makes the split from a stable foundation, and produces the base models as permanent neutral benchmark anchors.

10. **4-agent starter pool as Phase 1 target.** The full 16-agent combinatorial grid is demoted from "target population" to "optional long-term ceiling." The actual implementation target is 4 agents: `{CNN, GNN} × {cycle, territory}`, derived via the shared-base approach. Midpoint variants (CNN_mid, GNN_mid) are a Phase 2 expansion, only after the 4-agent pool shows visible strategic diversity.

11. **Section 13 restructured** around: Phase 0 (base training), Phase 1 (clone + fine-tune → 4 agents), Phase 2 (midpoint expansion → 6 agents), Phase 3 (cross-architecture mixing), Phase 4 (optional full grid / PBT).

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

#### 8.2.1 Two greedy variants (Greedy_1 and Greedy_2)

Implement **two explicitly different** greedy bots, not one canonical GreedyBot. These serve as:
- bootstrap opponents for neural agents (different priors baked in from early training)
- a diversity-validation check (they must play observably differently before the population grid is committed)
- independent benchmark anchors

**Greedy_1 — cycle and structure focused:**

Evaluation components:
- own cycle/scoring nodes minus opponent cycle/scoring nodes (heavy weight)
- largest connected component size (moderate weight)
- mobility / legal options count (light weight)
- territory/siege estimate (zero or minimal weight)

Strategy character: short-term cycle-builder. Prioritizes closing cycles quickly over spatial control.

**Greedy_2 — territory and frontier focused:**

Evaluation components:
- territory / siege estimate — reachable empty cells minus opponent-reachable empty cells (heavy weight)
- frontier size — own expandable border cells (moderate weight)
- opponent mobility penalty — negative weight on opponent's legal moves
- cycle/scoring nodes (light weight, as a secondary tiebreaker only)

Strategy character: spatial controller. Prioritizes area ownership and opponent restriction over immediate cycle scoring.

**Validation requirement before committing the population grid:**
Run Greedy_1 vs Greedy_2 head-to-head on the committed board size. If they play indistinguishably (similar opening patterns, similar mid-game positions, similar score distributions), the axis is not providing real diversity — collapse to one greedy and remove the greedy axis from the combinatorial grid. Only proceed with both variants if they produce observably different games.

#### 8.2.2 SearchBot

Before RL, add a stronger classical baseline:
- beam search, or
- depth-limited search, or
- lightweight MCTS with heuristic leaf eval

This is a structural addition absent from V1 and is mandatory.

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

### 13.0 Combinatorial population — ceiling vs. starter pool

The full combinatorial grid (`{CNN,GNN} × {G1,G2} × {R0..R3} = 16 agents`) from V2.2 is retained as the **long-term ceiling** — what the project can grow into if the pipeline works and diversity is still insufficient.

The **actual implementation target** is the 4-agent starter pool:

```
CNN_cycle     CNN_territory
GNN_cycle     GNN_territory
```

Expansion to 6 agents (+ midpoint variants) is a Phase 2 target, only after the starter pool shows visible strategic diversity. Expansion to the full grid is optional and late.

This section describes how those 4 agents are produced via the shared-base approach.

---

### 13.1 Phase 0 — shared base training (one per architecture)

Train one neutral base model per architecture family before producing any style variants.

#### Objective
Terminal win/loss only. No reward shaping. The base must not develop an implicit style before the split.

#### Opponent pool during base training
- ~50% own recent snapshots
- ~30% own older snapshots  
- ~20% fixed heuristic anchors (Greedy_1, Greedy_2, SearchBot)

Both greedy variants are included so the base is exposed to both strategic pressures without being nudged toward either.

#### Split timing — critical
Split the base **before convergence**, not after.

Target: base reliably beats Greedy but has not settled into a dominant opening or mid-game pattern. Observable signals:
- win rate vs Greedy: 65–80% (competent but not dominant)
- game trajectories still show variation in style across different opponents
- policy entropy still meaningfully above minimum

If the base is allowed to converge fully before splitting, both clones start with the same baked-in implicit strategy and fine-tuning becomes a weaker intervention. Split too early and the clones waste compute re-learning basic legality independently.

#### Base models as permanent artifacts
CNN_base and GNN_base are kept as frozen benchmark anchors throughout all subsequent phases. They answer: "what does competent-but-unstyled play look like?" — useful for isolating whether style variants are actually diverging.

#### Phase 0 gate
- [ ] CNN_base trains stably
- [ ] GNN_base trains stably
- [ ] Both beat Greedy reliably (≥65% win rate)
- [ ] Both are assessed as "competent but not converged" per the signals above
- [ ] Both checkpoints saved as permanent benchmark anchors

---

### 13.2 Phase 1 — clone and fine-tune → 4 agents

Clone each base into two style variants and fine-tune on shaped rewards.

#### Cloning procedure
1. Take the base checkpoint at the split point
2. Create two exact copies (identical weights, different agent IDs)
3. Assign each copy its shaped reward profile
4. Resume PPO training from the checkpoint with the new reward

No re-initialization. No random restarts. Each clone inherits full competence from the base and diverges from there.

#### Style profiles

**Cycle-focused (CNN_cycle, GNN_cycle)**
```
w_terminal  = 1.0   # always primary
w_cycle     = high  # own cycle/scoring nodes minus opponent
w_component = moderate
w_territory = low
w_mobility  = low
```
Expected character: pursues visible scoring earlier, prefers concrete local structure, accepts less spatial control for immediate cycle pressure.

**Territory-focused (CNN_territory, GNN_territory)**
```
w_terminal  = 1.0   # always primary
w_territory = high  # reachable area / siege estimate minus opponent
w_frontier  = moderate
w_mobility  = moderate (opponent penalty)
w_cycle     = low
```
Expected character: contests space earlier, values restriction, delays immediate score for spatial leverage.

#### Opponent pool during fine-tuning
- ~40% own recent snapshots (same agent)
- ~30% cross-style snapshots from the other 3 agents in the starter pool
- ~10% own older snapshots
- ~20% fixed heuristic/search anchors + both base models

Cross-style exposure during fine-tuning is important: each agent should face opponents with different strategic priorities from the start of divergence.

#### Divergence validation gate
Before declaring Phase 1 complete:
- [ ] All 4 agents train stably from their base checkpoint
- [ ] All 4 beat Greedy reliably
- [ ] CNN_cycle and CNN_territory play observably differently (different openings, mid-game shapes, score patterns)
- [ ] GNN_cycle and GNN_territory play observably differently
- [ ] CNN variants and GNN variants show architecture-level differences
- [ ] If any two agents are indistinguishable: diagnose before expanding

---

### 13.3 Phase 2 — midpoint expansion → 6 agents (conditional)

Only after Phase 1 shows visible diversity across the 4 agents.

Clone each base again and fine-tune on a balanced profile:

**Midpoint profile (CNN_mid, GNN_mid)**
```
w_terminal  = 1.0
w_cycle     = medium
w_territory = medium
w_mobility  = low to moderate
w_component = low to moderate
```

Cycle and territory terms should be of similar scale. Neither should dominate.

Expected character: avoids overcommitting to either immediate scoring or spatial control. Takes opportunities in either direction. Likely the most "human-like" sparring partner of the set.

**Why midpoint is not first:**
The two extremes must be confirmed real before a midpoint is interpretable. If cycle-focused and territory-focused agents are already indistinguishable, a midpoint will be meaningless.

#### Phase 2 gate
- [ ] Phase 1 divergence confirmed
- [ ] CNN_mid and GNN_mid clone from base and fine-tune stably
- [ ] Midpoint agents play observably differently from both extremes

---

### 13.4 Phase 3 — cross-architecture mixing

After all 6 agents (or the 4-agent pool if midpoint is skipped) are stable and diverse:
- Include cross-architecture opponents in each agent's pool
- GNN variants face CNN variant snapshots and vice versa
- All agents still face own-family snapshots and heuristic anchors

This is the disciplined version of "teach each other." Not policy imitation — richer opponent diversity spanning both architecture families.

---

### 13.5 Phase 4 — optional advanced methods (ceiling)

Only if Phases 1–3 are complete and strategic diversity is still insufficient.

**Step 1: hyperparameter-only PBT**
- Add hyperparameter variation across the population
- Keep reward shaping consistent within each agent (shaping is structural, not a hyperparameter)
- Keep terminal objective fixed

**Step 2: reward-shaping mutation (late diagnostic only)**
- Only if Step 1 runs and population still collapses
- Diagnose first — collapse after Phase 1 shaping diversity likely indicates training instability or degenerate game dynamics
- Allow small shaping weight perturbations only if no other explanation is found

**Full 16-agent grid**
- Only if the 6-agent pool is stable, diverse, and the research question requires more coverage
- Expand by adding G1/G2 bootstrap axis and additional R variants
- Do not expand before Phases 1–3 are validated

---

### 13.6 Compute scaling

| Phase | Agents | Minimum GPUs |
|---|---|---|
| Phase 0 | 2 bases | 2 |
| Phase 1 | 4 agents | 4 (one per agent) |
| Phase 2 | 6 agents | 4 (run 4 at a time) |
| Phase 3 | 6 agents | 4 |
| Phase 4 full grid | 16 agents | 8–16 |

For Phase 1 with 4 GPUs: run all 4 agents simultaneously, one per GPU. Fine-tuning from a shared base checkpoint is lightweight enough that this is feasible.

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

V2.3 restructures the training approach established in V2.2.

It keeps everything V2 through V2.2 established:
- unified bot interface, single-placement action decomposition, masked action space
- staged implementation: engine → bots → search → learning
- GNN-first architecture, CNN as comparison baseline
- two greedy variants (G1 cycle-focused, G2 territory-focused)
- board size curriculum (R=4 → 7 → 10)
- siege features computed but not mandatory in first observation layers
- PBT and advanced methods late and conditional

V2.3 changes how the training population is built:

**Shared-base-then-split:**
1. Train CNN_base and GNN_base on neutral terminal reward until competent but not converged
2. Clone each base into two style variants (cycle-focused, territory-focused)
3. Fine-tune each clone on its shaped reward from the base checkpoint
4. This produces 4 agents from 2 training runs, with divergence from a stable foundation

**4-agent starter pool as the Phase 1 target:**
- `CNN_cycle`, `CNN_territory`, `GNN_cycle`, `GNN_territory`
- Enough to give real strategic pressure from different directions
- Manageable implementation cost at Phase 1

**Progression:**
- Phase 0: train 2 bases
- Phase 1: clone + fine-tune → 4 agents, validate diversity
- Phase 2: add midpoint variants → 6 agents (only if Phase 1 diversity confirmed)
- Phase 3: cross-architecture mixing
- Phase 4: optional full 16-agent grid / PBT (ceiling, not target)

The 16-agent combinatorial grid from V2.2 is demoted to an optional ceiling. The base models are kept as permanent neutral benchmark anchors throughout all phases.
