# Cycle Control — AI Implementation Checklist V2.4.1

**Status:** operational checklist derived from `AI_DESIGN_V2.3`, tuned by practitioner evidence (Huang 2022, SIMPLE self-play, ML-Agents, Tablut AlphaZero reproduction, CleanRL, PettingZoo, Ben-Assayag & El-Yaniv 2021 arXiv:2107.08387)  
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


## 0.3 Practical tooling recommendations (tuned by practitioner evidence)

These are not random preferences. They reflect what current tooling most directly supports for this kind of project, drawn from documented self-play practice.

### Recommended default stack

**Environment layer**
- PettingZoo AEC as the multi-agent interface (sequential turn-based games are AEC's native use case; Chess and Connect Four in PettingZoo Classic use this model)
- Action masks stored in the observation dict as `{"observation": ..., "action_mask": ...}` — this is the PettingZoo Classic convention
- SuperSuit wrappers for any frame stacking, normalization, or vectorization
- Gymnasium single-agent wrapper only where the learner side needs it

**Learner layer**
- PPO with explicit masking: `sb3-contrib` `MaskablePPO` is the default; CleanRL `ppo.py` if a minimal from-scratch reference is needed
- Do NOT roll custom masking logic into a custom PPO — masking bugs are among the most common silent failures
- Reference for correct implementation: Huang et al., "The 37 Implementation Details of PPO" (ICLR 2022 Blog Track). Do not skip any of the 37 details silently

**GNN layer**
- PyTorch Geometric (PyG) as the default graph library
- Batching via PyG's `DataLoader` (block-diagonal sparse adjacency, no padding needed)
- Layer choice order to try: `GCNConv` → `GraphConv` → `GATConv` (GraphConv has a separate weight for self-connections, often better for node-label tasks like ours; GAT adds attention but increases compute)

**Logging/monitoring layer**
- TensorBoard or Weights & Biases for training curves — RL agents fail silently, logging is not optional
- Elo tracking as primary skill metric — episodic reward is meaningless in self-play (it depends on opponent)
- Keep the Elo table checked in to the repo or logged per run so progress is traceable across checkpoints

### Version pinning and reproducibility

- [ ] Record exact package versions (`pip freeze > requirements.lock` or equivalent) before the first stable training run
- [ ] PettingZoo keeps strict environment versioning (`v0`, `v1`, etc.) — pin the version you validated against
- [ ] Record the random seeds used in each run
- [ ] Record commit hash of the engine at each saved checkpoint

### Required validation steps before trusting a training run

- [ ] `pettingzoo.test.api_test(env, num_cycles=1000)` passes without warnings
- [ ] `pettingzoo.test.seed_test(env, num_cycles=10)` passes (seed determinism)
- [ ] If using parallel wrapper: `parallel_api_test(env, num_cycles=1_000_000)` passes
- [ ] Short "random vs random" smoke test runs to completion without crashes, assertions, or illegal moves

### Anti-drift rule

- [ ] Record the exact chosen stack in the repo before large implementation starts
- [ ] Record package versions used for the first stable run
- [ ] Document library choice rationale — any deviation from the recommended stack needs a written justification

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

### 7.0 Why GNN is the right choice here — empirical evidence

Ben-Assayag & El-Yaniv (2021) "Train on Small, Play the Large" (arXiv:2107.08387) directly validates our architecture and curriculum choices. Key findings:

- **Board-size transfer**: GNNs naturally process graphs of any size — a GNN trained on R=4 transfers to R=10 without architectural changes. CNNs with any fully-connected layers are fixed to one board size and cannot transfer (see Section 8 constraints).
- **Curriculum efficiency**: ScalableAlphaZero trained on small boards for 3 days defeated a standard AlphaZero trained on large boards for 30 days. Our R=4→7→10 curriculum is directly supported by this result.
- **Non-local pattern advantage**: GNN agents avoid the non-local pattern mistakes that CNN agents make. For Cycle Control, where cycle detection and siege recognition are inherently non-local (require reasoning about connected components across the board), this is a direct advantage.
- **No domain knowledge needed**: ScalableAlphaZero learned from self-play alone. Our design (no pre-baked siege channels in GNN v1) is consistent with this.

Practical implication: **the GNN base model should be trained on the full curriculum path (R=4 → R=7 → R=10), not just on R=10.** Each curriculum step initializes from the previous step's checkpoint. The GNN architecture handles this naturally because `edge_index` just changes between steps — the weight matrices don't need resizing.

### 7.1 Graph representation
- [ ] Build graph as a `torch_geometric.data.Data` object with fields: `x` (node features), `edge_index` (2 × num_edges), `batch` (for batched inputs)
- [ ] Use the real board topology as the graph — standard adjacency + mirror adjacency under the committed ruleset
- [ ] Include **both directions** for each undirected edge in `edge_index` (PyG convention — `[u,v]` and `[v,u]` must both be in the list)
- [ ] Verify graph construction on known small boards (R=3, R=4) by printing adjacency and checking against hand-computed values
- [ ] Verify `edge_index` is a `long` tensor (PyG will silently fail or misbehave if it's float)

### 7.2 Node features

Minimum required features (binary unless noted):
- [ ] own stone
- [ ] opponent stone
- [ ] empty
- [ ] on-board / valid-node flag (for batched graphs that pad to same size)
- [ ] local own-neighbor count (normalized to [0, 1])
- [ ] local opponent-neighbor count (normalized)
- [ ] move-number / game-progress signal (normalized scalar broadcast to all nodes)
- [ ] turn-phase one-hot (4 dims: opening, normal_1, normal_2, truncated_1) broadcast
- [ ] active-player indicator (broadcast)

**Do NOT include in v1:**
- [ ] Sieged-interior channels — these are a strong prior that may shortcut learning. Add them only if an ablation shows agents aren't learning siege concepts organically

### 7.3 Architecture recipe (starting point)

Based on PyG practice for node-level prediction on small graphs:

- [ ] Input projection: `Linear(num_node_features, hidden_dim=128)`
- [ ] 4-6 GNN blocks, each:
  - `GraphConv(hidden_dim, hidden_dim)` (GraphConv has separate weight for self-loop, better than GCNConv for node-label tasks)
  - `ReLU`
  - optional `BatchNorm` (PyG's `BatchNorm` for graph data) or `LayerNorm`
  - residual connection: `x = x + block(x)` — residuals stabilize training significantly
- [ ] Skip connection from input to final block output
- [ ] Shared trunk → two heads:
  - Policy head: `Linear(hidden_dim, 1)` per node → concatenate + append pass logit
  - Value head: `global_mean_pool(x, batch)` → `Linear(hidden_dim, 128)` → `Tanh` → `Linear(128, 1)` → `Tanh`

**Alternatives to try if GraphConv underperforms:**
- [ ] `GATConv` with 4 attention heads (captures which neighbors matter most — useful for territory reasoning)
- [ ] `SAGEConv` (GraphSAGE — more sample-efficient on large graphs)
- Do NOT start with `GCNConv` — it normalizes by degree, which weakens signal for irregular graphs like our mirror-adjacency topology

### 7.4 Policy/value output format
- [ ] Policy logits: shape `(batch_size, N+1)` where N = number of nodes, last index is pass
- [ ] Value: shape `(batch_size, 1)`, squashed to `[-1, +1]` via tanh
- [ ] Both heads share the trunk — do NOT build two separate networks (from Tablut reproduction: separate trunks causes catastrophic forgetting between roles; shared trunk with color-relative observations is the stable pattern)

### 7.5 Masking integration
- [ ] Mask is applied **after** policy logits are computed, by setting masked logits to `-inf` (NOT by multiplying probs by mask — that breaks gradient flow)
- [ ] Verify numerically: on a state with 3 legal actions, softmax over logits must produce exactly 0 for illegal indices (within float precision)
- [ ] Confirm gradient flows correctly through mask (Huang & Ontañón 2020 show this is subtle)

### 7.6 GNN implementation quality checks
- [ ] Forward pass works on batched states (use `DataLoader` from `torch_geometric.loader`)
- [ ] Output dimensions match `(batch_size, N+1)` policy and `(batch_size, 1)` value
- [ ] Parameter count sanity check: 500K-1M params for R=10 board
- [ ] Memory usage on a batch of 256 states should fit comfortably on one GPU
- [ ] Symmetry augmentation path is compatible (applying rotation/reflection to graph + remapping edge_index)
- [ ] Gradient-flow test passes (section 9.6)

### 7.7 Phase gate — GNN model ready
- [ ] Graph construction validated on R=3 and R=4
- [ ] Node features match spec
- [ ] Forward pass produces correct output shape
- [ ] Masking works numerically (verified, not assumed)
- [ ] Parameter count is within expected range
- [ ] Runs a full PPO update step without errors

---

## 8. Model family B — CNN baseline

This is the **comparison baseline**, not the presumed main model.

### 8.1 Board tensor representation
- [ ] Axial grid layout: shape `(C, H, W)` where `H = W = 2*R+1` for a hex-shaped region of radius R
- [ ] Two orientation slots per cell (up-triangle, down-triangle) handled as either:
  - separate channels (doubling channel count)
  - interleaved rows (doubling H)
  - Pick one and document the choice
- [ ] Off-board cells padded with zeros; include a separate "on-board" mask channel
- [ ] Channel set (match GNN node features so comparison is fair):
  - own stones
  - opponent stones
  - empty / on-board
  - off-board padding mask
  - own neighbor count (normalized)
  - opponent neighbor count (normalized)
  - move-number scalar broadcast
  - turn-phase one-hot (4 channels) broadcast
  - active-player flag broadcast
  - Total: ~13 channels (doubled if using separate-channel encoding for orientation)

### 8.2 Architecture recipe (starting point)

AlphaZero-style ResNet trunk adapted for small boards:

- [ ] Initial conv: `Conv2d(in_channels, 64, kernel_size=3, padding=1)` + BN + ReLU
- [ ] 4-6 residual blocks, each: `Conv3x3(64→64) → BN → ReLU → Conv3x3(64→64) → BN → add → ReLU`
- [ ] Policy head: `Conv1x1(64→2) → BN → ReLU → Flatten → Linear → N+1 logits`
- [ ] Value head: `Conv1x1(64→1) → BN → ReLU → Flatten → Linear(256) → ReLU → Linear(1) → Tanh`

### 8.3 CNN-specific constraints
- [ ] Do NOT require handcrafted siege/interior channels in v1
- [ ] Mask logits with `-inf` after policy head (same as GNN)
- [ ] Handle the off-board cells correctly — the policy head will produce logits for off-board indices, mask them out
- [ ] Parameter count comparable to GNN (500K-1M) so the architecture comparison is fair

**Critical board-size transfer constraint (from Ben-Assayag & El-Yaniv 2021 and the Hex GNN paper):**
- [ ] **Do NOT use a fully-connected (FC) layer in the feature extractor** — CNNs with FC layers are fixed to one board size and cannot transfer knowledge between R=4, R=7, R=10. This breaks the curriculum.
- [ ] Use a **fully convolutional** design up to the board-level feature map; the policy head can be a `Conv1x1` followed by flatten, but the feature extractor must be all-convolutional
- [ ] The U-Net architecture in particular breaks down on board-size transfer (confirmed by Hex GNN paper) — do NOT use it
- [ ] If the CNN cannot transfer between board sizes, it loses the curriculum efficiency advantage and must be trained separately at each board size — document this limitation clearly if FC is used anyway

### 8.4 Fair comparison requirements

For the CNN vs GNN comparison to mean anything:
- [ ] Same observation channels (modulo spatial vs graph format)
- [ ] Same training hyperparameters (lr, clip, entropy, etc.)
- [ ] Same opponent pool composition
- [ ] Same number of training steps
- [ ] Same evaluation suite
- [ ] Same random seeds for initialization comparison

### 8.5 Phase gate — CNN model ready
- [ ] Tensor representation validated (same information content as GNN)
- [ ] Forward pass produces correct shape `(batch_size, N+1)` policy and `(batch_size, 1)` value
- [ ] Off-board cells correctly masked
- [ ] Parameter count in expected range
- [ ] Compatible with the same mask / self-play / evaluation framework as GNN

---

## 9. Environment and training harness

### 9.1 Environment design
- [ ] Environment emits observation + legal mask in the PettingZoo Classic dict format: `{"observation": ..., "action_mask": ...}`
- [ ] Multi-placement turns handled correctly — `agent_selection` stays on the same player across both placements within a turn
- [ ] Pass action has a dedicated index in the action space (last index is fine)
- [ ] Environment is reproducible under `env.reset(seed=N)` — same seed produces same trajectory
- [ ] Environment is easy to wrap for self-play (AEC `agent_iter()` loop works)
- [ ] Run `pettingzoo.test.api_test(env, num_cycles=1000)` and fix every warning
- [ ] Run `pettingzoo.test.seed_test(env)` to confirm determinism
- [ ] Run `pettingzoo.test.max_cycles_test` to confirm the env terminates cleanly
- [ ] Run Gymnasium / SB3 environment checks on any single-agent wrapper used for training
- [ ] Smoke test: random vs random, 100 games, zero illegal moves, zero crashes

### 9.2 Framework choice
- [ ] Default RL library: `sb3-contrib` with `MaskablePPO`
- [ ] Alternative: CleanRL `ppo.py` if a from-scratch reference is wanted — both are fine, do not use both in the same training run
- [ ] If using `MaskablePPO`: use `MaskableEvalCallback` for evaluation (default `EvalCallback` will ignore masks)
- [ ] If using subprocess vectorized envs with `MaskablePPO`: confirm the `action_mask_fn` is correctly reachable from inside worker processes
- [ ] If not using `sb3-contrib`: document why and point to where masking is implemented instead
- [ ] Verify masking actually zeroes invalid action probabilities — log `policy.distribution.probs` on a test state where most actions are invalid, confirm they are zero
- [ ] Reference: Huang & Ontañón 2020, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms"

### 9.3 Starting hyperparameters (starting point, tune from here)

Based on PPO board-game practice (Huang 2022 "37 Implementation Details", SIMPLE, CleanRL defaults):

- [ ] `learning_rate = 3e-4` (Adam default; reduce to 1e-4 if training is unstable)
- [ ] `n_steps = 2048` per rollout (per env)
- [ ] `batch_size = 256`
- [ ] `n_epochs = 4` (4 epochs per PPO update; more can overfit the rollout)
- [ ] `gamma = 1.0` (deterministic finite game; no discounting needed for terminal rewards)
- [ ] `gae_lambda = 0.95`
- [ ] `clip_range = 0.2`
- [ ] `ent_coef = 0.01` at start, anneal to 0.001 over training (prevents early policy collapse, allows exploitation late — this is the SIMPLE recipe and it works)
- [ ] `vf_coef = 0.5`
- [ ] `max_grad_norm = 0.5`
- [ ] `target_kl = 0.02` (early-stop the update if KL divergence exceeds this — prevents naive-masking KL blowup)
- [ ] Advantage normalization ON (per-batch)
- [ ] Observation normalization: only if observations have unbounded continuous channels; for binary channels keep off

### 9.4 Self-play support
- [ ] Build self-play wrapper that converts 2-player AEC env into 1-player env for the current learner
- [ ] Support snapshot pools with tunable `window` size (how many past checkpoints to retain) — start with 16 iterations based on Tablut reproduction findings
- [ ] Support fixed heuristic/search anchors in the opponent pool
- [ ] Support color randomization per episode (prevents role specialization)
- [ ] Support `play_against_latest_model_ratio` parameter (fraction of games against current self vs older snapshot) — start with 0.5 based on ML-Agents defaults
- [ ] Support `swap_steps` — how often to swap opponent — start with 10_000 environment steps
- [ ] Support curriculum by board size (R=4 → 7 → 10)
- [ ] Support 8-fold symmetry augmentation of observations (D6-like symmetry of the hex board) — critical for sample efficiency per AlphaZero practice

### 9.5 Required logging (each run)

These metrics are the ones that actually tell you what is happening. Log all of them, track all of them:

**PPO internal health**
- [ ] `losses/policy_loss` — should fluctuate around zero, not monotonically grow
- [ ] `losses/value_loss` — should generally decrease over time; expect jumps when new checkpoint is saved (its prior evaluation becomes invalid)
- [ ] `losses/entropy_loss` — should drift gradually toward lower entropy (agent becoming more certain); if it crashes to near-zero early, increase `ent_coef`
- [ ] `losses/approx_kl` — should stay below `target_kl`; if constantly at limit, reduce `learning_rate` or `n_epochs`
- [ ] `losses/clip_fraction` — fraction of samples clipped by PPO's clip; 0.1-0.3 is healthy; >0.4 means updates are too aggressive

**Training dynamics**
- [ ] `charts/episodic_return` — mean reward per episode; noisy in self-play, use Elo as primary metric
- [ ] `charts/episodic_length` — average game length; stable length indicates stable policy
- [ ] `charts/learning_rate` — if using schedule, confirm it's decreasing as intended
- [ ] `charts/action_mask_fraction` — fraction of action space that was legal per step; logs whether masking is actually restricting the space

**Self-play dynamics (most important)**
- [ ] `selfplay/elo_vs_bank` — Elo rating of current learner against full checkpoint bank
- [ ] `selfplay/win_rate_vs_latest` — win rate against the most recent checkpoint
- [ ] `selfplay/win_rate_vs_greedy_1` — win rate against Greedy_1 (fixed anchor)
- [ ] `selfplay/win_rate_vs_greedy_2` — win rate against Greedy_2 (fixed anchor)
- [ ] `selfplay/win_rate_vs_random` — should be ~100% after minimal training; if not, pipeline is broken
- [ ] `selfplay/opponent_pool_size` — size of the snapshot pool currently being sampled from

**Reproducibility**
- [ ] Log all seeds
- [ ] Log checkpoint IDs
- [ ] Log opponent sampling composition per-episode (which opponent, which snapshot)
- [ ] Log all hyperparameters at run start
- [ ] Log git commit hash

### 9.6 Required debugging diagnostics

Build these into the harness from day one. They are cheap and they catch silent failures:

- [ ] **Masking unit test**: on a synthetic state with only 3 legal actions, verify `policy.distribution.probs` has zero weight on all other actions
- [ ] **Legality test**: 1000 random rollouts with the current policy, zero illegal moves triggered
- [ ] **Reset test**: run `env.reset(seed=N)` twice with same seed, verify identical trajectories
- [ ] **Gradient flow test**: on 10 training steps, verify gradients are non-zero on all layers (no dead layers from init bugs)
- [ ] **Value-head sanity**: evaluate value head on `initial_state()` — should be near 0 for a symmetric game

### 9.7 Phase gate — training harness ready
Do not start Phase 0 base training until:

- [ ] All 9.1 env tests pass
- [ ] 9.2 masking verified empirically (not just assumed from library)
- [ ] 9.3 hyperparameters recorded with rationale for any deviation
- [ ] 9.4 self-play wrapper works end-to-end (can run random-vs-random through it)
- [ ] 9.5 all required metrics are logged (not just the convenient ones)
- [ ] 9.6 all diagnostic tests pass

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
- [ ] Opponent pool composition during base training:
  - ~50% own snapshots (recent, window of last 8-16 checkpoints)
  - ~30% older own snapshots (anti-forgetting — applies from Tablut reproduction: larger buffer stabilizes self-play)
  - ~20% fixed heuristic anchors (Greedy_1, Greedy_2, SearchBot equally weighted)
- [ ] Both greedy variants included so the base is exposed to both strategic pressures without being nudged toward either
- [ ] Save checkpoint every N=100 training iterations (or every N=50 on small boards)
- [ ] Keep at least the last 16 checkpoints in the opponent bank (larger buffer stabilizes)
- [ ] Randomize color each episode (prevents role specialization)
- [ ] Apply 8-fold symmetry augmentation to observations (D6 hex symmetry)

### 10.2 Monitoring CNN_base for the split point
- [ ] Track `win_rate_vs_greedy_1` and `win_rate_vs_greedy_2` per evaluation cycle
- [ ] Track policy entropy (`-sum(p * log p)`) — must stay meaningfully above minimum before splitting
- [ ] Track Elo vs the full checkpoint bank — should be rising monotonically
- [ ] Track game trajectory diversity — opening moves, mid-game shapes; if variance collapses, the base is converging and you are near the split point
- [ ] Inspect games manually every 500 iterations — look for style lock-in (same opening every game, same response patterns)

### 10.3 CNN_base split condition
Save base checkpoint when ALL of these hold:
- [ ] win rate vs Greedy_1 AND Greedy_2 both in range 65-80%
- [ ] Elo is still rising (not plateaued)
- [ ] Policy entropy is still at least 50% of its initial value
- [ ] Game trajectories still show some variation across seeds

If base converges before all conditions hold (e.g., Elo plateaus at 60%), diagnose before splitting. Likely cause: opponent pool too weak, or architecture too small.

### 10.4 GNN_base
- [ ] Same procedure as CNN_base
- [ ] Use identical hyperparameters, opponent pool composition, evaluation cadence
- [ ] Save GNN_base at the same split-criteria trigger
- [ ] Note: GNN may train faster or slower than CNN — adjust checkpoint interval if needed, but do not change the split criteria

### 10.5 Base model artifacts
After split, both bases are frozen and become permanent benchmark anchors:
- [ ] CNN_base checkpoint saved with metadata (training iterations, hyperparameters, opponent pool stats, split-condition values)
- [ ] GNN_base checkpoint saved similarly
- [ ] Both added to the benchmark pool in Section 13.3
- [ ] Documented in the run log with git commit hash and package versions

### 10.6 Expected wall-clock for Phase 0

Rough estimates (4 GPUs, 100 games/sec self-play throughput):
- R=4 (~96 cells): base converges in 1-2 hours per architecture
- R=7 (~294 cells): 4-8 hours per architecture
- R=10 (~600 cells): 12-24 hours per architecture

**Curriculum protocol (Ben-Assayag & El-Yaniv validated):**
- [ ] Train R=4 base first until split condition met, save checkpoint
- [ ] Initialize R=7 base from R=4 checkpoint (GNN: `edge_index` changes, weights transfer; CNN: works only if fully convolutional — see Section 8.3)
- [ ] Train R=7 until split condition met, save checkpoint
- [ ] Initialize R=10 base from R=7 checkpoint
- [ ] Train R=10 until split condition met — this is the base used for cloning in Phase 1

This approach matches the "train on small, play the large" result. Total curriculum time is NOT the sum of each step — each step starts from a competent base, so convergence at each step is much faster than training from scratch.

Expected total Phase 0 time (GNN, curriculum path): ~6-10 hours on 4 GPUs.
Expected total Phase 0 time (CNN, fully convolutional): similar.
Expected total Phase 0 time (CNN, with FC layers, trained separately per board size): ~3× longer and loses curriculum advantage.

### 10.7 Phase 0 gate — BLOCKER for Phase 1
- [ ] Both bases train stably (loss curves, not diverging)
- [ ] Both beat Greedy_1 AND Greedy_2 in the 65-80% range
- [ ] Both assessed as "competent but not converged" per the signals above
- [ ] Both checkpoints saved and documented
- [ ] Split timing decision recorded with concrete numbers (win rates, entropy, Elo)
- [ ] Both bases added to benchmark pool

---

## 11. Phase 1 — clone and fine-tune → 4 agents

### 11.1 Cloning procedure (exact steps)
- [ ] Save the base model state_dict and optimizer state
- [ ] Load into a fresh PPO agent instance with a new agent ID
- [ ] Verify the loaded model produces identical outputs to the base on a test batch (sanity check — catches serialization bugs)
- [ ] Assign new shaping reward config
- [ ] Reset only the optimizer's learning rate schedule if using one; keep optimizer state (momentum) from base
- [ ] Do NOT reset the value head separately — the base's value estimates are a warm start

Repeat for each of the 4 clones:
- [ ] CNN_cycle ← CNN_base
- [ ] CNN_territory ← CNN_base
- [ ] GNN_cycle ← GNN_base
- [ ] GNN_territory ← GNN_base

### 11.2 Reward shaping profiles (starting weights)

Shaping is an auxiliary signal on the value head's training target. Terminal reward is always the primary signal at weight 1.0.

**Cycle-focused (CNN_cycle, GNN_cycle)**
- [ ] `w_terminal = 1.0` (primary, not shaped — terminal win/loss remains the true objective)
- [ ] `w_cycle = 0.3` (own cycle/scoring nodes minus opponent, per step)
- [ ] `w_component = 0.1` (largest own component size minus opponent's)
- [ ] `w_territory = 0.05` (minor)
- [ ] `w_mobility = 0.02` (minor)

**Territory-focused (CNN_territory, GNN_territory)**
- [ ] `w_terminal = 1.0`
- [ ] `w_territory = 0.3` (own reachable area / siege estimate minus opponent)
- [ ] `w_frontier = 0.1` (own expandable border cells)
- [ ] `w_mobility = 0.1` (opponent mobility penalty — higher = worse for us)
- [ ] `w_cycle = 0.05` (minor)

**Notes on shaping weights:**
- Shaping terms are scaled so their sum is ~0.5 (half the magnitude of terminal) — keeps terminal reward dominant
- Document exact weights; record any tuning adjustments
- If an agent's win rate drops significantly after cloning, the shaping may be too aggressive — reduce shaping weights

### 11.3 Opponent pool during fine-tuning per agent

Each of the 4 agents fine-tunes with the following opponent mix:
- [ ] ~40% own recent snapshots (window of last 16)
- [ ] ~30% cross-style snapshots from the other 3 agents in the starter pool (10% each)
- [ ] ~10% own older snapshots (anti-forgetting)
- [ ] ~10% fixed heuristic anchors (Greedy_1, Greedy_2, SearchBot)
- [ ] ~10% the base models (CNN_base, GNN_base) — these stay as fixed neutral reference points

Total: 100%. Cross-style exposure is important from the start — each agent should encounter opponents with different strategic priorities immediately after the clone split, not later.

### 11.4 Fine-tuning hyperparameter adjustments from Phase 0

- [ ] Reduce learning rate by 2-4x from Phase 0 default (e.g., 3e-4 → 1e-4) — fine-tuning from a trained base should not require large weight updates
- [ ] Keep `n_steps`, `batch_size`, `n_epochs` the same as Phase 0
- [ ] Keep `ent_coef` at its annealed value from Phase 0 (do NOT reset to high entropy — base already has a reasonable policy)
- [ ] Monitor for policy drift: if `approx_kl` exceeds 0.05 in early fine-tuning steps, reduce learning rate further

### 11.5 Phase 1 evaluation cadence
- [ ] Every 100 training iterations, run round-robin tournament: each of 4 agents vs each other (6 pairings × 20 games = 120 games per eval)
- [ ] Record pairwise win rates
- [ ] Record each agent's win rate vs Greedy_1, Greedy_2, Random, and both base models
- [ ] Record Elo using the full tournament history
- [ ] Inspect game trajectories: save 5 games per agent per eval for manual review

### 11.6 Divergence validation gate — BLOCKER for Phase 2

All of these must hold before declaring Phase 1 complete:

- [ ] All 4 agents train stably from base checkpoint
- [ ] All 4 beat Greedy_1 reliably (>70% win rate)
- [ ] All 4 beat Greedy_2 reliably (>70% win rate)
- [ ] All 4 beat their own base (>55%) — if a clone cannot beat its own base, fine-tuning is failing
- [ ] CNN_cycle vs CNN_territory: observable strategy difference (different openings, mid-game shapes, score patterns)
- [ ] GNN_cycle vs GNN_territory: observable strategy difference
- [ ] CNN family vs GNN family: observable architecture-level difference (different positional preferences, different response patterns)
- [ ] Pairwise win rates across all 4 agents form a non-trivial matrix (not all equal, not a strict total ordering)

**If any two agents are indistinguishable:**
- [ ] Diagnose: is the shaping too weak? Is the fine-tuning too short? Are the architectures too similar?
- [ ] Document the finding before expanding — a Phase 2 midpoint agent is meaningless if the Phase 1 extremes aren't distinct

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

## 14. Risks, sanity checks, and diagnostic signals

### 14.1 Rule risks
- [ ] Monitor first-player edge (Black should not win >60% at high skill — if it does, the rules favor the opener too much)
- [ ] Monitor draw degeneracy (>30% draws at high skill = rules reward blocking over building)
- [ ] Monitor small-board vs large-board mismatch — strategies that work at R=4 may not transfer to R=10; validate at each curriculum step
- [ ] Monitor mirror-adjacency pathologies (e.g., one-shot cycles of length 4 being too easy)

### 14.2 Heuristic risks
- [ ] Do NOT treat Greedy/Search evaluation as ground truth — they encode specific assumptions
- [ ] Do NOT assume territory intuition is correct without evidence — the siege mechanic may not dominate at high skill
- [ ] If trained agents play very differently from Greedy, that's signal, not failure

### 14.3 Model risks
- [ ] Check whether GNN underperforms despite theory (possible — GATConv or deeper nets may be needed)
- [ ] Check whether CNN overfits local motifs (CNN on padded axial grid has a spatial prior mismatch for hex topology)
- [ ] Check whether either family only looks good due to poor opponent diversity (eval against heuristics AND against the other family)
- [ ] If reward-shaping diversity is used: check whether surviving shaping profiles reflect real game structure or are artifacts of noisy self-play

### 14.4 Training failure modes — concrete diagnostic signals

Based on PPO and self-play practice (Huang 2022, SIMPLE blog, Tablut reproduction):

**Policy collapse (agent converges to single action / style too early)**
- [ ] **Signal**: `losses/entropy_loss` crashes toward zero within first 100 iterations
- [ ] **Fix**: increase `ent_coef` to 0.02-0.05; slow the anneal
- [ ] **Fix**: increase opponent diversity; add more random past snapshots

**Training instability (policy jumps erratically)**
- [ ] **Signal**: `losses/approx_kl` consistently at or above `target_kl` (0.02)
- [ ] **Signal**: `losses/clip_fraction` > 0.4 consistently
- [ ] **Fix**: reduce `learning_rate` by 2-3x
- [ ] **Fix**: reduce `n_epochs` from 4 to 2
- [ ] **Fix**: reduce `clip_range` from 0.2 to 0.1

**Value function divergence (agent is confidently wrong)**
- [ ] **Signal**: `losses/value_loss` grows over time instead of decreasing
- [ ] **Signal**: value estimates at game start far from 0 for a symmetric game
- [ ] **Fix**: increase `vf_coef` to 1.0
- [ ] **Fix**: use separate networks for policy and value (at cost of sample efficiency)
- [ ] **Fix**: check for reward scaling bugs

**Catastrophic forgetting (agent regresses vs past checkpoints)**
- [ ] **Signal**: win rate vs older snapshots drops below 50%
- [ ] **Signal**: Tablut reproduction specifically reported this for asymmetric games
- [ ] **Fix**: increase the past-checkpoint replay window from 8 to 16+ iterations
- [ ] **Fix**: add data augmentation (symmetry rotations)
- [ ] **Fix**: add 25% past-checkpoint games to the training opponent pool
- [ ] **Fix**: randomize color per game (prevents role specialization)

**Self-play stagnation (no progress against fixed benchmarks)**
- [ ] **Signal**: win rate vs Greedy plateaus below 70% for 500+ iterations
- [ ] **Signal**: Elo stops rising for extended periods
- [ ] **Fix**: increase opponent diversity
- [ ] **Fix**: check if mask is too restrictive (observation mask fraction too low)
- [ ] **Fix**: reduce `play_against_latest_model_ratio` from 0.5 to 0.3 — agent may be overfitting to recent self

**Masking bugs (silent — agent explores invalid actions)**
- [ ] **Signal**: non-zero probability on masked actions in any rollout
- [ ] **Signal**: occasional illegal-action errors at runtime that mysteriously pass masking
- [ ] **Fix**: unit test the mask integration before starting any RL run (Section 9.6)
- [ ] **Fix**: log `action_mask.sum(-1)` to confirm mask has non-zero legal actions at all steps

**Weird internal metagame (agent plays strangely)**
- [ ] **Signal**: trained agent loses to simpler strategies (e.g., random-frontier bot)
- [ ] **Signal**: opening moves are bizarre (e.g., agent always passes first turn)
- [ ] **Fix**: increase diversity of opponents during training
- [ ] **Fix**: add fixed heuristic anchors (Greedy_1, Greedy_2) to the pool — prevents drift into self-optimized nonsense
- [ ] **Fix**: audit games manually — if agent is playing "right" but losing, the eval may be flawed

**Search-depth sensitivity**
- [ ] **Signal**: SearchBot with depth 3 draws conclusions that SearchBot with depth 6 reverses
- [ ] **Fix**: verify main findings hold across multiple search budgets
- [ ] **Fix**: include a strong SearchBot in the benchmark pool

### 14.5 Research-validity sanity check
- [ ] Ask explicitly: does stronger play make the game look better, worse, or merely stranger?
- [ ] Do not answer the game-design question using only one training curve
- [ ] Manually play against trained agents — if human can beat them with obvious strategy, the training is insufficient
- [ ] If humans can't beat trained agents AND trained agents can't beat each other consistently, the game may genuinely be strategically rich

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
- [x] V2.4 practice-tuned additions documented in Section 17.0.

---

## 18. Short version

If everything here feels too large, the irreducible core is:

1. prepare the engine for AI,
2. build Random, Greedy_1, Greedy_2, Search — validate G1 ≠ G2 behaviorally,
3. analyze small and mid boards before trusting RL,
4. set up the training harness with sb3-contrib MaskablePPO, PettingZoo AEC env, PyG for graphs, and required metrics + diagnostics,
5. train CNN_base and GNN_base on neutral reward — save as frozen anchors when competent but not converged (win rate 65-80% vs Greedy, entropy not collapsed),
6. clone each base into cycle-focused and territory-focused variants — fine-tune each on shaped reward at lower learning rate,
7. validate that the 4 agents play observably differently,
8. optionally expand to 6 agents by adding midpoint variants from the same bases,
9. then cross-architecture mixing, then optional PBT/full grid if needed,
10. evaluate the **game** separately from the **agent**.

That is the minimum disciplined path consistent with V2.3 design + V2.4 practice tuning.
