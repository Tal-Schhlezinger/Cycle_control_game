# Cycle Control — AI Design Document

**Status:** design, pre-implementation. No code written yet. This document specifies the bot interface, RL environment, network architecture, training pipeline, and open research questions.

**Scope:** enable training of bots strong enough to evaluate whether the current rule set (v5 + mirror adjacency + neutrality + strict adjacency) produces balanced two-player games. The goal is **rule balance research**, not a superhuman bot.

---

## 1. Bot interface

### 1.1 Design constraints

1. A turn after the opening contains **up to 2 placements**. This is a multi-placement turn, not a single action.
2. The action space is large (600 cells on a radius-10 board); a huge fraction is illegal in any given state under strict adjacency + neutrality. Random-over-all-cells is unusable.
3. Every bot — random, greedy, PPO, MCTS — must implement **the same interface** so we can pit any pair against each other for eval and self-play.

### 1.2 Chosen approach: single `choose_action` per placement

A turn is decomposed into up to two independent `choose_action` calls. The engine drives the loop; the bot is not aware of "turn" vs "placement 1 of 2" as separate concepts. This matches how AEC-style environments work in PettingZoo and simplifies the bot API.

**The bot sees:** its color, the current full game state (board + turn-phase + whose turn), and the precomputed legal-action mask.
**The bot returns:** one integer — an index into the action space — where `-1` (or a dedicated `PASS` index) means pass.

#### Action space encoding

```
action_index in [0, N]    where N = topology.node_count()
    0..N-1  = place at node i (indices correspond to topology.all_nodes() ordering)
    N       = pass
```

Size: `N+1`. On R=10 this is 601. The index-to-node mapping is fixed by `topology.all_nodes()` which is deterministic and sorted.

#### Interface

```python
class Bot(Protocol):
    def choose_action(
        self,
        state: GameState,
        legal_mask: np.ndarray,   # shape (N+1,), bool; True = legal
        color: Player,
    ) -> int:
        ...

    def reset(self, seed: int | None = None) -> None:
        """Called once per game. Resets any per-game state."""
        ...
```

Contracts:
- `legal_mask[action_index] == True` iff that action is legal in `state`.
- Bot MUST return an index `i` where `legal_mask[i] == True`. Returning an illegal index is a bug, not a legal loss.
- The engine is responsible for the turn loop: it calls `choose_action` once, applies the move, checks if a second placement is available (phase is `NORMAL_1`, game not over), and calls `choose_action` a second time if so. The bot does not need to know this is the 1st vs 2nd placement — the state object carries the turn phase if the bot wants to use it.
- If `choose_action` returns the pass index, the turn ends immediately regardless of whether placement 1 or 2 was pending. This matches the existing engine behavior (partial pass legal).

#### Why this decomposition and not a "full-turn" action

A full-turn action would be `(node1, node2_or_pass, pass_or_placement)`. Action space size would be `O(N²)` which is ~360k on R=10. Intractable for flat policy nets. The single-placement interface keeps the action space at `N+1` and leaves the coordination between the two placements to the policy's value estimate — exactly how PettingZoo handles multi-phase turns.

### 1.3 The engine-driven turn loop

```python
def play_turn(engine, state, bot):
    """Drive one turn of `bot` (Black or White depending on state.active_player).

    Handles: opening (1 placement), normal (up to 2 placements), truncated (1 placement),
    partial pass (terminates turn early), full pass (terminates turn with 0 placements).
    """
    starting_phase = state.turn_phase
    placements_done = 0
    max_placements = {
        TurnPhase.OPENING: 1,
        TurnPhase.NORMAL_1: 2,
        TurnPhase.NORMAL_TRUNCATED_1: 1,
    }[starting_phase]

    while placements_done < max_placements and not state.game_over:
        legal_mask = build_legal_mask(engine, state)
        action = bot.choose_action(state, legal_mask, state.active_player)
        if action == PASS_INDEX:
            engine.apply_pass(state)
            return
        node = engine.topology.all_nodes()[action]
        engine.apply_placement(state, node)
        placements_done += 1
    # Turn ended naturally after max placements; engine has already advanced.
```

This loop lives in the RL environment wrapper. Nothing about it changes between random/greedy/PPO bots.

---

## 2. RL environment

### 2.1 Framework: PettingZoo AEC

Use PettingZoo's AEC (Agent-Environment-Cycle) API. Sequential turn-based, native support for action masks in the observation dict, and widely supported by RL libraries (CleanRL, Tianshou, RLlib, Stable-Baselines3 via SuperSuit wrappers). Matches Connect Four / Chess / Go environments in PettingZoo Classic.

From PettingZoo's documentation: the observation dict contains `observation` and `action_mask`, and the RL library handles masking through the mask key automatically.

### 2.2 Observation encoding

A stack of 2D "planes" shaped to the triangular grid. Two options for the grid:

**Option A — Axial grid (H=2*R+1, W=2*R+1, 2 orientations):**
- Shape: `(C, H, W, 2)` or flattened to `(2C, H, W)`
- Natural for axial coordinates
- Some cells in the rectangle are off-board (mask as 0)

**Option B — Rectangular padded grid with one "orientation" channel:**
- Shape: `(C, 2H, W)` where row parity encodes orientation
- Simpler for conv nets but wastes more cells

**Choice: Option A, flattened as `(2C, H, W)`.** Cleaner mapping to axial coords, conv kernels still work.

Channels (binary unless noted):
1. Own stones
2. Opponent stones
3. Empty & on-board
4. Off-board (padding mask)
5. Own neighbor count per cell (normalized 0..1) — helps with neutrality reasoning
6. Opponent neighbor count per cell (normalized 0..1)
7. Is-own-sieged-interior (1 if this cell is guaranteed own territory, 0 otherwise) — **expensive to compute, provides critical strategic info**
8. Is-opponent-sieged-interior
9. Turn phase one-hot (broadcast to full plane): opening / normal_1 / normal_2 / truncated_1 — 4 channels
10. Active player flag (broadcast): 1 if it's my turn, 0 otherwise — 1 channel

Total: ~13 channels. Each channel is a 2D plane over the axial grid × 2 orientations = 2 planes per channel logically, so final tensor is `(26, H, W)` where `H = W = 2*R+1`.

**The sieged-interior channels (7, 8) are the key strategic signal.** They are computed by the same algorithm that drives the greedy bot's eval (section 4) and encode the partial-siege insight from playtesting. Including them as input means the net doesn't need to learn siege detection from scratch — it just needs to learn what to do with that information.

### 2.3 Action space

`Discrete(N + 1)` where `N = topology.node_count()`. Index `N` is pass.

Action mask is a `Box(low=0, high=1, shape=(N+1,), dtype=bool)`. Engine computes it via `engine.legal_moves(state)`, then appends `True` at index `N` if `engine.can_pass(state)`.

### 2.4 Reward

Terminal reward only: `+1` for winner, `-1` for loser, `0` for draw. Zero intermediate reward.

**Rationale:** intermediate shaping (e.g. cycle-score diff) would bias toward the greedy strategy you identified as losing. The siege mechanic is a long-term ceiling effect; only terminal reward captures it faithfully.

**Illegal action handling:** PettingZoo's default is "illegal move = -1 for the mover, game ends." With action masking enabled (section 3), this should be unreachable, but we keep the default as a safety net. A masked policy cannot sample an illegal action in theory.

### 2.5 Multi-placement turn in AEC

PettingZoo AEC steps one agent at a time. Two ways to handle the 2-placement turn:

**A. Agent selector stays the same across both placements.** The env keeps `agent_selection` on the same player until both placements are made or a pass occurs. Simple; matches the existing engine's turn model.

**B. Expose each placement as a separate "agent step" but with the same agent name.** Equivalent to A from the bot's perspective, cleaner internal book-keeping.

**Choice: A.** The env holds an internal counter (0 or 1) tracking placements-this-turn. After each `step`, if the current agent still has a placement remaining and hasn't passed, `agent_selection` stays the same. Otherwise it flips.

---

## 3. Network architecture

### 3.1 Base network (used by all RL bots)

**Input:** `(26, H, W)` observation tensor.

**Trunk:** small residual CNN, AlphaZero-style.
- Initial conv: `3×3, 64` channels
- Residual blocks: 4–6 blocks of `(conv3×3, 64 → BN → ReLU → conv3×3, 64 → BN → add → ReLU)`

The residual trunk is consistent with practice from the Tablut AlphaZero reproduction, which used 8 blocks of 128 filters for a 9×9 board with action space 2592; our board is comparable in size (R=10 → 600 cells) and our action space is smaller (601), so 4–6 blocks of 64 filters is a reasonable starting point.

**Heads (two separate linear heads on top of trunk, AlphaZero-style):**

- **Policy head:** flatten trunk output → linear → `N+1` logits → softmax (after masking)
- **Value head:** flatten trunk output → linear → 128 → tanh → linear → 1 → tanh  (output in `[-1, +1]` for win probability)

**Action masking:** Replace logits corresponding to masked actions with `-∞` before softmax. Huang & Ontañón (2020) showed this approach zeroes the gradients through invalid actions and is the standard approach used in AlphaStar and OpenAI Five.

**Separate heads per player color:** We do NOT use separate heads per player. The observation is color-relative (channel 1 = own, channel 2 = opponent), so the network implicitly handles both sides. This is the same pattern Connect Four / Chess PettingZoo implementations use: the board is always oriented to the mover's perspective.

### 3.2 Parameter count estimate

`(26 × 3 × 3 × 64)` + ~6 × `(64 × 3 × 3 × 64 × 2)` + policy head + value head ≈ 500K params. Trainable on a single GPU in hours; on 4 GPUs (your setup) in tens of minutes per iteration.

---

## 4. Greedy bot (bootstrap, and baseline evaluator)

### 4.1 Purpose

1. Non-random opponent for early PPO training — beats random, giving PPO a real gradient signal
2. Baseline for Elo ladder — every trained model is evaluated vs. greedy
3. Sanity check: if PPO can't beat greedy after reasonable training, something is broken

### 4.2 Algorithm

One-ply greedy. For each legal action:
1. Clone state, apply action
2. Compute `eval(state, own_color)`
3. Pick action with highest eval; tiebreak by a per-bot deterministic RNG

### 4.3 Evaluation function

```
eval(state, own_color) = 
    w_cycle * (own_cycle_nodes - opp_cycle_nodes)
  + w_siege * (own_sieged_empty - opp_sieged_empty)
  + w_comp  * (own_large_component_size - opp_large_component_size)
  + w_mobility * (own_legal_moves - opp_legal_moves)
```

**`own_cycle_nodes`:** current scoring (existing `scoring_nodes`)

**`own_sieged_empty`:** count of empty cells where all boundary paths to the opponent are blocked — i.e., cells the opponent cannot reach under neutrality + strict adjacency rules. **This is the critical computation.** See section 4.4.

**`own_large_component_size`:** size of largest connected component of own stones. Proxy for "is the cluster big enough to eventually cycle."

**`own_legal_moves`:** `len(engine.legal_moves(state))` with active_player set to own_color. Proxy for future flexibility.

Starting weights (to be tuned empirically):
```
w_cycle = 3.0      # direct score
w_siege = 1.0      # per sieged cell = future point
w_comp = 0.1       # mild bonus
w_mobility = 0.05  # tiebreaker
```

`w_cycle > w_siege` means the bot values current cycles over future territory — this is intentionally suboptimal (greedy) so PPO has something to improve against. Tune later.

### 4.4 Sieged-area detector — algorithm

**Definition:** an empty cell `c` is "sieged by player P" if no legal sequence of opponent moves can reach `c`. Under strict adjacency + neutrality, this has a clean characterization:

**Local sufficient condition (fast):** `c` is sieged by P iff `opp_neighbors(c) >= own_neighbors(c) + 1` under neutrality → opponent cannot ever place there. This is necessary but not sufficient for full siege (doesn't account for reachability through other cells).

**Global condition (accurate, slower):** Flood-fill from each opponent stone through empty cells where neutrality allows opponent entry. Any empty cell NOT reached is sieged-by-P.

Implementation:
```
def sieged_by(state, P):
    opp = P.other()
    reachable = set(all opp stones)   # opponent currently occupies these
    frontier = set(all opp stones)
    while frontier:
        new_frontier = set()
        for u in frontier:
            for v in topology.neighbors(u):
                if v in reachable or not topology.is_on_board(v):
                    continue
                if state.board[v] != EMPTY:
                    continue
                # Can opp legally place at v, given current neighbor counts?
                own_n = count neighbors of v with color opp      # opp's own
                foe_n = count neighbors of v with color P        # P's stones (foe from opp's POV)
                if neutrality_enabled and own_n < foe_n:
                    continue
                if strict_adjacency_enabled and own_n == 0 and opp_has_stones:
                    continue
                reachable.add(v)
                new_frontier.add(v)
        frontier = new_frontier
    return {empty cells on board} - reachable
```

**Subtle point:** this computes "cells opponent can reach **right now**," not "cells opponent can ever reach after arbitrary play." The real definition is game-tree-deep; this approximation is sound for the neutrality + strict-adjacency rule because those rules are monotone — if opponent can't reach `c` now, adding more P stones nearby only makes it harder, never easier. So the approximation is an **under-estimate** of own sieged cells (safe: we never claim a cell as sieged when opponent could reach it). It can miss cells that become sieged only after P plays specific stones. That's fine for a heuristic eval.

**Complexity:** `O(V + E)` per call. V ≈ 600 at R=10, so essentially free.

### 4.5 Weight tuning

Initial tuning: hand-pick values, run `greedy_A vs greedy_B` with different weights, measure win rate. Pick the strongest variant as the canonical greedy baseline. This is also a mini-experiment in rule-balance: if no set of weights produces interesting games, the rules are broken.

Later: optionally run CMA-ES or grid search over the 4 weights, using win rate vs. a fixed reference opponent as the fitness.

---

## 5. Training pipeline

### 5.1 Stages

```
Stage 0: RandomBot (trivial, for sanity tests and unit tests)
    ↓
Stage 1: GreedyBot with tuned weights
    ↓
Stage 2: PPO + self-play vs. snapshots, bootstrapped against GreedyBot
    ↓
Stage 3 (optional): AlphaZero-style MCTS + self-play, seeded from Stage 2 weights
```

Each stage must be complete and evaluated before moving to the next.

### 5.2 Stage 2: PPO + self-play details

Follows the "SIMPLE" pattern (Foster, 2021) and PettingZoo's AgileRL self-play tutorial.

**Self-play wrapper:** converts the 2-player env into a 1-player env for PPO. The PPO agent plays one side (e.g. Black); the other side is sampled from the network bank on each reset.

**Network bank:** stores snapshots of the PPO agent at regular intervals (e.g. every 100 training iterations). On each env reset, the wrapper samples a random snapshot as the opponent, with probability:
- 50% — current agent (pure self-play)
- 30% — random past snapshot (anti-forgetting)
- 20% — GreedyBot (anti-catastrophic-loss-of-basic-competence)

The 20% GreedyBot mix prevents the "network plays weird strategies that only work against other networks" failure mode documented in Tablut. GreedyBot is a deterministic, understandable baseline that enforces basic competence.

**Color swapping:** on each reset, randomize whether PPO plays Black or White. Otherwise the network learns one side specifically, which is wasteful.

**Data augmentation:** the board has 6-fold rotational + reflection symmetry (12 symmetries total). Exploit by augmenting each observation with a random rotation/reflection from D₆ before passing to the network. Critical for sample efficiency on small boards; mostly-critical at R=10. (AlphaZero used this for Go.)

**Hyperparameters (starting point):**
- PPO from Stable-Baselines3 or CleanRL
- `lr = 3e-4` (standard)
- `gamma = 1.0` (full credit to terminal reward — no discounting needed in finite games)
- `gae_lambda = 0.95`
- `clip_range = 0.2`
- `entropy_coef = 0.01`, annealed to 0.001 over training (prevents policy collapse early, allows exploitation late)
- `n_steps = 2048` per rollout
- `batch_size = 256`
- `n_epochs = 4`
- `target_kl = 0.02` (safeguard against naive-masking KL explosion described by Huang)

**Anti-plateau measures:**
- Monitor win rate vs. GreedyBot every N iterations. If stalled, reduce lr or add entropy.
- Monitor policy entropy. If it collapses to near-zero early, increase entropy_coef.
- Keep the replay of past checkpoints larger rather than smaller (16+ past iterations as in Tablut reproduction) to stabilize against catastrophic forgetting.

**Evaluation:** run a tournament among the network bank + GreedyBot every N iterations, producing an Elo ladder. Progress = monotonic Elo growth.

### 5.3 Committed rule set

Training targets **one** rule configuration:

```
board_radius          = 10           # to be confirmed during prototyping
mirror_adjacency      = True
strict_adjacency_rule = True
neutrality_rule       = True
partial_credit_k      = 0            # off
```

Reasons for committing to this single variant rather than training across all 16 combinations of the 4 experimental flags:

1. **Not every combination is a playable game.** The baseline (no rules enabled) produces 0-0 draws by mutual blocking — already demonstrated. Some other combinations are likely similarly degenerate, and training on degenerate rules is wasted compute.

2. **Rule changes alter what "winning" means, so learned value is rule-specific.** The value head approximates win probability given the scoring function. Changing `partial_credit_k` or any rule that changes which positions score means the network's entire learned eval is invalidated. Cross-rule transfer isn't a realistic option; we'd need to retrain from scratch per variant.

3. **Strict adjacency is a training-feasibility requirement, not just a game design choice.** Without it, legal moves at turn 1 on a radius-10 board is ~600 and stays large for many turns. The policy distribution has to spread across hundreds of uniformly-plausible actions, which is slow to learn and produces weak gradients. With strict adjacency, legal moves start small (1-3) and grow with the player's cluster, giving the policy a naturally-concentrated distribution to learn. Action-space sparsity is what makes PPO tractable on this board size without game-specific architectural tricks.

4. **Committing to one rule set sharpens the research question.** Instead of "which of 16 variants is best," we ask "is this specific rule set a good game, and what strategies emerge?" That's a more tractable and more meaningful question given the playtesting evidence so far. If the committed variant turns out to be flawed (forced wins, degenerate equilibria, etc.), we retrain on an adjusted variant — but we don't pre-train on variants we have reason to believe are bad.

**Implication for Phase 2 (PBT):** the population's reward-shaping diversity (section 5.5) becomes even more important under a single rule set. Without cross-rule variation to explore, internal strategy diversity is the only mechanism producing a non-trivial research output.

**If Phase 1 reveals the committed rules are broken:** retrain from scratch on an adjusted variant. Budget one such retraining cycle (~8h on 4 GPUs) into the plan. Variants most likely to be tried as fallbacks, in order: (a) reduce to radius-7 if convergence is too slow; (b) add `partial_credit_k=3` if cycle-building proves too hard even for trained agents; (c) drop neutrality if the structural siege dynamic proves pathological in practice.

### 5.4 Compute budget estimate

- R=10 board, ~50 moves per game, self-play at ~100 games/sec on 4 GPUs with vectorized envs
- 1 training iteration = 2048 steps ≈ 40 games, ~0.5 sec
- 1000 iterations ≈ 40,000 games, ~10 minutes
- Full Phase 1 run on committed rule set: 10k iterations ≈ 100 minutes
- Phase 2 PBT (K=4 agents) on same rule set: ~7 hours
- Budget one full retraining cycle in case Phase 1 reveals the committed rules are broken: add ~8h contingency

Total realistic compute: ~15h on 4 GPUs for a research-grade output on the committed rule set.

These are rough estimates; real numbers come after the first prototype run.

---

## 5.5 Population-Based Training (evolutionary extension)

### Motivation

Section 5.2 self-play has three known failure modes for games like Cycle Control:

1. **Strategy collapse** — all self-play runs converge to the same local optimum, missing alternative strategies entirely. A solo agent that learns "rush cycles" will never discover "build sieges" because once the first strategy works well enough, the gradient is too small to escape.
2. **Non-transitive dominance cycles** — if rock-paper-scissors-like dynamics exist between strategies (siege-builder beats cycle-rusher, cycle-rusher beats passive defender, passive defender beats siege-builder), vanilla self-play oscillates instead of converging. This is empirically documented as the primary failure mode of naive self-play in non-transitive games.
3. **Hyperparameter fragility** — PPO's behavior is famously sensitive to entropy coefficient, learning rate, and clip range. One bad setting per rule variant and we draw wrong conclusions about the rules.

All three of these are directly addressed by **Population-Based Training (PBT)** (Jaderberg et al. 2017), used in AlphaStar and FTW for Quake III. This is the formal name for the "split training multiple times, each learns different things, tournament, repeat" pattern described in the user's question. AgileRL provides a production implementation; we can build a lighter version tailored to our needs.

### Core PBT loop

```
Initialize a population of K policies (typically K=4 to K=8), each with:
    - Same network architecture
    - Same observation/action spaces
    - Different initialization seed
    - Different hyperparameters sampled from a perturbation schedule
    - (Optional) Different reward-shaping mix (see "strategy-diversity" variant below)

Training loop (outer):
    for each generation in 1..G:
        # Exploit phase
        for each agent A in population:
            train A with PPO for N_inner steps, opponents sampled from population

        # Evaluation phase
        run round-robin tournament among population (and optional fixed refs: Greedy, Random)
        compute fitness = win rate or Elo

        # Exploit / explore phase
        rank population by fitness
        for each low-ranked agent (bottom 25%):
            - copy weights from a random top-ranked agent (exploit)
            - perturb hyperparameters (explore): multiply lr, entropy_coef, etc. by random factor in [0.8, 1.25]
            - (if strategy-diversity variant) perturb reward-shaping weights similarly
```

Generation length (N_inner) must be long enough for an agent to express its current strategy — probably 500-2000 PPO iterations per generation. Total generations G = 20-50.

### What specifically to diversify

Three axes, in order of expected payoff:

1. **Reward-shaping weights (most important for this game).** All agents maximize terminal win, but internally their value head is trained on shaped reward during training. Different agents in the population use different shaping weights — some value current cycles more, some value sieged territory more, some value mobility more. The shaping doesn't affect terminal judgment — win rate is always the fitness signal. The population naturally explores strategy space because agents with shaping misaligned with the true game lose the tournament.

    Starting distribution per agent:
    ```
    agent 0: w_cycle=3.0, w_siege=0.5, w_comp=0.1, w_mobility=0.05  # greedy-cycle
    agent 1: w_cycle=1.0, w_siege=2.0, w_comp=0.3, w_mobility=0.1   # territory-focused
    agent 2: w_cycle=2.0, w_siege=1.5, w_comp=0.2, w_mobility=0.1   # balanced
    agent 3: w_cycle=0.0, w_siege=3.0, w_comp=0.0, w_mobility=0.0   # pure-siege
    ...
    ```

    After each generation, losing agents copy weights from winners AND get their shaping weights perturbed. Over time, the population converges to shaping that produces winning strategies — revealing WHICH internal priorities correspond to winning play.

2. **Standard PPO hyperparameters.** Learning rate, entropy coefficient, clip range, GAE lambda. Standard PBT practice.

3. **Network seed / init.** Cheapest to vary; provides a floor of diversity even if other axes are fixed.

### Strategy-diversity variant

For our specific research question — "what rule set produces interesting games?" — we don't want the population to collapse to a single dominant strategy too quickly. Two practical tweaks:

- **Diversity-preserving selection.** Instead of pure top-K selection, use fitness-sharing: an agent's effective fitness is reduced if many other agents in the population play similarly to it (measured e.g. by action-distribution distance on a fixed eval position set). This keeps the population from collapsing to a single strategy family.
- **Periodic niche re-seeding.** Every G/4 generations, re-initialize 1-2 agents with extreme shaping values (e.g. pure-siege, pure-cycle) to ensure these strategies keep getting tested throughout training. Based on the "late bloomer" correction in FIRE-PBT — some strategies take many generations before they become competitive.

### What PBT tells us about rule balance

The research output per rule variant becomes richer:

- **Final population fitness matrix** — who beats whom in the final population. A healthy rule set produces a population with non-trivial intransitivities (some matchups close to 50-50, indicating multiple viable strategies). A degenerate rule set produces a strict total ordering (one strategy dominates all others).
- **Which shaping weights survived** — tells us what internal priorities correspond to winning play under that rule set. E.g. if every surviving agent has `w_siege > w_cycle`, that's evidence siege-building is the dominant strategy under that rule set. If shapings are diverse across survivors, the rule set supports multiple viable styles.
- **Behavioral diversity score** — computed over the final population's games, how varied are typical game trajectories. Higher is better for "interesting game" evaluation.

These metrics are more informative than a single scalar Elo per rule variant.

### Compute cost

- Population size K = 4: ~4× the solo budget. For 5 rule variants: 4 × 8h = ~32h on 4 GPUs
- Population size K = 8: ~8× solo budget, ~64h
- In practice: start K=4 (feasible on 4 GPUs, one agent per GPU)

### When to switch on PBT

**Phase 1 (solo PPO per rule variant):** validates the pipeline works at all. 1 agent per rule variant, ~8h total. If no agent beats GreedyBot, fix the pipeline before adding complexity. If agents plateau with varied strategies already, PBT may be unnecessary.

**Phase 2 (PBT per rule variant):** once solo PPO validated. Produces the rule-balance research output.

**Skip PBT entirely if:** Phase 1 agents all converge to the same strategy AND one rule variant produces obviously balanced games — then solo is enough and PBT is over-engineering.

### Integration with the existing design

PBT is a **wrapper around Stage 2** (section 5.2). The self-play mechanism, action masking, network architecture, and evaluation metrics are all unchanged. What changes:

- `ppo_selfplay.py` → `pbt_trainer.py`, which orchestrates K PPO learners
- Each PPO learner has its own `SelfPlayWrapper` instance, but they all share the **population** as their opponent pool instead of each maintaining a solo checkpoint bank
- `tournament.py` runs periodically to compute fitness AND to drive opponent sampling

New file: `cycle_control/ai/pbt_trainer.py`.

### Confidence and risks

- **Confidence this helps:** ~70%. PBT has strong empirical support across many games; the failure modes it addresses clearly apply here. Main risk is compute budget — K=4 is tight on 4 GPUs.
- **Confidence in the "strategy-diversity" variant specifically:** ~55%. Reward-shaping diversity as the primary evolution axis is a natural fit for this research project, but less battle-tested than pure hyperparameter-PBT. If it doesn't work, fall back to standard PBT (hyperparameter perturbation only) + manual reward-shaping ablation studies.
- **Risk: evaluation overhead.** Round-robin tournaments among K=8 agents = 28 pairings × several games = significant compute. Mitigation: use short eval games (radius-5 instead of radius-10) or subsample pairings.
- **Do NOT use PBT as a substitute for fixing bugs.** If solo PPO fails mysteriously, diagnose before scaling up.

---

## 6. Evaluation

### 6.1 Elo ladder

Every checkpoint + GreedyBot + RandomBot form an Elo pool. Round-robin tournament every N iterations. Standard Elo update rule. Rating is the research output, not the reward.

### 6.2 Rule-balance metrics

Per rule variant, run 1000 games of the strongest model vs. itself and record:
- **Game length (moves until terminal):** too short = rules force early resolution; too long = low signal
- **Final score difference:** if 80%+ of games are decided by < 5 points the game is knife-edge; if 80%+ are decided by > 30 points the rules create runaway winners
- **Drawing rate:** very high draws = degenerate (both players can force a tie)
- **First-move (Black) win rate:** should be ~50-55%. If ~90%, Black has a forced win; if ~30%, White does.

These are the real outputs of the project.

---

## 7. Implementation plan (files)

Proposed module layout under `cycle_control/ai/`:

```
cycle_control/ai/
    __init__.py
    action_space.py    # action_index <-> (node | pass) mapping; legal_mask builder
    bot_interface.py   # Bot protocol; play_turn turn loop helper
    bots/
        random_bot.py
        greedy_bot.py
    siege.py           # sieged_by algorithm
    features.py        # observation encoder: state -> (C, H, W) tensor
    env.py             # PettingZoo AEC env: CycleControlEnv
    network.py         # ResNet trunk + policy + value heads (PyTorch)
    ppo_selfplay.py    # self-play wrapper + training loop
    elo.py             # Elo ladder bookkeeping
    tournament.py      # round-robin tournament runner
    cli.py             # CLI entry points (train, eval, play)
```

Plus a `/tests_ai/` directory for unit tests of each module.

### 7.1 Implementation order

1. `action_space.py` + `bot_interface.py` + `random_bot.py` — scaffolding, testable without RL
2. `siege.py` + unit tests vs. hand-crafted positions
3. `greedy_bot.py` using existing `scoring_nodes` + `siege.py`
4. Tournament runner: RandomBot vs. GreedyBot, confirm GreedyBot wins ~95%+
5. `features.py` + `env.py` — PettingZoo env on committed rule set; validate via PettingZoo's `api_test`
6. `network.py` — forward pass on dummy batch, shape-check
7. **Phase 1:** `ppo_selfplay.py` — train one solo agent on the committed rule set. Confirm it beats GreedyBot within reasonable budget.
8. Evaluate Phase 1 result. If agent strategy collapsed to a single style OR rule-balance questions remain unanswered, proceed to Phase 2. If the rules are revealed to be broken (forced win for one side, consistent degenerate outcomes), adjust rules and retrain.
9. **Phase 2 (conditional):** `pbt_trainer.py` — population-based training with reward-shaping diversity on the committed rule set.
10. Produce research output: strategy analysis, shaping-survival distribution, example games.

Do not skip step 4 or 7 — they are the "is this actually working" gates. Step 8 is the decision gate for Phase 2 vs. rule adjustment vs. ship.

Plus a `cycle_control/ai/pbt_trainer.py` added if we proceed to Phase 2.

---

## 8. Open questions / research risks

1. **Does the siege-interior channel help or does it shortcut learning?** If the greedy bot's eval is already mostly captured by the observation channels, PPO might just learn to mimic greedy and not discover better strategy. Ablation: train once with channels 7-8, once without; compare Elo at convergence.

2. **Is mirror adjacency too big a state-space change for the same architecture?** The neighbor graph changes structurally. The observation encoding is the same, but the conv kernels that implicitly learn spatial patterns may need re-learning. Likely fine; worth monitoring.

3. **Will the bots discover the siege-building strategy, or overfit to cycle-making?** This is the core research question. If trained bots play greedy-cycle-style and lose to handcrafted siege strategies, the architecture is missing something. Test: have humans (you) play vs. trained bots, check if humans can beat them with siege tactics.

4. **Is catastrophic forgetting going to be a problem?** Tablut reproduction saw role-imbalance even on a symmetric game. Anti-forgetting mitigations (large past-checkpoint pool, GreedyBot mix-in) are prophylactic.

5. **Partial vs. full action space exploration:** on R=10 there are ~600 actions but under strict adjacency + neutrality mid-game, often only 10-30 are legal. The policy must learn not to distribute probability uniformly over the mask; entropy regularization needs careful tuning to keep exploration without overcommitting.

6. **Impartial-game-like parity issues?** Cycle Control is partisan (distinct colors), but cycles have a parity property (girth 6 with side-only adjacency, girth 4 with mirror). If the network struggles with parity, it'll show up as slow cycle-completion learning. Mitigation: the sieged-interior and neighbor-count channels should help.

7. **Rule-balance research validity:** trained-model strength is a noisy signal for rule balance. The real question is whether the game, played by two strong agents, produces varied, non-degenerate outcomes. We don't know how strong "strong enough" is a priori. May need to play the trained model ourselves and judge qualitatively.

---

## 9. Summary

- **Committed rule set:** mirror adjacency + strict adjacency + neutrality, `partial_credit_k=0`, board radius TBD from prototyping. Chosen because: (a) the other 15 combinations include known-degenerate variants, (b) rule changes invalidate the value head, (c) strict adjacency is a training-feasibility requirement (action-space sparsity), and (d) single-variant commitment sharpens the research question.
- **Bot interface:** `choose_action(state, legal_mask, color) -> int`. Single placement per call. Engine drives the multi-placement turn loop. Pass is a dedicated action index.
- **Env:** PettingZoo AEC, `(26, H, W)` observation with siege-interior channels, `Discrete(N+1)` action space with mask.
- **Network:** ResNet trunk + policy head (masked softmax) + value head (tanh). ~500K params.
- **Training Phase 1:** Stage 0 Random → Stage 1 Greedy (with hand-tuned eval) → Stage 2 solo PPO self-play vs. past snapshots + 20% Greedy mix.
- **Training Phase 2 (conditional):** Population-Based Training — K=4 agents with different reward-shaping weights, round-robin tournaments drive exploit/explore cycles. Addresses non-transitive strategy cycles and produces strategy-diversity research output.
- **Key pre-RL engineering work:** sieged-area detector (section 4.4). Reused by greedy eval, observation encoder, and rule-balance metrics. Build and unit-test it first.
- **Research output:** is the committed rule set a good game, what strategies emerge, what shaping priorities correspond to winning play.

**Implementation-readiness confidence: ~85%.** Main uncertainty is section 4.4 (siege detector correctness) and whether the committed rule set produces a well-defined learning target (we assume it does based on playtesting). PBT is opt-in based on Phase 1 results.
