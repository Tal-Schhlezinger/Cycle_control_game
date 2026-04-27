# Cycle Control — Potential Optimizations Book

Generated from the uploaded `cycle_control_game_1.zip`.

## 0. Scope

This document analyzes potential speed optimizations in the current Cycle Control codebase.

The focus is **quality-preserving optimization**: same rules, same legal move set, same scoring, same search result for the same search depth and evaluation.  
When an idea changes behavior or uses an approximation, I mark it clearly.

I use the following symbols:

| Symbol | Meaning |
|---|---|
| `V` | number of board nodes |
| `E` | number of board edges |
| `Δ` | max degree; currently at most `3` side-only, `6` with mirror adjacency |
| `L` | number of currently legal placement moves |
| `H` | move-history length |
| `b` | search branching factor |
| `d` | search depth |

Because `Δ` is bounded, many graph operations that look like `O(E)` are effectively `O(V)` on this board family. That said, Python constant factors matter heavily here because the AI repeatedly scans the board.

---

## 1. Online research summary

The external research supports the direction already implied by the profile:

1. **Bridge-based cycle scoring is theoretically good.** Tarjan-style bridge detection is already linear, `O(V + E)`, so the algorithmic class is not the problem. The problem is repeated recomputation from scratch.
2. **Alpha-beta is exact, not approximate.** It returns the same minimax value as full minimax at the same depth, while pruning branches that cannot affect the result.
3. **Move ordering matters a lot for alpha-beta.** Good ordering increases cutoffs. The most relevant exact upgrades are principal-variation move reuse, hash/transposition move first, killer moves, and history heuristic.
4. **Transposition tables are a major exact search optimization.** They avoid re-searching repeated board positions. For this game, transpositions should be common because a two-placement turn can often reach the same final board by placing the two cells in the opposite order.
5. **Bitboard / bitset representations are a natural fit.** The board is fixed-size and graph-based. Per-player occupancy bitsets, empty bitsets, and precomputed neighbor masks would replace many Python `dict`, `set`, and tuple operations with integer bit operations.

References used are listed at the end.

---

## 2. Current bottleneck profile

I profiled one `Greedy1` vs `Greedy2` game at committed `R=3` and one `SearchBot(depth=2)` vs `Greedy2` game at committed `R=2`.

### Greedy game profile, committed R=3

The dominant cost was:

| Hot path | Meaning |
|---|---|
| `GreedyBot.choose_action` | called once per bot placement; evaluates many candidate moves |
| `GreedyBot.evaluate` | recomputes features from scratch per candidate |
| `MoveEngine.legal_moves` → `is_legal_placement` | repeated full-board scans |
| `mobility_for` | calls legal move scans for active player |
| `scoring_nodes` → `_find_bridges` | recomputes cycle score from scratch |
| `frontier_count` | recomputes frontier from scratch |

### Search game profile, committed R=2

The dominant cost was:

| Hot path | Meaning |
|---|---|
| `SearchBot._root` / `_minimax` | expected search tree cost |
| `SearchBot._leaf_eval` | called many times |
| `mobility_for` | still calls `legal_moves`, dominating leaf evaluation |
| `SearchState.apply` | good direction, but still calls full engine placement logic |
| `MoveEngine.legal_moves` → `is_legal_placement` | repeated full-board scans |

### Main conclusion

The current code already has one correct idea: **search should mutate and undo instead of cloning.**

The same idea should be pushed further:

> Do not copy whole game states inside hot AI loops. Apply the move, remember exactly what changed, evaluate, then undo.

This is the single most important theme.

---

## 3. Highest-priority optimization roadmap

### P0 — correctness fixes before speed claims

#### 3.1 Replace lazy `end_on_no_legal_moves`

Current `MoveEngine._check_end_conditions` only checks no-legal-moves when the board is nearly full. That is a speed shortcut, but it is not logically safe under strict adjacency + neutrality. The game can theoretically become stuck while many cells remain empty.

**Exact replacement:**

Maintain exact legal counts or legal sets per player incrementally.

On every placement, only the placed node and its neighbors can change local legality because legality depends on occupancy and neighbor counts. With mirror adjacency, this is at most `1 + Δ ≤ 7` directly affected cells for placement occupancy, plus strict-adjacency “first own stone” transition after a player’s first stone.

Expected result:

| Current | Exact optimized |
|---|---|
| Often skips check, can be wrong | Always correct |
| Full check: `O(V·Δ)` per player | Usually `O(Δ²)` local update |
| Fast but unsafe | Fast and safe |

#### 3.2 Fix committed-rules constructors in analysis scripts

`greedy_validation.make_engine()` and `search_analysis.make_committed_engine()` manually set some committed options but do not use `RulesConfig.committed()`. This risks benchmarking the wrong game. This is not a speed issue, but optimization data is useless if the rules differ from the target ruleset.

**Exact fix:** use `RulesConfig.committed(board_radius=radius)` everywhere that says “committed.”

---

## 4. Architecture-level optimization strategy

### 4.1 Add indexed topology

The project already has `ActionSpace`, but indexing should become a core topology feature.

Add to `BoardTopology`:

```python
self.node_to_idx: dict[Node, int]
self.idx_to_node: tuple[Node, ...]
self.neighbor_indices: tuple[tuple[int, ...], ...]
self.neighbor_mask: tuple[int, ...]  # bit i set if neighbor
```

Then hot logic can use integer indices instead of tuple nodes.

Benefits:

| Area | Current | Optimized |
|---|---:|---:|
| Node lookup | dict/set of tuples | array index / integer |
| Neighbor iteration | tuple nodes + dict board lookup | tuple ints + list/array lookup |
| Occupancy | dict `Node -> NodeState` | `black_bits`, `white_bits`, `empty_bits` |
| Frontier | scan all nodes | bit operations over neighbor masks |
| Action mask | scan legal moves + node lookup | direct boolean/int bit mask |

This is exact. It changes representation, not rules.

### 4.2 Add a fast board/state layer

Keep `GameState` for clarity and persistence, but add either:

1. `FastGameState` used by AI/search, or
2. extra cached fields inside `GameState`.

Recommended fields:

```python
black_bits: int
white_bits: int
empty_bits: int
occupied_count: int
stone_count: dict[Player, int]
frontier_bits: dict[Player, int]
legal_bits: dict[Player, int]
zobrist_hash: int
```

If you want minimum disruption, keep canonical `state.board` and build `FastState` from it for bots. If you want maximum speed, make the indexed state canonical and derive the dict view only for UI/persistence.

### 4.3 Delta-based apply/undo everywhere hot

Search already has `SearchDelta`. Greedy should use the same model.

Current Greedy:

```text
for each candidate:
    clone full GameState
    apply move
    recompute features
```

Better Greedy:

```text
ss = SearchState(engine, state)
for each candidate:
    delta = ss.apply(node)
    score = evaluate_incremental(ss)
    ss.undo(delta)
```

This preserves exact board behavior if `ss.apply()` calls the same legality/turn logic or a verified equivalent.

### 4.4 Incremental legal move sets

Legal placement depends on:

- whether node is empty
- supply
- own neighbor count
- opponent neighbor count
- whether player has any stone yet
- active player only for supply/turn context

For spatial legality, after placing at node `p`, only empty cells adjacent to `p` can have neighbor counts changed. The placed node itself also becomes illegal.

So legal sets should be maintained as bitsets:

```python
legal_bits[player] = all currently legal placement cells for player
```

After placement:

1. remove placed bit from both legal sets
2. update candidate legality for each empty neighbor of placed node
3. if this was player’s first stone, recompute strict-adjacency legality for that player once

Expected improvement:

| Current `legal_moves` | Incremental legal bits |
|---|---|
| `O(V·Δ)` scan | `O(1)` to return count if stored |
| list construction `O(L)` | bit iteration `O(L)` only when list needed |
| repeated in AI leaf eval | cheap count / mask |

This is exact.

### 4.5 Incremental scoring: choose carefully

The current scoring algorithm is already linear per call. Full dynamic bridge maintenance is complicated and not worth implementing first.

Recommended staged plan:

1. **Stage A:** precompute indexed adjacency and avoid tuple/frozenset allocation in `_find_bridges`.
2. **Stage B:** score both players in one evaluation pass when possible, avoid duplicate `scoring_nodes` calls.
3. **Stage C:** cache scoring results by Zobrist hash for search states.
4. **Stage D:** local dirty-component rescoring. When a player places one stone, only that player’s connected component around the placed stone can change. Recompute scoring only for that component.

Stage D is exact and much simpler than fully dynamic bridge algorithms.

---

# 5. File-by-file function analysis

## 5.1 `cycle_control/topology.py`

### Overall

Topology is mostly construction-time. It is not the main runtime bottleneck, but it should own the indexed graph representation because every hot module needs it.

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `BoardTopology.__init__` | `O(V + E + sanity)` | Yes | Build `node_to_idx`, `idx_to_node`, `neighbor_indices`, and `neighbor_mask` once. This increases construction cost slightly but speeds all gameplay/search. |
| `_vertex_in_hex` | `O(1)` | No meaningful | Already trivial. |
| `_triangle_in_hex` | `O(1)` | Minor | Inline vertex checks if topology construction ever matters. Not urgent. |
| `_compute_nodes` | `O(R²)` = `O(V)` | Minor | Could derive exact coordinate ranges tighter, but current construction is fine. |
| `_compute_neighbors` | `O(V·Δ)` | Yes | Build both node-neighbors and index-neighbors here. Store neighbor bitmasks. |
| `_sanity_check` | `O(V·Δ + bounded BFS)` | Minor | Gate behind debug flag for production/tournament runs. Keep enabled in tests. |
| `_shortest_cycle_up_to` | roughly `O(V·Δ^depth)` bounded | Minor | Since it is construction-only, not urgent. Could early-exit immediately when a violating cycle is found. |
| `_compute_girth` | bounded BFS | Minor | Not used in hot path. Cache result if exposed. |
| `iterate_nodes` | `O(V)` when consumed | Yes indirectly | Prefer `all_nodes()` or indexed arrays in hot loops to avoid generator overhead. |
| `get_neighbors` | `O(1)` | Yes indirectly | Add `get_neighbor_indices(idx)` for hot loops. |
| `is_on_board` | `O(1)` but called often | Yes | In hot paths, validate by index or by direct `node in _node_set` after caller type safety. For internal engine calls generated from topology, skip full tuple validation. |
| `all_nodes` | `O(1)` | No meaningful | Good. |
| `node_count` | `O(1)` | No meaningful | Good. |
| `__repr__` | `O(1)` | No | Irrelevant. |

### Specific recommendation

`is_on_board()` appears in profiles because `is_legal_placement()` calls it for every candidate. Split legality into two methods:

```python
is_legal_placement_external(state, node)  # validates arbitrary user input
is_legal_placement_idx(state, idx)        # hot internal path, assumes valid idx
```

The UI/sandbox can use the external path. Bots/search should use the indexed path.

---

## 5.2 `cycle_control/state.py`

### Overall

`GameState.clone()` is convenient but becomes expensive when used inside AI loops. The state also lacks cached counts/sets, forcing scans.

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `Player.other` | `O(1)` | Minor | Could use a constant dict, but current enum comparison is fine. |
| `NodeState.from_player` | `O(1)` | Minor | Fine. In hot indexed code, replace enum states with small ints. |
| `PlacementEntry.to_dict` | `O(1)` | No | Persistence only. |
| `PlacementEntry.from_dict` | `O(1)` | No | Persistence only. |
| `PassEntry.to_dict` | `O(1)` | No | Persistence only. |
| `PassEntry.from_dict` | `O(1)` | No | Persistence only. |
| `history_entry_from_dict` | `O(1)` | No | Persistence only. |
| `GameState.clone` | `O(V + H)` | Yes, high impact | Do not use in hot AI loops. Replace with delta apply/undo. If cloning remains needed, optionally omit history/redo for simulation clones. |
| `GameState.move_count` | `O(1)` | No | Good. |

### Specific recommendation

Add a lightweight clone mode:

```python
def clone_for_search(self) -> GameState:
    return GameState(
        board=dict(self.board),
        active_player=self.active_player,
        turn_phase=self.turn_phase,
        stones_remaining=dict(self.stones_remaining),
        consecutive_pass_count=self.consecutive_pass_count,
        current_turn=self.current_turn,
        move_history=[],
        redo_stack=[],
        game_over=self.game_over,
        winner=self.winner,
    )
```

But the better solution is still delta apply/undo.

---

## 5.3 `cycle_control/rules.py`

### Overall

Rules are not a runtime bottleneck.

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `RulesConfig.__post_init__` | validation cost | No | Construction only. |
| `RulesConfig.committed` | `O(1)` | No | Use this more, especially in scripts. |
| `supply_enabled` | `O(1)` | Minor | Fine. |
| `enabled_end_conditions` | `O(1)` | Minor | Could return tuple, not important. |
| `_validate` | `O(1)` | No | Construction only. Remove dead/suspicious final check if cleaning code. |
| `to_dict` | `O(1)` | No | Persistence only. |
| `from_dict` | `O(1)` | No | Persistence only. |

### Specific recommendation

No performance work needed here. The important thing is **consistency**: use `RulesConfig.committed()` in all analysis scripts instead of manually reconstructing the committed ruleset.

---

## 5.4 `cycle_control/engine.py`

### Overall

This is a hot file. The biggest issue is full-board rescanning.

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `MoveEngine.__init__` | `O(1)` | No | Good. |
| `initial_state` | `O(V)` | Yes | Also initialize cached empty count, player stone counts, occupancy bitsets, legal bitsets if added. |
| `is_legal_placement` | `O(Δ + V)` worst due `has_own_stones` scan | Yes, very high | Store `stone_count[player]`. Then strict-adjacency test becomes `O(Δ)`. Add indexed version to skip `is_on_board`. |
| `legal_moves` | `O(V·(Δ+V))` worst currently; practically `O(V·Δ + V·own-scan)` | Yes, very high | Maintain `legal_bits[player]`. Return list by iterating set bits only. |
| `can_pass` | `O(1)` | No | Good. |
| `apply_placement` | `O(legality + end check + turn advance)` | Yes | Use indexed legality; update cached board data and legal sets locally; avoid global end scans. |
| `apply_pass` | `O(end check)` | Minor | Fine unless end check remains expensive. |
| `_placements_this_turn` | `O(1)` | No | Good. |
| `_advance_after_placement` | `O(turn phase)` | Minor | Good. |
| `_end_turn` | includes `_compute_turn_phase_at_start` | Yes | If cached `empty_count` exists, `_compute_turn_phase_at_start` becomes `O(1)`. |
| `_compute_turn_phase_at_start` | `O(V)` due empty scan | Yes | Maintain `empty_count`. |
| `_check_end_conditions` | `O(V)` board full + possibly `O(V·legality)` legal checks | Yes, critical | Maintain `empty_count`, `stones_remaining`, and exact `legal_count[player]`. Also remove unsafe lazy no-legal shortcut. |
| `_count_legal_moves_for` | `O(V·legality)` and mutates active player temporarily | Yes | Return cached `legal_count[player]`. Avoid mutating `state.active_player`. |
| `_determine_winner` | `O(V+E)` per player | Yes | Use cached scoring if available; at least score both players with indexed adjacency. |
| `can_undo` | `O(1)` | No | Good. |
| `can_redo` | `O(1)` | No | Good. |
| `undo` | `O(H·move_cost)` replay | Yes | For UI this is acceptable. For performance, store inverse deltas in history and undo in `O(changed fields)`. |
| `redo` | one apply | Minor | If undo becomes delta-based, redo should also use stored deltas or replay one move. |
| `sandbox_place` | `O(1)` currently but would break caches | Yes if caches added | Update all cached board fields/legal/scoring invalidation. |
| `sandbox_remove` | `O(1)` currently but would break caches | Yes if caches added | Same as sandbox place. |

### Most important engine refactor

Add a lower-level exact primitive:

```python
_apply_placement_unchecked_idx(state, idx) -> Delta
_undo_delta(state, delta) -> None
```

Then public `apply_placement()` becomes:

```python
idx = topology.node_to_idx[node]
if not is_legal_placement_idx(...): raise
_apply_placement_unchecked_idx(...)
```

Search and greedy can use the indexed path after the legal mask already proves legality.

---

## 5.5 `cycle_control/scoring.py`

### Overall

The chosen algorithm is reasonable. The issue is not Tarjan; the issue is calling it repeatedly and using expensive Python objects.

| Function | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `_find_bridges` | `O(Vp + Ep)` for player's induced graph | Yes | Use integer indices and encode undirected edge as `(min_idx, max_idx)` integer pair or packed int, not `frozenset`. Avoid dicts where arrays work. |
| `scoring_nodes` | `O(V + Ep)` | Yes, high | Avoid scanning all topology nodes by storing per-player occupied set/bitset. Cache by `(board_hash, player, partial_credit_k)`. For local updates, recompute only affected connected component. |
| `score` | same as `scoring_nodes` | Yes | If only count needed, avoid materializing full `set[Node]`. Add `score_count(...)` returning int directly. |

### Exact local scoring strategy

When player places node `p`:

1. Find the connected component of player stones containing `p`.
2. Recompute bridge/cycle scoring only inside that component.
3. Remove old scoring contribution for dirty component, add new one.

Why this is exact:

- Opponent scoring cannot change when this player places a stone, because opponent induced subgraph is unchanged.
- Other components of this player cannot change, because a new stone only connects components adjacent to `p`.

Worst case is still `O(V + E)`, but normal case is much smaller.

### Avoid `frozenset` bridges

Current bridge set stores each bridge as `frozenset((u, v))`. That allocates many objects.

With indexed nodes:

```python
edge_id = min(u, v) * V + max(u, v)
```

Then bridge membership is integer-set lookup.

---

## 5.6 `cycle_control/persistence.py`

### Overall

Not hot unless saving/loading huge experiments.

| Function | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `_node_to_str` | `O(1)` | Minor | Fine. |
| `_str_to_node` | `O(1)` | Minor | Fine. Validate length if hardening. |
| `_winner_to_json` | `O(1)` | No | Good. |
| `_winner_from_json` | `O(1)` | No | Good. |
| `serialize_state` | `O(V + H)` | Minor | For large experiment logs, serialize compact indexed board arrays or only move history. |
| `deserialize_state` | `O(V + H)` | Minor | If adding fast state, rebuild caches here. |
| `save_to_file` | `O(serialized size)` | Minor | Use compact separators for large batch files. |
| `load_from_file` | `O(file size)` | Minor | Good. |

### Recommendation

Keep human-readable saves for UI. For tournaments, add a separate compact experiment log format instead of optimizing this one too much.

---

## 5.7 `cycle_control/debug.py`

| Function | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `connected_components` | `O(V + E)` | Minor | Use player occupancy bitset and indexed adjacency if debug is used frequently. |
| `debug_summary` | `O(V + E)` plus scoring | Minor | Cache scoring if called during UI redraw; otherwise fine. |

Debug helpers are not priority unless you call them during every frame or every AI node.

---

## 5.8 `cycle_control/ai_hooks.py`

| Function / class | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `legal_moves` | same as engine | Yes | Benefits automatically if engine legal cache is added. |
| `clone` | `O(V + H)` | Yes | Do not use in hot loops. Prefer delta apply/undo. |
| `apply_move` | apply cost | Yes | Benefits automatically from indexed apply. |
| `evaluate` | two scoring calls | Yes | Use cached score or a count-only scoring function. |
| `BotRNG.__init__` | `O(1)` | No | Good. |
| `BotRNG.seed_rng` | `O(1)` | No | Good. |
| `BotRNG.random` | `O(1)` | No | Good. |
| `BotRNG.choice` | `O(1)` average | No | Good. |

---

## 5.9 `cycle_control/ai/action_space.py`

### Overall

This file already has the right concept, but `build_legal_mask()` does not use the precomputed map inside `ActionSpace`.

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `pass_index` | `O(1)` | No | Good. |
| `action_space_size` | `O(1)` | No | Good. |
| `action_index_to_node` | `O(1)` after tuple lookup | Minor | Good. |
| `node_to_action_index` | `O(V)` linear search | Yes | Replace global function implementation with topology/action-space node map. |
| `build_legal_mask` | `O(legal_moves + L·V)` because it calls linear `node_to_action_index` | Yes, high | Use `ActionSpace._node_to_idx`, or move mapping into topology. |
| `ActionSpace.__init__` | `O(V)` | Good | Already precomputes map. |
| `ActionSpace.index_to_node` | `O(1)` | No | Good. |
| `ActionSpace.node_to_index` | `O(1)` | No | Good. |
| `ActionSpace.build_mask` | currently calls slow global function | Yes | Implement method directly using `self._node_to_idx`, or use engine legal bits. |

### Concrete safe fix

```python
def build_mask(self, engine, state):
    mask = np.zeros(self.size, dtype=bool)
    if state.game_over:
        return mask
    for node in engine.legal_moves(state):
        mask[self._node_to_idx[node]] = True
    if engine.can_pass(state):
        mask[self.pass_index] = True
    return mask
```

That is a small exact improvement immediately.

Better later: engine returns `legal_bits`, and mask construction becomes bit expansion.

---

## 5.10 `cycle_control/ai/bot_interface.py`

| Function | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `play_turn` | `O(actions per turn × mask/build/apply)` | Yes | Reuse `ActionSpace`; already does. Bigger win comes from faster mask and apply. |
| `auto_fill` | repeated `legal_moves`, can be `O(V²)` | Yes | Use legal bitset and repeatedly choose lowest set bit. Update incrementally. |
| `play_game` | full game driver | Yes | Avoid extra `engine.legal_moves(state)` before `play_turn`; build mask once and let `play_turn` use it, or expose `has_legal_moves` as cached count. |

### Problem in `play_game`

It calls `engine.legal_moves(state)` every turn, then `play_turn()` builds a legal mask again. This duplicates work.

Exact improvement:

```text
legal_mask = action_space.build_mask(...)
if no placement legal:
    stuck logic
else:
    play_turn_with_initial_mask(...)
```

Or simpler: add `engine.has_legal_moves(state)` using cached legal count.

---

## 5.11 `cycle_control/ai/siege.py`

### Overall

This file contains some of the most expensive non-search logic. It computes reachability by repeated full scans.

| Function | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `_can_player_place_at` | `O(Δ + V)` due own-stone scan | Yes | Pass `has_own_stones` as precomputed bool. Use neighbor counts and bitsets. |
| `reachable_empty_cells` | documented `O(V²)` | Yes, high | Replace repeated full-board scans with a queue/fixed-point frontier. When a cell becomes reachable, only its neighbors may become newly reachable. |
| `sieged_against` | `O(reachable + V)` | Yes | Use empty bitset minus reachable bitset. |
| `sieged_for` | two reachability calls | Yes | Compute both players reachability; cache per state hash. |
| `territory_score` | reachability cost | Yes | Return `bit_count(reachable_bits)`. |
| `exclusive_territory` | two reachability calls | Yes | Bitset subtraction + `bit_count`. |
| `frontier_count` | `O(V·Δ)` | Yes | Maintain frontier bits incrementally, already partly done in `SearchState`. |

### Exact queue-based reachability

Instead of:

```text
repeat:
    scan every empty cell
```

Use:

```text
reachable = initial legal cells
queue = reachable cells
while queue:
    u = queue.pop()
    for each empty neighbor/nearby candidate v:
        if v not reachable and can_place_given_reachable(v):
            add v
```

Because the rules are monotone for adding own stones, this finds the same fixed point. It avoids repeatedly rescanning unrelated cells.

Worst case remains potentially `O(V·Δ·updates)`, but practical cost drops sharply.

---

## 5.12 `cycle_control/ai/bots/random_bot.py`

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `RandomBot.__init__` | `O(1)` | No | Good. |
| `RandomBot.reset` | `O(1)` | No | Good. |
| `RandomBot.choose_action` | `O(action_space_size)` due `np.flatnonzero` | Minor | If legal actions are stored as a list/bitset, choose directly. Current cost is acceptable. |
| `FrontierRandomBot.__init__` | `O(1)` but bug | Yes | Store `self._topology = topology`. Current constructor accepts topology but ignores it. |
| `FrontierRandomBot.reset` | `O(1)` | No | Good. |
| `FrontierRandomBot.choose_action` | `O(V + L·Δ)` | Yes | Store `ActionSpace` once. Use occupancy bitset/frontier legal intersection. |
| `FrontierRandomBot.attach_topology` | `O(1)` | Minor | If constructor stores topology, this becomes optional compatibility only. |

### Exact bug/performance fix

Current constructor comment says topology is accepted, but it never assigns `self._topology`. That means the bot degrades to uniform random unless `attach_topology()` is called.

Fix:

```python
self._topology = topology
self._action_space = ActionSpace(topology) if topology is not None else None
```

---

## 5.13 `cycle_control/ai/bots/greedy_bot.py`

### Overall

This is one of the largest optimization targets.

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `cycle_score_diff` | two full scoring calls | Yes | Cache scoring by state hash; compute dirty component only; score count directly. |
| `largest_component_size` | `O(V + E)` | Yes | Maintain DSU-like component data for monotone placements, or recompute only player's dirty merged component. |
| `component_size_diff` | two component scans | Yes | Use cached largest component per player. |
| `mobility_for` | active player exact `O(V·legality)`, non-active approximate `O(V+frontier)` | Yes | Maintain exact legal counts for both players. This improves speed and quality. |
| `mobility_diff` | two mobility calls | Yes | Use cached exact counts. |
| `territory_diff` | two territory closures | Yes | Cache reachability by hash or queue-based closure. |
| `exclusive_territory_diff` | very expensive | Yes | Use bitset reachability and cache. |
| `frontier_diff` | two full frontier scans | Yes | Use maintained `frontier_bits[player].bit_count()`. |
| `GreedyWeights.describe` | tiny | No | Good. |
| `GreedyBot.__init__` | `O(V)` action space | No | Good. |
| `GreedyBot.reset` | `O(1)` | No | Good. |
| `GreedyBot.evaluate` | weighted sum of expensive features | Yes | Evaluate from an `EvalCache` / `SearchState` with incremental features. Skip all zero-weight features already done well. |
| `GreedyBot.choose_action` | `O(L × (clone + apply + eval))` | Yes, very high | Replace clone-per-candidate with delta apply/undo. Reuse `SearchState`. Avoid list conversion for pass detection. |
| `Greedy1.__init__` | `O(V)` inherited | No | Good. |
| `Greedy2.__init__` | `O(V)` inherited | No | Good. |

### Biggest exact Greedy improvement

Current:

```python
trial = state.clone()
engine.apply_placement(trial, node)
score = self.evaluate(trial, color)
```

Replace with:

```python
ss = SearchState(engine, state)
for node in candidates:
    delta = ss.apply(node)
    score = self.evaluate_search_state(ss, color)
    ss.undo(delta)
```

This avoids `O(V+H)` clone cost per candidate.

### Quality warning

`mobility_for()` is currently exact only when `player == state.active_player`. For the non-active player, it uses frontier approximation. That is faster, but not the same quality under neutrality. If the goal is exact evaluation, replace it with cached exact legal counts.

---

## 5.14 `cycle_control/ai/search_utils.py`

### Overall

This file is directionally correct. It should become the central hot-path state mutation layer.

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `SearchDelta.__init__` | `O(1)` plus supply dict copy | Yes | Store only changed supply entry instead of full `dict`. Add zobrist/hash/legal/frontier deltas. |
| `SearchState.__init__` | `O(V·Δ)` frontier counts | Yes | If engine/state already maintains frontier bits, initialize in `O(1)`. |
| `SearchState.frontier_diff` | `O(1)` | Good | Keep. |
| `SearchState.apply` | local frontier update + engine apply | Yes | Avoid full `engine.apply_placement` overhead after legality is known. Use unchecked indexed apply. Also update legal bits, zobrist, score cache invalidation. |
| `SearchState.undo` | `O(1)` mostly | Yes | Restore extra cached fields. Avoid assigning full supply dict if changed field only. |
| `_MoveCtx.__init__` | `O(1)` | Minor | Fine. |
| `_MoveCtx.__enter__` | apply cost | Minor | Fine. |
| `_MoveCtx.__exit__` | undo cost | Minor | Fine. |
| `SearchState.move` | `O(1)` | Minor | Context manager overhead is nonzero; avoid inside deepest loops if profiling shows cost. |
| `_count_frontier` | `O(V·Δ)` | Yes | Use bitsets or cached frontier. |
| `apply_and_save` | apply cost | Yes | Prefer `SearchState.apply`; compatibility only. |
| `undo_placement` | `O(1)` mostly | Yes | Extend to cached fields or retire in favor of `SearchState.undo`. |

### Specific improvement

`SearchDelta.old_supply = dict(state.stones_remaining)` is small now, but it is still unnecessary work inside search. Store:

```python
old_supply_for_player: int | None
```

Only if supply is enabled.

---

## 5.15 `cycle_control/ai/bots/search_bot.py`

### Overall

The search bot has the right base algorithm, but it is missing the standard exact upgrades that make alpha-beta strong.

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `terminal_value` | `O(1)` | No | Good. |
| `SearchStats.describe` | `O(1)` | No | Good. |
| `SearchBot.__init__` | `O(V)` action space | Yes minor | Add transposition table, killer/history tables, zobrist keys. |
| `SearchBot.reset` | `O(1)` | Minor | Decide whether TT persists across games. Usually clear per game for deterministic experiments. |
| `_leaf_eval` | `O(V)` due opponent mobility | Yes, high | Use exact cached legal count / mobility. Then leaf eval becomes `O(1)`. |
| `choose_action` | root setup + iterative deepening | Yes | Current iterative deepening only used with time budget. Use it always for ordering. Reuse previous depth best move. |
| `_root` | `O(b^d)` reduced by alpha-beta | Yes | Use principal variation / TT best move first. Store root scores for next iteration. |
| `_minimax` | `O(b^d)` worst | Yes, very high | Add transposition table with zobrist hash, depth, bound type, best move. Add killer/history move ordering. Consider negamax simplification. |
| `_order` | currently applies every move and calls leaf eval | Yes | Use cheap static ordering first; apply/eval only for top moves or root. TT/PV move first. |
| `_order_nodes` | same as `_order` | Yes | Same. |
| `_time_up` | `O(1)` but called very often | Minor | Check time every N nodes, not every node, for lower overhead. Keep hard deadline safety. |

### Exact search upgrades

#### A. Transposition table

Key should include:

- board occupancy
- active player
- turn phase
- supply if relevant
- pass count if pass rules matter
- current turn only if rules/evaluation use it; currently probably not needed

Store:

```python
TTEntry = {
    "depth": depth,
    "value": value,
    "flag": EXACT | LOWER | UPPER,
    "best_move": idx,
}
```

This is exact if bound logic is implemented correctly.

#### B. Zobrist hashing

Maintain a 64-bit hash incrementally:

```text
hash ^= piece_square[player][idx] when placing/unplacing
hash ^= side_to_move when active player changes
hash ^= phase_key[phase] when phase changes
```

This makes transposition lookup `O(1)`.

#### C. Move ordering

Priority order for this game:

1. TT/PV move from previous search
2. immediate scoring-cycle creation moves
3. moves that reduce opponent legal count
4. moves increasing own exact legal count/frontier
5. killer moves at same depth
6. history heuristic

The exact result is unchanged. Only the search order changes.

#### D. Time check throttling

Current `_time_up()` is called frequently. A common exact optimization is:

```python
if (nodes_visited & 1023) == 0:
    check_time()
```

This slightly reduces responsiveness but not search correctness when no time budget is used. With a hard UI budget, use a small interval like 256 or 1024 nodes.

---

## 5.16 `cycle_control/ai/tournament.py`

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `MatchResult.total_games` | `O(1)` | No | Good. |
| `a_win_rate` | `O(1)` | No | Good. |
| `b_win_rate` | `O(1)` | No | Good. |
| `draw_rate` | `O(1)` | No | Good. |
| `summary` | `O(1)` | Minor | Guard division by zero if called before games. |
| `run_match` | `O(n_games × game_cost)` | Yes | Parallelize independent games. Also avoid reusing same engine if future engine gets mutable caches. |
| `RoundRobinResult.win_rate_matrix` | `O(n²)` | No | Fine. |
| `RoundRobinResult.pretty_print` | `O(n²)` | No | Fine. |
| `round_robin` | `O(n² × games)` | Yes | Parallelize pairings/games. Avoid duplicate symmetric pairings if color swapping already handles fairness. |
| `elo_update` | `O(1)` | No | Good. |
| `elo_from_round_robin` | `O(total games)` | Minor | Can aggregate repeated wins with formula loop still fine. Not a bottleneck. |

### Exact tournament speedup

Games are independent. Use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`.

Do not use threads for CPU-bound Python search because the GIL will limit benefit.

---

## 5.17 `cycle_control/ai/bots/__init__.py`, `cycle_control/ai/__init__.py`, `cycle_control/__init__.py`

These are package export files.

Optimization is irrelevant except for correctness/ergonomics:

- Ensure every intended public bot/helper is exported.
- Avoid importing heavy modules at package import time if startup ever matters.

---

## 5.18 `cycle_control/testrunner.py`

### Overall

This is not a gameplay bottleneck, but it repeatedly recomputes scoring/legal moves for assertions.

| Function | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `_winner_str` | `O(1)` | No | Good. |
| `dispatch` | command-dependent | Minor | For multiple scoring assertions in one test step group, cache scoring result per `(state_hash, player, k)`. |
| `run_test` | `O(commands × dispatch)` | Minor | Fine. Rebuild topology once per spec already. |
| `run_tests_from_file` | `O(number of tests)` | Minor | Could parallelize files/specs if suite grows. |
| `main` | CLI only | No | Good. |

Testing code should prioritize clarity over micro-optimization.

---

## 5.19 `ui.py`

### Overall

The UI redraws the full board each time. That is fine for small boards but will become slow with larger radii or frequent AI visualization.

| Function / method | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `triangle_to_pixels` | `O(1)` | Yes | Precompute triangle vertices for each node once per topology/radius. |
| `_point_in_triangle` | `O(1)` | Minor | Good. |
| `CycleControlUI.__init__` | startup | Minor | Precompute pixel geometry and node centers. |
| `_build_ui` | startup | No | Not hot. |
| `_redraw` | `O(V + scoring)` | Yes | Precompute geometry; update only changed cells; cache scoring sets unless board changed. |
| `_update_status` | scoring likely | Yes | Use cached score counts from engine/eval cache. |
| `_find_clicked_node` | `O(V)` hit test | Yes | Precompute bounding boxes; use spatial hash/grid; or store canvas item tags mapping item to node. |
| `_on_left_click` | event + apply + redraw | Minor | Benefits from faster clicked-node lookup and incremental redraw. |
| `_on_right_click` | event + redraw | Minor | Same. |
| `_on_pass` | apply + redraw | Minor | Same. |
| `_on_undo` | current engine undo replay may be `O(H)` | Yes | Delta undo would improve UI too. |
| `_on_redo` | one apply | Minor | Fine. |
| `_on_apply_modes` | rebuild engine | No | Not hot. |
| `_sync_mode_vars_from_rules` | tiny | No | Good. |
| `_on_restart` | `O(V)` initial state + redraw | Minor | Fine. |
| `_on_save` | persistence | No | User-driven. |
| `_on_load` | load + rebuild | Minor | Rebuild fast caches if added. |
| `main` | startup | No | Good. |

### UI-specific recommendation

Use canvas tags:

```python
canvas.create_polygon(..., tags=("cell", f"node:{idx}"))
```

Then click events can identify the clicked item directly instead of testing every triangle.

---

## 5.20 `greedy_validation.py`

| Function | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `make_engine` | small | Correctness first | Use `RulesConfig.committed(radius)`. Current version does not disable pass/end conditions like committed factory. |
| `run_validation` | tournament cost | Yes | Parallelize matches. Record final states only if needed. |
| `analyze_and_verdict` | `O(1)` | No | Good. |
| `main` | CLI | No | Good. |

This script should not be optimized before fixing the committed-rules mismatch.

---

## 5.21 `search_analysis.py`

| Function | Current cost | Can optimize? | Exact optimization |
|---|---:|---|---|
| `make_committed_engine` | small | Correctness first | Use `RulesConfig.committed(radius)`. |
| `run_analysis` | tournament/search cost | Yes | Parallelize independent games/pairings. Save raw results for reproducibility. |
| `analyze_findings` | small | Minor | Fine. |
| `main` | CLI orchestration | Minor | For multi-radius, parallelize per radius if CPU available. |

Again: first fix committed-rules construction.

---

## 5.22 `tests_ai.py` and `tests_builtin.py`

These are test suites, not runtime code.

Optimization priority:

1. Keep clarity.
2. Add performance regression tests for the hot paths.
3. Add correctness tests for every cache/delta optimization.

Recommended new tests:

| Test | Purpose |
|---|---|
| `test_indexed_legal_moves_matches_engine_scan` | proves cached legal set equals old legal scan |
| `test_delta_apply_undo_roundtrip_hash_and_board` | proves search mutation is reversible |
| `test_cached_score_matches_scoring_nodes_random_boards` | protects scoring cache/local scoring |
| `test_transposition_table_does_not_change_best_move` | compares SearchBot with/without TT at same depth |
| `test_no_legal_moves_detected_before_board_nearly_full` | catches current lazy-end-condition weakness |
| `test_action_space_build_mask_uses_precomputed_mapping` | prevents regression to linear node lookup |

---

# 6. Optimization plan by implementation phase

## Phase 1 — small safe wins

1. Fix `ActionSpace.build_mask()` to use precomputed node mapping.
2. Fix `FrontierRandomBot.__init__` to store topology and action space.
3. Replace repeated `any(s == own_state for s in state.board.values())` with maintained stone counts or a helper count.
4. Add `empty_count` to state or engine-derived fast state.
5. Use `RulesConfig.committed()` in scripts.
6. Add tests for committed-rules equivalence.

Expected impact: noticeable speedup, low risk.

## Phase 2 — exact legal cache

1. Add indexed topology.
2. Add `FastState` or cached fields:
   - occupancy bits
   - empty bits
   - stone counts
   - legal bits per player
3. Make `legal_moves()` and `build_legal_mask()` consume legal bits.
4. Replace lazy no-legal-moves with exact legal counts.

Expected impact: large speedup, fixes correctness risk.

## Phase 3 — Greedy and eval refactor

1. Convert Greedy to delta apply/undo.
2. Add exact cached mobility/frontier.
3. Add score cache by board hash.
4. Add count-only scoring function.

Expected impact: very large Greedy speedup and cleaner evaluation quality.

## Phase 4 — SearchBot serious upgrade

1. Add Zobrist hash.
2. Add transposition table.
3. Add PV/hash move ordering.
4. Add killer/history heuristics.
5. Use exact cached legal counts in `_leaf_eval`.
6. Throttle time checks.

Expected impact: largest SearchBot speedup. Same search depth result should remain unchanged except for tie-breaking if move order changes equal-score choice. To preserve exact deterministic tie behavior, sort tied best moves by original action order before RNG.

## Phase 5 — local scoring

1. Reimplement scoring on indexed adjacency.
2. Replace bridge `frozenset` with packed edge ids.
3. Add local dirty-component rescoring.
4. Validate against old `scoring_nodes()` on random boards.

Expected impact: important for deeper search and larger boards.

## Phase 6 — tournament parallelism

1. Parallelize games.
2. Store reproducible seeds per game.
3. Keep each process isolated to avoid mutable cache leaks.

Expected impact: near-linear speedup across CPU cores for batch experiments.

---

# 7. Exact vs approximate optimization table

| Optimization | Same quality? | Notes |
|---|---|---|
| Indexed topology | Yes | Representation-only change. |
| Bitsets for occupancy/legal/frontier | Yes | Exact if kept synchronized. |
| Delta apply/undo | Yes | Exact if undo restores all changed fields. |
| Legal move cache | Yes | Exact if local invalidation is complete. |
| Score cache by hash | Yes except hash collision risk | Store full verification key if you want zero practical concern. |
| Zobrist + transposition table | Yes except hash collision risk | Standard approach; can store extra board signature to be safer. |
| Move ordering | Yes | Does not change minimax value. May change tie-breaking unless handled. |
| Killer/history heuristic | Yes | Ordering only. |
| Iterative deepening | Yes | With fixed final depth, final result should match depth search; tie-order may differ. |
| Time-check throttling | Mostly | Exact when no timeout. With timeout, result always depends on cutoff timing. |
| Frontier approximation for mobility | No | Faster but not exact under neutrality/strict adjacency. |
| Lazy no-legal-moves check | No | Can miss early stuck states. Replace it. |
| Auto-fill when stuck | Changes game process | May be intended experiment rule, but it is not neutral search optimization. |

---

# 8. Suggested new internal APIs

## 8.1 `IndexedBoardTopology`

Could be folded into `BoardTopology`.

```python
class BoardTopology:
    idx_to_node: tuple[Node, ...]
    node_to_idx: dict[Node, int]
    neighbor_indices: tuple[tuple[int, ...], ...]
    neighbor_mask: tuple[int, ...]
```

## 8.2 `FastState`

```python
@dataclass
class FastState:
    black_bits: int
    white_bits: int
    empty_bits: int
    active_player: Player
    turn_phase: TurnPhase
    consecutive_pass_count: int
    current_turn: int
    stones_remaining_black: int | None
    stones_remaining_white: int | None
    legal_black: int
    legal_white: int
    frontier_black: int
    frontier_white: int
    zobrist: int
```

## 8.3 `Delta`

```python
@dataclass(slots=True)
class Delta:
    idx: int
    old_active_player: Player
    old_turn_phase: TurnPhase
    old_current_turn: int
    old_pass_count: int
    old_game_over: bool
    old_winner: object
    old_black_bits: int
    old_white_bits: int
    old_empty_bits: int
    old_legal_black: int
    old_legal_white: int
    old_frontier_black: int
    old_frontier_white: int
    old_zobrist: int
    old_supply_black: int | None
    old_supply_white: int | None
```

For performance, start by storing full old bit fields. Later, store XOR masks instead.

---

# 9. Blunt verdict

The current program is architecturally clean, but performance is still mostly “prototype Python.”

The main issue is not one bad function. It is this pattern:

```text
scan whole board
clone whole state
recompute whole score
repeat inside AI loops
```

The correct direction is:

```text
precompute topology
represent board as indexed bits
maintain exact legal/frontier/count caches
apply move locally
remember changed fields
undo exactly
cache searched positions
```

The first optimizations I would actually implement are:

1. Fix committed-rules constructors.
2. Fix `ActionSpace.build_mask()` linear lookup.
3. Fix `FrontierRandomBot` topology storage.
4. Add indexed topology.
5. Add exact legal bits and remove lazy no-legal-moves.
6. Convert Greedy to delta apply/undo.
7. Add Zobrist + transposition table to SearchBot.

That order gives the best mix of correctness, speed, and low implementation risk.

---

# 10. References

- CP-Algorithms, “Finding bridges in a graph in O(N+M).” Used for confirming that the current bridge/cycle-scoring basis is already asymptotically linear.
- Wikipedia, “Alpha–beta pruning.” Used for exactness and complexity behavior of alpha-beta compared with minimax.
- Chessprogramming Wiki, “Move Ordering.” Used for move-ordering priorities such as PV move, hash move, killer moves, and history heuristics.
- Chessprogramming Wiki, “Bitboards.” Used for board-set representation strategy.
- Chessprogramming Wiki, “Transposition Table” and “Zobrist Hashing.” Used for repeated-position caching strategy.
- Dogeystamp, “Chess engine, pt. 4: α-β pruning and better search.” Used as a practical overview of alpha-beta, transposition tables, move ordering, iterative deepening, and time-check throttling.
