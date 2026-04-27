# Cycle Control Program Book

**Source analyzed:** `cycle_control_game_1.zip`  
**Project:** Cycle Control, graph-version research prototype  
**Main language:** Python  
**Primary purpose:** deterministic game engine + AI/search/testing harness for a two-player abstract strategy game on triangular cells.

---

## 1. Executive summary

This project is not just a playable UI. It is mainly a **research engine** for testing whether the Cycle Control rules create interesting strategic play. The code is organized around a clean core:

1. `topology.py` defines the board graph.
2. `rules.py` defines legal rule configurations.
3. `state.py` defines the mutable game state.
4. `engine.py` applies moves, enforces legality, handles turns, end conditions, undo/redo, and sandbox edits.
5. `scoring.py` computes which occupied nodes belong to cycles.
6. `persistence.py` saves and loads game states.
7. `testrunner.py`, `tests_builtin.py`, and `tests_ai.py` validate behavior.
8. `cycle_control/ai/` adds action masks, bot protocols, baselines, greedy bots, search, siege/territory analysis, and tournament tools.
9. `ui.py` gives a simple Tkinter interface for manual inspection.
10. `greedy_validation.py` and `search_analysis.py` are research scripts for bot/rules analysis.

The architecture is mostly solid: the core game model is separate from UI and AI. The main weak points are performance hotspots and a few consistency bugs between the “committed ruleset” idea and the scripts that claim to use it.

---

## 2. Complexity notation used in this book

The board is sparse and has constant maximum degree.

| Symbol | Meaning |
|---|---|
| `R` | board radius |
| `V` | number of board nodes/triangles. For normal boards, `V = 6R²`. |
| `E` | graph edges. Since max degree is at most 3 or 6, `E = O(V)`. |
| `Δ` | max node degree, `3` side-only or `6` with mirror adjacency. Treated as constant. |
| `L` | number of legal placement moves in a state, `L <= V`. |
| `A` | action-space size, `A = V + 1` including pass. |
| `H` | move history length. |
| `M` | number of gameplay actions in one game. |
| `G` | number of games in a match/tournament. |
| `B` | search branching factor, at most `V`, usually legal moves. |
| `d` | search depth. |
| `C` | number of commands in a JSON test spec. |

Because `E = O(V)`, many graph operations written as `O(V + E)` are effectively `O(V)` for this topology. I still write `O(V + E)` when the graph algorithm conceptually depends on both vertices and edges.

---

## 3. General architecture

### 3.1 Layer diagram

```text
Documentation / handoff
    README.md
    AI_DESIGN*.md
    AI_IMPLEMENTATION_CHECKLIST*.md
    IMPLEMENTATION_NOTES_PHASE*.md
    PROJECT_HANDOFF.json

Manual UI / scripts
    ui.py
    greedy_validation.py
    search_analysis.py

AI layer
    cycle_control/ai/action_space.py
    cycle_control/ai/bot_interface.py
    cycle_control/ai/bots/*.py
    cycle_control/ai/search_utils.py
    cycle_control/ai/siege.py
    cycle_control/ai/tournament.py
    cycle_control/ai_hooks.py

Core game layer
    cycle_control/topology.py
    cycle_control/rules.py
    cycle_control/state.py
    cycle_control/engine.py
    cycle_control/scoring.py
    cycle_control/persistence.py
    cycle_control/debug.py
    cycle_control/testrunner.py

Tests
    tests_builtin.py
    tests_ai.py
    tests/test_basic.json
```

### 3.2 Core data flow

```text
RulesConfig + BoardTopology
        ↓
MoveEngine.initial_state()
        ↓
GameState
        ↓
MoveEngine.apply_placement/apply_pass
        ↓
MoveEngine._check_end_conditions
        ↓
scoring_nodes / score
        ↓
winner or ongoing state
```

### 3.3 AI data flow

```text
GameState + MoveEngine
        ↓
ActionSpace.build_mask()
        ↓
Bot.choose_action(state, legal_mask, color)
        ↓
play_turn / play_game
        ↓
MoveEngine.apply_placement/apply_pass
```

### 3.4 Scoring model

The official score is the number of a player’s occupied nodes that belong to at least one simple cycle in that player’s induced subgraph. The implementation computes bridges with Tarjan’s bridge algorithm. A node scores if it has at least one incident same-color edge that is **not** a bridge. This is a good strategy: in an undirected graph, non-bridge edges are exactly the edges that participate in cycles, so any endpoint of a non-bridge same-color edge is cycle-connected.

### 3.5 Turn model

The engine has a special opening turn with one placement. Normal turns allow up to two placements. A pass ends the turn. The phase is frozen at turn start, so a turn does not dynamically change from two placements to one placement halfway through except by the explicit `NORMAL_2` transition.

### 3.6 Major architectural verdict

The separation between core rules, state, engine, scoring, AI, and UI is good. The project is maintainable and testable. The weak parts are:

1. **Several scripts claim to use committed rules but do not call `RulesConfig.committed()`.** They enable neutrality/strict/mirror but leave pass and end conditions at defaults.
2. **`build_legal_mask()` is slower than it needs to be.** It recomputes node indices with a linear search for each legal move.
3. **`FrontierRandomBot(topology=...)` ignores its constructor topology argument.** It only works in frontier mode if `attach_topology()` is called later.
4. **`cycle_control/ai/bots/__init__.py` exports `default_territory_eval`, which does not exist.** That is a real public API bug.
5. **`SearchBot` is described as negamax but implemented as minimax.** Not fatal, but the documentation is inaccurate.
6. **`end_on_no_legal_moves` is lazily checked only near the end of the board.** This is a performance optimization, but if a ruleset can produce early mutual lockout, the game may fail to terminate when it logically should.
7. **`ui._on_apply_modes()` does not preserve `end_on_no_legal_moves`.** Applying UI modes can silently reset this rule to default false.

---

# 4. File-by-file code book

## 4.1 `cycle_control/topology.py`

### Responsibility

Defines the board graph. A node is `(q, r, o)`, where `q,r` are axial coordinates and `o` is triangle orientation. It computes all valid nodes inside the hex-shaped region and all adjacency edges. It also validates graph invariants at construction time.

### Key design

The topology is immutable after construction. `BoardTopology` precomputes:

- sorted node tuple
- node set
- neighbor dictionary
- sanity checks: bipartite orientation, degree bounds, minimum girth

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `BoardTopology.__init__(radius, mirror_adjacency)` | Validate radius, compute nodes, node set, neighbors, run sanity checks. | `O(V)` for normal construction sanity because girth check is bounded to constant depth here. | `O(V + E)` |
| `_vertex_in_hex(q,r,R)` | Check axial hex condition `abs(q), abs(r), abs(q+r) <= R`. | `O(1)` | `O(1)` |
| `_triangle_in_hex(q,r,o,R)` | Build the triangle’s three corner vertices and require all to be inside hex. | `O(1)` | `O(1)` |
| `_compute_nodes()` | Scan a bounding square of coordinates and keep triangles whose vertices are inside. Sort nodes. | `O(R² log R)` due sorting, equivalent `O(V log V)`, but coordinate scan itself is `O(V)`. | `O(V)` |
| `_compute_neighbors()` | For each node, generate side-neighbor candidates plus optional mirror-neighbor candidates, then filter against node set. | `O(V·Δ) = O(V)` | `O(V + E)` |
| `_sanity_check()` | Check orientation bipartiteness, degree bounds, and too-short cycles. | `O(V + E)` with the current bounded-depth use. | `O(V)` temporary BFS structures |
| `_shortest_cycle_up_to(max_len)` | For each node, bounded BFS looking for a short cycle. | `O(V·Δ^k)` where `k≈max_len/2`; for constant `max_len`, `O(V)`. If `max_len` grows with board size, up to `O(V(V+E))`. | `O(V)` per BFS root |
| `_compute_girth()` | Calls `_shortest_cycle_up_to(2R+4)`. This is a convenience exact-ish girth scan for realistic sizes. | Worst `O(V(V+E))`, effectively `O(V²)` on this sparse graph. | `O(V)` |
| `iterate_nodes()` | Yield from the precomputed sorted node tuple. | Full iteration `O(V)`; creating iterator `O(1)`. | `O(1)` |
| `get_neighbors(node)` | Dictionary lookup of precomputed neighbors. | `O(1)` lookup, returns up to `Δ` neighbors. | `O(1)` |
| `is_on_board(node)` | Validate tuple shape/types/orientation, then membership in node set. | `O(1)` | `O(1)` |
| `all_nodes()` | Return precomputed tuple. | `O(1)` | `O(1)` |
| `node_count()` | Return length of precomputed tuple. | `O(1)` | `O(1)` |
| `__repr__()` | Format radius and node count. | `O(1)` | `O(1)` |

### Notes

This file is one of the strongest parts of the project. The precomputed graph representation is the correct choice for performance and simplicity.

---

## 4.2 `cycle_control/state.py`

### Responsibility

Defines the mutable game-state data model and simple enums. It does not enforce rules; that belongs to `engine.py`.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `Player.other()` | Return the opposite enum. | `O(1)` | `O(1)` |
| `NodeState.from_player(p)` | Convert player enum to matching node-state enum. | `O(1)` | `O(1)` |
| `PlacementEntry.to_dict()` | Serialize player/node into JSON-friendly dict. | `O(1)` | `O(1)` |
| `PlacementEntry.from_dict(d)` | Parse player/node from dict. | `O(1)` | `O(1)` |
| `PassEntry.to_dict()` | Serialize pass entry. | `O(1)` | `O(1)` |
| `PassEntry.from_dict(d)` | Parse pass entry. | `O(1)` | `O(1)` |
| `history_entry_from_dict(d)` | Dispatch by `type` field to placement/pass parser. | `O(1)` | `O(1)` |
| `GameState.clone()` | Shallow-copy board dict, supply dict, history list, redo list, scalar state. Entries themselves are immutable-style dataclasses. | `O(V + H)` | `O(V + H)` |
| `GameState.move_count()` | Return `len(move_history)`. | `O(1)` | `O(1)` |

### Notes

The state is intentionally mutable, and engine methods mutate it in place. That is fine for UI and performance, but AI code must clone or use exact undo deltas when exploring futures.

---

## 4.3 `cycle_control/rules.py`

### Responsibility

Defines the rules configuration and validates incompatible settings. This file owns rule flags, not the engine.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `RulesConfig.__post_init__()` | Run `_validate()` after dataclass construction. | Same as `_validate`, effectively `O(1)`. | `O(1)` |
| `RulesConfig.committed(board_radius)` | Factory for the intended committed V2.3 ruleset: pass disabled, no-legal-moves ending, neutrality, strict adjacency, mirror adjacency. | `O(1)` | `O(1)` |
| `supply_enabled()` | Check `stones_per_player is not None`. | `O(1)` | `O(1)` |
| `enabled_end_conditions()` | Build a list of enabled end-condition names. | `O(1)` | `O(1)` |
| `_validate()` | Reject bad radius/supply and inconsistent end-condition combinations. | `O(1)` | `O(1)` |
| `to_dict()` | Serialize all rule flags. | `O(1)` | `O(1)` |
| `from_dict(d)` | Construct config from dict with defaults for missing fields. | `O(1)` | `O(1)` |

### Critical issue

`RulesConfig.committed()` exists, but `greedy_validation.py` and `search_analysis.py` do **not** use it. They set neutrality/strict/mirror manually and leave pass/end-condition defaults. That means their “committed” analysis is probably not actually testing the committed ruleset.

---

## 4.4 `cycle_control/scoring.py`

### Responsibility

Computes official score: a player’s occupied nodes that belong to at least one cycle in their induced same-color graph.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `_find_bridges(adj)` | Iterative Tarjan bridge-finding algorithm. Uses discovery/low-link values without recursion. | `O(Vp + Ep)` for player subgraph. Worst `O(V + E)`. | `O(Vp + Ep)` |
| `scoring_nodes(topology, board, player, partial_credit_k)` | Build player induced adjacency, find bridges, score nodes adjacent to non-bridge edges. Optional partial credit adds all nodes in components of size `>= k`. | `O(V + E)` | `O(V + E)` |
| `score(...)` | Return length of `scoring_nodes`. | `O(V + E)` | `O(V + E)` because it builds the set |

### Notes

The algorithmic choice is correct. It avoids enumerating cycles, which would be much slower. For this graph, scoring is effectively linear in board size.

Minor cleanup: `sys` is imported but unused.

---

## 4.5 `cycle_control/engine.py`

### Responsibility

The central game authority. It owns move legality, applying placements/passes, turn phase transitions, end conditions, winner calculation, undo/redo, and sandbox edits.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `MoveEngine.__init__(rules, topology)` | Ensure rules radius and mirror mode match topology. Store references. | `O(1)` | `O(1)` |
| `initial_state()` | Create empty board dict for all nodes, initialize active player, phase, supply, counters, history. | `O(V)` | `O(V)` |
| `is_legal_placement(state,node)` | Reject game-over/off-board/occupied/no-supply. Then enforce neutrality and strict adjacency if enabled. Strict adjacency scans whole board to see if player has any stones. | `O(1)` without strict adjacency; `O(V)` with strict adjacency because of `has_own_stones`. | `O(1)` |
| `legal_moves(state)` | Scan every node and call `is_legal_placement`. | `O(V)` without strict adjacency; `O(V²)` with strict adjacency as written. | `O(L)` |
| `can_pass(state)` | Check pass flag and not game-over. | `O(1)` | `O(1)` |
| `apply_placement(state,node)` | Validate, place stone, update supply/history/redo/pass counter, check end, advance phase. | Common `O(V)` or `O(V²)` depending legality/end checks; final scoring can add `O(V+E)`. | `O(1)` extra, not counting history append |
| `apply_pass(state)` | Validate pass, record pass entry, update pass counter, check end, end turn. | Common `O(V)` due end checks; can be `O(V²)` if no-legal check runs. | `O(1)` |
| `_placements_this_turn(state)` | Infer whether current turn already has one placement from phase. | `O(1)` | `O(1)` |
| `_advance_after_placement(state)` | Transition opening/normal/truncated phases after placement. | `O(V)` when ending turn because turn phase counts empty cells. | `O(1)` |
| `_end_turn(state)` | Switch active player, increment turn, compute next phase. | `O(V)` due `_compute_turn_phase_at_start`. | `O(1)` |
| `_compute_turn_phase_at_start(state)` | Determine max placements from supply and empty count. | `O(V)` because it counts empty nodes. | `O(1)` |
| `_check_end_conditions(state)` | Check enabled end conditions. If triggered, score both players and set winner. Lazy no-legal check only runs near full board. | `O(V)` normally; `O(V²)` if no-legal check calls `legal_moves` for both under strict adjacency; plus `O(V+E)` scoring when triggered. | `O(L)` temporary for legal move lists |
| `_count_legal_moves_for(state,player)` | Temporarily switch active player, call `legal_moves`, restore active player. | Same as `legal_moves`: `O(V)` or `O(V²)`. | `O(L)` |
| `_determine_winner(state)` | Score black and white, compare. | `O(V + E)` | `O(V + E)` |
| `can_undo(state)` | Check history length. | `O(1)` | `O(1)` |
| `can_redo(state)` | Check redo length. | `O(1)` | `O(1)` |
| `undo(state)` | Rebuild from initial state and replay all history except last action. Push undone action to redo stack. | Worst `O(H·cost(apply))`, so up to `O(HV²)` under strict rules. | `O(V + H)` |
| `redo(state)` | Pop redo action and apply it. Preserve remaining redo stack because apply clears redo. | Same as one apply: up to `O(V²)`. | `O(Hr)` to copy redo stack |
| `sandbox_place(state,node,color)` | Validate board node and color, then directly set board cell. | `O(1)` | `O(1)` |
| `sandbox_remove(state,node)` | Validate board node, then set empty. | `O(1)` | `O(1)` |

### Critical notes

- `legal_moves()` is more expensive than it should be under strict adjacency. The repeated `any(s == own_state for s in state.board.values())` scan can be avoided by maintaining occupied counts per player or computing `has_own_stones` once per scan.
- Undo is correctness-first, not performance-first. That is acceptable for UI/testing but bad for deep AI search. The project correctly adds `search_utils.py` for faster search undo.
- The lazy `end_on_no_legal_moves` check is unsafe if early lockout is possible. Either prove early mutual lockout cannot happen under committed rules, or check it whenever legal-move absence is suspected.

---

## 4.6 `cycle_control/persistence.py`

### Responsibility

JSON serialization/deserialization for game states and rules. It deliberately does not persist `redo_stack`.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `_node_to_str(n)` | Convert tuple node to comma string. | `O(1)` | `O(1)` |
| `_str_to_node(s)` | Split comma string and parse ints. | `O(1)` | `O(1)` |
| `_winner_to_json(w)` | Convert winner enum/draw/none to JSON value. | `O(1)` | `O(1)` |
| `_winner_from_json(s)` | Convert JSON winner value back to `Player`, `draw`, or `None`. | `O(1)` | `O(1)` |
| `serialize_state(state,rules)` | Serialize rules, board, state scalars, supply, history. Sort board items for stable output. | `O(V log V + H)` because board is sorted. | `O(V + H)` |
| `deserialize_state(data)` | Validate schema/required fields, parse rules, board, supply, history. | `O(V + H)` | `O(V + H)` |
| `save_to_file(path,state,rules)` | Serialize and write JSON. | `O(V log V + H)` | `O(V + H)` |
| `load_from_file(path)` | Load JSON and deserialize. | `O(file_size + V + H)` | `O(V + H)` |

### Notes

This is simple and good. Missing validation: after loading, it does not verify that the board exactly matches the topology implied by rules. The UI reconstructs topology separately, so malformed saves could create strange states.

---

## 4.7 `cycle_control/testrunner.py`

### Responsibility

Runs JSON-defined tests. This is separate from `unittest` and supports scripted move/assertion sequences.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `_winner_str(w)` | Normalize winner to string/none. | `O(1)` | `O(1)` |
| `dispatch(engine,state,cmd)` | Interpret one JSON command: place/pass/sandbox/undo/redo/assertions. Unknown commands fail. | Depends on command. `place/pass/undo/redo` use engine costs; score assertions use `O(V+E)`; legal-move assertions can be `O(V²)`. | Usually `O(1)` to `O(V)` temporary |
| `run_test(spec,verbose)` | Build rules/topology/engine/state, run commands, catch errors, return log/result. | `O(V + Σ command costs)` | `O(V + C)` for state and log |
| `run_tests_from_file(path,verbose)` | Load one JSON file, run one or many specs. | `O(file_size + Σ test costs)` | `O(file_size + results)` |
| `main()` | CLI wrapper around `run_tests_from_file`. | Same as tests run. | Same as tests run. |

### Notes

The JSON runner is very useful for reproducible rule-level examples. It is not a replacement for the larger unit test files; it complements them.

---

## 4.8 `cycle_control/ai_hooks.py`

### Responsibility

Small compatibility layer exposing simple bot/AI helper functions without requiring callers to know engine internals.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `legal_moves(engine,state)` | Forward to engine. | Same as `engine.legal_moves`: `O(V)` or `O(V²)`. | `O(L)` |
| `clone(state)` | Forward to `state.clone`. | `O(V + H)` | `O(V + H)` |
| `apply_move(engine,state,move)` | Dispatch `{'type':'place'}` or `{'type':'pass'}` to engine. | Same as chosen engine apply. | `O(1)` extra |
| `evaluate(engine,state,player)` | Score player and opponent, return own/opponent/diff dict. | `O(V + E)` | `O(V + E)` |
| `BotRNG.__init__(seed)` | Create independent `random.Random`. | `O(1)` | `O(1)` |
| `BotRNG.seed_rng(seed)` | Reseed RNG. | `O(1)` | `O(1)` |
| `BotRNG.random()` | Draw float. | `O(1)` | `O(1)` |
| `BotRNG.choice(seq)` | Choose item from sequence. | `O(1)` | `O(1)` |

---

## 4.9 `cycle_control/debug.py`

### Responsibility

Debug helpers that summarize a player’s connected components and scoring nodes. Not official game logic.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `connected_components(topology,board,player)` | DFS/BFS over the player’s induced subgraph. | `O(V + E)` | `O(V)` |
| `debug_summary(topology,state,player)` | Compute components, scoring nodes, occupied count, largest component. | `O(V + E)` | `O(V)` |

---

## 4.10 `cycle_control/__init__.py`

### Responsibility

Public API re-export file. It imports and exposes the core objects and helpers.

### Functions/classes

No local functions. It defines `__all__`.

### Complexity

Import-time cost is the cost of importing all referenced modules. This is convenient but somewhat heavy because importing `cycle_control` pulls in test runner/debug/AI hooks too.

---

# 5. AI layer

## 5.1 `cycle_control/ai/action_space.py`

### Responsibility

Maps game placements to integer action indices for AI. Convention: indices `0..V-1` are placements in topology order, index `V` is pass.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `pass_index(topology)` | Return `topology.node_count()`. | `O(1)` | `O(1)` |
| `action_space_size(topology)` | Return `node_count + 1`. | `O(1)` | `O(1)` |
| `action_index_to_node(topology,index)` | Range-check; return `None` for pass, else `all_nodes()[index]`. | `O(1)` | `O(1)` |
| `node_to_action_index(topology,node)` | Linear search through all nodes. | `O(V)` | `O(1)` |
| `build_legal_mask(engine,state)` | Allocate boolean mask, set true for each legal move and pass if allowed. Calls `node_to_action_index` for each legal node. | `O(cost(legal_moves) + L·V)`. As written, can be `O(V²)` even if legality is cheap. | `O(A)` |
| `ActionSpace.__init__(topology)` | Store topology, size, pass index, and precompute node-to-index dict. | `O(V)` | `O(V)` |
| `ActionSpace.index_to_node(index)` | Fast index-to-node lookup. | `O(1)` | `O(1)` |
| `ActionSpace.node_to_index(node)` | Dict lookup. | `O(1)` average | `O(1)` |
| `ActionSpace.build_mask(engine,state)` | Currently delegates to module-level `build_legal_mask`, so it does **not** use the precomputed dict. | Same as `build_legal_mask`. | `O(A)` |

### Critical issue

`ActionSpace` precomputes a fast node→index map, but `build_mask()` does not use it. This wastes performance in every AI turn. The direct fix is to implement `ActionSpace.build_mask()` with `self._node_to_idx[node]` instead of calling the global function.

---

## 5.2 `cycle_control/ai/bot_interface.py`

### Responsibility

Defines the bot protocol and drives turns/games. Bots choose one action at a time; the interface handles multi-placement turns.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `Bot.choose_action(...)` | Protocol method only; implemented by bots. | Depends on bot. | Depends on bot. |
| `Bot.reset(seed)` | Protocol method only. | Depends on bot. | Depends on bot. |
| `play_turn(engine,state,bot,action_space)` | For active player, repeatedly build mask, ask bot, validate returned action, apply placement/pass until turn ends. | Up to 2 bot calls per normal turn. `O(P·(mask_cost + bot_cost + apply_cost))`, `P<=2`. | `O(A)` for mask |
| `auto_fill(engine,state)` | While current player has legal moves, place the first legal move in topology order. | Up to `V` placements; each legal scan can be `O(V²)` under strict, so worst `O(V³)`. | `O(L)` per loop |
| `play_game(engine,bot_black,bot_white,seed,max_turns,auto_fill_when_stuck)` | Create state, reset bots, loop turns until game over or max turns. Handles stuck players/pass/auto-fill. | `O(M·turn_cost)` plus possible auto-fill. | `O(V + H + A)` |

### Notes

The bot contract is good: illegal bot actions are treated as bugs, not as game moves. The `auto_fill` behavior is deterministic but should be treated as a rules/testing helper, not a strategic player.

---

## 5.3 `cycle_control/ai/siege.py`

### Responsibility

Computes future reachability/territory under monotone placement rules. This is used for strategic features, not official scoring.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `_can_player_place_at(rules,topology,board,v,player,extra_own)` | Test spatial legality for a hypothetical player, treating `extra_own` as already occupied by that player. Ignores turn/supply. | `O(V)` in strict mode because it scans board for own stones; otherwise `O(Δ) = O(1)`. | `O(1)` |
| `reachable_empty_cells(engine,state,player)` | Fixed-point expansion: repeatedly add empty cells the player could place given already reachable cells. | Documented `O(V²)`, but with strict-mode board scan inside helper it can be `O(V³)` as written. | `O(V)` |
| `sieged_against(engine,state,player)` | Empty cells minus reachable cells. | Same as `reachable_empty_cells` plus `O(V)`. | `O(V)` |
| `sieged_for(engine,state,player)` | Own reachable minus opponent reachable. | Two reachability computations. `O(V²)` intended, possibly `O(V³)` as written. | `O(V)` |
| `territory_score(engine,state,player)` | Count reachable cells. | Same as reachability. | `O(V)` |
| `exclusive_territory(engine,state,player)` | Count `sieged_for`. | Same as `sieged_for`. | `O(V)` |
| `frontier_count(engine,state,player)` | Count empty cells adjacent to at least one own stone. | `O(V·Δ)=O(V)` | `O(1)` |

### Critical note

The docstring says reachability is `O(V²)`, but `_can_player_place_at()` does a full-board scan for strict adjacency each call. That can push worst-case to `O(V³)`. It can be fixed by precomputing whether the player has real stones once per reachability call.

---

## 5.4 `cycle_control/ai/search_utils.py`

### Responsibility

Fast apply/undo utilities for search. This avoids cloning the full state at every search node.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `SearchDelta.__init__(state,node,f_black_delta,f_white_delta)` | Save enough old state to undo a placement later. Copies supply dict. | `O(1)` because supply has two players; history lengths only. | `O(1)` |
| `SearchState.__init__(engine,state)` | Store engine/state and compute initial frontier count for both players. | `O(V)` | `O(1)` plus frontier dict |
| `SearchState.frontier_diff(player)` | Return own frontier count minus opponent’s. | `O(1)` | `O(1)` |
| `SearchState.apply(node)` | Estimate frontier deltas around node, save delta, call engine placement, update frontier counts. | Intended `O(Δ)` plus engine apply cost. In practice engine apply can be `O(V)` or `O(V²)`. | `O(1)` |
| `SearchState.undo(delta)` | Restore board cell, trim history/redo, restore scalar fields/supply/frontier counts. | `O(number of trimmed entries)`, usually `O(1)` in depth-first search. | `O(1)` |
| `_MoveCtx.__init__(ss,node)` | Store context references. | `O(1)` | `O(1)` |
| `_MoveCtx.__enter__()` | Apply move and return delta. | Same as `SearchState.apply`. | `O(1)` |
| `_MoveCtx.__exit__(...)` | Undo move. | Same as `SearchState.undo`. | `O(1)` |
| `SearchState.move(node)` | Return context manager. | `O(1)` | `O(1)` |
| `_count_frontier(engine,state,player)` | Scan empty cells and check adjacency to own stone. | `O(V·Δ)=O(V)` | `O(1)` |
| `apply_and_save(engine,state,node)` | Compatibility helper: save delta then apply placement. | Same as engine apply. | `O(1)` |
| `undo_placement(state,delta)` | Compatibility helper: restore fields from delta. | Usually `O(1)`. | `O(1)` |

### Notes

This is the right direction for search performance. However, because it still calls `engine.apply_placement`, it inherits engine legality and end-check costs. A truly high-performance search engine would need a lighter internal transition function or cached legality data.

---

## 5.5 `cycle_control/ai/bots/random_bot.py`

### Responsibility

Baseline random bots.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `RandomBot.__init__(seed,name)` | Store name and independent RNG. | `O(1)` | `O(1)` |
| `RandomBot.reset(seed)` | Reinitialize RNG if seed provided. | `O(1)` | `O(1)` |
| `RandomBot.choose_action(state,legal_mask,color)` | Use `np.flatnonzero` then random choice. | `O(A)` | `O(L)` due legal index list |
| `FrontierRandomBot.__init__(topology,seed,name)` | Store name/RNG. Intended to accept topology. | `O(1)` | `O(1)` |
| `FrontierRandomBot.reset(seed)` | Reinitialize RNG if seed provided. | `O(1)` | `O(1)` |
| `FrontierRandomBot.choose_action(...)` | Prefer legal placement adjacent to own stone if topology attached; otherwise uniform random. | `O(A + V + L·Δ) = O(V)` | `O(V)` for own nodes/frontier list |
| `FrontierRandomBot.attach_topology(topology)` | Store topology for frontier behavior. | `O(1)` | `O(1)` |

### Critical bug

`FrontierRandomBot.__init__(topology=...)` accepts a topology parameter but does not assign it to `self._topology`. So passing topology to the constructor does nothing. It only uses frontier behavior after `attach_topology()`.

---

## 5.6 `cycle_control/ai/bots/greedy_bot.py`

### Responsibility

One-ply heuristic bots. `Greedy1` is cycle/structure focused. `Greedy2` is territory/frontier focused, though default `Greedy2` now uses frontier instead of expensive full siege territory.

### Evaluation features

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `cycle_score_diff(engine,state,player)` | Score player and opponent, subtract. | `O(V + E)` | `O(V + E)` |
| `largest_component_size(engine,state,player)` | DFS over player induced components, track largest. | `O(V + E)` | `O(V)` |
| `component_size_diff(...)` | Largest own component minus opponent’s. | `O(V + E)` | `O(V)` |
| `mobility_for(engine,state,player)` | If player is active, exact `legal_moves`; otherwise approximate by empty frontier cells. | Active exact: `O(V)` or `O(V²)` under strict. Non-active approximation: `O(V)`. | `O(V)` |
| `mobility_diff(...)` | Own mobility minus opponent mobility. | Up to exact `legal_moves` cost + `O(V)`. | `O(V)` |
| `territory_diff(...)` | Reachable territory diff. | Intended `O(V²)`, possibly `O(V³)` as written in strict mode. | `O(V)` |
| `exclusive_territory_diff(...)` | Exclusive future territory diff. | Same as two reachability calls; intended `O(V²)`, possibly `O(V³)`. | `O(V)` |
| `frontier_diff(...)` | Empty frontier count diff. | `O(V)` | `O(1)` |

### Bot classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `GreedyWeights.describe()` | Join nonzero weight fields into string. | `O(1)` | `O(1)` |
| `GreedyBot.__init__(engine,weights,seed,name)` | Store engine/weights/name/RNG/action space. | `O(V)` because `ActionSpace` precomputes map. | `O(V)` |
| `GreedyBot.reset(seed)` | Reinitialize RNG if seed provided. | `O(1)` | `O(1)` |
| `GreedyBot.evaluate(state,player)` | Weighted sum of enabled feature functions. | Depends on weights. Default Greedy1 can hit exact mobility; default Greedy2 is roughly `O(V+E)`. | Depends on features, usually `O(V)` |
| `GreedyBot.choose_action(...)` | For each candidate, clone state, apply action/pass, evaluate result, choose best with RNG tie-break. Avoids pass if any placement exists. | `O(L·(clone + apply + eval))`. Worst can be `O(L·V²)` or worse if expensive territory weights enabled. | `O(V + H)` per trial clone, not all retained |
| `Greedy1.__init__(...)` | Configure cycle-heavy default weights. | Same as `GreedyBot.__init__`: `O(V)`. | `O(V)` |
| `Greedy2.__init__(...)` | Configure frontier/opponent-mobility default weights. | Same as `GreedyBot.__init__`: `O(V)`. | `O(V)` |

### Notes

The greedy architecture is clear and useful for research. The major cost is clone-per-candidate. That is acceptable on small boards but not scalable. If you want strong bots on larger boards, reuse `SearchState`-style apply/undo for greedy evaluations too.

---

## 5.7 `cycle_control/ai/bots/search_bot.py`

### Responsibility

Depth-limited minimax with alpha-beta pruning and optional iterative deepening under time budget.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `terminal_value(state,player)` | Return large win/loss/zero value from winner field. | `O(1)` | `O(1)` |
| `SearchStats.describe()` | Format search counters. | `O(1)` | `O(1)` |
| `SearchBot.__init__(...)` | Store engine, depth, budget, ordering flag, RNG, action space, stats. | `O(V)` because `ActionSpace` precomputes map. | `O(V)` |
| `SearchBot.reset(seed)` | Reset RNG/stats. | `O(1)` | `O(1)` |
| `_leaf_eval(ss,player)` | Evaluate frontier diff and opponent mobility approximation. | `O(V)` due `mobility_for` for opponent approximation. | `O(V)` temporary own-stone set in mobility |
| `choose_action(state,legal_mask,color)` | Convert mask to legal indices, remove pass if placements exist, build `SearchState`, run fixed depth or iterative deepening. | `O(A + search_cost)` | `O(V + d)` |
| `_root(ss,color,depth,indices)` | Score each root action via apply/minimax/undo; choose best. | `O(B·subtree_cost)` | `O(d)` recursion plus state |
| `_minimax(ss,depth,alpha,beta,color_to_move,search_side)` | Recursive alpha-beta minimax over legal placements, using apply/undo. | Worst `O(B^d · node_cost)`. With pruning best case roughly `O(B^(d/2))`, but not guaranteed. | `O(d)` recursion |
| `_order(ss,color,indices)` | One-ply score root actions for move ordering. | `O(B·leaf_eval/apply cost)` | `O(B)` |
| `_order_nodes(ss,color,nodes)` | Same as `_order`, but receives nodes directly. | `O(B·leaf_eval/apply cost)` | `O(B)` |
| `_time_up()` | Compare elapsed time to budget. | `O(1)` | `O(1)` |

### Critical notes

- The module docstring says “negamax,” but the implementation is standard minimax with maximizing/minimizing branches.
- Recursive search ignores pass actions. It searches placement actions only. At root, pass is stripped if any placement exists. That matches the project’s “don’t pass unless forced” policy but is not a full rules search when pass is strategically legal.
- Search depth counts placement actions, not full player turns. This matters because normal turns may contain two placements.

---

## 5.8 `cycle_control/ai/tournament.py`

### Responsibility

Runs matches and round-robin tournaments between bots, then computes simple Elo ratings.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `MatchResult.total_games()` | Sum result counters. | `O(1)` | `O(1)` |
| `MatchResult.a_win_rate()` | Compute A wins / total. | `O(1)` | `O(1)` |
| `MatchResult.b_win_rate()` | Compute B wins / total. | `O(1)` | `O(1)` |
| `MatchResult.draw_rate()` | Compute draws / total. | `O(1)` | `O(1)` |
| `MatchResult.summary()` | Format result line. | `O(1)` | `O(1)` |
| `run_match(engine,bot_a,bot_b,n_games,swap_colors,base_seed,record_games)` | Run repeated games, optionally alternate colors, accumulate outcomes. | `O(G·game_cost)` | `O(G)` if recording games, else `O(1)` plus game state |
| `RoundRobinResult.win_rate_matrix()` | Build matrix of A win rates for all stored pairings. | `O(N²)` for `N` bots. | `O(N²)` |
| `RoundRobinResult.pretty_print()` | Format win-rate matrix. | `O(N²)` | `O(N²)` output string |
| `round_robin(engine_factory,bot_factories,bot_names,n_games_per_pair,base_seed,verbose)` | For every ordered pair of bots, create fresh engine/bots and run match. | `O(N²·G·game_cost)` | `O(N²)` results |
| `elo_update(rating_a,rating_b,score_a,k)` | Standard Elo update. | `O(1)` | `O(1)` |
| `elo_from_round_robin(rr,initial,k)` | Replay abstract wins/losses/draws through Elo update. | `O(total recorded game counts)` | `O(N)` ratings |

---

## 5.9 `cycle_control/ai/__init__.py`

### Responsibility

Public re-export file for the AI module.

### Functions/classes

No local functions. It imports/export action-space, bot interface, bot classes, siege analysis, and tournament helpers.

### Note

The docstring says `SearchBot` is “not yet implemented,” but it is implemented and exported. The docstring is stale.

---

## 5.10 `cycle_control/ai/bots/__init__.py`

### Responsibility

Re-exports bot classes from `greedy_bot.py`, `random_bot.py`, and `search_bot.py`.

### Functions/classes

No local functions.

### Critical bug

`__all__` includes `default_territory_eval`, but no such symbol is imported or defined. This can break `from cycle_control.ai.bots import *` or mislead API users.

---

# 6. UI and research scripts

## 6.1 `ui.py`

### Responsibility

Tkinter UI for manual game play and inspection. It draws triangular cells, handles clicks/buttons, displays score/status, supports sandbox mode, save/load, undo/redo, and rule-mode toggles.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `triangle_to_pixels(q,r,o,cell_size,ox,oy)` | Convert axial triangle vertices to screen coordinates. | `O(1)` | `O(1)` |
| `_point_in_triangle(px,py,verts)` | Barycentric point-in-triangle test. | `O(1)` | `O(1)` |
| `CycleControlUI.__init__(root,radius)` | Build rules/topology/engine/state, initialize Tk variables, build UI, redraw. | `O(V)` | `O(V)` |
| `_build_ui()` | Construct Tk frames, canvas, buttons, checkboxes, labels. | `O(1)` relative to board size. | `O(1)` |
| `_redraw()` | Clear canvas, compute centering, compute scoring sets, draw all triangles and markers. | `O(V + E)` | `O(V)` temporary pixel/scoring sets |
| `_update_status(b_score,w_score)` | Update label text. | `O(1)` | `O(1)` |
| `_find_clicked_node(x,y)` | Recompute board centering, then scan all triangles and test hit. | `O(V)` | `O(V)` temporary pixels |
| `_on_left_click(event)` | Find clicked node; in sandbox cycle state, otherwise apply placement. Redraw. | `O(V + apply_cost + redraw)` | `O(V)` |
| `_on_right_click(event)` | Sandbox-only remove clicked node. Redraw. | `O(V)` for hit-test + redraw. | `O(V)` |
| `_on_pass()` | Apply pass and redraw. | `O(apply_pass + V)` | `O(V)` |
| `_on_undo()` | Undo and redraw. | `O(undo + V)` | `O(V + H)` |
| `_on_redo()` | Redo and redraw. | `O(redo + V)` | `O(V)` |
| `_on_apply_modes()` | Build new rules/topology/engine from checkbox modes, restart state, redraw. | `O(V)` | `O(V)` |
| `_sync_mode_vars_from_rules()` | Reflect loaded rules in checkbox variables. | `O(1)` | `O(1)` |
| `_on_restart()` | Confirm, reset state, redraw. | `O(V)` | `O(V)` |
| `_on_save()` | Ask filename, save JSON. | `O(V log V + H)` | `O(V + H)` |
| `_on_load()` | Ask filename, load rules/state, rebuild topology/engine, redraw. | `O(V + H)` plus file read. | `O(V + H)` |
| `main()` | Parse optional radius, create Tk root and UI. | `O(V)` startup. | `O(V)` |

### Critical issue

`_on_apply_modes()` does not pass `end_on_no_legal_moves` when rebuilding `RulesConfig`, so that flag resets to default `False`. If a saved/loaded/committed configuration uses no-legal-moves ending, applying UI modes can silently change game termination behavior.

---

## 6.2 `greedy_validation.py`

### Responsibility

Research script comparing `Greedy1` and `Greedy2`, with sanity checks against `RandomBot`.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `make_engine(radius, committed_ruleset)` | Build rules/topology/engine. If committed flag true, manually sets neutrality/strict/mirror. | `O(V)` | `O(V)` |
| `run_validation(radius,n_games,seed,verbose)` | Run Greedy1 vs Greedy2, then both vs Random, print verdict. | `O(G·game_cost)` | `O(G)` if recording head-to-head games |
| `analyze_and_verdict(h2h,vs_rand_1,vs_rand_2)` | Compute win rates, draw rates, imbalance, text verdict. | `O(1)` | `O(1)` output list |
| `main()` | CLI parse, run validation, choose exit code. | Same as `run_validation`. | Same as `run_validation` |

### Critical issue

`make_engine(committed_ruleset=True)` does not call `RulesConfig.committed()`. It sets only the balance flags and leaves pass/end-condition defaults. That likely invalidates the claim that this script tests the committed ruleset.

---

## 6.3 `search_analysis.py`

### Responsibility

Research script running round-robin tournaments between Random, Greedy, and Search bots. Reports win matrix, Elo, draw rates, and interpretation.

### Functions/classes

| Function / method | Strategy | Time complexity | Space complexity |
|---|---|---:|---:|
| `make_committed_engine(radius)` | Build rules/topology/engine with neutrality/strict/mirror manually set. | `O(V)` | `O(V)` |
| `run_analysis(radius,n_games,include_search_d3,seed)` | Build bot factories, run round-robin, print matrix/Elo/draw stats. | `O(N²·G·game_cost)` | `O(N²)` results |
| `analyze_findings(rr,elo,bot_names)` | Produce interpretation text from Elo/draw/search deltas. | `O(N log N + matches)` | `O(N)` |
| `main()` | CLI parse; run single-radius or multi-radius analysis. | Same as selected analyses. | Same as selected analyses. |

### Critical issue

Same committed-rules problem as `greedy_validation.py`: `make_committed_engine()` does not call `RulesConfig.committed()`. The name is misleading.

---

# 7. Test files

## 7.1 `tests_builtin.py`

### Responsibility

Large `unittest` suite covering core topology, rules, legality, turns, scoring, undo/redo, persistence, sandbox, JSON runner, AI hooks, mirror adjacency, neutrality, strict adjacency, partial credit, and combined balance modes.

### Test class coverage

| Test class | What it verifies | Complexity comment |
|---|---|---|
| `TestTopology` | node counts, bipartite property, degrees, girth, adjacency, board predicate, sorting | Mostly construction `O(V)` to `O(V²)` depending girth calls. |
| `TestRulesConfig` | config validation and roundtrip | `O(1)`. |
| `TestLegalPlacement` | placement legality and sorted legal moves | Legal move scans up to `O(V²)` under strict cases. |
| `TestTurnStructure` | opening and normal two-placement turn behavior | Engine apply costs. |
| `TestTruncation` | supply/board truncation and frozen phase | Engine apply costs. |
| `TestSupplyExhaustionEnd` | end on all stones placed | Engine apply/scoring costs. |
| `TestPassing` | pass counter and pass-disabled behavior | Engine pass costs. |
| `TestScoring` | cycle scoring, branch bridges, independent colors | `O(V+E)` scoring. |
| `TestUndoRedo` | replay-based undo/redo behavior | Undo can be `O(H·apply_cost)`. |
| `TestPersistence` | save/load schema and redo omission | `O(V log V + H)`. |
| `TestSandbox` | sandbox bypasses turn/supply/history | Mostly `O(1)` operations. |
| `TestGameResolution` | winner/draw by score | Scoring `O(V+E)`. |
| `TestJSONRunner` | JSON test runner actions/assertions | Sum of command costs. |
| `TestAIHooks` | simple AI hook correctness/RNG | Hook costs. |
| `TestMirrorAdjacency` | mirror graph degree/girth/symmetry/scoring | Topology/scoring costs. |
| `TestNeutralityRule` | neutrality placement restriction | Engine legality. |
| `TestStrictAdjacency` | strict adjacency placement restriction | Engine legality. |
| `TestPartialCreditScoring` | partial component scoring | Scoring + component BFS. |
| `TestBalanceModesCompose` | combined rule modes run/persist/validate | Engine + persistence. |

### Notes

The test coverage is broad and useful. The tests are also a good informal specification.

---

## 7.2 `tests_ai.py`

### Responsibility

AI-specific `unittest` suite covering action space, bot interface, siege/territory, random/greedy/search bots, tournament, and committed rules behavior.

### Test class coverage

| Test class | What it verifies | Complexity comment |
|---|---|---|
| `TestActionSpace` | action count, pass index, node-index roundtrip, legal masks | Mask construction can be `O(V²)`. |
| `TestBotInterface` | turn loop, partial pass, full game, illegal bot action rejection | Game/turn costs. |
| `TestSiege` | reachability, neutrality, strict adjacency, territory/frontier | Reachability can be expensive: intended `O(V²)`. |
| `TestRandomBot` | random legal choice, deterministic seeding, full game | Game costs. |
| `TestGreedyBots` | legal greedy actions, deterministic eval, greedy vs random, pass avoidance, mobility | Greedy action selection can be expensive. |
| `TestTournament` | match counting, color swapping, round robin, Elo monotonicity | Tournament costs. |
| `TestSearchBot` | search legal picks, termination, beats baselines, timeout/stats/terminal detection | Exponential in depth but small board/tests. |
| `TestCommittedRuleset` | factory, termination, no draws random-vs-random, full coverage, auto-fill, roundtrip, no-legal-moves | Game simulation costs. |

---

## 7.3 `tests/test_basic.json`

### Responsibility

Example JSON-driven tests. It gives non-Python test specs for the custom runner.

### Complexity

Each JSON command delegates to `testrunner.dispatch`, so cost is the sum of engine/scoring/assertion costs used by the spec.

---

# 8. Documentation and non-code files

The repository includes many design/checklist/history files:

| File group | Role |
|---|---|
| `README.md` | Current project overview, run instructions, rules summary, known limitations. |
| `AI_DESIGN*.md` | Iterative AI design documents. These appear to record changing plans. Use the latest version as most relevant unless a section says otherwise. |
| `AI_IMPLEMENTATION_CHECKLIST*.md` | Implementation checklists across AI versions. The latest checklist is probably the active roadmap. |
| `IMPLEMENTATION_NOTES_PHASE*.md` | Phase-by-phase implementation notes. Useful for understanding why code changed. |
| `PROJECT_HANDOFF.json` | Structured handoff for future implementation, including planned environment/training/GNN/PPO work. |

### Important documentation drift

There is drift between docs and code:

- `cycle_control/ai/__init__.py` says SearchBot is not implemented, but it is.
- README/UI help disagree about scoring marker colors.
- The scripts use “committed” language but do not use `RulesConfig.committed()`.

Documentation drift matters here because this project is meant to be a research engine. If experiments are run under the wrong rule assumptions, the conclusions become unreliable.

---

# 9. Main runtime scenarios

## 9.1 Manual UI game

```text
ui.py main()
  → CycleControlUI.__init__
      → RulesConfig
      → BoardTopology
      → MoveEngine
      → engine.initial_state
      → _build_ui
      → _redraw
  → user click
      → _find_clicked_node
      → engine.apply_placement / sandbox_place / sandbox_remove
      → _redraw
```

## 9.2 Bot game

```text
play_game
  → engine.initial_state
  → reset bots
  → loop until game_over
      → engine.legal_moves
      → play_turn
          → ActionSpace.build_mask
          → bot.choose_action
          → engine.apply_placement or apply_pass
```

## 9.3 Greedy bot move

```text
GreedyBot.choose_action
  → legal indices from mask
  → remove pass unless forced
  → for each candidate:
      → trial = state.clone()
      → engine.apply_placement(trial, node)
      → evaluate(trial, bot color)
  → choose max score, random among ties
```

## 9.4 Search bot move

```text
SearchBot.choose_action
  → legal indices from mask
  → remove pass unless forced
  → SearchState(engine, state)
  → _root
      → order candidate moves by leaf eval
      → for each root move:
          → SearchState.apply
          → _minimax(depth-1)
          → SearchState.undo
```

## 9.5 JSON test

```text
run_tests_from_file
  → load JSON
  → run_test
      → RulesConfig + BoardTopology + MoveEngine + initial_state
      → for each command:
          → dispatch
```

---

# 10. Performance hotspots

## 10.1 `legal_moves` under strict adjacency

Current behavior:

```text
legal_moves scans V nodes
  each is_legal_placement may scan V board values to check has_own_stones
```

Worst result: `O(V²)`.

Better design:

- Add `black_count` and `white_count` to state, or compute `has_own_stones` once in `legal_moves` and pass it into a helper.
- Then legality becomes `O(Δ)` and legal move scan becomes `O(V)`.

## 10.2 `ActionSpace.build_mask`

Current behavior:

```text
for node in legal_moves:
    mask[node_to_action_index(node)] = True   # linear search O(V)
```

Worst result: `O(L·V)` after legal move generation.

Better design:

- Use `ActionSpace._node_to_idx` inside `ActionSpace.build_mask()`.
- Or change global `build_legal_mask()` to optionally receive a mapping.

## 10.3 Greedy clone-per-candidate

Current behavior:

```text
for each legal candidate:
    clone full state O(V+H)
    apply move
    evaluate
```

This is easy to write and safe, but expensive. A better version would use `SearchState.apply/undo` or an equivalent lightweight trial mutation.

## 10.4 Siege reachability strict-mode scan

`_can_player_place_at()` checks `has_own_stones_real = any(...)` each call. In a fixed-point loop this is repeated many times. Precompute it once per player per reachability calculation.

## 10.5 Search still pays engine costs

`SearchState.apply()` avoids cloning, but it still calls `engine.apply_placement()`. That means it still pays full legality/end-condition cost. For deep search, a dedicated fast transition layer would help.

---

# 11. Correctness and consistency issues to fix

## Highest priority

1. **Fix committed rules usage in analysis scripts.**
   - Replace manual `RulesConfig(...)` in `greedy_validation.py` and `search_analysis.py` with `RulesConfig.committed(radius)`.
   - Build topology using `mirror_adjacency=rules.mirror_adjacency`.

2. **Fix `cycle_control/ai/bots/__init__.py`.**
   - Remove `default_territory_eval` from `__all__`, or actually define/import it.

3. **Fix `FrontierRandomBot.__init__`.**
   - Add `self._topology = topology`.

4. **Fix `ActionSpace.build_mask`.**
   - Use the precomputed node-index map.

5. **Fix UI rule preservation.**
   - `_on_apply_modes()` must include `end_on_no_legal_moves=self.rules.end_on_no_legal_moves`.

## Medium priority

6. **Rename SearchBot docstring or implementation.**
   - Either call it minimax, or convert to true negamax.

7. **Prove or remove lazy no-legal-moves check.**
   - The optimization is dangerous unless proven safe under committed rules.

8. **Clean stale documentation.**
   - Update AI `__init__.py` docstring.
   - Align README and UI marker descriptions.

9. **Remove unused imports.**
   - `sys` in `scoring.py`.
   - `MoveError` in `search_utils.py`.
   - `apply_and_save`, `undo_placement`, `cycle_score_diff` imports in `search_bot.py` if unused.
   - `sys` in `search_analysis.py` if unused.

---

# 12. Suggested next architecture improvements

## 12.1 Add a state-side occupancy count

Add to `GameState`:

```python
stone_counts: dict[Player, int]
```

Or derive once per `legal_moves`. This directly improves strict-adjacency legality.

## 12.2 Separate rule legality from tactical search transition

The engine is correctness-first. Search needs speed-first. Keep `MoveEngine` authoritative, but add a `FastRulesView` or `SearchRulesCache` that caches:

- empty cells
- own counts
- neighbor color counts
- legal frontier sets
- stone counts

## 12.3 Use one source for committed rules

Every script/test/UI mode that claims “committed” should call the same factory. Otherwise tournament conclusions cannot be trusted.

## 12.4 Add architecture tests for committed rules

Add tests that assert:

```python
make_committed_engine(radius).rules.to_dict() == RulesConfig.committed(radius).to_dict()
```

for every script/helper that creates committed engines.

---

# 13. Compact file responsibility index

| File | Responsibility |
|---|---|
| `cycle_control/topology.py` | Board graph generation and adjacency. |
| `cycle_control/rules.py` | Rule flags, validation, committed rules factory. |
| `cycle_control/state.py` | Enums, history entries, mutable game state. |
| `cycle_control/engine.py` | Legal moves, apply placement/pass, turns, end conditions, undo/redo, sandbox. |
| `cycle_control/scoring.py` | Cycle-node scoring via bridge detection. |
| `cycle_control/persistence.py` | JSON save/load. |
| `cycle_control/testrunner.py` | JSON scripted tests. |
| `cycle_control/ai_hooks.py` | Simple external AI helper interface. |
| `cycle_control/debug.py` | Debug summaries and connected components. |
| `cycle_control/ai/action_space.py` | Node/pass action indexing and legal masks. |
| `cycle_control/ai/bot_interface.py` | Bot protocol, turn loop, game loop. |
| `cycle_control/ai/siege.py` | Future reachability/territory/siege analysis. |
| `cycle_control/ai/search_utils.py` | Fast apply/undo and frontier tracking for search. |
| `cycle_control/ai/bots/random_bot.py` | Random baseline bots. |
| `cycle_control/ai/bots/greedy_bot.py` | One-ply heuristic bots. |
| `cycle_control/ai/bots/search_bot.py` | Depth-limited alpha-beta minimax bot. |
| `cycle_control/ai/tournament.py` | Match, round robin, Elo. |
| `ui.py` | Tkinter manual UI. |
| `greedy_validation.py` | Greedy variant research script. |
| `search_analysis.py` | Search-vs-greedy tournament research script. |
| `tests_builtin.py` | Core engine/unit tests. |
| `tests_ai.py` | AI/unit tests. |
| `tests/test_basic.json` | JSON-runner example tests. |

---

# 14. Bottom-line assessment

The program has a good research-engine architecture. The core model is properly separated from UI and AI. The scoring algorithm is efficient and correct for the stated cycle-node objective. The testing culture is strong.

The biggest risk is **experimental validity**, not basic code organization. Some scripts are probably not testing the actual committed ruleset they claim to test. Fix that before trusting bot tournament conclusions.

The biggest performance issue is repeated full-board scanning during legality/mask/greedy/search operations. The next serious engineering step should be cached legality/frontier state, not more AI complexity.
