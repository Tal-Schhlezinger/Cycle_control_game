# Cycle Control — Pre-RL Action Checklist for Claude Code

**Purpose:** finite, scoped list of fixes to leave the codebase sound before RL development continues.
**Source:** items from `cycle_control_program_book.md` and `cycle_control_optimization_book.md` audits, prioritized by impact and effort.
**Read this entire document before starting.** Some fixes are correctness-critical; others are cleanup. The order matters.

---

## Ground rules

1. **One change at a time.** Each item below is a discrete change. Make it, run the test suite (`python -m unittest tests_builtin.py tests_ai.py`), confirm 136/136 pass, commit/snapshot, move to next.
2. **Do not refactor while fixing.** If you spot a separate cleanup opportunity, write it down, don't do it.
3. **Each item has acceptance criteria.** A fix is done when those criteria pass, not before.
4. **Do not add features.** This checklist is purely fixes. Adding new tests is allowed (and expected); adding new functionality is not.

---

## Critical fixes — block all further bot tournament work

These items must be completed before any new tournament data is generated. Tournament results without these fixes are suspect.

### Item 1: Use `RulesConfig.committed()` in analysis scripts

**Files:** `greedy_validation.py`, `search_analysis.py`

**Problem:** Both scripts construct `RulesConfig(...)` manually with neutrality, strict adjacency, and mirror adjacency, but leave pass and end-condition flags at defaults. The "committed" tournament data may not actually use the committed ruleset.

**Fix:**

In `greedy_validation.py`, replace the `make_engine` function:

```python
def make_engine(radius: int, committed_ruleset: bool = True) -> MoveEngine:
    if committed_ruleset:
        rules = RulesConfig.committed(board_radius=radius)
    else:
        rules = RulesConfig(board_radius=radius)
    topology = BoardTopology(radius, mirror_adjacency=rules.mirror_adjacency)
    return MoveEngine(rules, topology)
```

In `search_analysis.py`, replace `make_committed_engine`:

```python
def make_committed_engine(radius: int) -> MoveEngine:
    rules = RulesConfig.committed(board_radius=radius)
    topology = BoardTopology(radius, mirror_adjacency=rules.mirror_adjacency)
    return MoveEngine(rules, topology)
```

**Acceptance criteria:**
- Both scripts run successfully on R=3 with reduced game count
- A new test in `tests_ai.py` named `test_committed_engine_factories_match` asserts that engines from both scripts produce `RulesConfig.committed(R).to_dict() == engine.rules.to_dict()` for R in {2, 3, 4}
- All 136+1 tests pass

---

### Item 2: Fix `FrontierRandomBot.__init__` to store topology

**File:** `cycle_control/ai/bots/random_bot.py`

**Problem:** Constructor accepts `topology=None` but never assigns `self._topology = topology`. Passing topology to the constructor does nothing; only `attach_topology()` works.

**Fix:** In the `__init__` method around line 47-51, after the comment "topology is accepted for future use but not required", add:

```python
self._topology = topology
```

Also remove `attach_topology` if it becomes redundant — but only if no other code calls it (grep first).

**Acceptance criteria:**
- New test `test_frontier_random_bot_constructor_topology` in `tests_ai.py`:
  ```python
  def test_frontier_random_bot_constructor_topology(self):
      from cycle_control.topology import BoardTopology
      topo = BoardTopology(3, mirror_adjacency=True)
      bot = FrontierRandomBot(topology=topo, seed=0)
      assert bot._topology is topo
  ```
- Test passes
- All 137+1 tests pass

---

### Item 3: Fix `ActionSpace.build_mask` to use precomputed dict

**File:** `cycle_control/ai/action_space.py`

**Problem:** `ActionSpace.build_mask` (around line 112) just calls the module-level `build_legal_mask`, which uses linear search via `node_to_action_index`. The precomputed `_node_to_idx` dict is unused. Every bot move pays an unnecessary `O(L·V)` cost.

**Fix:** Rewrite `ActionSpace.build_mask` to use `self._node_to_idx`:

```python
def build_mask(self, engine: MoveEngine, state: GameState) -> np.ndarray:
    mask = np.zeros(self.size, dtype=bool)
    if state.game_over:
        return mask
    for node in engine.legal_moves(state):
        mask[self._node_to_idx[node]] = True
    if engine.can_pass(state):
        mask[self.pass_index] = True
    return mask
```

The standalone `build_legal_mask` function should remain (other code may use it), but consider adding a comment that `ActionSpace.build_mask` is preferred for performance.

**Acceptance criteria:**
- Existing tests still pass (current tests don't measure performance, they measure correctness)
- New test `test_action_space_build_mask_uses_dict` confirms behavior matches `build_legal_mask` byte-for-byte:
  ```python
  def test_action_space_build_mask_matches_global(self):
      engine = _make_committed_engine(radius=3)
      state = engine.initial_state()
      engine.apply_placement(state, engine.topology.all_nodes()[5])
      aspace = ActionSpace(engine.topology)
      m1 = build_legal_mask(engine, state)
      m2 = aspace.build_mask(engine, state)
      assert np.array_equal(m1, m2)
  ```
- All 138+1 tests pass

---

### Item 4: Fix `_on_apply_modes` in `ui.py` to preserve `end_on_no_legal_moves`

**File:** `ui.py`

**Problem:** Around line 354-365, the `RulesConfig(...)` constructor call inside `_on_apply_modes` does not include `end_on_no_legal_moves`. When the user clicks "Apply modes" in the UI, the field silently resets to `False`. Manual playtests on the UI may not actually be playing committed rules.

**Fix:** Add `end_on_no_legal_moves=self.rules.end_on_no_legal_moves` to the kwargs of the `RulesConfig(...)` constructor call.

**Acceptance criteria:**
- Manual test: launch UI, configure rules with `end_on_no_legal_moves=True`, click "Apply modes", verify the engine's rules still have `end_on_no_legal_moves=True`
- (Optional) Add an automated UI test if the test infrastructure for Tkinter exists; otherwise skip
- All existing tests pass

---

### Item 5: Remove `default_territory_eval` from `__all__` of bots package

**File:** `cycle_control/ai/bots/__init__.py`

**Problem:** Line 10 exports `default_territory_eval` in `__all__`, but this name is no longer imported (it was removed when SearchBot was refactored to use `_leaf_eval` method). Any code doing `from cycle_control.ai.bots import default_territory_eval` will fail at import time.

**Fix:** Remove `"default_territory_eval"` from the `__all__` list. The line should become:

```python
__all__ = [
    "RandomBot", "FrontierRandomBot",
    "GreedyBot", "Greedy1", "Greedy2",
    "SearchBot", "SearchStats",
]
```

Same fix may need to be applied to `cycle_control/ai/__init__.py` if that file also re-exports `default_territory_eval`.

**Acceptance criteria:**
- `python -c "from cycle_control.ai.bots import *; print('ok')"` succeeds
- All existing tests pass

---

## Cleanup fixes — should be done soon, not blocking

These items improve code quality but don't affect correctness or experimental validity. Do them after the critical fixes above.

### Item 6: Reconcile SearchBot docstring vs. implementation

**File:** `cycle_control/ai/bots/search_bot.py`

**Problem:** Module docstring at line 1 says "negamax with alpha-beta pruning". The class docstring at line 52, method name `_minimax`, and the actual implementation are minimax with explicit maximizing/minimizing branches. The "negamax" claim is inaccurate.

**Fix:** Rename "negamax" to "minimax" everywhere in the docstring. The implementation is sound — it's just labelled wrong. Specifically:
- Line 1: `"""SearchBot — minimax with alpha-beta pruning.` (was "negamax")
- Anywhere else "negamax" appears, replace with "minimax"

Do NOT convert the implementation to negamax. The current code is clear and correct as minimax; converting to negamax would introduce bugs for no benefit.

**Acceptance criteria:**
- `grep -n "negamax" cycle_control/ai/bots/search_bot.py` returns nothing
- All tests pass

---

### Item 7: Update `cycle_control/ai/__init__.py` docstring

**File:** `cycle_control/ai/__init__.py`

**Problem:** Module docstring says certain sections of the AI checklist are "not yet implemented" when they are implemented. The list claims `SearchBot` is not implemented, but it is.

**Fix:** Read the current docstring, then update it to reflect reality:
- AI baselines: implemented (Random, FrontierRandom, Greedy_1, Greedy_2, SearchBot)
- Tournament: implemented
- Siege analysis: implemented
- NOT yet implemented: PettingZoo environment, GNN/CNN architectures, training loop

**Acceptance criteria:**
- Docstring accurately describes what is and is not implemented
- All tests pass

---

### Item 8: Remove unused imports

**Problem:** Various `import` statements have become unused as code evolved.

**Files and unused imports to investigate:**
- `cycle_control/scoring.py`: `sys` may be unused
- `cycle_control/ai/search_utils.py`: `MoveError` may be unused
- `cycle_control/ai/bots/search_bot.py`: check `apply_and_save`, `undo_placement`, `cycle_score_diff`, `mobility_for` imports — some may be unused since the refactor to `SearchState`-based eval
- `search_analysis.py`: `sys` may be unused

**Fix:** For each file, run a tool like `pyflakes` or visually inspect imports. Remove any that are not used in the file. Verify by running tests after each file's changes.

**Acceptance criteria:**
- `pyflakes` (or equivalent) reports no unused imports in the affected files
- All tests pass

---

## Items deferred to optimization roadmap (DO NOT IMPLEMENT)

The following items appeared in the program/optimization books but are out of scope for this checklist. Document their existence here so they're not forgotten, but do NOT implement them in this pass:

1. **Indexed topology + bitset state representation** — large refactor, defer to Phase 2 of optimization roadmap.
2. **Exact incremental legal-move tracking** — replaces lazy `end_on_no_legal_moves` check with exact `O(1)` version. Required for full correctness but not blocking. Defer to Phase 2.
3. **Zobrist hashing + transposition tables** — for SearchBot. Defer to Phase 4. Most beneficial at search depth >= 4.
4. **Greedy bot delta apply/undo** — would significantly speed up greedy bots. Defer to Phase 3.
5. **Reachability flood-fill `O(V³)` worst case** — `_can_player_place_at` recomputes `has_own_stones` per call. Fixable with one cached value, but `siege.py` is rarely called in hot paths. Defer.
6. **Persistence: validate board against topology on load** — defensive, defer.
7. **Read-only public API consolidation in `cycle_control/__init__.py`** — defer.

---

## Final verification before declaring done

After all critical fixes (items 1-5) and ideally the cleanup fixes (items 6-8) are complete:

1. **Run full test suite:** `python -m unittest tests_builtin.py tests_ai.py` — expect ≥136 tests pass (more if regression tests were added per acceptance criteria).
2. **Run analysis scripts at small radius to confirm they work:**
   - `python greedy_validation.py --radius 3 --games 10`
   - `python search_analysis.py --radius 3 --games 10`
   Both should run cleanly. The Elo numbers may differ slightly from prior runs because the rules are now actually committed.
3. **Verify committed-rules invariant:**
   ```python
   from cycle_control.rules import RulesConfig
   from greedy_validation import make_engine
   from search_analysis import make_committed_engine
   for R in [2, 3, 4]:
       expected = RulesConfig.committed(board_radius=R).to_dict()
       assert make_engine(R, committed_ruleset=True).rules.to_dict() == expected
       assert make_committed_engine(R).rules.to_dict() == expected
   print("committed rules invariant holds for all scripts")
   ```
4. **Document the work:** add a short `IMPLEMENTATION_NOTES_PHASE6.md` listing what was fixed, before/after Elo numbers if available, and confirmation that 136+ tests pass.

Once all of this is green, the codebase is ready to continue to RL development.

---

## Estimated effort

- Items 1-5 (critical): ~1-2 hours of focused work, mostly small edits + new tests
- Items 6-8 (cleanup): ~1 hour
- Verification + notes: ~30 minutes
- **Total: ~3 hours of careful work**

If any item takes substantially longer, stop and ask before continuing. The codebase should be left in a strictly-better state, not a half-refactored state.
