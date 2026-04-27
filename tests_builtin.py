"""Built-in unittest suite for Cycle Control.

Run with:
    python -m unittest tests_builtin
or:
    python tests_builtin.py
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from cycle_control.engine import MoveEngine, MoveError
from cycle_control.persistence import (
    PersistenceError, deserialize_state, load_from_file,
    save_to_file, serialize_state,
)
from cycle_control.rules import RulesConfig
from cycle_control.scoring import scoring_nodes
from cycle_control.state import (
    GameState, NodeState, PassEntry, PlacementEntry, Player, TurnPhase,
)
from cycle_control.testrunner import AssertionFailed, run_test
from cycle_control.topology import BoardTopology


# ====================================================================
# Topology
# ====================================================================

class TestTopology(unittest.TestCase):
    def test_radius_1_has_6_triangles(self):
        self.assertEqual(BoardTopology(1).node_count(), 6)

    def test_radius_2_has_24_triangles(self):
        self.assertEqual(BoardTopology(2).node_count(), 24)

    def test_radius_3_has_54_triangles(self):
        self.assertEqual(BoardTopology(3).node_count(), 54)

    def test_bipartite_by_orientation(self):
        for R in (1, 2, 3):
            t = BoardTopology(R)
            for node in t.iterate_nodes():
                for n in t.get_neighbors(node):
                    self.assertNotEqual(node[2], n[2])

    def test_no_isolated_nodes(self):
        for R in (1, 2, 3):
            t = BoardTopology(R)
            for node in t.iterate_nodes():
                self.assertGreaterEqual(len(t.get_neighbors(node)), 1)

    def test_degrees_in_1_to_3(self):
        for R in (1, 2, 3):
            t = BoardTopology(R)
            for node in t.iterate_nodes():
                self.assertIn(len(t.get_neighbors(node)), (1, 2, 3))

    def test_girth_at_least_6(self):
        for R in (1, 2, 3):
            t = BoardTopology(R)
            g = t._compute_girth()
            if g != float("inf"):
                self.assertGreaterEqual(g, 6)

    def test_r1_forms_single_6_cycle(self):
        t = BoardTopology(1)
        self.assertEqual(t._compute_girth(), 6)
        for node in t.iterate_nodes():
            self.assertEqual(len(t.get_neighbors(node)), 2)

    def test_adjacency_formula_spot_check(self):
        t = BoardTopology(3)
        self.assertEqual(
            set(t.get_neighbors((0, 0, 0))),
            {(0, 0, 1), (-1, 0, 1), (0, -1, 1)},
        )
        self.assertEqual(
            set(t.get_neighbors((0, 0, 1))),
            {(0, 0, 0), (1, 0, 0), (0, 1, 0)},
        )

    def test_adjacency_symmetric(self):
        t = BoardTopology(2)
        for u in t.iterate_nodes():
            for v in t.get_neighbors(u):
                self.assertIn(u, t.get_neighbors(v))

    def test_is_on_board(self):
        t = BoardTopology(1)
        self.assertTrue(t.is_on_board((0, 0, 0)))
        self.assertFalse(t.is_on_board((10, 10, 0)))
        self.assertFalse(t.is_on_board((0, 0, 2)))
        self.assertFalse(t.is_on_board("not a tuple"))
        self.assertFalse(t.is_on_board((0, 0)))
        self.assertFalse(t.is_on_board((0.5, 0, 0)))

    def test_invalid_radius(self):
        with self.assertRaises(ValueError):
            BoardTopology(0)
        with self.assertRaises(ValueError):
            BoardTopology(-1)

    def test_sorted_node_order(self):
        nodes = list(BoardTopology(3).iterate_nodes())
        self.assertEqual(nodes, sorted(nodes))


# ====================================================================
# RulesConfig
# ====================================================================

class TestRulesConfig(unittest.TestCase):
    def test_default_is_valid(self):
        RulesConfig()  # should not raise

    def test_reject_no_end_conditions(self):
        with self.assertRaises(ValueError):
            RulesConfig(
                end_on_consecutive_passes=False,
                end_on_all_stones_placed=False,
                end_on_board_full=False,
            )

    def test_reject_all_stones_placed_without_supply(self):
        with self.assertRaises(ValueError):
            RulesConfig(
                stones_per_player=None,
                end_on_all_stones_placed=True,
                end_on_board_full=False,
            )

    def test_reject_consecutive_passes_sole_with_pass_disabled(self):
        with self.assertRaises(ValueError):
            RulesConfig(
                pass_enabled=False,
                end_on_consecutive_passes=True,
                end_on_all_stones_placed=False,
                end_on_board_full=False,
            )

    def test_accept_consecutive_passes_sole_with_pass_enabled(self):
        RulesConfig(
            pass_enabled=True,
            end_on_consecutive_passes=True,
            end_on_all_stones_placed=False,
            end_on_board_full=False,
        )

    def test_accept_board_full_only_with_pass_disabled(self):
        RulesConfig(
            pass_enabled=False,
            end_on_consecutive_passes=False,
            end_on_all_stones_placed=False,
            end_on_board_full=True,
        )

    def test_roundtrip_dict(self):
        r = RulesConfig(board_radius=2, stones_per_player=20,
                        end_on_all_stones_placed=True)
        r2 = RulesConfig.from_dict(r.to_dict())
        self.assertEqual(r.to_dict(), r2.to_dict())

    def test_reject_zero_stones(self):
        with self.assertRaises(ValueError):
            RulesConfig(stones_per_player=0)


# ====================================================================
# Legal placement
# ====================================================================

class TestLegalPlacement(unittest.TestCase):
    def setUp(self):
        self.rules = RulesConfig(board_radius=2)
        self.topology = BoardTopology(2)
        self.engine = MoveEngine(self.rules, self.topology)
        self.state = self.engine.initial_state()

    def test_opening_phase_at_start(self):
        self.assertIs(self.state.turn_phase, TurnPhase.OPENING)
        self.assertIs(self.state.active_player, Player.BLACK)
        self.assertEqual(self.state.current_turn, 1)

    def test_legal_on_empty(self):
        self.assertTrue(self.engine.is_legal_placement(self.state, (0, 0, 0)))

    def test_illegal_off_board(self):
        self.assertFalse(self.engine.is_legal_placement(self.state, (100, 100, 0)))

    def test_illegal_on_occupied(self):
        self.engine.apply_placement(self.state, (0, 0, 0))
        self.assertFalse(self.engine.is_legal_placement(self.state, (0, 0, 0)))

    def test_raises_on_illegal(self):
        with self.assertRaises(MoveError):
            self.engine.apply_placement(self.state, (100, 100, 0))

    def test_legal_moves_is_sorted(self):
        moves = self.engine.legal_moves(self.state)
        self.assertEqual(moves, sorted(moves))
        self.assertEqual(len(moves), self.topology.node_count())


# ====================================================================
# Turn structure
# ====================================================================

class TestTurnStructure(unittest.TestCase):
    def setUp(self):
        self.rules = RulesConfig(board_radius=2)
        self.topology = BoardTopology(2)
        self.engine = MoveEngine(self.rules, self.topology)
        self.state = self.engine.initial_state()

    def test_opening_one_placement(self):
        self.engine.apply_placement(self.state, (0, 0, 0))
        self.assertIs(self.state.active_player, Player.WHITE)
        self.assertIs(self.state.turn_phase, TurnPhase.NORMAL_1)
        self.assertEqual(self.state.current_turn, 2)

    def test_normal_2_placements(self):
        self.engine.apply_placement(self.state, (0, 0, 0))    # Black opening
        self.engine.apply_placement(self.state, (1, 0, 0))    # White 1/2
        self.assertIs(self.state.turn_phase, TurnPhase.NORMAL_2)
        self.assertIs(self.state.active_player, Player.WHITE)

        self.engine.apply_placement(self.state, (-1, 0, 0))   # White 2/2
        self.assertIs(self.state.turn_phase, TurnPhase.NORMAL_1)
        self.assertIs(self.state.active_player, Player.BLACK)
        self.assertEqual(self.state.current_turn, 3)


# ====================================================================
# Truncation
# ====================================================================

class TestTruncation(unittest.TestCase):
    def test_truncated_by_supply(self):
        rules = RulesConfig(board_radius=2, stones_per_player=2,
                            end_on_all_stones_placed=True)
        topology = BoardTopology(2)
        engine = MoveEngine(rules, topology)
        state = engine.initial_state()

        engine.apply_placement(state, (0, 0, 0))    # Black opening; Black has 1 left
        engine.apply_placement(state, (1, 0, 0))    # White 1/2
        engine.apply_placement(state, (-1, 0, 0))   # White 2/2

        self.assertIs(state.active_player, Player.BLACK)
        self.assertIs(state.turn_phase, TurnPhase.NORMAL_TRUNCATED_1)
        self.assertEqual(state.stones_remaining[Player.BLACK], 1)

    def test_phase_frozen_at_turn_start(self):
        # NORMAL_1 placement still leads to NORMAL_2 even if a future
        # condition could have truncated.
        rules = RulesConfig(board_radius=2, stones_per_player=10,
                            end_on_all_stones_placed=True)
        topology = BoardTopology(2)
        engine = MoveEngine(rules, topology)
        state = engine.initial_state()
        engine.apply_placement(state, (0, 0, 0))   # Black opening
        engine.apply_placement(state, (1, 0, 0))   # White 1/2 -> NORMAL_2
        self.assertIs(state.turn_phase, TurnPhase.NORMAL_2)


# ====================================================================
# Supply exhaustion as end condition
# ====================================================================

class TestSupplyExhaustionEnd(unittest.TestCase):
    def test_all_stones_placed_ends_game(self):
        rules = RulesConfig(
            board_radius=2,
            stones_per_player=1,
            pass_enabled=False,
            end_on_consecutive_passes=False,
            end_on_all_stones_placed=True,
            end_on_board_full=False,
        )
        topology = BoardTopology(2)
        engine = MoveEngine(rules, topology)
        state = engine.initial_state()

        engine.apply_placement(state, (0, 0, 0))
        self.assertFalse(state.game_over)
        engine.apply_placement(state, (1, 0, 0))
        self.assertTrue(state.game_over)


# ====================================================================
# Passing
# ====================================================================

class TestPassing(unittest.TestCase):
    def setUp(self):
        self.rules = RulesConfig(board_radius=2)
        self.topology = BoardTopology(2)
        self.engine = MoveEngine(self.rules, self.topology)
        self.state = self.engine.initial_state()

    def test_full_turn_pass_increments_counter(self):
        self.engine.apply_pass(self.state)
        self.assertEqual(self.state.consecutive_pass_count, 1)

    def test_two_full_passes_end_game(self):
        self.engine.apply_pass(self.state)  # Black opening-pass
        self.engine.apply_pass(self.state)  # White full-turn pass
        self.assertTrue(self.state.game_over)

    def test_placement_resets_pass_counter(self):
        self.engine.apply_pass(self.state)
        self.assertEqual(self.state.consecutive_pass_count, 1)
        self.engine.apply_placement(self.state, (0, 0, 0))
        self.assertEqual(self.state.consecutive_pass_count, 0)

    def test_partial_pass_does_not_increment(self):
        self.engine.apply_placement(self.state, (0, 0, 0))   # Black opening
        self.engine.apply_placement(self.state, (1, 0, 0))   # White 1/2
        self.engine.apply_pass(self.state)                    # White partial pass
        self.assertEqual(self.state.consecutive_pass_count, 0)

    def test_pass_disabled_raises(self):
        rules = RulesConfig(board_radius=2, pass_enabled=False,
                            end_on_consecutive_passes=False,
                            end_on_board_full=True)
        engine = MoveEngine(rules, BoardTopology(2))
        state = engine.initial_state()
        with self.assertRaises(MoveError):
            engine.apply_pass(state)


# ====================================================================
# Scoring
# ====================================================================

class TestScoring(unittest.TestCase):
    def test_empty_scores_zero(self):
        t = BoardTopology(2)
        board = {n: NodeState.EMPTY for n in t.iterate_nodes()}
        self.assertEqual(len(scoring_nodes(t, board, Player.BLACK)), 0)

    def test_single_stone_no_cycle(self):
        t = BoardTopology(2)
        board = {n: NodeState.EMPTY for n in t.iterate_nodes()}
        board[(0, 0, 0)] = NodeState.BLACK
        self.assertEqual(len(scoring_nodes(t, board, Player.BLACK)), 0)

    def test_minimal_6_cycle(self):
        """The R=1 board IS a 6-cycle. Filling all 6 => all 6 score."""
        t = BoardTopology(1)
        board = {n: NodeState.BLACK for n in t.iterate_nodes()}
        self.assertEqual(len(scoring_nodes(t, board, Player.BLACK)), 6)

    def test_cycle_with_bridge_branch(self):
        """A 6-cycle plus one extra Black stone dangling off. The dangler is
        a bridge-connected branch; it must NOT score."""
        t = BoardTopology(2)
        board = {n: NodeState.EMPTY for n in t.iterate_nodes()}
        r1_nodes = list(BoardTopology(1).iterate_nodes())
        for n in r1_nodes:
            board[n] = NodeState.BLACK
        # Find a node outside R=1 adjacent to an R=1 node.
        r1_set = set(r1_nodes)
        branch = None
        for u in r1_nodes:
            for v in t.get_neighbors(u):
                if v not in r1_set and t.is_on_board(v):
                    branch = v
                    break
            if branch is not None:
                break
        self.assertIsNotNone(branch)
        board[branch] = NodeState.BLACK

        sc = scoring_nodes(t, board, Player.BLACK)
        self.assertEqual(len(sc), 6)
        self.assertNotIn(branch, sc)
        for n in r1_nodes:
            self.assertIn(n, sc)

    def test_colors_independent(self):
        """Black and White induced subgraphs are independent."""
        t = BoardTopology(1)
        board = {n: NodeState.BLACK for n in t.iterate_nodes()}
        self.assertEqual(len(scoring_nodes(t, board, Player.WHITE)), 0)


# ====================================================================
# Undo / redo
# ====================================================================

class TestUndoRedo(unittest.TestCase):
    def setUp(self):
        self.rules = RulesConfig(board_radius=2)
        self.topology = BoardTopology(2)
        self.engine = MoveEngine(self.rules, self.topology)
        self.state = self.engine.initial_state()

    def test_undo_single_placement(self):
        self.engine.apply_placement(self.state, (0, 0, 0))
        self.engine.undo(self.state)
        self.assertEqual(self.state.board[(0, 0, 0)], NodeState.EMPTY)
        self.assertIs(self.state.turn_phase, TurnPhase.OPENING)
        self.assertIs(self.state.active_player, Player.BLACK)
        self.assertEqual(self.state.current_turn, 1)

    def test_redo_after_undo(self):
        self.engine.apply_placement(self.state, (0, 0, 0))
        self.engine.undo(self.state)
        self.engine.redo(self.state)
        self.assertEqual(self.state.board[(0, 0, 0)], NodeState.BLACK)
        self.assertIs(self.state.active_player, Player.WHITE)

    def test_undo_crosses_turn_boundary(self):
        self.engine.apply_placement(self.state, (0, 0, 0))   # Black opening
        self.engine.apply_placement(self.state, (1, 0, 0))   # White 1/2
        self.assertIs(self.state.turn_phase, TurnPhase.NORMAL_2)
        self.assertEqual(self.state.current_turn, 2)
        self.engine.undo(self.state)
        self.assertIs(self.state.turn_phase, TurnPhase.NORMAL_1)
        self.assertIs(self.state.active_player, Player.WHITE)
        self.assertEqual(self.state.current_turn, 2)

    def test_undo_decreases_turn_number(self):
        self.engine.apply_placement(self.state, (0, 0, 0))
        self.assertEqual(self.state.current_turn, 2)
        self.engine.undo(self.state)
        self.assertEqual(self.state.current_turn, 1)

    def test_undo_pass(self):
        self.engine.apply_pass(self.state)
        self.assertEqual(self.state.consecutive_pass_count, 1)
        self.engine.undo(self.state)
        self.assertEqual(self.state.consecutive_pass_count, 0)
        self.assertIs(self.state.turn_phase, TurnPhase.OPENING)

    def test_new_placement_clears_redo(self):
        self.engine.apply_placement(self.state, (0, 0, 0))
        self.engine.undo(self.state)
        self.assertTrue(self.engine.can_redo(self.state))
        self.engine.apply_placement(self.state, (1, 0, 0))
        self.assertFalse(self.engine.can_redo(self.state))

    def test_multiple_undos_and_redos(self):
        self.engine.apply_placement(self.state, (0, 0, 0))
        self.engine.apply_placement(self.state, (1, 0, 0))
        self.engine.apply_placement(self.state, (-1, 0, 0))
        self.engine.undo(self.state)
        self.engine.undo(self.state)
        self.engine.undo(self.state)
        self.assertEqual(self.state.board[(0, 0, 0)], NodeState.EMPTY)
        self.engine.redo(self.state)
        self.assertEqual(self.state.board[(0, 0, 0)], NodeState.BLACK)
        self.engine.redo(self.state)
        self.engine.redo(self.state)
        self.assertEqual(self.state.board[(-1, 0, 0)], NodeState.WHITE)

    def test_undo_on_empty_history_raises(self):
        with self.assertRaises(MoveError):
            self.engine.undo(self.state)

    def test_redo_on_empty_stack_raises(self):
        with self.assertRaises(MoveError):
            self.engine.redo(self.state)


# ====================================================================
# Persistence
# ====================================================================

class TestPersistence(unittest.TestCase):
    def test_roundtrip_via_file(self):
        rules = RulesConfig(board_radius=2, stones_per_player=10,
                            end_on_all_stones_placed=True)
        topology = BoardTopology(2)
        engine = MoveEngine(rules, topology)
        state = engine.initial_state()
        engine.apply_placement(state, (0, 0, 0))
        engine.apply_placement(state, (1, 0, 0))
        engine.apply_placement(state, (-1, 0, 0))

        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        path = f.name
        f.close()
        try:
            save_to_file(path, state, rules)
            rules2, state2 = load_from_file(path)
            self.assertEqual(rules.to_dict(), rules2.to_dict())
            self.assertEqual(state.board, state2.board)
            self.assertIs(state.active_player, state2.active_player)
            self.assertIs(state.turn_phase, state2.turn_phase)
            self.assertEqual(state.current_turn, state2.current_turn)
            self.assertEqual(state.stones_remaining, state2.stones_remaining)
            self.assertEqual(state.consecutive_pass_count,
                             state2.consecutive_pass_count)
            self.assertEqual(len(state.move_history), len(state2.move_history))
        finally:
            os.unlink(path)

    def test_schema_mismatch_errors(self):
        with self.assertRaises(PersistenceError):
            deserialize_state({"schema_version": 999, "rules": {}})

    def test_missing_schema_version_errors(self):
        with self.assertRaises(PersistenceError):
            deserialize_state({"rules": {}})

    def test_redo_stack_not_persisted(self):
        rules = RulesConfig(board_radius=2)
        engine = MoveEngine(rules, BoardTopology(2))
        state = engine.initial_state()
        engine.apply_placement(state, (0, 0, 0))
        engine.undo(state)
        self.assertEqual(len(state.redo_stack), 1)
        data = serialize_state(state, rules)
        self.assertNotIn("redo_stack", data)
        _, restored = deserialize_state(data)
        self.assertEqual(len(restored.redo_stack), 0)


# ====================================================================
# Sandbox
# ====================================================================

class TestSandbox(unittest.TestCase):
    def test_sandbox_place_bypasses_turn(self):
        rules = RulesConfig(board_radius=2)
        engine = MoveEngine(rules, BoardTopology(2))
        state = engine.initial_state()
        engine.sandbox_place(state, (0, 0, 0), NodeState.WHITE)
        engine.sandbox_place(state, (1, 0, 0), NodeState.WHITE)
        self.assertEqual(state.board[(0, 0, 0)], NodeState.WHITE)
        self.assertIs(state.active_player, Player.BLACK)
        self.assertEqual(len(state.move_history), 0)
        # Resume normal play.
        engine.apply_placement(state, (0, 0, 1))
        self.assertEqual(state.board[(0, 0, 1)], NodeState.BLACK)

    def test_sandbox_remove(self):
        rules = RulesConfig(board_radius=2)
        engine = MoveEngine(rules, BoardTopology(2))
        state = engine.initial_state()
        engine.sandbox_place(state, (0, 0, 0), NodeState.BLACK)
        engine.sandbox_remove(state, (0, 0, 0))
        self.assertEqual(state.board[(0, 0, 0)], NodeState.EMPTY)

    def test_sandbox_not_in_move_count(self):
        rules = RulesConfig(board_radius=2)
        engine = MoveEngine(rules, BoardTopology(2))
        state = engine.initial_state()
        engine.sandbox_place(state, (0, 0, 0), NodeState.BLACK)
        engine.sandbox_place(state, (1, 0, 0), NodeState.WHITE)
        self.assertEqual(state.move_count(), 0)
        engine.apply_placement(state, (0, 0, 1))
        self.assertEqual(state.move_count(), 1)

    def test_sandbox_does_not_touch_supply(self):
        rules = RulesConfig(board_radius=2, stones_per_player=5,
                            end_on_all_stones_placed=True)
        engine = MoveEngine(rules, BoardTopology(2))
        state = engine.initial_state()
        engine.sandbox_place(state, (0, 0, 0), NodeState.BLACK)
        self.assertEqual(state.stones_remaining[Player.BLACK], 5)


# ====================================================================
# Game resolution
# ====================================================================

class TestGameResolution(unittest.TestCase):
    def test_black_wins(self):
        rules = RulesConfig(board_radius=1)
        topology = BoardTopology(1)
        engine = MoveEngine(rules, topology)
        state = engine.initial_state()
        for node in topology.iterate_nodes():
            engine.sandbox_place(state, node, NodeState.BLACK)
        engine._determine_winner(state)
        self.assertIs(state.winner, Player.BLACK)

    def test_draw(self):
        rules = RulesConfig(board_radius=1)
        topology = BoardTopology(1)
        engine = MoveEngine(rules, topology)
        state = engine.initial_state()
        engine._determine_winner(state)
        self.assertEqual(state.winner, "draw")


# ====================================================================
# JSON test runner
# ====================================================================

class TestJSONRunner(unittest.TestCase):
    def test_basic_test_spec_passes(self):
        spec = {
            "name": "smoke",
            "rules": {"board_radius": 2},
            "moves": [
                {"cmd": "assert_active_player", "expected": "black"},
                {"cmd": "assert_turn_phase", "expected": "opening"},
                {"cmd": "place", "node": [0, 0, 0]},
                {"cmd": "assert_active_player", "expected": "white"},
                {"cmd": "assert_turn_phase", "expected": "normal_1"},
                {"cmd": "assert_turn_number", "expected": 2},
                {"cmd": "assert_move_count", "expected": 1},
            ],
        }
        result = run_test(spec)
        self.assertTrue(result["passed"], msg=result["error"])

    def test_unknown_cmd_fails_and_stops(self):
        spec = {"moves": [{"cmd": "nonsense_command"}]}
        result = run_test(spec)
        self.assertFalse(result["passed"])
        self.assertIn("unknown command", result["error"])

    def test_wrong_assertion_fails(self):
        spec = {"moves": [
            {"cmd": "assert_active_player", "expected": "white"}  # actually black
        ]}
        result = run_test(spec)
        self.assertFalse(result["passed"])

    def test_sandbox_plus_assert_scoring(self):
        spec = {
            "name": "sandbox-6-cycle",
            "rules": {"board_radius": 1},
            "moves": [
                {"cmd": "sandbox_place", "node": [0, 0, 0], "color": "black"},
                {"cmd": "sandbox_place", "node": [-1, 0, 0], "color": "black"},
                {"cmd": "sandbox_place", "node": [0, -1, 0], "color": "black"},
                {"cmd": "sandbox_place", "node": [-1, 0, 1], "color": "black"},
                {"cmd": "sandbox_place", "node": [0, -1, 1], "color": "black"},
                {"cmd": "sandbox_place", "node": [-1, -1, 1], "color": "black"},
                {"cmd": "assert_score", "player": "black", "expected": 6},
                {"cmd": "assert_score", "player": "white", "expected": 0},
            ],
        }
        result = run_test(spec)
        self.assertTrue(result["passed"], msg=result["error"])


# ====================================================================
# AI hooks
# ====================================================================

class TestAIHooks(unittest.TestCase):
    def test_evaluate(self):
        from cycle_control.ai_hooks import evaluate
        rules = RulesConfig(board_radius=1)
        topology = BoardTopology(1)
        engine = MoveEngine(rules, topology)
        state = engine.initial_state()
        for node in topology.iterate_nodes():
            engine.sandbox_place(state, node, NodeState.BLACK)
        ev = evaluate(engine, state, Player.BLACK)
        self.assertEqual(ev["own"], 6)
        self.assertEqual(ev["opponent"], 0)
        self.assertEqual(ev["diff"], 6)

    def test_botrng_deterministic(self):
        from cycle_control.ai_hooks import BotRNG
        a = BotRNG(seed=42)
        b = BotRNG(seed=42)
        self.assertEqual(a.random(), b.random())

    def test_botrng_per_bot(self):
        from cycle_control.ai_hooks import BotRNG
        a = BotRNG(seed=1)
        b = BotRNG(seed=2)
        self.assertNotEqual(a.random(), b.random())


# ====================================================================
# Experimental balance modes
# ====================================================================

class TestMirrorAdjacency(unittest.TestCase):
    def test_degree_up_to_6(self):
        t = BoardTopology(3, mirror_adjacency=True)
        degrees = [len(t.get_neighbors(n)) for n in t.iterate_nodes()]
        self.assertEqual(max(degrees), 6)
        # Interior node should have all 6.
        self.assertEqual(len(t.get_neighbors((0, 0, 0))), 6)

    def test_bipartite_preserved(self):
        t = BoardTopology(3, mirror_adjacency=True)
        for u in t.iterate_nodes():
            for v in t.get_neighbors(u):
                self.assertNotEqual(u[2], v[2])

    def test_girth_4(self):
        t = BoardTopology(3, mirror_adjacency=True)
        self.assertEqual(t._compute_girth(), 4)

    def test_mirror_adjacency_symmetric(self):
        t = BoardTopology(3, mirror_adjacency=True)
        for u in t.iterate_nodes():
            for v in t.get_neighbors(u):
                self.assertIn(u, t.get_neighbors(v))

    def test_mirror_neighbors_spot_check(self):
        t = BoardTopology(3, mirror_adjacency=True)
        neighs = set(t.get_neighbors((0, 0, 0)))
        # side
        for s in [(0, 0, 1), (-1, 0, 1), (0, -1, 1)]:
            self.assertIn(s, neighs)
        # mirror
        for m in [(-1, -1, 1), (1, -1, 1), (-1, 1, 1)]:
            self.assertIn(m, neighs)

    def test_rules_mismatch_raises(self):
        rules = RulesConfig(board_radius=2, mirror_adjacency=True)
        plain = BoardTopology(2, mirror_adjacency=False)
        with self.assertRaises(ValueError):
            MoveEngine(rules, plain)

    def test_4_cycle_scores(self):
        """Under mirror adjacency, 4 stones can close a cycle."""
        rules = RulesConfig(board_radius=2, mirror_adjacency=True)
        t = BoardTopology(2, mirror_adjacency=True)
        engine = MoveEngine(rules, t)
        state = engine.initial_state()
        # The 4-cycle up(0,0,0) - down(0,0,1) - up(0,1,0) - down(-1,0,1).
        for n in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (-1, 0, 1)]:
            engine.sandbox_place(state, n, NodeState.BLACK)
        sc = scoring_nodes(t, state.board, Player.BLACK)
        self.assertEqual(len(sc), 4)


class TestNeutralityRule(unittest.TestCase):
    def setUp(self):
        self.rules = RulesConfig(board_radius=2, neutrality_rule=True)
        self.topology = BoardTopology(2)
        self.engine = MoveEngine(self.rules, self.topology)
        self.state = self.engine.initial_state()

    def test_opening_anywhere_legal(self):
        # No stones on board, neutrality trivially holds (0 >= 0).
        self.assertTrue(self.engine.is_legal_placement(self.state, (0, 0, 0)))

    def test_cannot_parachute_into_opponent_territory(self):
        # Set up: White has 2 adjacent stones to some cell. Black has 0.
        # Black (to move) cannot place there: 0 < 2.
        target = (0, 0, 0)
        nbrs = self.topology.get_neighbors(target)
        # Sandbox 2 opponent stones adjacent, then make it Black's turn.
        self.engine.sandbox_place(self.state, nbrs[0], NodeState.WHITE)
        self.engine.sandbox_place(self.state, nbrs[1], NodeState.WHITE)
        # Still Black's opening.
        self.assertFalse(self.engine.is_legal_placement(self.state, target))

    def test_allowed_when_balanced(self):
        target = (0, 0, 0)
        nbrs = self.topology.get_neighbors(target)
        self.engine.sandbox_place(self.state, nbrs[0], NodeState.BLACK)
        self.engine.sandbox_place(self.state, nbrs[1], NodeState.WHITE)
        # 1 own >= 1 opp → legal for Black.
        self.assertTrue(self.engine.is_legal_placement(self.state, target))


class TestStrictAdjacency(unittest.TestCase):
    def setUp(self):
        self.rules = RulesConfig(board_radius=2, strict_adjacency_rule=True)
        self.topology = BoardTopology(2)
        self.engine = MoveEngine(self.rules, self.topology)
        self.state = self.engine.initial_state()

    def test_first_placement_anywhere(self):
        # Black has 0 stones, rule suspended.
        self.assertTrue(self.engine.is_legal_placement(self.state, (0, 0, 0)))

    def test_second_placement_must_touch_own(self):
        self.engine.apply_placement(self.state, (0, 0, 0))  # B opens
        # White has 0 stones, rule suspended — any empty cell legal.
        self.assertTrue(self.engine.is_legal_placement(self.state, (-2, -1, 1)))
        self.engine.apply_placement(self.state, (-2, -1, 1))  # W 1/2
        # White has 1 stone at (-2, -1, 1). Next placement must touch it.
        # (1, -1, 0) is on board, empty, but not adjacent to (-2, -1, 1).
        self.assertFalse(self.engine.is_legal_placement(self.state, (1, -1, 0)))
        # (-1, -1, 0) IS adjacent to (-2, -1, 1). Must be legal.
        self.assertTrue(self.engine.is_legal_placement(self.state, (-1, -1, 0)))

    def test_black_second_move_must_touch_own(self):
        self.engine.apply_placement(self.state, (0, 0, 0))    # B opens
        self.engine.apply_placement(self.state, (-2, -1, 1))  # W 1/2 (suspended)
        self.engine.apply_placement(self.state, (-2, 0, 0))   # W 2/2 (adj to W stone)
        # Black to move, has 1 stone at (0, 0, 0). Must touch it.
        # (-1, 1, 0) is on board, empty, not adjacent to (0, 0, 0).
        self.assertFalse(self.engine.is_legal_placement(self.state, (-1, 1, 0)))


class TestPartialCreditScoring(unittest.TestCase):
    def test_component_of_size_3_scores_with_k_3(self):
        t = BoardTopology(2)
        board = {n: NodeState.EMPTY for n in t.iterate_nodes()}
        # A path of 3: pick 3 connected nodes in a line.
        path = [(0, 0, 0), (0, 0, 1), (0, 1, 0)]
        for n in path:
            board[n] = NodeState.BLACK
        # Without partial credit: 0 (path is all bridges).
        self.assertEqual(len(scoring_nodes(t, board, Player.BLACK, 0)), 0)
        # With K=3: all 3 score.
        self.assertEqual(len(scoring_nodes(t, board, Player.BLACK, 3)), 3)

    def test_component_below_k_does_not_score(self):
        t = BoardTopology(2)
        board = {n: NodeState.EMPTY for n in t.iterate_nodes()}
        board[(0, 0, 0)] = NodeState.BLACK
        board[(0, 0, 1)] = NodeState.BLACK  # size-2 component
        self.assertEqual(len(scoring_nodes(t, board, Player.BLACK, 3)), 0)
        self.assertEqual(len(scoring_nodes(t, board, Player.BLACK, 2)), 2)

    def test_cycle_plus_partial_credit_union(self):
        """6-cycle plus a separate 3-path. With K=3: 9 score (6 + 3)."""
        t = BoardTopology(3)
        board = {n: NodeState.EMPTY for n in t.iterate_nodes()}
        r1_nodes = set(BoardTopology(1).iterate_nodes())
        for n in r1_nodes:
            board[n] = NodeState.BLACK

        # Find a 3-node path outside R=1 and with no neighbor inside R=1
        # (so it stays a disconnected component).
        r1_or_neighbors = set(r1_nodes)
        for n in r1_nodes:
            r1_or_neighbors.update(t.get_neighbors(n))

        path: list = []
        for start in t.iterate_nodes():
            if start in r1_or_neighbors:
                continue
            visited = {start}
            path_attempt = [start]
            while len(path_attempt) < 3:
                last = path_attempt[-1]
                nxt = next((nb for nb in t.get_neighbors(last)
                            if nb not in visited and nb not in r1_or_neighbors),
                           None)
                if nxt is None:
                    break
                path_attempt.append(nxt)
                visited.add(nxt)
            if len(path_attempt) == 3:
                path = path_attempt
                break
        self.assertEqual(len(path), 3, "could not find a disconnected 3-path")

        for n in path:
            board[n] = NodeState.BLACK
        self.assertEqual(len(scoring_nodes(t, board, Player.BLACK, 0)), 6)
        self.assertEqual(len(scoring_nodes(t, board, Player.BLACK, 3)), 9)


class TestBalanceModesCompose(unittest.TestCase):
    def test_all_four_enabled_runs(self):
        rules = RulesConfig(
            board_radius=2,
            neutrality_rule=True,
            strict_adjacency_rule=True,
            mirror_adjacency=True,
            partial_credit_k=3,
        )
        t = BoardTopology(2, mirror_adjacency=True)
        engine = MoveEngine(rules, t)
        state = engine.initial_state()
        # Black opening: both placement restrictions suspended (no own stones
        # + trivial neutrality 0>=0). Any legal cell is fine.
        engine.apply_placement(state, (0, 0, 0))
        self.assertEqual(state.board[(0, 0, 0)], NodeState.BLACK)

    def test_rules_persist_round_trip(self):
        rules = RulesConfig(
            board_radius=2,
            neutrality_rule=True,
            strict_adjacency_rule=True,
            mirror_adjacency=True,
            partial_credit_k=5,
        )
        r2 = RulesConfig.from_dict(rules.to_dict())
        self.assertTrue(r2.neutrality_rule)
        self.assertTrue(r2.strict_adjacency_rule)
        self.assertTrue(r2.mirror_adjacency)
        self.assertEqual(r2.partial_credit_k, 5)

    def test_reject_negative_k(self):
        with self.assertRaises(ValueError):
            RulesConfig(board_radius=2, partial_credit_k=-1)


if __name__ == "__main__":
    unittest.main()
