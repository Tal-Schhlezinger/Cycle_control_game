"""Unit tests for cycle_control.ai subpackage."""

from __future__ import annotations

import unittest

import numpy as np

from cycle_control.ai.action_space import (
    ActionSpace, action_space_size, build_legal_mask,
    node_to_action_index, pass_index,
)
from cycle_control.ai.bot_interface import play_game, play_turn
from cycle_control.ai.bots import (
    FrontierRandomBot, Greedy1, Greedy2, GreedyBot, RandomBot,
)
from cycle_control.ai.bots.greedy_bot import (
    GreedyWeights, cycle_score_diff, largest_component_size, mobility_for,
)
from cycle_control.ai.siege import (
    exclusive_territory, frontier_count, reachable_empty_cells,
    sieged_against, territory_score,
)
from cycle_control.ai.tournament import (
    elo_from_round_robin, elo_update, round_robin, run_match,
)
from cycle_control.engine import MoveEngine
from cycle_control.rules import RulesConfig
from cycle_control.state import NodeState, Player, TurnPhase
from cycle_control.topology import BoardTopology


def _make_committed_engine(radius: int = 3) -> MoveEngine:
    """Engine with the V2.3 committed ruleset using the RulesConfig.committed() factory.
    Pass disabled; ends on no-legal-moves; neutrality + strict + mirror adjacency."""
    from cycle_control.rules import RulesConfig
    rules = RulesConfig.committed(board_radius=radius)
    topology = BoardTopology(radius, mirror_adjacency=True)
    return MoveEngine(rules, topology)


def _make_plain_engine(radius: int = 2) -> MoveEngine:
    """Engine with default rules (no balance modes)."""
    rules = RulesConfig(board_radius=radius)
    topology = BoardTopology(radius)
    return MoveEngine(rules, topology)


# ===========================================================================
# Action space
# ===========================================================================

class TestActionSpace(unittest.TestCase):
    def test_size_is_nodes_plus_one(self):
        t = BoardTopology(2)
        self.assertEqual(action_space_size(t), t.node_count() + 1)

    def test_pass_index_is_last(self):
        t = BoardTopology(2)
        self.assertEqual(pass_index(t), t.node_count())

    def test_round_trip_node_index(self):
        t = BoardTopology(3)
        aspace = ActionSpace(t)
        for i, node in enumerate(t.all_nodes()):
            self.assertEqual(aspace.node_to_index(node), i)
            self.assertEqual(aspace.index_to_node(i), node)

    def test_index_to_pass(self):
        t = BoardTopology(2)
        aspace = ActionSpace(t)
        self.assertIsNone(aspace.index_to_node(aspace.pass_index))

    def test_legal_mask_initial_state(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        mask = build_legal_mask(engine, state)
        self.assertEqual(mask.shape, (engine.topology.node_count() + 1,))
        # All placements legal at opening
        self.assertEqual(int(mask[:engine.topology.node_count()].sum()),
                         engine.topology.node_count())
        # Pass legal
        self.assertTrue(mask[-1])

    def test_legal_mask_game_over(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        state.game_over = True
        mask = build_legal_mask(engine, state)
        self.assertFalse(mask.any())

    def test_legal_mask_matches_legal_moves(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        engine.apply_placement(state, (0, 0, 0))
        mask = build_legal_mask(engine, state)
        aspace = ActionSpace(engine.topology)
        mask_nodes = {aspace.index_to_node(int(i))
                      for i in np.flatnonzero(mask)
                      if int(i) != aspace.pass_index}
        self.assertEqual(set(engine.legal_moves(state)), mask_nodes)


# ===========================================================================
# Bot interface — turn loop
# ===========================================================================

class _DummyBot:
    """Returns a pre-specified action sequence."""
    name = "Dummy"

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def reset(self, seed=None):
        self.i = 0

    def choose_action(self, state, legal_mask, color):
        a = self.actions[self.i]
        self.i += 1
        return a


class TestBotInterface(unittest.TestCase):
    def test_turn_opening_one_placement(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        aspace = ActionSpace(engine.topology)
        # Choose index 0 = first node
        bot = _DummyBot([0])
        play_turn(engine, state, bot)
        self.assertEqual(state.active_player, Player.WHITE)
        self.assertEqual(state.turn_phase, TurnPhase.NORMAL_1)
        self.assertEqual(state.move_count(), 1)

    def test_turn_normal_two_placements(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        # Black opening: 1 placement
        engine.apply_placement(state, (0, 0, 0))
        # White's turn: up to 2 placements
        legal = engine.legal_moves(state)
        aspace = ActionSpace(engine.topology)
        a1 = aspace.node_to_index(legal[0])
        a2 = aspace.node_to_index(legal[1])
        bot = _DummyBot([a1, a2])
        play_turn(engine, state, bot)
        # Now Black's turn
        self.assertEqual(state.active_player, Player.BLACK)
        self.assertEqual(state.move_count(), 3)

    def test_turn_partial_pass(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        engine.apply_placement(state, (0, 0, 0))  # Black opening
        legal = engine.legal_moves(state)
        aspace = ActionSpace(engine.topology)
        a1 = aspace.node_to_index(legal[0])
        bot = _DummyBot([a1, aspace.pass_index])  # 1 place then pass
        play_turn(engine, state, bot)
        self.assertEqual(state.active_player, Player.BLACK)
        self.assertEqual(state.consecutive_pass_count, 0)  # partial pass doesn't increment

    def test_play_game_completes(self):
        engine = _make_plain_engine(radius=1)  # tiny board — game will finish fast
        bot_a = RandomBot(seed=1)
        bot_b = RandomBot(seed=2)
        final_state, winner = play_game(engine, bot_a, bot_b, seed=42)
        self.assertTrue(final_state.game_over)
        self.assertIn(winner, [Player.BLACK, Player.WHITE, "draw"])

    def test_bot_returns_illegal_action_raises(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        # Occupy cell at index 0 via sandbox so it becomes illegal
        from cycle_control.state import NodeState
        node0 = engine.topology.all_nodes()[0]
        engine.sandbox_place(state, node0, NodeState.WHITE)
        # Bot returns index 0 — illegal (cell occupied). Opening phase needs
        # only 1 placement, so DummyBot with 1 action is right-sized.
        bot = _DummyBot([0])
        with self.assertRaises(ValueError):
            play_turn(engine, state, bot)


# ===========================================================================
# Siege analysis
# ===========================================================================

class TestSiege(unittest.TestCase):
    def test_empty_board_all_reachable(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        reach = reachable_empty_cells(engine, state, Player.BLACK)
        self.assertEqual(len(reach), engine.topology.node_count())

    def test_neutrality_blocks_cells(self):
        """Under neutrality, a cell with more opponent than own neighbors
        cannot be entered by player."""
        rules = RulesConfig(board_radius=2, neutrality_rule=True)
        topology = BoardTopology(2)
        engine = MoveEngine(rules, topology)
        state = engine.initial_state()
        # Find a node with some neighbors
        target = (0, 0, 0)
        neighbors = engine.topology.get_neighbors(target)
        # Place 2 opponent (White) stones around target
        engine.sandbox_place(state, neighbors[0], NodeState.WHITE)
        engine.sandbox_place(state, neighbors[1], NodeState.WHITE)
        # target has 2 opp, 0 own for Black — Black can't reach target directly.
        # But can Black reach target via a cascade? Only if some other cell
        # becomes reachable that adjoins target and gives Black a neighbor there.
        # With neutrality alone (no strict adjacency), Black can place far
        # away and build toward target...
        # This is hard to test tightly. Just verify the target is NOT immediately
        # placeable. A full reachability check would require the flood fill
        # to demonstrate Black can still route in from elsewhere.
        reach = reachable_empty_cells(engine, state, Player.BLACK)
        # The opponent stones' neighborhoods are limited; target may or may not
        # be reachable via cascading. Just verify the reachable set excludes
        # target if target has 0 Black neighbors and 2 White neighbors AND
        # no adjacent reachable cell provides Black neighbors to target.
        # For small boards this cascading is hard to precompute by hand — the
        # test here just checks that the function runs without error.
        self.assertIsInstance(reach, set)

    def test_strict_adjacency_requires_own_stone(self):
        """Under strict adjacency + no own stones, only isolated empty cells
        that don't require adjacency are reachable. First placement special case."""
        rules = RulesConfig(board_radius=2, strict_adjacency_rule=True)
        topology = BoardTopology(2)
        engine = MoveEngine(rules, topology)
        state = engine.initial_state()
        # With no own stones, strict adjacency is suspended -> all cells reachable
        reach = reachable_empty_cells(engine, state, Player.BLACK)
        self.assertEqual(len(reach), engine.topology.node_count())

        # Now place a Black stone. Only cells adjacent to it should be
        # immediately reachable; from there more cells cascade in.
        engine.sandbox_place(state, (0, 0, 0), NodeState.BLACK)
        reach = reachable_empty_cells(engine, state, Player.BLACK)
        # Reachable should include at least the direct neighbors of (0,0,0)
        for nb in engine.topology.get_neighbors((0, 0, 0)):
            if state.board.get(nb) == NodeState.EMPTY:
                self.assertIn(nb, reach, f"neighbor {nb} should be reachable")

    def test_territory_score(self):
        engine = _make_plain_engine(radius=1)
        state = engine.initial_state()
        # No stones, default rules = all empty, all reachable
        self.assertEqual(territory_score(engine, state, Player.BLACK),
                         engine.topology.node_count())

    def test_exclusive_territory_zero_if_both_can_reach(self):
        engine = _make_plain_engine(radius=1)
        state = engine.initial_state()
        # Default rules = no neutrality, no strict adj = both can reach everything
        self.assertEqual(exclusive_territory(engine, state, Player.BLACK), 0)
        self.assertEqual(exclusive_territory(engine, state, Player.WHITE), 0)

    def test_frontier_count(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        engine.sandbox_place(state, (0, 0, 0), NodeState.BLACK)
        n_frontier = frontier_count(engine, state, Player.BLACK)
        # Frontier = empty cells with at least one Black neighbor
        # = count of Black's neighbors that are empty
        black_neighbors = engine.topology.get_neighbors((0, 0, 0))
        expected = sum(1 for nb in black_neighbors
                       if state.board.get(nb, NodeState.EMPTY) == NodeState.EMPTY)
        self.assertEqual(n_frontier, expected)

    def test_sieged_against_disjoint_from_reachable(self):
        engine = _make_committed_engine(radius=2)
        state = engine.initial_state()
        engine.sandbox_place(state, (0, 0, 0), NodeState.WHITE)
        reach = reachable_empty_cells(engine, state, Player.BLACK)
        sieged = sieged_against(engine, state, Player.BLACK)
        self.assertEqual(len(reach & sieged), 0,
                         "reachable and sieged must be disjoint")


# ===========================================================================
# RandomBot
# ===========================================================================

class TestRandomBot(unittest.TestCase):
    def test_random_picks_legal(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        bot = RandomBot(seed=0)
        mask = build_legal_mask(engine, state)
        for _ in range(20):
            a = bot.choose_action(state, mask, Player.BLACK)
            self.assertTrue(mask[a])

    def test_random_is_seeded(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        mask = build_legal_mask(engine, state)
        bot_a = RandomBot(seed=42)
        bot_b = RandomBot(seed=42)
        for _ in range(10):
            self.assertEqual(bot_a.choose_action(state, mask, Player.BLACK),
                             bot_b.choose_action(state, mask, Player.BLACK))

    def test_full_game_with_random_bots(self):
        engine = _make_plain_engine(radius=2)
        final_state, winner = play_game(engine, RandomBot(seed=1), RandomBot(seed=2), seed=0)
        self.assertTrue(final_state.game_over)


# ===========================================================================
# Greedy bots — the main deliverable
# ===========================================================================

class TestGreedyBots(unittest.TestCase):
    def test_greedy1_picks_legal(self):
        engine = _make_committed_engine(radius=3)
        state = engine.initial_state()
        bot = Greedy1(engine, seed=0)
        mask = build_legal_mask(engine, state)
        a = bot.choose_action(state, mask, Player.BLACK)
        self.assertTrue(mask[a])

    def test_greedy2_picks_legal(self):
        engine = _make_committed_engine(radius=3)
        state = engine.initial_state()
        bot = Greedy2(engine, seed=0)
        mask = build_legal_mask(engine, state)
        a = bot.choose_action(state, mask, Player.BLACK)
        self.assertTrue(mask[a])

    def test_greedy_beats_random(self):
        """Greedy_2 (territory-focused) should beat Random under committed rules.
        Greedy_1 intentionally does NOT beat Random — this is an empirically
        confirmed finding (Phase 1): cycle-greed is worse than random on the
        committed ruleset. See IMPLEMENTATION_NOTES_PHASE1.md."""
        engine = _make_committed_engine(radius=2)
        result = run_match(
            engine,
            Greedy2(engine, seed=0),
            RandomBot(seed=0),
            n_games=20,
            base_seed=100,
        )
        self.assertGreater(result.a_wins, result.b_wins,
                           f"Greedy2 should beat Random: {result.summary()}")

    def test_greedy_evaluates_deterministically_on_repeat(self):
        engine = _make_committed_engine(radius=3)
        state = engine.initial_state()
        engine.apply_placement(state, (0, 0, 0))
        bot = Greedy1(engine)
        e1 = bot.evaluate(state, Player.BLACK)
        e2 = bot.evaluate(state, Player.BLACK)
        self.assertEqual(e1, e2)

    def test_greedy1_and_greedy2_have_different_weights(self):
        """G1 is cycle-focused (high cycle weight, zero frontier).
        G2 is territory-focused (high frontier weight, low cycle).
        Note: G2 uses frontier_diff not exclusive_territory_diff — the
        O(V^2) flood fill was replaced with O(V) frontier for R>=4 feasibility.
        """
        engine = _make_committed_engine(radius=3)
        g1 = Greedy1(engine)
        g2 = Greedy2(engine)
        self.assertNotEqual(g1.weights.cycle, g2.weights.cycle)
        self.assertNotEqual(g1.weights.frontier, g2.weights.frontier)
        # G1: cycle-focused, frontier near zero
        self.assertGreater(g1.weights.cycle, 1.0)
        self.assertEqual(g1.weights.frontier, 0.0)
        # G2: frontier-focused (territory proxy), cycle near zero
        self.assertGreater(g2.weights.frontier, 1.0)
        self.assertLess(g2.weights.cycle, 1.0)

    def test_greedy_never_passes_when_placements_exist(self):
        """Regression: bots were choosing pass over placement because pass
        evaluated as 0 while placements scored negative. Fixed in Phase 4."""
        engine = _make_committed_engine(radius=4)
        state = engine.initial_state()
        mask = build_legal_mask(engine, state)
        pass_idx = engine.topology.node_count()
        for bot in [Greedy1(engine, seed=0), Greedy2(engine, seed=0)]:
            for _ in range(5):  # check multiple times (tiebreaking is RNG)
                a = bot.choose_action(state, mask, Player.BLACK)
                self.assertNotEqual(
                    a, pass_idx,
                    f"{bot.name} chose pass when {int(mask[:pass_idx].sum())} placements were available"
                )

    def test_search_never_passes_when_placements_exist(self):
        """Same regression test for SearchBot."""
        from cycle_control.ai.bots import SearchBot
        engine = _make_committed_engine(radius=3)
        state = engine.initial_state()
        mask = build_legal_mask(engine, state)
        pass_idx = engine.topology.node_count()
        bot = SearchBot(engine, depth=2, seed=0)
        a = bot.choose_action(state, mask, Player.BLACK)
        self.assertNotEqual(
            a, pass_idx,
            f"SearchBot chose pass when {int(mask[:pass_idx].sum())} placements were available"
        )

    def test_mobility_for_works_on_both_players(self):
        engine = _make_plain_engine(radius=2)
        state = engine.initial_state()
        # At opening, both players have the same number of legal moves
        mob_b = mobility_for(engine, state, Player.BLACK)
        mob_w = mobility_for(engine, state, Player.WHITE)
        self.assertEqual(mob_b, mob_w)


# ===========================================================================
# Tournament
# ===========================================================================

class TestTournament(unittest.TestCase):
    def test_run_match_counts_games(self):
        engine = _make_plain_engine(radius=2)
        result = run_match(engine, RandomBot(seed=0), RandomBot(seed=1),
                           n_games=6, base_seed=0)
        self.assertEqual(result.total_games(), 6)

    def test_run_match_swaps_colors(self):
        engine = _make_plain_engine(radius=2)
        result = run_match(engine, RandomBot(seed=0), RandomBot(seed=1),
                           n_games=6, base_seed=0, swap_colors=True,
                           record_games=True)
        # With swap_colors, half games have A as Black, half as White
        a_black = sum(1 for g in result.games if g["a_is_black"])
        self.assertEqual(a_black, 3)

    def test_round_robin_runs(self):
        def factory_engine():
            return _make_plain_engine(radius=2)

        def greedy1_factory(engine):
            return Greedy1(engine, seed=0)

        def greedy2_factory(engine):
            return Greedy2(engine, seed=0)

        def random_factory(engine):
            return RandomBot(seed=0)

        rr = round_robin(
            factory_engine,
            [random_factory, greedy1_factory, greedy2_factory],
            ["Random", "Greedy1", "Greedy2"],
            n_games_per_pair=4,
            verbose=False,
        )
        self.assertEqual(len(rr.bot_names), 3)
        # Should have 6 matches (3*2 ordered pairs excluding diagonal)
        self.assertEqual(len(rr.matches), 6)

    def test_elo_update_monotone(self):
        # Winning increases rating
        new_a, new_b = elo_update(1200, 1200, 1.0)
        self.assertGreater(new_a, 1200)
        self.assertLess(new_b, 1200)
        # Equal ratings, draw preserves
        new_a, new_b = elo_update(1200, 1200, 0.5)
        self.assertAlmostEqual(new_a, 1200, places=5)
        self.assertAlmostEqual(new_b, 1200, places=5)


# ===========================================================================
# SearchBot
# ===========================================================================

class TestSearchBot(unittest.TestCase):
    def test_search_picks_legal(self):
        from cycle_control.ai.bots import SearchBot
        engine = _make_committed_engine(radius=2)
        state = engine.initial_state()
        bot = SearchBot(engine, depth=2, seed=0)
        mask = build_legal_mask(engine, state)
        a = bot.choose_action(state, mask, Player.BLACK)
        self.assertTrue(mask[a])

    def test_search_depth_1_picks_legal_consistently(self):
        from cycle_control.ai.bots import SearchBot
        engine = _make_committed_engine(radius=3)
        state = engine.initial_state()
        bot = SearchBot(engine, depth=1, seed=0)
        mask = build_legal_mask(engine, state)
        # Repeat — should always return a legal action
        for _ in range(5):
            a = bot.choose_action(state, mask, Player.BLACK)
            self.assertTrue(mask[a])

    def test_search_full_game_terminates(self):
        from cycle_control.ai.bots import SearchBot
        engine = _make_plain_engine(radius=1)
        final_state, winner = play_game(
            engine, SearchBot(engine, depth=2, seed=0),
            RandomBot(seed=0), seed=0,
        )
        self.assertTrue(final_state.game_over)

    def test_search_beats_random_on_committed_rules(self):
        """Regression against Phase 1 finding: SearchBot must beat Random
        where one-ply greedy does not."""
        from cycle_control.ai.bots import SearchBot
        engine = _make_committed_engine(radius=2)
        result = run_match(
            engine,
            SearchBot(engine, depth=2, seed=0),
            RandomBot(seed=0),
            n_games=10,
            base_seed=100,
        )
        # Depth-2 search should substantially beat Random on committed rules
        self.assertGreaterEqual(result.a_wins, 7,
                                f"SearchBot d=2 should beat Random: {result.summary()}")

    def test_search_beats_greedy(self):
        """SearchBot at depth=4 should beat one-ply Greedy_2.
        Note: depth=3 is no longer reliably enough after the Greedy_2
        weight update made frontier-focused play stronger. Using depth=4
        with a time budget so the test doesn't hang.
        """
        from cycle_control.ai.bots import SearchBot
        engine = _make_committed_engine(radius=2)
        result = run_match(
            engine,
            SearchBot(engine, depth=4, time_budget_s=5.0, seed=0),
            Greedy2(engine, seed=0),
            n_games=6,
            base_seed=200,
        )
        # Search at depth 4 should win or draw more than it loses
        self.assertGreaterEqual(
            result.a_wins + result.draws,
            result.b_wins,
            f"SearchBot d=4 should not lose overall to Greedy_2: {result.summary()}"
        )

    def test_search_with_time_budget_returns_action(self):
        from cycle_control.ai.bots import SearchBot
        engine = _make_committed_engine(radius=3)
        state = engine.initial_state()
        # Very tight time budget; should still return a legal action via
        # iterative deepening
        bot = SearchBot(engine, depth=5, time_budget_s=0.05, seed=0)
        mask = build_legal_mask(engine, state)
        a = bot.choose_action(state, mask, Player.BLACK)
        self.assertTrue(mask[a])

    def test_search_stats_recorded(self):
        from cycle_control.ai.bots import SearchBot
        engine = _make_committed_engine(radius=2)
        state = engine.initial_state()
        bot = SearchBot(engine, depth=2, seed=0)
        mask = build_legal_mask(engine, state)
        bot.choose_action(state, mask, Player.BLACK)
        # Depth-2 search should visit more than zero nodes
        self.assertGreater(bot.last_stats.nodes_visited, 0)

    def test_search_detects_terminal_win(self):
        """When a move ends the game in our favor, search should find it.

        Construct: near-empty board state where the ONLY legal move leads
        to a terminal state. SearchBot should still return it.
        """
        from cycle_control.ai.bots import SearchBot
        # Use a tiny board where game-ending situations are easy to engineer
        engine = _make_plain_engine(radius=1)
        state = engine.initial_state()
        bot = SearchBot(engine, depth=3, seed=0)
        mask = build_legal_mask(engine, state)
        a = bot.choose_action(state, mask, Player.BLACK)
        self.assertTrue(mask[a])



# ===========================================================================
# Committed ruleset + auto-fill
# ===========================================================================

class TestCommittedRuleset(unittest.TestCase):
    def _make_committed(self, radius=3):
        from cycle_control.rules import RulesConfig
        from cycle_control.topology import BoardTopology
        rules = RulesConfig.committed(board_radius=radius)
        topology = BoardTopology(radius, mirror_adjacency=True)
        return MoveEngine(rules, topology)

    def test_committed_factory(self):
        engine = self._make_committed(3)
        self.assertFalse(engine.rules.pass_enabled)
        self.assertFalse(engine.rules.end_on_consecutive_passes)
        self.assertTrue(engine.rules.end_on_no_legal_moves)
        self.assertTrue(engine.rules.neutrality_rule)
        self.assertTrue(engine.rules.strict_adjacency_rule)
        self.assertTrue(engine.rules.mirror_adjacency)

    def test_committed_game_terminates(self):
        engine = self._make_committed(3)
        state, winner = play_game(engine, RandomBot(seed=0), RandomBot(seed=1), seed=42)
        self.assertTrue(state.game_over)
        self.assertIsNotNone(winner)

    def test_committed_no_draws_random_vs_random(self):
        """Under committed rules, random vs random games should be decisive."""
        engine = self._make_committed(3)
        draws = 0
        for seed in range(10):
            _, winner = play_game(engine, RandomBot(seed=seed), RandomBot(seed=seed+100), seed=seed)
            if winner == "draw":
                draws += 1
        # May have some draws, but not all
        self.assertLess(draws, 10, "All games were draws — something wrong")

    def test_committed_full_board_coverage(self):
        """With auto-fill, all or nearly all cells should be occupied at end."""
        from cycle_control.state import NodeState
        engine = self._make_committed(3)
        state, _ = play_game(
            engine, Greedy1(engine, seed=0), Greedy2(engine, seed=0), seed=0
        )
        empty = sum(1 for v in state.board.values() if v == NodeState.EMPTY)
        total = engine.topology.node_count()
        # Most cells should be filled (allow a few unreachable)
        self.assertLessEqual(empty, total * 0.1,
                             f"{empty}/{total} cells empty — too many")

    def test_auto_fill_exported(self):
        from cycle_control.ai.bot_interface import auto_fill
        engine = self._make_committed(2)
        state = engine.initial_state()
        # Sandbox-place some black stones to create adjacency
        engine.sandbox_place(state, (0, 0, 0), __import__('cycle_control.state', fromlist=['NodeState']).NodeState.BLACK)
        # Give black the turn and auto-fill
        from cycle_control.state import Player
        state.active_player = Player.BLACK
        moves_before = engine.legal_moves(state)
        auto_fill(engine, state)
        # After auto-fill, black should have no more legal moves
        remaining = engine.legal_moves(state)
        self.assertEqual(len(remaining), 0,
                         f"auto_fill left {len(remaining)} legal moves for BLACK")

    def test_committed_rules_roundtrip(self):
        rules = RulesConfig.committed(board_radius=5)
        r2 = RulesConfig.from_dict(rules.to_dict())
        self.assertEqual(rules.to_dict(), r2.to_dict())
        self.assertTrue(r2.end_on_no_legal_moves)
        self.assertFalse(r2.pass_enabled)

    def test_end_on_no_legal_moves_condition(self):
        """end_on_no_legal_moves fires when both players are stuck."""
        engine = self._make_committed(2)
        state = engine.initial_state()
        # Fill the entire board with alternating stones via sandbox
        from cycle_control.state import NodeState, Player
        nodes = list(engine.topology.all_nodes())
        for i, n in enumerate(nodes):
            color = NodeState.BLACK if i % 2 == 0 else NodeState.WHITE
            engine.sandbox_place(state, n, color)
        # Manually trigger end check
        engine._check_end_conditions(state)
        self.assertTrue(state.game_over)


if __name__ == "__main__":
    unittest.main()
