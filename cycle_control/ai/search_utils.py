"""Fast apply/undo for search — delta-based with incremental frontier tracking.

A placement at `node` for `player` changes:
  - board[node]: EMPTY -> player_color
  - frontier of player: node is no longer frontier, its empty neighbors become frontier
  - frontier of opponent: node was potentially opponent's frontier, may no longer be
  - move_history, redo_stack, pass_count, active_player, current_turn,
    turn_phase, game_over, winner, stones_remaining

Incremental frontier tracking reduces leaf eval from O(V) to O(neighbors) = O(6).
"""

from __future__ import annotations

from ..engine import MoveEngine, MoveError
from ..state import GameState, NodeState, Player
from ..topology import Node


class SearchDelta:
    """Saved state before a placement, used to undo it."""
    __slots__ = (
        'node', 'old_cell',
        'old_history_len', 'old_redo_len',
        'old_pass_count', 'old_active_player',
        'old_turn', 'old_phase',
        'old_game_over', 'old_winner',
        'old_supply',
        # incremental frontier tracking
        'frontier_black_delta', 'frontier_white_delta',
    )

    def __init__(self, state: GameState, node: Node,
                 f_black_delta: int = 0, f_white_delta: int = 0):
        self.node = node
        self.old_cell = state.board.get(node, NodeState.EMPTY)
        self.old_history_len = len(state.move_history)
        self.old_redo_len = len(state.redo_stack)
        self.old_pass_count = state.consecutive_pass_count
        self.old_active_player = state.active_player
        self.old_turn = state.current_turn
        self.old_phase = state.turn_phase
        self.old_game_over = state.game_over
        self.old_winner = state.winner
        self.old_supply = dict(state.stones_remaining)
        self.frontier_black_delta = f_black_delta
        self.frontier_white_delta = f_white_delta


class SearchState:
    """Wraps GameState with incremental frontier counts for fast leaf eval.

    Usage:
        ss = SearchState(engine, game_state)
        with ss.move(node):
            score = ss.frontier_diff(Player.BLACK)
        # state and frontier counts are restored after the with block
    """

    def __init__(self, engine: MoveEngine, state: GameState):
        self.engine = engine
        self.state = state
        # Compute initial frontier counts
        self.frontier = {
            Player.BLACK: _count_frontier(engine, state, Player.BLACK),
            Player.WHITE: _count_frontier(engine, state, Player.WHITE),
        }

    def frontier_diff(self, player: Player) -> int:
        return self.frontier[player] - self.frontier[player.other()]

    def apply(self, node: Node) -> SearchDelta:
        """Apply placement at node, update frontier counts, return delta."""
        topology = self.engine.topology
        state = self.state
        player = state.active_player
        own_color = NodeState.from_player(player)
        opp_color = NodeState.from_player(player.other())

        # Compute frontier deltas BEFORE applying the move
        f_black_delta = 0
        f_white_delta = 0

        neighbors = topology.get_neighbors(node)

        # This node was potentially in both players' frontiers — placing here
        # removes it from whichever frontier it was in.
        # A cell is in P's frontier if it's empty AND adjacent to a P stone.
        old_cell = state.board.get(node, NodeState.EMPTY)
        if old_cell == NodeState.EMPTY:
            # Was this node in Black's frontier?
            black_adj = any(state.board.get(nb) == NodeState.BLACK for nb in neighbors)
            white_adj = any(state.board.get(nb) == NodeState.WHITE for nb in neighbors)
            if black_adj:
                f_black_delta -= 1  # node is no longer empty, leaves Black frontier
            if white_adj:
                f_white_delta -= 1  # same for White

        # Placing here: new empty neighbors of node may enter our frontier,
        # or leave opponent's frontier if node was their only own-color neighbor.
        for nb in neighbors:
            if state.board.get(nb, NodeState.EMPTY) != NodeState.EMPTY:
                continue
            nb_neighbors = topology.get_neighbors(nb)
            # Does nb currently have an own (player) neighbor other than node?
            has_own_already = any(
                state.board.get(n2) == own_color
                for n2 in nb_neighbors if n2 != node
            )
            # Does nb lose its only opp-color neighbor?
            opp_neighbors_of_nb = [
                n2 for n2 in nb_neighbors
                if state.board.get(n2) == opp_color
            ]

            # nb will now have node as an own-color neighbor
            if not has_own_already:
                # nb enters our frontier
                if player == Player.BLACK:
                    f_black_delta += 1
                else:
                    f_white_delta += 1

            # node (now our color) displaces any opp-color count at nb
            # nb was in opponent's frontier if it had any opp-color neighbor
            # After placement, node is no longer empty so it can't be in opp frontier
            # But nb might LOSE from opp frontier if node was opp's only neighbor
            # ... actually node was EMPTY before, so it was never an opp-color stone.
            # opp frontier of nb is unchanged by this placement (we add own, not remove opp).

        delta = SearchDelta(state, node, f_black_delta, f_white_delta)

        # Apply move via engine
        self.engine.apply_placement(state, node)

        # Update frontier counts
        self.frontier[Player.BLACK] += f_black_delta
        self.frontier[Player.WHITE] += f_white_delta

        return delta

    def undo(self, delta: SearchDelta) -> None:
        """Restore state and frontier counts."""
        state = self.state
        state.board[delta.node] = delta.old_cell
        del state.move_history[delta.old_history_len:]
        del state.redo_stack[delta.old_redo_len:]
        state.consecutive_pass_count = delta.old_pass_count
        state.active_player = delta.old_active_player
        state.current_turn = delta.old_turn
        state.turn_phase = delta.old_phase
        state.game_over = delta.old_game_over
        state.winner = delta.old_winner
        state.stones_remaining = delta.old_supply
        # Restore frontier counts
        self.frontier[Player.BLACK] -= delta.frontier_black_delta
        self.frontier[Player.WHITE] -= delta.frontier_white_delta

    class _MoveCtx:
        def __init__(self, ss, node):
            self.ss = ss
            self.node = node
            self.delta = None

        def __enter__(self):
            self.delta = self.ss.apply(self.node)
            return self.delta

        def __exit__(self, *_):
            self.ss.undo(self.delta)

    def move(self, node: Node):
        return self._MoveCtx(self, node)


def _count_frontier(engine: MoveEngine, state: GameState, player: Player) -> int:
    """Initial O(V) frontier count. Called once at SearchState construction."""
    own = NodeState.from_player(player)
    count = 0
    for v in engine.topology.iterate_nodes():
        if state.board.get(v, NodeState.EMPTY) != NodeState.EMPTY:
            continue
        if any(state.board.get(nb) == own
               for nb in engine.topology.get_neighbors(v)):
            count += 1
    return count


# Simple (non-incremental) versions kept for compatibility with existing SearchBot
def apply_and_save(engine: MoveEngine, state: GameState, node: Node) -> SearchDelta:
    delta = SearchDelta(state, node)
    engine.apply_placement(state, node)
    return delta


def undo_placement(state: GameState, delta: SearchDelta) -> None:
    state.board[delta.node] = delta.old_cell
    del state.move_history[delta.old_history_len:]
    del state.redo_stack[delta.old_redo_len:]
    state.consecutive_pass_count = delta.old_pass_count
    state.active_player = delta.old_active_player
    state.current_turn = delta.old_turn
    state.turn_phase = delta.old_phase
    state.game_over = delta.old_game_over
    state.winner = delta.old_winner
    state.stones_remaining = delta.old_supply
