"""MoveEngine: legality, placement, pass, undo/redo, sandbox.

Undo strategy: replay from initial_state through move_history minus the last
entry. Simple, robust, correctness-by-construction. Sandbox actions are NOT
in move_history and are NOT reproduced by replay (v5 Section 7).
"""

from __future__ import annotations

from .rules import RulesConfig
from .scoring import scoring_nodes
from .state import (
    GameState, HistoryEntry, NodeState, PassEntry, PlacementEntry, Player, TurnPhase,
)
from .topology import BoardTopology, Node


class MoveError(Exception):
    """Raised when an illegal move is attempted."""


class MoveEngine:
    def __init__(self, rules: RulesConfig, topology: BoardTopology):
        if rules.board_radius != topology.radius:
            raise ValueError(
                f"rules.board_radius={rules.board_radius} does not match "
                f"topology.radius={topology.radius}"
            )
        if rules.mirror_adjacency != topology.mirror_adjacency:
            raise ValueError(
                f"rules.mirror_adjacency={rules.mirror_adjacency} does not match "
                f"topology.mirror_adjacency={topology.mirror_adjacency}"
            )
        self.rules = rules
        self.topology = topology

    # ----- initial state -----

    def initial_state(self) -> GameState:
        board: dict[Node, NodeState] = {
            n: NodeState.EMPTY for n in self.topology.iterate_nodes()
        }
        stones: dict[Player, int] = {}
        if self.rules.supply_enabled():
            stones = {
                Player.BLACK: self.rules.stones_per_player,  # type: ignore[dict-item]
                Player.WHITE: self.rules.stones_per_player,  # type: ignore[dict-item]
            }
        return GameState(
            board=board,
            active_player=Player.BLACK,
            turn_phase=TurnPhase.OPENING,
            stones_remaining=stones,
            consecutive_pass_count=0,
            current_turn=1,
            move_history=[],
            redo_stack=[],
            game_over=False,
            winner=None,
        )

    # ----- legality -----

    def is_legal_placement(self, state: GameState, node: Node) -> bool:
        if state.game_over:
            return False
        if not self.topology.is_on_board(node):
            return False
        if state.board.get(node, NodeState.EMPTY) is not NodeState.EMPTY:
            return False
        if self.rules.supply_enabled():
            if state.stones_remaining.get(state.active_player, 0) <= 0:
                return False

        # Experimental placement restrictions.
        player = state.active_player
        own_state = NodeState.from_player(player)
        opp_state = NodeState.from_player(player.other())

        if self.rules.neutrality_rule or self.rules.strict_adjacency_rule:
            own_neighbors = 0
            opp_neighbors = 0
            for nb in self.topology.get_neighbors(node):
                s = state.board.get(nb, NodeState.EMPTY)
                if s == own_state:
                    own_neighbors += 1
                elif s == opp_state:
                    opp_neighbors += 1

            if self.rules.neutrality_rule:
                if own_neighbors < opp_neighbors:
                    return False

            if self.rules.strict_adjacency_rule:
                # Suspended when placer has no stones on board (needed for
                # opening and each player's first placement).
                has_own_stones = any(
                    s == own_state for s in state.board.values()
                )
                if has_own_stones and own_neighbors == 0:
                    return False

        return True

    def legal_moves(self, state: GameState) -> list[Node]:
        if state.game_over:
            return []
        return [n for n in self.topology.iterate_nodes() if self.is_legal_placement(state, n)]

    def can_pass(self, state: GameState) -> bool:
        return self.rules.pass_enabled and not state.game_over

    # ----- apply: placement -----

    def apply_placement(self, state: GameState, node: Node) -> None:
        if not self.is_legal_placement(state, node):
            raise MoveError(f"illegal placement at {node!r}")

        player = state.active_player
        # 1. board
        state.board[node] = NodeState.from_player(player)
        # 2. supply
        if self.rules.supply_enabled():
            state.stones_remaining[player] -= 1
        # 3. history
        state.move_history.append(PlacementEntry(player=player, node=node))
        # 4. clear redo
        state.redo_stack.clear()
        # 5. reset consecutive pass counter (rule 3: any placement resets)
        state.consecutive_pass_count = 0
        # 6. end-condition check BEFORE turn advance
        if self._check_end_conditions(state):
            return
        # 7. advance turn phase
        self._advance_after_placement(state)

    # ----- apply: pass -----

    def apply_pass(self, state: GameState) -> None:
        if not self.can_pass(state):
            raise MoveError("pass is not allowed in current state")

        player = state.active_player
        placements_before = self._placements_this_turn(state)

        state.move_history.append(PassEntry(
            player=player,
            placements_before_pass=placements_before,
        ))
        state.redo_stack.clear()

        # Full-turn pass (0 placements) increments counter; partial pass does not.
        if placements_before == 0:
            state.consecutive_pass_count += 1
        else:
            state.consecutive_pass_count = 0

        if self._check_end_conditions(state):
            return

        self._end_turn(state)

    # ----- helpers -----

    def _placements_this_turn(self, state: GameState) -> int:
        """Placements already made in the current (ongoing) turn."""
        if state.turn_phase is TurnPhase.NORMAL_2:
            return 1
        # OPENING, NORMAL_1, NORMAL_TRUNCATED_1 all have 0 placements yet.
        return 0

    def _advance_after_placement(self, state: GameState) -> None:
        phase = state.turn_phase
        if phase is TurnPhase.OPENING:
            self._end_turn(state)
        elif phase is TurnPhase.NORMAL_1:
            # phase is frozen at turn start; we move to placement 2.
            state.turn_phase = TurnPhase.NORMAL_2
        elif phase is TurnPhase.NORMAL_2:
            self._end_turn(state)
        elif phase is TurnPhase.NORMAL_TRUNCATED_1:
            self._end_turn(state)

    def _end_turn(self, state: GameState) -> None:
        state.active_player = state.active_player.other()
        state.current_turn += 1
        state.turn_phase = self._compute_turn_phase_at_start(state)

    def _compute_turn_phase_at_start(self, state: GameState) -> TurnPhase:
        """Compute turn phase at start of a non-opening turn.

        Rule: phase is determined once at turn start from pre-turn state
        (supply and board-empty-count). It is NOT recomputed mid-turn.
        """
        max_placements = 2
        if self.rules.supply_enabled():
            supply_left = state.stones_remaining.get(state.active_player, 0)
            if supply_left < max_placements:
                max_placements = supply_left
        empty_nodes = sum(1 for v in state.board.values() if v is NodeState.EMPTY)
        if empty_nodes < max_placements:
            max_placements = empty_nodes

        if max_placements >= 2:
            return TurnPhase.NORMAL_1
        if max_placements == 1:
            return TurnPhase.NORMAL_TRUNCATED_1
        # max_placements == 0: no placements possible. End conditions should
        # trigger shortly (board_full or all_stones_placed). Use NORMAL_1 as
        # a harmless placeholder; legal_moves() will return [].
        return TurnPhase.NORMAL_1

    def _check_end_conditions(self, state: GameState) -> bool:
        """Check all enabled end conditions. If any triggered, set game_over
        and winner, and return True."""
        triggered = False

        if self.rules.end_on_consecutive_passes:
            if state.consecutive_pass_count >= 2:
                triggered = True

        if self.rules.end_on_all_stones_placed and self.rules.supply_enabled():
            if all(v == 0 for v in state.stones_remaining.values()):
                triggered = True

        if self.rules.end_on_board_full:
            if not any(v is NodeState.EMPTY for v in state.board.values()):
                triggered = True

        if self.rules.end_on_no_legal_moves:
            # This check is expensive (calls legal_moves for both players).
            # Skip it unless the board is mostly filled — under committed rules
            # with strict adjacency + neutrality, neither player runs out of
            # moves until late in the game when territories are fully partitioned.
            empty_count = sum(1 for v in state.board.values() if v is NodeState.EMPTY)
            total = len(state.board)
            if empty_count <= max(4, total // 8):  # only check when <=12.5% empty
                black_moves = self._count_legal_moves_for(state, Player.BLACK)
                white_moves = self._count_legal_moves_for(state, Player.WHITE)
                if black_moves == 0 and white_moves == 0:
                    triggered = True

        if triggered:
            state.game_over = True
            self._determine_winner(state)
        return triggered

    def _count_legal_moves_for(self, state: GameState, player: Player) -> int:
        """Count legal placements for `player` without modifying state."""
        saved = state.active_player
        state.active_player = player
        count = len(self.legal_moves(state))
        state.active_player = saved
        return count

    def _determine_winner(self, state: GameState) -> None:
        k = self.rules.partial_credit_k
        b = len(scoring_nodes(self.topology, state.board, Player.BLACK, k))
        w = len(scoring_nodes(self.topology, state.board, Player.WHITE, k))
        if b > w:
            state.winner = Player.BLACK
        elif w > b:
            state.winner = Player.WHITE
        else:
            state.winner = "draw"

    # ----- undo / redo -----

    def can_undo(self, state: GameState) -> bool:
        return len(state.move_history) > 0

    def can_redo(self, state: GameState) -> bool:
        return len(state.redo_stack) > 0

    def undo(self, state: GameState) -> None:
        """Revert exactly ONE action (placement or pass).

        Preserves any pre-existing redo_stack entries and appends the newly
        undone action on top, so consecutive undos build a stack of actions
        that redo() can replay in LIFO order.
        """
        if not self.can_undo(state):
            raise MoveError("nothing to undo")
        last = state.move_history[-1]
        new_history = state.move_history[:-1]
        existing_redo = list(state.redo_stack)

        replayed = self.initial_state()
        for entry in new_history:
            if isinstance(entry, PlacementEntry):
                self.apply_placement(replayed, entry.node)
            elif isinstance(entry, PassEntry):
                self.apply_pass(replayed)
            else:
                raise AssertionError(f"unknown history entry: {entry!r}")

        # Copy replayed fields back into state.
        state.board = replayed.board
        state.active_player = replayed.active_player
        state.turn_phase = replayed.turn_phase
        state.stones_remaining = replayed.stones_remaining
        state.consecutive_pass_count = replayed.consecutive_pass_count
        state.current_turn = replayed.current_turn
        state.move_history = replayed.move_history
        state.game_over = replayed.game_over
        state.winner = replayed.winner
        # Restore prior redo entries (replay cleared them) and push the
        # newly undone action on top.
        state.redo_stack = existing_redo
        state.redo_stack.append(last)

    def redo(self, state: GameState) -> None:
        """Re-apply one action from the redo stack."""
        if not self.can_redo(state):
            raise MoveError("nothing to redo")
        entry = state.redo_stack.pop()
        # apply_* will clear redo_stack; preserve what's left.
        preserved = list(state.redo_stack)
        if isinstance(entry, PlacementEntry):
            self.apply_placement(state, entry.node)
        elif isinstance(entry, PassEntry):
            self.apply_pass(state)
        else:
            raise AssertionError(f"unknown history entry: {entry!r}")
        state.redo_stack = preserved

    # ----- sandbox (out-of-history) -----

    def sandbox_place(self, state: GameState, node: Node, color: NodeState) -> None:
        """Set node to BLACK or WHITE bypassing all turn/supply rules.

        Does NOT touch: history, turn phase, active player, pass counter,
        supply, game_over, winner.
        """
        if not self.topology.is_on_board(node):
            raise MoveError(f"node {node!r} not on board")
        if color not in (NodeState.BLACK, NodeState.WHITE):
            raise MoveError("sandbox_place color must be BLACK or WHITE")
        state.board[node] = color

    def sandbox_remove(self, state: GameState, node: Node) -> None:
        """Set node to EMPTY bypassing all turn/supply rules."""
        if not self.topology.is_on_board(node):
            raise MoveError(f"node {node!r} not on board")
        state.board[node] = NodeState.EMPTY
