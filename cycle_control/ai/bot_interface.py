"""Bot interface and turn-loop helpers.

A Bot implements `choose_action(state, legal_mask, color) -> int` returning
a single action index. The engine drives the multi-placement turn loop; the
bot is called once per placement. A pass terminates the turn immediately.
"""

from __future__ import annotations

from typing import Optional, Protocol, Union

import numpy as np

from ..engine import MoveEngine, MoveError
from ..state import GameState, Player, TurnPhase
from .action_space import ActionSpace


class Bot(Protocol):
    """Protocol for any bot.

    Implementations must:
      - `choose_action` must return an integer index where `legal_mask[i]`
        is True. Returning an illegal index is a bug, not a legal loss.
      - `reset(seed)` is called once per game by the turn driver.
    """

    def choose_action(
        self,
        state: GameState,
        legal_mask: np.ndarray,
        color: Player,
    ) -> int: ...

    def reset(self, seed: Optional[int] = None) -> None: ...


def play_turn(
    engine: MoveEngine,
    state: GameState,
    bot: Bot,
    action_space: Optional[ActionSpace] = None,
) -> None:
    """Run one turn for the active player using `bot`.

    Handles OPENING (1 placement), NORMAL_1 -> NORMAL_2 (up to 2 placements),
    NORMAL_TRUNCATED_1 (1 placement), and partial/full pass termination.
    """
    if action_space is None:
        action_space = ActionSpace(engine.topology)

    max_placements = {
        TurnPhase.OPENING: 1,
        TurnPhase.NORMAL_1: 2,
        TurnPhase.NORMAL_2: 1,          # mid-turn recovery (shouldn't normally start here)
        TurnPhase.NORMAL_TRUNCATED_1: 1,
    }[state.turn_phase]

    placements_done = 0
    start_player = state.active_player
    start_turn = state.current_turn

    while (
        placements_done < max_placements
        and not state.game_over
        and state.active_player == start_player
        and state.current_turn == start_turn
    ):
        legal_mask = action_space.build_mask(engine, state)
        if not legal_mask.any():
            # No legal action: nothing the bot can do. Shouldn't occur
            # under normal rules because pass is typically available.
            break

        action = bot.choose_action(state, legal_mask, state.active_player)

        # Validate — illegal returns are a bug
        if action < 0 or action >= action_space.size:
            raise ValueError(
                f"bot returned action index {action} outside [0, {action_space.size})"
            )
        if not legal_mask[action]:
            raise ValueError(
                f"bot returned illegal action index {action}; legal count={int(legal_mask.sum())}"
            )

        if action == action_space.pass_index:
            engine.apply_pass(state)
            return  # pass ends the turn

        node = action_space.index_to_node(action)
        try:
            engine.apply_placement(state, node)
        except MoveError as e:
            raise RuntimeError(
                f"apply_placement failed for bot-chosen action {action} "
                f"(node {node}) despite legal_mask claiming legal: {e}"
            )

        placements_done += 1


def auto_fill(engine: MoveEngine, state: GameState) -> None:
    """Fill all cells the current active player can legally reach, in
    topology order, without any bot decision.

    Called when the committed ruleset is in use (pass disabled) and one
    player runs out of legal moves. The opponent keeps placing in topology
    order — expanding their cluster as far as strict adjacency + neutrality
    allow — until no legal placements remain for them either.

    Deterministic (topology order is sorted), no evaluation needed.
    """
    while not state.game_over:
        moves = engine.legal_moves(state)
        if not moves:
            break
        engine.apply_placement(state, moves[0])


def play_game(
    engine: MoveEngine,
    bot_black: Bot,
    bot_white: Bot,
    seed: Optional[int] = None,
    max_turns: int = 10_000,
    auto_fill_when_stuck: bool = True,
) -> tuple[GameState, Union[Player, str, None]]:
    """Play one full game between two bots.

    When `auto_fill_when_stuck=True` (default) and pass is disabled:
    if the active player has no legal placements, the OPPONENT is given
    the turn and auto-fills all cells it can reach (in topology order,
    no bot decision). This repeats until both players are stuck, at which
    point end_on_no_legal_moves fires and the game ends.

    If pass is enabled, a stuck player passes normally instead.

    Returns (final_state, winner) where winner is Player, "draw", or None
    if max_turns was exceeded without termination.
    """
    state = engine.initial_state()
    action_space = ActionSpace(engine.topology)

    if seed is not None:
        bot_black.reset(seed=seed * 2 + 1)
        bot_white.reset(seed=seed * 2 + 2)
    else:
        bot_black.reset()
        bot_white.reset()

    turn_count = 0
    while not state.game_over and turn_count < max_turns:
        bot = bot_black if state.active_player == Player.BLACK else bot_white
        legal = engine.legal_moves(state)

        if not legal:
            if auto_fill_when_stuck and not engine.rules.pass_enabled:
                # Active player is stuck. Give the turn to the opponent and
                # auto-fill until the opponent is also stuck or game ends.
                state.active_player = state.active_player.other()
                auto_fill(engine, state)
                # After auto-fill, the end condition (both stuck) should
                # have fired inside auto_fill via _check_end_conditions.
                # If the game isn't over, it means the original player now
                # has moves again (possible if opponent's fill changed
                # adjacency). Continue the main loop normally.
            elif engine.rules.pass_enabled and engine.can_pass(state):
                engine.apply_pass(state)
            # else: genuinely stuck with no options — end condition should
            # have fired. Let the while condition handle it.
        else:
            play_turn(engine, state, bot, action_space)

        turn_count += 1

    return state, state.winner
