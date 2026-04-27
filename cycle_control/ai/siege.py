"""Territory and siege analysis.

Core question: given the current board, which empty cells can a given player
P eventually place at (possibly after placing other cells first)?

Under monotone rules (neutrality + strict adjacency), adding more P stones
never makes it *harder* for P to reach further cells — so a fixed-point
iteration from "cells P can currently place at" correctly computes the
reachable set. Cells NOT in the reachable set are sieged against P.

Algorithm:
  1. Let reachable = {}.
  2. For each empty cell v: if P could legally place at v right now, given
     the current stones plus any stones already in `reachable`, add v to
     reachable.
  3. Repeat until no new cells are added.

A cell is "sieged against P" iff it is empty and not in reachable(P).
A cell is "sieged for P" iff it is sieged against P's opponent AND not
sieged against P itself (i.e., P can reach it but opponent cannot).
"""

from __future__ import annotations

from typing import Optional

from ..engine import MoveEngine
from ..rules import RulesConfig
from ..state import GameState, NodeState, Player
from ..topology import BoardTopology, Node


def _can_player_place_at(
    rules: RulesConfig,
    topology: BoardTopology,
    board: dict[Node, NodeState],
    v: Node,
    player: Player,
    extra_own: Optional[set[Node]] = None,
) -> bool:
    """Would `player` be able to legally place at `v`, if additionally they
    had stones at every cell in `extra_own`?

    Ignores turn order and supply — only spatial legality matters for
    reachability analysis.
    """
    if board.get(v, NodeState.EMPTY) is not NodeState.EMPTY:
        return False
    if extra_own is None:
        extra_own = set()
    if v in extra_own:
        # Treating v as already occupied by player
        return False

    own_state = NodeState.from_player(player)
    opp_state = NodeState.from_player(player.other())

    own_n = 0
    opp_n = 0
    for nb in topology.get_neighbors(v):
        if nb in extra_own:
            own_n += 1
            continue
        s = board.get(nb, NodeState.EMPTY)
        if s == own_state:
            own_n += 1
        elif s == opp_state:
            opp_n += 1

    if rules.neutrality_rule:
        if own_n < opp_n:
            return False

    if rules.strict_adjacency_rule:
        # Rule is suspended when player has no stones on board.
        has_own_stones_real = any(s == own_state for s in board.values())
        has_extras = len(extra_own) > 0
        if (has_own_stones_real or has_extras) and own_n == 0:
            return False

    return True


def reachable_empty_cells(
    engine: MoveEngine,
    state: GameState,
    player: Player,
) -> set[Node]:
    """Set of empty cells that `player` can eventually occupy via some
    sequence of legal placements, without modifying any opponent stones.

    This is the closure of "can place here now" under iterative expansion.
    Complexity: O(V^2) worst case.
    """
    rules = engine.rules
    topology = engine.topology
    board = state.board

    reachable: set[Node] = set()

    # Keep iterating until no new cells are added (monotone fixed point).
    changed = True
    while changed:
        changed = False
        for v in topology.iterate_nodes():
            if v in reachable:
                continue
            if board.get(v, NodeState.EMPTY) is not NodeState.EMPTY:
                continue
            if _can_player_place_at(rules, topology, board, v, player, extra_own=reachable):
                reachable.add(v)
                changed = True
    return reachable


def sieged_against(
    engine: MoveEngine,
    state: GameState,
    player: Player,
) -> set[Node]:
    """Empty cells that `player` can NEVER reach under the current board.

    These cells are "sieged against `player`" — i.e., permanently denied.
    """
    reachable = reachable_empty_cells(engine, state, player)
    all_empty = {
        n for n in engine.topology.iterate_nodes()
        if state.board.get(n, NodeState.EMPTY) is NodeState.EMPTY
    }
    return all_empty - reachable


def sieged_for(
    engine: MoveEngine,
    state: GameState,
    player: Player,
) -> set[Node]:
    """Empty cells that `player` can reach but opponent cannot.

    These are "owned future territory" for `player`.
    """
    own_reach = reachable_empty_cells(engine, state, player)
    opp_reach = reachable_empty_cells(engine, state, player.other())
    return own_reach - opp_reach


def territory_score(
    engine: MoveEngine,
    state: GameState,
    player: Player,
) -> int:
    """Count of empty cells that `player` can reach (upper bound on future
    territory). This is the greedy-bot territory feature."""
    return len(reachable_empty_cells(engine, state, player))


def exclusive_territory(
    engine: MoveEngine,
    state: GameState,
    player: Player,
) -> int:
    """Count of empty cells reachable by `player` but NOT by opponent."""
    return len(sieged_for(engine, state, player))


def frontier_count(
    engine: MoveEngine,
    state: GameState,
    player: Player,
) -> int:
    """Count of empty cells adjacent to at least one of `player`'s stones.

    This is the "expandable border" of player's position.
    """
    own_state = NodeState.from_player(player)
    count = 0
    for v in engine.topology.iterate_nodes():
        if state.board.get(v, NodeState.EMPTY) is not NodeState.EMPTY:
            continue
        for nb in engine.topology.get_neighbors(v):
            if state.board.get(nb) == own_state:
                count += 1
                break
    return count
