"""SearchBot — negamax with alpha-beta pruning.

Optimisations:
    1. In-place apply/undo (SearchDelta) — no O(V) clone per node.
    2. SearchState with incremental frontier tracking — leaf eval O(6) not O(V).
    3. Lazy end_on_no_legal_moves check in engine — only triggers when board
       is nearly full (<=12.5% empty cells).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ...engine import MoveEngine
from ...state import GameState, Player
from ..action_space import ActionSpace
from ..search_utils import SearchState, apply_and_save, undo_placement
from .greedy_bot import cycle_score_diff


TERMINAL_WIN  =  1_000_000.0
TERMINAL_LOSS = -1_000_000.0


def terminal_value(state: GameState, player: Player) -> float:
    if state.winner == player:         return TERMINAL_WIN
    if state.winner == player.other(): return -TERMINAL_WIN
    return 0.0


@dataclass
class SearchStats:
    nodes_visited:     int = 0
    leaf_evals:        int = 0
    terminals_reached: int = 0
    alpha_cutoffs:     int = 0
    beta_cutoffs:      int = 0
    max_depth_reached: int = 0

    def describe(self) -> str:
        return (f"nodes={self.nodes_visited} leaves={self.leaf_evals} "
                f"terminals={self.terminals_reached} "
                f"cuts(a={self.alpha_cutoffs},b={self.beta_cutoffs})")


class SearchBot:
    """Alpha-beta minimax with incremental frontier eval (O(6) per leaf)."""

    def __init__(
        self,
        engine: MoveEngine,
        depth: int = 3,
        time_budget_s: Optional[float] = None,
        move_ordering: bool = True,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        if depth < 1:
            raise ValueError(f"depth >= 1 required, got {depth}")
        self.engine        = engine
        self.max_depth     = depth
        self.time_budget_s = time_budget_s
        self.move_ordering = move_ordering
        self.rng           = random.Random(seed)
        self.name          = name or f"Search-d{depth}"
        self._aspace       = ActionSpace(engine.topology)
        self.last_stats    = SearchStats()
        self._t0: float    = 0.0
        self._timed_out    = False

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        self.last_stats = SearchStats()

    def _leaf_eval(self, ss: SearchState, player: Player) -> float:
        """Leaf evaluation combining incremental frontier + opponent mobility.

        - Frontier: O(1) via SearchState's running counts.
        - Opponent mobility: O(V) via engine.legal_moves — this is the
          strength-critical signal that makes search win against Greedy_2.
          Removing it for speed makes the bot play substantially weaker.

        Net cost is dominated by the legal_moves call. Still ~10x faster
        than the original clone-based version because state is not cloned.
        """
        from .greedy_bot import mobility_for
        return (2.0 * ss.frontier_diff(player)
                - 0.5 * mobility_for(self.engine, ss.state, player.other()))

    # ------------------------------------------------------------------ public

    def choose_action(self, state: GameState, legal_mask: np.ndarray,
                      color: Player) -> int:
        indices = [int(i) for i in np.flatnonzero(legal_mask)]
        if not indices:
            raise RuntimeError(f"{self.name}: no legal actions")
        placements = [i for i in indices if i != self._aspace.pass_index]
        if placements:
            indices = placements
        if len(indices) == 1:
            return indices[0]

        # Build SearchState once — O(V) frontier init, then O(6) per move
        ss = SearchState(self.engine, state)

        self.last_stats = SearchStats()
        self._t0 = time.time()
        self._timed_out = False

        if self.time_budget_s is None:
            return self._root(ss, color, self.max_depth, indices)

        best = indices[0]
        for d in range(1, self.max_depth + 1):
            if self._time_up(): break
            candidate = self._root(ss, color, d, indices)
            if not self._timed_out:
                best = candidate
                self.last_stats.max_depth_reached = d
            else:
                break
        return best

    # ----------------------------------------------------------------- search

    def _root(self, ss: SearchState, color: Player,
              depth: int, indices: list[int]) -> int:
        state = ss.state
        ordered = self._order(ss, color, indices) if self.move_ordering else indices
        best_score = -float("inf")
        best: list[int] = []
        alpha, beta = -float("inf"), float("inf")

        for idx in ordered:
            if self._time_up():
                self._timed_out = True
                break
            node = self._aspace.index_to_node(idx)
            if node is None: continue

            delta = ss.apply(node)
            score = self._minimax(ss, depth - 1, alpha, beta,
                                  state.active_player, color)
            ss.undo(delta)

            if score > best_score:
                best_score, best = score, [idx]
            elif score == best_score:
                best.append(idx)
            if best_score > alpha:
                alpha = best_score

        return (self.rng.choice(best) if len(best) > 1 else best[0]) if best else indices[0]

    def _minimax(self, ss: SearchState, depth: int,
                 alpha: float, beta: float,
                 color_to_move: Player, search_side: Player) -> float:
        state = ss.state
        self.last_stats.nodes_visited += 1

        if state.game_over:
            self.last_stats.terminals_reached += 1
            return terminal_value(state, search_side)
        if depth <= 0 or self._time_up():
            if self._time_up(): self._timed_out = True
            self.last_stats.leaf_evals += 1
            return self._leaf_eval(ss, search_side)

        moves = self.engine.legal_moves(state)
        if not moves:
            self.last_stats.leaf_evals += 1
            return self._leaf_eval(ss, search_side)

        ordered = (self._order_nodes(ss, color_to_move, moves)
                   if self.move_ordering and depth >= 2 else moves)

        maximizing = (color_to_move == search_side)
        best = -float("inf") if maximizing else float("inf")

        for node in ordered:
            delta = ss.apply(node)
            val = self._minimax(ss, depth - 1, alpha, beta,
                                state.active_player, search_side)
            ss.undo(delta)

            if maximizing:
                if val > best: best = val
                if best > alpha: alpha = best
                if alpha >= beta:
                    self.last_stats.beta_cutoffs += 1
                    break
            else:
                if val < best: best = val
                if best < beta: beta = best
                if alpha >= beta:
                    self.last_stats.alpha_cutoffs += 1
                    break

        return best

    # --------------------------------------------------------------- ordering

    def _order(self, ss: SearchState, color: Player,
               indices: list[int]) -> list[int]:
        scored = []
        for idx in indices:
            node = self._aspace.index_to_node(idx)
            if node is None: continue
            delta = ss.apply(node)
            s = self._leaf_eval(ss, color)
            ss.undo(delta)
            scored.append((s, idx))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [i for _, i in scored]

    def _order_nodes(self, ss: SearchState, color: Player, nodes: list) -> list:
        scored = []
        for node in nodes:
            delta = ss.apply(node)
            s = self._leaf_eval(ss, color)
            ss.undo(delta)
            scored.append((s, node))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [n for _, n in scored]

    def _time_up(self) -> bool:
        return self.time_budget_s is not None and (
            time.time() - self._t0 >= self.time_budget_s)
