"""One-ply greedy bots.

Two distinct variants per checklist Section 4.2 / 4.2b:
    Greedy1 — cycle and structure focused
    Greedy2 — territory and frontier focused

Both play the same interface. For each legal action, the bot clones the
state, applies the action, evaluates the resulting position from the
bot's own color perspective, and picks the action maximizing eval.
Ties broken deterministically by action index (with optional RNG for
independent-bot variation).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...engine import MoveEngine
from ...scoring import scoring_nodes
from ...state import GameState, NodeState, Player
from ..action_space import ActionSpace
from ..siege import (
    exclusive_territory, frontier_count, territory_score,
)


# ---------------------------------------------------------------------------
# Evaluation features
# ---------------------------------------------------------------------------

def cycle_score_diff(engine: MoveEngine, state: GameState, player: Player) -> int:
    """Own scoring nodes minus opponent scoring nodes."""
    own = len(scoring_nodes(engine.topology, state.board, player, engine.rules.partial_credit_k))
    opp = len(scoring_nodes(engine.topology, state.board, player.other(), engine.rules.partial_credit_k))
    return own - opp


def largest_component_size(engine: MoveEngine, state: GameState, player: Player) -> int:
    own_state = NodeState.from_player(player)
    visited: set = set()
    best = 0
    for start in engine.topology.iterate_nodes():
        if state.board.get(start) != own_state or start in visited:
            continue
        # BFS
        comp_size = 0
        frontier = [start]
        while frontier:
            u = frontier.pop()
            if u in visited:
                continue
            visited.add(u)
            comp_size += 1
            for nb in engine.topology.get_neighbors(u):
                if state.board.get(nb) == own_state and nb not in visited:
                    frontier.append(nb)
        if comp_size > best:
            best = comp_size
    return best


def component_size_diff(engine: MoveEngine, state: GameState, player: Player) -> int:
    return largest_component_size(engine, state, player) - largest_component_size(engine, state, player.other())


def mobility_for(engine: MoveEngine, state: GameState, player: Player) -> int:
    """Number of legal placements for `player`.

    Exact for the current active player (legal_moves already computed).
    Fast O(V) frontier approximation for the opponent player, to avoid the
    expensive legal_moves scan on every cloned eval state.
    """
    if state.active_player == player:
        return len(engine.legal_moves(state))
    # Fast frontier count for non-active player: count empty cells that the
    # player could reach (adjacent to an own stone, or board is empty).
    # This preserves semantic intent of "how much space does this player have"
    # without the full legality check expense.
    own_state = NodeState.from_player(player)
    own_stones = {n for n, s in state.board.items() if s == own_state}
    if not own_stones:
        # No own stones yet: suspended strict adjacency, can place anywhere
        return sum(1 for s in state.board.values() if s is NodeState.EMPTY)
    reachable: set = set()
    for n in own_stones:
        for nb in engine.topology.get_neighbors(n):
            if state.board.get(nb, NodeState.EMPTY) is NodeState.EMPTY:
                reachable.add(nb)
    return len(reachable)


def mobility_diff(engine: MoveEngine, state: GameState, player: Player) -> int:
    return mobility_for(engine, state, player) - mobility_for(engine, state, player.other())


def territory_diff(engine: MoveEngine, state: GameState, player: Player) -> int:
    return territory_score(engine, state, player) - territory_score(engine, state, player.other())


def exclusive_territory_diff(engine: MoveEngine, state: GameState, player: Player) -> int:
    return exclusive_territory(engine, state, player) - exclusive_territory(engine, state, player.other())


def frontier_diff(engine: MoveEngine, state: GameState, player: Player) -> int:
    return frontier_count(engine, state, player) - frontier_count(engine, state, player.other())


# ---------------------------------------------------------------------------
# Base greedy bot
# ---------------------------------------------------------------------------

@dataclass
class GreedyWeights:
    """Evaluation weights. Set unused weights to 0."""
    cycle: float = 0.0
    component: float = 0.0
    mobility: float = 0.0
    territory: float = 0.0
    exclusive_territory: float = 0.0
    frontier: float = 0.0
    opp_mobility_penalty: float = 0.0  # subtracts opponent mobility directly

    def describe(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if v != 0.0) or "zero"


class GreedyBot:
    """One-ply greedy bot with configurable evaluation.

    Subclasses define specific weight profiles. RNG is per-bot and used only
    for tiebreaking on equal evaluations.
    """

    def __init__(
        self,
        engine: MoveEngine,
        weights: GreedyWeights,
        seed: Optional[int] = None,
        name: str = "Greedy",
    ):
        self.engine = engine
        self.weights = weights
        self.name = name
        self.rng = random.Random(seed)
        self._action_space = ActionSpace(engine.topology)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)

    def evaluate(self, state: GameState, player: Player) -> float:
        """Compute weighted eval. Higher = better for `player`."""
        w = self.weights
        score = 0.0
        if w.cycle != 0.0:
            score += w.cycle * cycle_score_diff(self.engine, state, player)
        if w.component != 0.0:
            score += w.component * component_size_diff(self.engine, state, player)
        if w.mobility != 0.0:
            score += w.mobility * mobility_diff(self.engine, state, player)
        if w.territory != 0.0:
            score += w.territory * territory_diff(self.engine, state, player)
        if w.exclusive_territory != 0.0:
            score += w.exclusive_territory * exclusive_territory_diff(self.engine, state, player)
        if w.frontier != 0.0:
            score += w.frontier * frontier_diff(self.engine, state, player)
        if w.opp_mobility_penalty != 0.0:
            # Penalize opponent mobility (higher opp mobility = worse for us)
            score -= w.opp_mobility_penalty * mobility_for(self.engine, state, player.other())
        return score

    def choose_action(
        self,
        state: GameState,
        legal_mask: np.ndarray,
        color: Player,
    ) -> int:
        """Pick the action maximizing evaluation after the move."""
        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            raise RuntimeError(f"{self.name}: no legal actions")

        # Separate placements from pass. Greedy bots should only pass when
        # NO placements are available — evaluating pass as "0" incorrectly
        # makes it look equal to or better than placements on an empty board.
        placement_indices = [int(i) for i in legal_indices
                             if int(i) != self._action_space.pass_index]
        pass_available = self._action_space.pass_index in [int(i) for i in legal_indices]

        # Use placements if any exist; fall back to pass only if forced.
        candidates = placement_indices if placement_indices else (
            [self._action_space.pass_index] if pass_available else []
        )
        if not candidates:
            raise RuntimeError(f"{self.name}: no candidates after filtering")

        best_score = None
        best_actions: list[int] = []

        for idx in candidates:
            idx = int(idx)
            if idx == self._action_space.pass_index:
                # Evaluate pass properly: clone and apply, then measure the
                # resulting position from our perspective. Simply using the
                # current state gives 0, which incorrectly makes pass look
                # equal to or better than any placement that shifts the board.
                trial = state.clone()
                try:
                    self.engine.apply_pass(trial)
                except Exception:
                    continue
                score = self.evaluate(trial, color)
            else:
                node = self._action_space.index_to_node(idx)
                # Clone and apply placement
                trial = state.clone()
                try:
                    self.engine.apply_placement(trial, node)
                except Exception:
                    # Shouldn't happen if mask was correct, but be defensive
                    continue
                score = self.evaluate(trial, color)

            if best_score is None or score > best_score:
                best_score = score
                best_actions = [idx]
            elif score == best_score:
                best_actions.append(idx)

        # Tiebreak by RNG among equally-valued actions.
        # RNG is seeded per-game for reproducibility.
        return self.rng.choice(best_actions) if len(best_actions) > 1 else best_actions[0]


# ---------------------------------------------------------------------------
# The two concrete variants per checklist Section 4.2 / 4.2b
# ---------------------------------------------------------------------------

class Greedy1(GreedyBot):
    """Cycle and structure focused.

    Evaluation: heavy cycle_score_diff, moderate component_diff, light mobility.
    Territory is explicitly zero — this bot should play short-term cycle-rush.
    """

    DEFAULT_WEIGHTS = GreedyWeights(
        cycle=3.0,
        component=0.5,
        mobility=0.1,
        territory=0.0,
        exclusive_territory=0.0,
        frontier=0.0,
        opp_mobility_penalty=0.0,
    )

    def __init__(
        self,
        engine: MoveEngine,
        weights: Optional[GreedyWeights] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            engine,
            weights or self.DEFAULT_WEIGHTS,
            seed=seed,
            name="Greedy1-Cycle",
        )


class Greedy2(GreedyBot):
    """Territory and frontier focused.

    Evaluation: heavy frontier_diff + opponent mobility penalty + cycle tiebreaker.

    Note: the original design used exclusive_territory_diff (O(V²) flood fill)
    which is too slow for boards R>=4. Replaced with frontier_diff (O(V)) which
    captures the same strategic intent — "control expandable border" — without
    the flood-fill cost. On small boards (R<=3) results are nearly identical.
    If you want the full flood-fill version, pass weights=GreedyWeights(
    exclusive_territory=2.0, frontier=0.5, ...) explicitly.
    """

    DEFAULT_WEIGHTS = GreedyWeights(
        cycle=0.2,
        component=0.0,
        mobility=0.0,
        territory=0.0,
        exclusive_territory=0.0,   # disabled — too slow at R>=4
        frontier=2.0,               # promoted to primary signal
        opp_mobility_penalty=0.5,   # heavier — penalising opponent reach is cheap
    )

    def __init__(
        self,
        engine: MoveEngine,
        weights: Optional[GreedyWeights] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            engine,
            weights or self.DEFAULT_WEIGHTS,
            seed=seed,
            name="Greedy2-Territory",
        )
