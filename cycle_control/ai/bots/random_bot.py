"""Random baseline bots."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from ...state import GameState, NodeState, Player
from ..action_space import ActionSpace


class RandomBot:
    """Picks uniformly at random among legal actions (including pass)."""

    def __init__(self, seed: Optional[int] = None, name: str = "Random"):
        self.name = name
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)

    def choose_action(
        self,
        state: GameState,
        legal_mask: np.ndarray,
        color: Player,
    ) -> int:
        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            raise RuntimeError("RandomBot: no legal actions available")
        return int(self.rng.choice(legal_indices.tolist()))


class FrontierRandomBot:
    """Random among legal placements adjacent to at least one own stone.

    Falls back to uniform random if no such placement exists (opening moves,
    isolated positions). This is a smarter-than-random baseline that
    avoids the worst cases of random play on large boards.
    """

    def __init__(
        self,
        topology=None,
        seed: Optional[int] = None,
        name: str = "FrontierRandom",
    ):
        # topology is accepted for future use but not required
        self.name = name
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)

    def choose_action(
        self,
        state: GameState,
        legal_mask: np.ndarray,
        color: Player,
    ) -> int:
        # Determine own stones
        own_state = NodeState.from_player(color)
        own_nodes = {n for n, s in state.board.items() if s == own_state}

        if not own_nodes:
            # No own stones yet: uniform random
            legal_indices = np.flatnonzero(legal_mask)
            return int(self.rng.choice(legal_indices.tolist()))

        # Prefer actions whose node is adjacent to own stones.
        # We need access to topology.get_neighbors, so we need it at construction
        # or via state. Easiest: walk legal mask and for each legal placement,
        # check board adjacency.
        # But mask contains indices; we need index->node. We rebuild via
        # iterating the topology from engine-side info stored in state... the
        # state doesn't carry topology. So this bot should be constructed
        # with topology. We'll accept topology as optional and degrade to
        # uniform random if not given.
        if not hasattr(self, "_topology") or self._topology is None:
            legal_indices = np.flatnonzero(legal_mask)
            return int(self.rng.choice(legal_indices.tolist()))

        topology = self._topology
        action_space = ActionSpace(topology)

        frontier_indices = []
        for idx in np.flatnonzero(legal_mask):
            node = action_space.index_to_node(int(idx))
            if node is None:
                continue  # pass
            if any(nb in own_nodes for nb in topology.get_neighbors(node)):
                frontier_indices.append(int(idx))

        if frontier_indices:
            return int(self.rng.choice(frontier_indices))
        # Fall back to uniform random legal
        return int(self.rng.choice(np.flatnonzero(legal_mask).tolist()))

    def attach_topology(self, topology) -> None:
        """Call this once after construction to enable frontier behavior."""
        self._topology = topology
