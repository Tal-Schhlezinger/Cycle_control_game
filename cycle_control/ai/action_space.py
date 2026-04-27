"""Action space for AI bots.

Maps between node placements and integer action indices.

Convention:
    index in [0, N-1]  -> place at topology.all_nodes()[index]
    index = N          -> pass

The total action space size is `N + 1` where N = topology.node_count().
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..engine import MoveEngine
from ..state import GameState
from ..topology import BoardTopology, Node


def pass_index(topology: BoardTopology) -> int:
    """Index of the pass action."""
    return topology.node_count()


def action_space_size(topology: BoardTopology) -> int:
    """Total number of discrete actions, including pass."""
    return topology.node_count() + 1


def action_index_to_node(topology: BoardTopology, index: int) -> Optional[Node]:
    """Return the node for a placement action, or None if the action is pass.

    Raises IndexError for out-of-range indices.
    """
    if index < 0 or index > topology.node_count():
        raise IndexError(
            f"action index {index} out of range [0, {topology.node_count()}]"
        )
    if index == topology.node_count():
        return None
    return topology.all_nodes()[index]


def node_to_action_index(topology: BoardTopology, node: Node) -> int:
    """Return the action index for placing at `node`.

    Raises ValueError if the node is not on the board.
    """
    nodes = topology.all_nodes()
    # Linear search; acceptable for O(V) in construction paths.
    # If this shows up in profiles, build a dict once per topology.
    for i, n in enumerate(nodes):
        if n == node:
            return i
    raise ValueError(f"node {node!r} not on board")


def build_legal_mask(engine: MoveEngine, state: GameState) -> np.ndarray:
    """Return a boolean numpy array of shape (N+1,) indicating legal actions.

    `mask[i] = True` iff action index `i` is legal in `state`.
    `mask[N]` (pass) is True iff `engine.can_pass(state)`.

    The mask is computed for the CURRENT active player. If state.game_over,
    all entries are False.
    """
    size = action_space_size(engine.topology)
    mask = np.zeros(size, dtype=bool)
    if state.game_over:
        return mask

    # Placement actions
    for node in engine.legal_moves(state):
        mask[node_to_action_index(engine.topology, node)] = True

    # Pass action
    if engine.can_pass(state):
        mask[pass_index(engine.topology)] = True

    return mask


class ActionSpace:
    """Convenience wrapper bundling topology + mapping functions."""

    def __init__(self, topology: BoardTopology):
        self.topology = topology
        self.size = action_space_size(topology)
        self.pass_index = pass_index(topology)
        # Precompute node -> index mapping for fast lookup
        self._node_to_idx: dict[Node, int] = {
            n: i for i, n in enumerate(topology.all_nodes())
        }

    def index_to_node(self, index: int) -> Optional[Node]:
        if index == self.pass_index:
            return None
        if index < 0 or index > self.pass_index:
            raise IndexError(f"action index {index} out of range")
        return self.topology.all_nodes()[index]

    def node_to_index(self, node: Node) -> int:
        try:
            return self._node_to_idx[node]
        except KeyError:
            raise ValueError(f"node {node!r} not on board")

    def build_mask(self, engine: MoveEngine, state: GameState) -> np.ndarray:
        return build_legal_mask(engine, state)
