"""Debug / analysis helpers (not official scoring)."""

from __future__ import annotations

from .scoring import scoring_nodes
from .state import NodeState, Player
from .topology import BoardTopology


def connected_components(topology: BoardTopology, board: dict, player: Player) -> list[set]:
    """Connected components of player's induced subgraph."""
    target = NodeState.from_player(player)
    own = {n for n in topology.iterate_nodes() if board.get(n) == target}
    components: list[set] = []
    visited: set = set()
    for start in sorted(own):
        if start in visited:
            continue
        comp: set = set()
        stack = [start]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.add(u)
            for v in topology.get_neighbors(u):
                if v in own and v not in visited:
                    stack.append(v)
        components.append(comp)
    return components


def debug_summary(topology: BoardTopology, state, player: Player) -> dict:
    comps = connected_components(topology, state.board, player)
    sc = scoring_nodes(topology, state.board, player)
    target = NodeState.from_player(player)
    occupied = sum(1 for v in state.board.values() if v == target)
    return {
        "player": player.value,
        "occupied": occupied,
        "components": len(comps),
        "largest_component": max((len(c) for c in comps), default=0),
        "scoring_nodes": len(sc),
    }
