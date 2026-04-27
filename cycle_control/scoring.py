"""Scoring: a player's node scores iff it belongs to a block (BCC) with
at least 2 edges in that player's induced subgraph.

Equivalently: a node scores iff at least one of its incident edges in the
induced subgraph is NOT a bridge. We compute bridges via Tarjan's bridge
algorithm (iterative, to avoid recursion-depth issues on large boards).

Every block with at least 2 edges contains a cycle.
"""

from __future__ import annotations

import sys

from .state import NodeState, Player
from .topology import BoardTopology, Node


def _find_bridges(adj: dict[Node, list[Node]]) -> set[frozenset[Node]]:
    """Tarjan's bridge algorithm, iterative. Returns set of undirected edges
    (as frozensets of 2 nodes) that are bridges."""
    bridges: set[frozenset[Node]] = set()
    disc: dict[Node, int] = {}
    low: dict[Node, int] = {}
    counter = 0

    for root in list(adj.keys()):
        if root in disc:
            continue
        # iterative DFS
        stack: list[tuple[Node, Node | None, int]] = [(root, None, 0)]
        # The third field is the index of next neighbor to explore.
        disc[root] = low[root] = counter
        counter += 1
        while stack:
            u, parent, idx = stack[-1]
            neighs = adj.get(u, [])
            if idx < len(neighs):
                stack[-1] = (u, parent, idx + 1)
                v = neighs[idx]
                if v == parent:
                    continue
                if v in disc:
                    # back edge
                    if disc[v] < low[u]:
                        low[u] = disc[v]
                else:
                    disc[v] = low[v] = counter
                    counter += 1
                    stack.append((v, u, 0))
            else:
                stack.pop()
                if parent is not None:
                    if low[u] < low[parent]:
                        low[parent] = low[u]
                    if low[u] > disc[parent]:
                        bridges.add(frozenset((parent, u)))
    return bridges


def scoring_nodes(
    topology: BoardTopology,
    board: dict[Node, NodeState],
    player: Player,
    partial_credit_k: int = 0,
) -> set[Node]:
    """Set of player's stones that score.

    Base: stones belonging to a cycle in the induced subgraph (bridge-based).
    If partial_credit_k > 0: additionally include stones in connected
    components of size >= partial_credit_k. Final result is the union.
    """
    target = NodeState.from_player(player)
    own = [n for n in topology.iterate_nodes() if board.get(n, NodeState.EMPTY) == target]
    if not own:
        return set()
    own_set = set(own)
    adj = {u: [v for v in topology.get_neighbors(u) if v in own_set] for u in own}

    bridges = _find_bridges(adj)

    scoring: set[Node] = set()
    for u in own:
        for v in adj[u]:
            if frozenset((u, v)) not in bridges:
                scoring.add(u)
                break

    if partial_credit_k > 0:
        # Connected components via BFS.
        seen: set[Node] = set()
        for root in own:
            if root in seen:
                continue
            comp: list[Node] = []
            stack = [root]
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                comp.append(u)
                for v in adj[u]:
                    if v not in seen:
                        stack.append(v)
            if len(comp) >= partial_credit_k:
                scoring.update(comp)

    return scoring


def score(
    topology: BoardTopology,
    board: dict[Node, NodeState],
    player: Player,
    partial_credit_k: int = 0,
) -> int:
    return len(scoring_nodes(topology, board, player, partial_credit_k))
