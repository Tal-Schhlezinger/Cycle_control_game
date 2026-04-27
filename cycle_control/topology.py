"""Board topology for Cycle Control (graph version).

Node identity: (q, r, o) where q, r are axial coordinates and o in {0, 1}
is the orientation bit (0 = up triangle, 1 = down triangle).

Adjacency (FIXED per v5 Section 3.2):
    (q, r, 0) neighbors: (q, r, 1), (q-1, r, 1), (q, r-1, 1)
    (q, r, 1) neighbors: (q, r, 0), (q+1, r, 0), (q, r+1, 0)

On-board predicate: a triangle is on board iff all 3 of its corner vertices
satisfy max(|a|, |b|, |a+b|) <= radius. Triangle vertices:
    o=0: (q, r), (q+1, r), (q, r+1)
    o=1: (q, r+1), (q+1, r+1), (q+1, r)

Construction-time invariants (checked once; not per-move):
    - bipartite by orientation
    - girth >= 6
    - every node has degree 1, 2, or 3 (no isolated nodes)
"""

from __future__ import annotations

import math
from typing import Iterator

Node = tuple[int, int, int]


class BoardTopology:
    def __init__(self, radius: int, mirror_adjacency: bool = False):
        if not isinstance(radius, int) or isinstance(radius, bool) or radius < 1:
            raise ValueError(f"board radius must be integer >= 1, got {radius!r}")
        self.radius = radius
        self.mirror_adjacency = bool(mirror_adjacency)
        self._nodes: tuple[Node, ...] = self._compute_nodes()
        self._node_set: frozenset[Node] = frozenset(self._nodes)
        self._neighbors: dict[Node, tuple[Node, ...]] = self._compute_neighbors()
        self._sanity_check()

    # ----- construction -----

    @staticmethod
    def _vertex_in_hex(q: int, r: int, R: int) -> bool:
        return abs(q) <= R and abs(r) <= R and abs(q + r) <= R

    @classmethod
    def _triangle_in_hex(cls, q: int, r: int, o: int, R: int) -> bool:
        if o == 0:
            vs = ((q, r), (q + 1, r), (q, r + 1))
        else:
            vs = ((q, r + 1), (q + 1, r + 1), (q + 1, r))
        return all(cls._vertex_in_hex(vq, vr, R) for vq, vr in vs)

    def _compute_nodes(self) -> tuple[Node, ...]:
        R = self.radius
        nodes: list[Node] = []
        for q in range(-R - 1, R + 2):
            for r in range(-R - 1, R + 2):
                for o in (0, 1):
                    if self._triangle_in_hex(q, r, o, R):
                        nodes.append((q, r, o))
        nodes.sort()
        return tuple(nodes)

    def _compute_neighbors(self) -> dict[Node, tuple[Node, ...]]:
        """Compute adjacency.

        Side-adjacency (always): two triangles sharing an edge.
            (q, r, 0) side-neighbors: (q, r, 1), (q-1, r, 1), (q, r-1, 1)
            (q, r, 1) side-neighbors: (q, r, 0), (q+1, r, 0), (q, r+1, 0)

        Mirror-adjacency (optional): triangle directly opposite across a
        shared vertex (point-reflection). Derived by reflecting each vertex
        through the shared point:
            (q, r, 0) mirror-neighbors: (q-1, r-1, 1), (q+1, r-1, 1), (q-1, r+1, 1)
            (q, r, 1) mirror-neighbors: (q-1, r+1, 0), (q+1, r+1, 0), (q+1, r-1, 0)

        Mirror edges also connect opposite-orientation triangles, so the
        graph remains bipartite by the orientation bit.
        """
        neighbors: dict[Node, tuple[Node, ...]] = {}
        for node in self._nodes:
            q, r, o = node
            if o == 0:
                side = [(q, r, 1), (q - 1, r, 1), (q, r - 1, 1)]
                mirror = [(q - 1, r - 1, 1), (q + 1, r - 1, 1), (q - 1, r + 1, 1)]
            else:
                side = [(q, r, 0), (q + 1, r, 0), (q, r + 1, 0)]
                mirror = [(q - 1, r + 1, 0), (q + 1, r + 1, 0), (q + 1, r - 1, 0)]
            candidates = side + (mirror if self.mirror_adjacency else [])
            valid = tuple(sorted(n for n in candidates if n in self._node_set))
            neighbors[node] = valid
        return neighbors

    # ----- invariants -----

    def _sanity_check(self) -> None:
        # 1. bipartite by orientation (holds for side AND mirror edges)
        for node, neighs in self._neighbors.items():
            for n in neighs:
                if node[2] == n[2]:
                    raise AssertionError(f"non-bipartite adjacency: {node} -- {n}")

        # 2. degree bounds. side-only: 1..3. mirror-extended: 1..6.
        max_deg = 6 if self.mirror_adjacency else 3
        for node, neighs in self._neighbors.items():
            d = len(neighs)
            if d < 1 or d > max_deg:
                raise AssertionError(
                    f"node {node} has degree {d}, expected 1..{max_deg}"
                )

        # 3. girth lower bound. Use bounded-depth BFS so check is fast
        #    even for large boards. side-only: girth >= 6. mirror-ext: girth >= 4.
        expected_min_girth = 4 if self.mirror_adjacency else 6
        shortest = self._shortest_cycle_up_to(expected_min_girth + 1)
        if shortest is not None and shortest < expected_min_girth:
            raise AssertionError(
                f"girth {shortest} < expected {expected_min_girth}"
            )

    def _shortest_cycle_up_to(self, max_len: int) -> int | None:
        """Return the shortest cycle length if it is <= max_len + 1 from some
        BFS root, otherwise None. Bounded BFS keeps this fast for large V.

        For girth check we only need to find cycles *shorter* than expected.
        We search up to depth ceil(max_len / 2) which is sufficient to catch
        all cycles of length <= max_len.
        """
        max_depth = (max_len + 1) // 2
        best: int | None = None
        for start in self._nodes:
            dist: dict[Node, int] = {start: 0}
            parent: dict[Node, Node | None] = {start: None}
            queue: list[Node] = [start]
            depth = 0
            while queue and depth < max_depth:
                next_queue: list[Node] = []
                for u in queue:
                    for v in self._neighbors[u]:
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            next_queue.append(v)
                        elif parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if best is None or cycle_len < best:
                                best = cycle_len
                queue = next_queue
                depth += 1
        return best

    # ----- public API -----

    def _compute_girth(self) -> float:
        """Convenience: exact girth (or math.inf if acyclic). Uses
        bounded-depth BFS up to a generous bound for speed."""
        # Search far enough to cover any realistic girth on this family.
        bound = 2 * self.radius + 4
        g = self._shortest_cycle_up_to(bound)
        return g if g is not None else math.inf

    def iterate_nodes(self) -> Iterator[Node]:
        yield from self._nodes

    def get_neighbors(self, node: Node) -> tuple[Node, ...]:
        return self._neighbors.get(node, ())

    def is_on_board(self, node) -> bool:
        """True iff node is a syntactically valid (q, r, o) triple within this board."""
        if not isinstance(node, tuple) or len(node) != 3:
            return False
        q, r, o = node
        for x in (q, r, o):
            if not isinstance(x, int) or isinstance(x, bool):
                return False
        if o not in (0, 1):
            return False
        return node in self._node_set

    def all_nodes(self) -> tuple[Node, ...]:
        return self._nodes

    def node_count(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        return f"BoardTopology(radius={self.radius}, nodes={len(self._nodes)})"
