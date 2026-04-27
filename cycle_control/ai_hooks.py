"""AI / bot hooks per v5 Section 10."""

from __future__ import annotations

import random
from typing import Optional

from .engine import MoveEngine
from .scoring import scoring_nodes
from .state import GameState, Player


def legal_moves(engine: MoveEngine, state: GameState) -> list:
    """Return list of legal placement nodes (sorted)."""
    return engine.legal_moves(state)


def clone(state: GameState) -> GameState:
    return state.clone()


def apply_move(engine: MoveEngine, state: GameState, move: dict) -> None:
    """move = {'type': 'place', 'node': (q, r, o)} or {'type': 'pass'}."""
    t = move.get("type")
    if t == "place":
        engine.apply_placement(state, tuple(move["node"]))
    elif t == "pass":
        engine.apply_pass(state)
    else:
        raise ValueError(f"unknown move type: {t!r}")


def evaluate(engine: MoveEngine, state: GameState, player: Player) -> dict:
    """Return {'own': int, 'opponent': int, 'diff': int}."""
    k = engine.rules.partial_credit_k
    own = len(scoring_nodes(engine.topology, state.board, player, k))
    opp = len(scoring_nodes(engine.topology, state.board, player.other(), k))
    return {"own": own, "opponent": opp, "diff": own - opp}


class BotRNG:
    """Per-bot RNG with seed_rng method (v5 Section 10.5)."""
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def seed_rng(self, seed: int) -> None:
        self.rng.seed(seed)

    def random(self) -> float:
        return self.rng.random()

    def choice(self, seq):
        return self.rng.choice(seq)
