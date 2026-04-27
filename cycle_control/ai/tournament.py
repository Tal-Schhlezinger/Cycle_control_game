"""Round-robin tournament runner and Elo ladder."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..engine import MoveEngine
from ..state import Player
from .bot_interface import Bot, play_game


# ---------------------------------------------------------------------------
# Match results
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    bot_a_name: str
    bot_b_name: str
    # Totals across all games played
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    unresolved: int = 0
    total_moves: int = 0
    wall_time_s: float = 0.0
    # Optional per-game details
    games: list[dict] = field(default_factory=list)

    def total_games(self) -> int:
        return self.a_wins + self.b_wins + self.draws + self.unresolved

    def a_win_rate(self) -> float:
        n = self.total_games()
        if n == 0:
            return 0.0
        return self.a_wins / n

    def b_win_rate(self) -> float:
        n = self.total_games()
        if n == 0:
            return 0.0
        return self.b_wins / n

    def draw_rate(self) -> float:
        n = self.total_games()
        if n == 0:
            return 0.0
        return self.draws / n

    def summary(self) -> str:
        n = self.total_games()
        return (
            f"{self.bot_a_name} vs {self.bot_b_name}: "
            f"A={self.a_wins}/{n} ({self.a_win_rate():.1%})  "
            f"B={self.b_wins}/{n} ({self.b_win_rate():.1%})  "
            f"D={self.draws}/{n} ({self.draw_rate():.1%})  "
            f"avg_moves={self.total_moves/n:.1f}  "
            f"time={self.wall_time_s:.1f}s"
        )


def run_match(
    engine: MoveEngine,
    bot_a: Bot,
    bot_b: Bot,
    n_games: int,
    swap_colors: bool = True,
    base_seed: int = 0,
    record_games: bool = False,
) -> MatchResult:
    """Run n_games between bot_a and bot_b.

    If swap_colors=True (default), half the games have bot_a as Black, half
    as White, to cancel any first-player advantage.
    """
    result = MatchResult(bot_a_name=getattr(bot_a, "name", "BotA"),
                         bot_b_name=getattr(bot_b, "name", "BotB"))
    t_start = time.time()

    for i in range(n_games):
        if swap_colors and (i % 2 == 1):
            # bot_a plays White, bot_b plays Black
            bot_black, bot_white = bot_b, bot_a
            a_is_black = False
        else:
            bot_black, bot_white = bot_a, bot_b
            a_is_black = True

        final_state, winner = play_game(
            engine, bot_black, bot_white, seed=base_seed + i * 1009,
        )
        result.total_moves += final_state.move_count()

        if winner is None:
            result.unresolved += 1
            outcome = "unresolved"
        elif winner == "draw":
            result.draws += 1
            outcome = "draw"
        elif winner == Player.BLACK:
            if a_is_black:
                result.a_wins += 1
            else:
                result.b_wins += 1
            outcome = "black"
        elif winner == Player.WHITE:
            if a_is_black:
                result.b_wins += 1
            else:
                result.a_wins += 1
            outcome = "white"
        else:
            result.unresolved += 1
            outcome = f"unknown:{winner}"

        if record_games:
            result.games.append({
                "game_index": i,
                "a_is_black": a_is_black,
                "moves": final_state.move_count(),
                "outcome": outcome,
                "winner_raw": str(winner),
            })

    result.wall_time_s = time.time() - t_start
    return result


# ---------------------------------------------------------------------------
# Round-robin
# ---------------------------------------------------------------------------

@dataclass
class RoundRobinResult:
    bot_names: list[str]
    # matrix[i][j] = MatchResult for bot_i (as A) vs bot_j (as B)
    matches: dict = field(default_factory=dict)

    def win_rate_matrix(self) -> list[list[float]]:
        n = len(self.bot_names)
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                m = self.matches.get((i, j))
                if m is not None:
                    mat[i][j] = m.a_win_rate()
        return mat

    def pretty_print(self) -> str:
        n = len(self.bot_names)
        col_w = max(12, max(len(s) for s in self.bot_names) + 2)
        header = " " * col_w + "".join(f"{s:>{col_w}}" for s in self.bot_names)
        lines = [header]
        mat = self.win_rate_matrix()
        for i, name in enumerate(self.bot_names):
            row = f"{name:<{col_w}}"
            for j in range(n):
                if i == j:
                    row += f"{'—':>{col_w}}"
                else:
                    row += f"{mat[i][j]:>{col_w}.1%}"
            lines.append(row)
        return "\n".join(lines)


def round_robin(
    engine_factory: Callable[[], MoveEngine],
    bot_factories: list[Callable[[MoveEngine], Bot]],
    bot_names: list[str],
    n_games_per_pair: int = 20,
    base_seed: int = 0,
    verbose: bool = True,
) -> RoundRobinResult:
    """Run a full round-robin tournament.

    `bot_factories[i](engine) -> Bot` constructs a fresh bot for each match,
    ensuring no state leaks between matches.
    """
    n = len(bot_factories)
    assert len(bot_names) == n

    result = RoundRobinResult(bot_names=list(bot_names))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            engine = engine_factory()
            bot_a = bot_factories[i](engine)
            bot_b = bot_factories[j](engine)
            m = run_match(
                engine, bot_a, bot_b,
                n_games=n_games_per_pair,
                swap_colors=True,
                base_seed=base_seed + i * 100 + j,
            )
            result.matches[(i, j)] = m
            if verbose:
                print(f"  {m.summary()}")
    return result


# ---------------------------------------------------------------------------
# Elo ladder (simple)
# ---------------------------------------------------------------------------

def elo_update(rating_a: float, rating_b: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    """Update two ratings after a game.

    score_a in {0, 0.5, 1} = loss / draw / win for A.
    Returns new (rating_a, rating_b).
    """
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)
    return new_a, new_b


def elo_from_round_robin(rr: RoundRobinResult, initial: float = 1200.0, k: float = 32.0) -> dict[str, float]:
    """Compute Elo ratings from a round-robin result.

    Naive implementation: process each match result as a series of abstract
    games weighted by win/draw/loss counts. Not optimal but good enough for
    ranking purposes.
    """
    ratings = {name: initial for name in rr.bot_names}
    for (i, j), match in rr.matches.items():
        name_a = rr.bot_names[i]
        name_b = rr.bot_names[j]
        # Treat each game as a unit update
        for _ in range(match.a_wins):
            ra, rb = elo_update(ratings[name_a], ratings[name_b], 1.0, k)
            ratings[name_a], ratings[name_b] = ra, rb
        for _ in range(match.b_wins):
            ra, rb = elo_update(ratings[name_a], ratings[name_b], 0.0, k)
            ratings[name_a], ratings[name_b] = ra, rb
        for _ in range(match.draws):
            ra, rb = elo_update(ratings[name_a], ratings[name_b], 0.5, k)
            ratings[name_a], ratings[name_b] = ra, rb
    return ratings
