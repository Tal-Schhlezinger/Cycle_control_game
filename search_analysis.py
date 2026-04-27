"""Search-before-learning analysis per checklist Section 6.

Runs a round-robin tournament among:
    - RandomBot
    - Greedy_1 (cycle-focused one-ply)
    - Greedy_2 (territory-focused one-ply)
    - SearchBot depth=2
    - SearchBot depth=3

on the committed ruleset at a small board (R=3 by default), to characterize
the shape of the game at different search budgets.

Research questions this answers:
    1. Does SearchBot produce meaningfully different play from Greedy?
       (If not, the game may not have much tactical depth.)
    2. Does deeper search monotonically dominate shallower search?
       (If not, this is a signal of non-transitivity or instability.)
    3. Is draw rate high at deeper search?
       (High draws = rules may still be defender-dominated.)
    4. Is first-move (Black) advantage significant?
       (Large asymmetry = rules favor one side structurally.)

Usage:
    python search_analysis.py [--radius R] [--games N]
"""

from __future__ import annotations

import argparse
import sys
import time

from cycle_control.ai.bots import (
    Greedy1, Greedy2, RandomBot, SearchBot,
)
from cycle_control.ai.tournament import (
    elo_from_round_robin, round_robin,
)
from cycle_control.engine import MoveEngine
from cycle_control.rules import RulesConfig
from cycle_control.topology import BoardTopology


def make_committed_engine(radius: int) -> MoveEngine:
    rules = RulesConfig(
        board_radius=radius,
        neutrality_rule=True,
        strict_adjacency_rule=True,
        mirror_adjacency=True,
    )
    topology = BoardTopology(radius, mirror_adjacency=True)
    return MoveEngine(rules, topology)


def run_analysis(
    radius: int,
    n_games: int,
    include_search_d3: bool,
    seed: int,
) -> dict:
    print(f"Board: radius={radius}")
    print(f"Rules: neutrality + strict adjacency + mirror adjacency")
    print(f"Games per pairing: {n_games}")
    print()

    def engine_factory():
        return make_committed_engine(radius)

    # Define bot factories
    def f_random(eng): return RandomBot(seed=seed)
    def f_greedy1(eng): return Greedy1(eng, seed=seed)
    def f_greedy2(eng): return Greedy2(eng, seed=seed)
    def f_search_d2(eng): return SearchBot(eng, depth=2, seed=seed)
    def f_search_d3(eng): return SearchBot(eng, depth=3, seed=seed,
                                           time_budget_s=10.0)

    bot_factories = [f_random, f_greedy1, f_greedy2, f_search_d2]
    bot_names = ["Random", "Greedy_1", "Greedy_2", "Search_d2"]

    if include_search_d3:
        bot_factories.append(f_search_d3)
        bot_names.append("Search_d3")

    t0 = time.time()
    rr = round_robin(
        engine_factory, bot_factories, bot_names,
        n_games_per_pair=n_games,
        base_seed=seed,
        verbose=True,
    )
    duration = time.time() - t0

    print()
    print("=" * 70)
    print("WIN-RATE MATRIX (row bot vs column bot, cell = row's win rate)")
    print("=" * 70)
    print(rr.pretty_print())
    print()

    # Elo ratings
    elo = elo_from_round_robin(rr, initial=1200.0, k=32.0)
    print("=" * 70)
    print("ELO RATINGS")
    print("=" * 70)
    for name, rating in sorted(elo.items(), key=lambda x: -x[1]):
        print(f"  {name:12s}  {rating:.0f}")
    print()

    # Game-level statistics
    print("=" * 70)
    print("DRAW RATES (row vs column)")
    print("=" * 70)
    n = len(bot_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            m = rr.matches.get((i, j))
            if m is None:
                continue
            print(f"  {bot_names[i]:12s} vs {bot_names[j]:12s}: "
                  f"{m.draw_rate():.1%} draws, avg {m.total_moves/m.total_games():.1f} moves")
    print()

    print(f"Total analysis time: {duration:.1f}s")

    return {
        "rr": rr,
        "elo": elo,
        "duration_s": duration,
    }


def analyze_findings(rr, elo, bot_names) -> list[str]:
    """Produce a short interpretive report."""
    lines = []
    lines.append("=" * 70)
    lines.append("INTERPRETATION")
    lines.append("=" * 70)

    # Rank
    ranked = sorted(elo.items(), key=lambda x: -x[1])
    top = ranked[0][0]
    lines.append(f"Top Elo: {top}")

    # Check monotonicity
    expected_order = ["Search_d3", "Search_d2", "Greedy_2", "Greedy_1", "Random"]
    present_order = [name for name in expected_order if name in elo]
    actual_by_elo = [name for name, _ in ranked]
    
    if actual_by_elo == present_order:
        lines.append("Elo order matches expected skill monotonicity.")
    else:
        lines.append(f"Elo order does NOT match naive expectation:")
        lines.append(f"  Expected (by design): {present_order}")
        lines.append(f"  Actual (by Elo):      {actual_by_elo}")
        lines.append("  This is a signal of non-transitivity or evaluation mismatch.")

    # Draw rate summary
    total_draws = sum(m.draws for m in rr.matches.values())
    total_games = sum(m.total_games() for m in rr.matches.values())
    overall_draw_rate = total_draws / max(1, total_games)
    lines.append(f"Overall draw rate across all pairings: {overall_draw_rate:.1%}")

    if overall_draw_rate > 0.4:
        lines.append("  HIGH draw rate — rules may favor blocking over building.")
    elif overall_draw_rate < 0.05:
        lines.append("  Very low draw rate — games are decisive.")
    else:
        lines.append("  Draw rate is in a reasonable range.")

    # Search improvement over greedy
    if "Search_d2" in elo and "Greedy_2" in elo:
        delta = elo["Search_d2"] - elo["Greedy_2"]
        lines.append(f"Search_d2 Elo - Greedy_2 Elo = {delta:+.0f}")
        if delta < 50:
            lines.append("  Search barely improves on Greedy — game may be shallow.")
        elif delta > 200:
            lines.append("  Search substantially improves on Greedy — game has real tactical depth.")
        else:
            lines.append("  Search improves over Greedy as expected.")

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Search-before-learning analysis (checklist Section 6)"
    )
    parser.add_argument("--radius", type=int, default=2,
                        help="Board radius (default: 2; use 3 for more data)")
    parser.add_argument("--games", type=int, default=10,
                        help="Games per pairing (default: 10)")
    parser.add_argument("--include-d3", action="store_true",
                        help="Include SearchBot depth=3 (slow)")
    parser.add_argument("--multi-radius", action="store_true",
                        help="Run across R=2,3,4,5 with auto game counts (no search at R>=4)")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    print("=" * 70)
    print("Cycle Control — Search-before-learning analysis")
    print("=" * 70)

    if args.multi_radius:
        # Auto-scaled game counts per radius
        configs = [
            (2, 20, True),
            (3, 10, True),
            (4,  8, False),
            (5,  6, False),
        ]
        summary_rows = []
        for radius, n_games, include_search in configs:
            print(f"\n=== R={radius} ({'search included' if include_search else 'greedy-only'}) ===")
            out = run_analysis(
                radius=radius,
                n_games=n_games,
                include_search_d3=False,
                seed=args.seed,
            )
            # Suppress re-printing interpretation inline; collect for summary
            elo = out["elo"]
            summary_rows.append((radius, elo, out["rr"]))

        print("\n" + "=" * 70)
        print("CROSS-RADIUS SUMMARY")
        print("=" * 70)
        all_bots = ["Search_d2", "Greedy_2", "Random", "Greedy_1"]
        print(f"{'Bot':14s}  {'R=2':>6}  {'R=3':>6}  {'R=4':>6}  {'R=5':>6}")
        for bot in all_bots:
            row = f"{bot:14s}"
            for radius, elo, _ in summary_rows:
                row += f"  {elo.get(bot, 0):>6.0f}" if bot in elo else f"  {'N/A':>6}"
            print(row)
        print()
        print(f"{'Draw rate':14s}", end="")
        for radius, elo, rr in summary_rows:
            total_draws = sum(m.draws for m in rr.matches.values())
            total_games = sum(m.total_games() for m in rr.matches.values())
            dr = total_draws / max(1, total_games)
            print(f"  {dr:>5.0%}  ", end="")
        print()
        print(f"{'Avg moves':14s}", end="")
        for radius, elo, rr in summary_rows:
            total_moves = sum(m.total_moves for m in rr.matches.values())
            total_games = sum(m.total_games() for m in rr.matches.values())
            am = total_moves / max(1, total_games)
            print(f"  {am:>6.0f}  ", end="")
        print()
    else:
        out = run_analysis(
            radius=args.radius,
            n_games=args.games,
            include_search_d3=args.include_d3,
            seed=args.seed,
        )
        bot_names = out["rr"].bot_names
        for line in analyze_findings(out["rr"], out["elo"], bot_names):
            print(line)


if __name__ == "__main__":
    main()
