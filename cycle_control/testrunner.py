"""JSON-driven test runner.

Test spec format (single test):
{
  "name": "...",
  "description": "...",
  "rules": {...},          // RulesConfig dict; if missing, defaults used
  "moves": [
    {"cmd": "place", "node": [q, r, o]},
    {"cmd": "pass"},
    {"cmd": "sandbox_place", "node": [q, r, o], "color": "black"},
    {"cmd": "sandbox_remove", "node": [q, r, o]},
    {"cmd": "undo"},
    {"cmd": "redo"},
    {"cmd": "snapshot"},
    {"cmd": "assert_score", "player": "black", "expected": 6},
    {"cmd": "assert_active_player", "expected": "white"},
    {"cmd": "assert_turn_phase", "expected": "normal_1"},
    {"cmd": "assert_turn_number", "expected": 2},
    {"cmd": "assert_move_count", "expected": 3},
    {"cmd": "assert_game_over", "expected": false},
    {"cmd": "assert_winner", "expected": "draw"},
    {"cmd": "assert_node_state", "node": [0,0,0], "expected": "black"},
    {"cmd": "assert_scoring_node", "player": "black", "node": [0,0,0]},
    {"cmd": "assert_not_scoring_node", "player": "black", "node": [0,0,0]},
    {"cmd": "assert_stones_remaining", "player": "black", "expected": 10},
    {"cmd": "assert_pass_counter", "expected": 1},
    {"cmd": "assert_legal_moves", "expected": [[0,0,0], ...]}
  ]
}
"""

from __future__ import annotations

import json
import sys

from .engine import MoveEngine, MoveError
from .rules import RulesConfig
from .scoring import scoring_nodes
from .state import NodeState, Player, TurnPhase
from .topology import BoardTopology


class TestRunnerError(Exception):
    pass


class AssertionFailed(Exception):
    pass


def _winner_str(w) -> str | None:
    if w is None:
        return None
    if hasattr(w, "value"):
        return w.value
    return w  # "draw"


def dispatch(engine: MoveEngine, state, cmd: dict) -> None:
    c = cmd.get("cmd")
    if c is None:
        raise TestRunnerError(f"command missing 'cmd' field: {cmd!r}")

    if c == "place":
        engine.apply_placement(state, tuple(cmd["node"]))
    elif c == "pass":
        engine.apply_pass(state)
    elif c == "sandbox_place":
        color = NodeState(cmd["color"])
        engine.sandbox_place(state, tuple(cmd["node"]), color)
    elif c == "sandbox_remove":
        engine.sandbox_remove(state, tuple(cmd["node"]))
    elif c == "undo":
        engine.undo(state)
    elif c == "redo":
        engine.redo(state)
    elif c in ("snapshot", "snapshot_board"):
        # Informational. Real implementations may print; here we no-op.
        pass

    elif c == "assert_score":
        player = Player(cmd["player"])
        expected = int(cmd["expected"])
        k = engine.rules.partial_credit_k
        actual = len(scoring_nodes(engine.topology, state.board, player, k))
        if actual != expected:
            raise AssertionFailed(
                f"assert_score: player={player.value} expected={expected} actual={actual}"
            )

    elif c == "assert_active_player":
        expected = Player(cmd["expected"])
        if state.active_player != expected:
            raise AssertionFailed(
                f"assert_active_player: expected={expected.value} "
                f"actual={state.active_player.value}"
            )

    elif c == "assert_turn_phase":
        expected = TurnPhase(cmd["expected"])
        if state.turn_phase != expected:
            raise AssertionFailed(
                f"assert_turn_phase: expected={expected.value} "
                f"actual={state.turn_phase.value}"
            )

    elif c == "assert_turn_number":
        expected = int(cmd["expected"])
        if state.current_turn != expected:
            raise AssertionFailed(
                f"assert_turn_number: expected={expected} actual={state.current_turn}"
            )

    elif c == "assert_move_count":
        # Excludes sandbox actions (they are not in move_history).
        expected = int(cmd["expected"])
        actual = state.move_count()
        if actual != expected:
            raise AssertionFailed(
                f"assert_move_count: expected={expected} actual={actual}"
            )

    elif c == "assert_game_over":
        expected = bool(cmd["expected"])
        if state.game_over != expected:
            raise AssertionFailed(
                f"assert_game_over: expected={expected} actual={state.game_over}"
            )

    elif c == "assert_winner":
        expected = cmd["expected"]
        actual = _winner_str(state.winner)
        if actual != expected:
            raise AssertionFailed(
                f"assert_winner: expected={expected!r} actual={actual!r}"
            )

    elif c == "assert_node_state":
        node = tuple(cmd["node"])
        expected = NodeState(cmd["expected"])
        actual = state.board.get(node)
        actual_val = actual.value if actual is not None else "MISSING"
        if actual != expected:
            raise AssertionFailed(
                f"assert_node_state: node={node} expected={expected.value} actual={actual_val}"
            )

    elif c == "assert_scoring_node":
        node = tuple(cmd["node"])
        player = Player(cmd["player"])
        k = engine.rules.partial_credit_k
        sc = scoring_nodes(engine.topology, state.board, player, k)
        if node not in sc:
            raise AssertionFailed(
                f"assert_scoring_node: node={node} player={player.value} NOT in scoring set"
            )

    elif c == "assert_not_scoring_node":
        node = tuple(cmd["node"])
        player = Player(cmd["player"])
        k = engine.rules.partial_credit_k
        sc = scoring_nodes(engine.topology, state.board, player, k)
        if node in sc:
            raise AssertionFailed(
                f"assert_not_scoring_node: node={node} player={player.value} IS in scoring set"
            )

    elif c == "assert_stones_remaining":
        player = Player(cmd["player"])
        expected = int(cmd["expected"])
        actual = state.stones_remaining.get(player, 0)
        if actual != expected:
            raise AssertionFailed(
                f"assert_stones_remaining: player={player.value} "
                f"expected={expected} actual={actual}"
            )

    elif c == "assert_pass_counter":
        expected = int(cmd["expected"])
        if state.consecutive_pass_count != expected:
            raise AssertionFailed(
                f"assert_pass_counter: expected={expected} "
                f"actual={state.consecutive_pass_count}"
            )

    elif c == "assert_legal_moves":
        expected = sorted(tuple(n) for n in cmd["expected"])
        actual = sorted(engine.legal_moves(state))
        if actual != expected:
            raise AssertionFailed(
                f"assert_legal_moves: expected={expected} actual={actual}"
            )

    else:
        # Unknown command: FAIL AND STOP per v5 Section 8.
        raise TestRunnerError(f"unknown command: {c!r}")


def run_test(spec: dict, verbose: bool = False) -> dict:
    name = spec.get("name", "<unnamed>")
    description = spec.get("description", "")
    rules_dict = spec.get("rules", {})
    rules = RulesConfig.from_dict(rules_dict) if rules_dict else RulesConfig()
    topology = BoardTopology(rules.board_radius, mirror_adjacency=rules.mirror_adjacency)
    engine = MoveEngine(rules, topology)
    state = engine.initial_state()

    log: list[str] = []
    passed = True
    error: str | None = None

    def _log(msg: str):
        log.append(msg)
        if verbose:
            print(f"  {msg}")

    moves = spec.get("moves", [])
    try:
        for i, cmd in enumerate(moves):
            _log(f"[{i:03d}] {cmd}")
            dispatch(engine, state, cmd)
    except AssertionFailed as e:
        passed = False
        error = f"AssertionFailed: {e}"
        _log(error)
    except TestRunnerError as e:
        passed = False
        error = f"TestRunnerError: {e}"
        _log(error)
    except MoveError as e:
        passed = False
        error = f"MoveError: {e}"
        _log(error)
    except Exception as e:
        passed = False
        error = f"{type(e).__name__}: {e}"
        _log(error)

    return {
        "name": name,
        "description": description,
        "passed": passed,
        "error": error,
        "log": log,
    }


def run_tests_from_file(path: str, verbose: bool = False) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    tests = data if isinstance(data, list) else [data]
    results = [run_test(t, verbose=verbose) for t in tests]
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run JSON-based Cycle Control tests.")
    parser.add_argument("files", nargs="+", help="JSON test file(s)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    all_results: list[dict] = []
    for path in args.files:
        results = run_tests_from_file(path, verbose=args.verbose)
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"[{status}] {r['name']}  ({path})")
            if not r["passed"]:
                print(f"    {r['error']}")
        all_results.extend(results)

    n_pass = sum(1 for r in all_results if r["passed"])
    n_total = len(all_results)
    print(f"\n{n_pass}/{n_total} tests passed")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
