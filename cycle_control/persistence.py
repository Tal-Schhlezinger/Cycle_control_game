"""JSON persistence with schema_version. redo_stack is NOT persisted."""

from __future__ import annotations

import json
from typing import Any

from .rules import RulesConfig
from .state import (
    GameState, NodeState, PassEntry, PlacementEntry, Player, TurnPhase,
    history_entry_from_dict,
)

SCHEMA_VERSION = 1


class PersistenceError(Exception):
    """Raised on schema mismatch or malformed save file."""


def _node_to_str(n) -> str:
    return f"{n[0]},{n[1]},{n[2]}"


def _str_to_node(s: str):
    parts = s.split(",")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _winner_to_json(w):
    if w is None:
        return None
    if isinstance(w, Player):
        return w.value
    return w  # "draw"


def _winner_from_json(s):
    if s is None:
        return None
    if s == "draw":
        return "draw"
    return Player(s)


def serialize_state(state: GameState, rules: RulesConfig) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "rules": rules.to_dict(),
        "board": {
            _node_to_str(n): st.value for n, st in sorted(state.board.items())
        },
        "active_player": state.active_player.value,
        "turn_phase": state.turn_phase.value,
        "stones_remaining": {p.value: v for p, v in state.stones_remaining.items()},
        "consecutive_pass_count": state.consecutive_pass_count,
        "current_turn": state.current_turn,
        "move_history": [e.to_dict() for e in state.move_history],
        "game_over": state.game_over,
        "winner": _winner_to_json(state.winner),
    }


def deserialize_state(data: dict) -> tuple[RulesConfig, GameState]:
    if not isinstance(data, dict):
        raise PersistenceError(f"expected dict, got {type(data).__name__}")
    if "schema_version" not in data:
        raise PersistenceError("missing 'schema_version'")
    if data["schema_version"] != SCHEMA_VERSION:
        raise PersistenceError(
            f"schema version mismatch: file is {data['schema_version']}, "
            f"expected {SCHEMA_VERSION}"
        )

    required = {
        "rules", "board", "active_player", "turn_phase", "stones_remaining",
        "consecutive_pass_count", "current_turn", "move_history",
        "game_over", "winner",
    }
    missing = required - set(data.keys())
    if missing:
        raise PersistenceError(f"missing fields: {sorted(missing)}")

    try:
        rules = RulesConfig.from_dict(data["rules"])
        board = {_str_to_node(k): NodeState(v) for k, v in data["board"].items()}
        state = GameState(
            board=board,
            active_player=Player(data["active_player"]),
            turn_phase=TurnPhase(data["turn_phase"]),
            stones_remaining={Player(k): int(v)
                              for k, v in data["stones_remaining"].items()},
            consecutive_pass_count=int(data["consecutive_pass_count"]),
            current_turn=int(data["current_turn"]),
            move_history=[history_entry_from_dict(e) for e in data["move_history"]],
            redo_stack=[],  # NOT persisted
            game_over=bool(data["game_over"]),
            winner=_winner_from_json(data["winner"]),
        )
    except PersistenceError:
        raise
    except Exception as e:
        raise PersistenceError(f"failed to parse save: {e}") from e

    return rules, state


def save_to_file(path: str, state: GameState, rules: RulesConfig) -> None:
    data = serialize_state(state, rules)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_from_file(path: str) -> tuple[RulesConfig, GameState]:
    with open(path, "r") as f:
        data = json.load(f)
    return deserialize_state(data)
