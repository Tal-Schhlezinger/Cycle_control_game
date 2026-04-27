"""Game state, player/node enums, history entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

Node = tuple[int, int, int]


class Player(Enum):
    BLACK = "black"
    WHITE = "white"

    def other(self) -> "Player":
        return Player.WHITE if self is Player.BLACK else Player.BLACK


class NodeState(Enum):
    EMPTY = "empty"
    BLACK = "black"
    WHITE = "white"

    @classmethod
    def from_player(cls, p: Player) -> "NodeState":
        return cls.BLACK if p is Player.BLACK else cls.WHITE


class TurnPhase(Enum):
    OPENING = "opening"
    NORMAL_1 = "normal_1"              # first of 2 placements
    NORMAL_2 = "normal_2"              # second of 2 placements
    NORMAL_TRUNCATED_1 = "truncated_1"  # only 1 placement possible this turn


@dataclass
class PlacementEntry:
    player: Player = Player.BLACK
    node: Node = (0, 0, 0)
    type: str = field(default="place", init=False)

    def to_dict(self) -> dict:
        return {"type": "place", "player": self.player.value, "node": list(self.node)}

    @classmethod
    def from_dict(cls, d: dict) -> "PlacementEntry":
        return cls(
            player=Player(d["player"]),
            node=tuple(d["node"]),
        )


@dataclass
class PassEntry:
    player: Player = Player.BLACK
    placements_before_pass: int = 0  # 0 or 1
    type: str = field(default="pass", init=False)

    def to_dict(self) -> dict:
        return {
            "type": "pass",
            "player": self.player.value,
            "placementsBeforePass": self.placements_before_pass,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PassEntry":
        return cls(
            player=Player(d["player"]),
            placements_before_pass=int(d.get("placementsBeforePass", 0)),
        )


HistoryEntry = Union[PlacementEntry, PassEntry]


def history_entry_from_dict(d: dict) -> HistoryEntry:
    t = d.get("type")
    if t == "place":
        return PlacementEntry.from_dict(d)
    if t == "pass":
        return PassEntry.from_dict(d)
    raise ValueError(f"unknown history entry type: {t!r}")


WinnerT = Union[Player, str, None]  # Player, "draw", or None if ongoing


@dataclass
class GameState:
    """Mutable game state. Engine methods modify in place."""
    board: dict[Node, NodeState] = field(default_factory=dict)
    active_player: Player = Player.BLACK
    turn_phase: TurnPhase = TurnPhase.OPENING
    stones_remaining: dict[Player, int] = field(default_factory=dict)
    consecutive_pass_count: int = 0
    current_turn: int = 1
    move_history: list[HistoryEntry] = field(default_factory=list)
    redo_stack: list[HistoryEntry] = field(default_factory=list)
    game_over: bool = False
    winner: WinnerT = None

    def clone(self) -> "GameState":
        return GameState(
            board=dict(self.board),
            active_player=self.active_player,
            turn_phase=self.turn_phase,
            stones_remaining=dict(self.stones_remaining),
            consecutive_pass_count=self.consecutive_pass_count,
            current_turn=self.current_turn,
            move_history=list(self.move_history),
            redo_stack=list(self.redo_stack),
            game_over=self.game_over,
            winner=self.winner,
        )

    def move_count(self) -> int:
        """Number of replayable normal gameplay actions in history.
        Sandbox actions are NOT in move_history and are NOT counted."""
        return len(self.move_history)
