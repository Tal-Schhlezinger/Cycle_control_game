"""RulesConfig: rules configuration with validation.

Reject rules (v5 Section 3.1):
    - no end conditions enabled
    - end_on_all_stones_placed enabled while supply disabled
    - end_on_consecutive_passes is the only enabled end condition AND pass_enabled = false
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RulesConfig:
    board_radius: int = 3
    stones_per_player: int | None = None  # None = unlimited (supply disabled)
    pass_enabled: bool = True
    end_on_consecutive_passes: bool = True
    end_on_all_stones_placed: bool = False  # requires supply enabled
    end_on_board_full: bool = True

    # Experimental balance modes. All default off; any combination is allowed.
    neutrality_rule: bool = False          # placer needs own-neighbors >= opp-neighbors at target
    strict_adjacency_rule: bool = False    # placer must touch at least one own stone (if any on board)
    mirror_adjacency: bool = False         # adds edges to opposite-orientation triangles across shared vertex
    partial_credit_k: int = 0              # 0 = off; else also score nodes in components of size >= k

    # End condition: game ends when neither player has any legal placement.
    # Use this instead of end_on_consecutive_passes when pass is disabled.
    # Under strict adjacency + neutrality the board can have unreachable empty
    # cells, so "no empty cells" != "no legal moves". This condition is correct.
    end_on_no_legal_moves: bool = False

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def committed(cls, board_radius: int = 4) -> "RulesConfig":
        """Factory: the committed V2.3 ruleset for normal play.

        Pass disabled; game ends when neither player can place anywhere.
        """
        return cls(
            board_radius=board_radius,
            pass_enabled=False,
            end_on_consecutive_passes=False,
            end_on_board_full=False,
            end_on_no_legal_moves=True,
            neutrality_rule=True,
            strict_adjacency_rule=True,
            mirror_adjacency=True,
        )

    def supply_enabled(self) -> bool:
        return self.stones_per_player is not None

    def enabled_end_conditions(self) -> list[str]:
        out: list[str] = []
        if self.end_on_consecutive_passes:
            out.append("consecutive_passes")
        if self.end_on_all_stones_placed:
            out.append("all_stones_placed")
        if self.end_on_board_full:
            out.append("board_full")
        if self.end_on_no_legal_moves:
            out.append("no_legal_moves")
        return out

    def _validate(self) -> None:
        if (not isinstance(self.board_radius, int)
                or isinstance(self.board_radius, bool)
                or self.board_radius < 1):
            raise ValueError(
                f"board_radius must be integer >= 1, got {self.board_radius!r}"
            )

        if self.stones_per_player is not None:
            if (not isinstance(self.stones_per_player, int)
                    or isinstance(self.stones_per_player, bool)
                    or self.stones_per_player < 1):
                raise ValueError(
                    f"stones_per_player must be positive int or None, "
                    f"got {self.stones_per_player!r}"
                )

        enabled = self.enabled_end_conditions()
        if not enabled:
            raise ValueError("at least one end condition must be enabled")

        if self.end_on_all_stones_placed and not self.supply_enabled():
            raise ValueError(
                "end_on_all_stones_placed=True requires stones_per_player to be set"
            )

        if enabled == ["consecutive_passes"] and not self.pass_enabled:
            raise ValueError(
                "consecutive_passes as sole end condition requires pass_enabled=True"
            )

        if (not isinstance(self.partial_credit_k, int)
                or isinstance(self.partial_credit_k, bool)
                or self.partial_credit_k < 0):
            raise ValueError(
                f"partial_credit_k must be int >= 0, got {self.partial_credit_k!r}"
            )

        # end_on_no_legal_moves doesn't require pass to be disabled, but it
        # is the natural companion: if pass is enabled AND no_legal_moves is
        # the only end condition, games can still pass-loop. Warn if suspicious.
        if self.enabled_end_conditions() == [] and not self.end_on_no_legal_moves:
            pass  # already caught by "at least one" check below

    def to_dict(self) -> dict:
        return {
            "board_radius": self.board_radius,
            "stones_per_player": self.stones_per_player,
            "pass_enabled": self.pass_enabled,
            "end_on_consecutive_passes": self.end_on_consecutive_passes,
            "end_on_all_stones_placed": self.end_on_all_stones_placed,
            "end_on_board_full": self.end_on_board_full,
            "end_on_no_legal_moves": self.end_on_no_legal_moves,
            "neutrality_rule": self.neutrality_rule,
            "strict_adjacency_rule": self.strict_adjacency_rule,
            "mirror_adjacency": self.mirror_adjacency,
            "partial_credit_k": self.partial_credit_k,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RulesConfig":
        return cls(
            board_radius=d.get("board_radius", 3),
            stones_per_player=d.get("stones_per_player", None),
            pass_enabled=d.get("pass_enabled", True),
            end_on_consecutive_passes=d.get("end_on_consecutive_passes", True),
            end_on_all_stones_placed=d.get("end_on_all_stones_placed", False),
            end_on_board_full=d.get("end_on_board_full", True),
            end_on_no_legal_moves=d.get("end_on_no_legal_moves", False),
            neutrality_rule=d.get("neutrality_rule", False),
            strict_adjacency_rule=d.get("strict_adjacency_rule", False),
            mirror_adjacency=d.get("mirror_adjacency", False),
            partial_credit_k=d.get("partial_credit_k", 0),
        )
