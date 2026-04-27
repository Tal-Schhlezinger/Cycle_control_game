"""Cycle Control AI module.

Implements the bot interface, baseline bots, territory/siege analysis, and
tournament harness per AI_IMPLEMENTATION_CHECKLIST_V2.4.1.md Sections 4, 5,
4.4 (tournament), and 4.2c (Greedy variant validation).

Not yet implemented: SearchBot (Section 4.3), PettingZoo env (Section 9),
GNN / CNN networks (Sections 7, 8), training harness (Sections 10-12).
"""

from .action_space import (
    ActionSpace, action_index_to_node, action_space_size,
    build_legal_mask, node_to_action_index, pass_index,
)
from .bot_interface import Bot, play_game, play_turn
from .bots import (
    FrontierRandomBot, Greedy1, Greedy2, GreedyBot, RandomBot,
    SearchBot, SearchStats,
)
from .siege import (
    exclusive_territory, frontier_count, reachable_empty_cells,
    sieged_against, sieged_for, territory_score,
)
from .tournament import (
    MatchResult, RoundRobinResult, elo_from_round_robin, elo_update,
    round_robin, run_match,
)

__all__ = [
    # action space
    "ActionSpace", "action_index_to_node", "action_space_size",
    "build_legal_mask", "node_to_action_index", "pass_index",
    # bot interface
    "Bot", "play_game", "play_turn",
    # bots
    "RandomBot", "FrontierRandomBot", "GreedyBot", "Greedy1", "Greedy2",
    "SearchBot", "SearchStats",
    # siege analysis
    "exclusive_territory", "frontier_count", "reachable_empty_cells",
    "sieged_against", "sieged_for", "territory_score",
    # tournament
    "MatchResult", "RoundRobinResult", "elo_from_round_robin", "elo_update",
    "round_robin", "run_match",
]
