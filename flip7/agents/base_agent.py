from __future__ import annotations
from typing import TYPE_CHECKING
from flip7.action import Action  # re-exported for backwards compatibility

if TYPE_CHECKING:
    from flip7.game_state import GameState

__all__ = ["Action", "BaseAgent"]


class BaseAgent:
    def act(self, game_state: "GameState", legal_actions: list[Action]) -> Action:
        raise NotImplementedError

    def observe(
        self,
        game_state: "GameState",
        action: Action,
        next_game_state: "GameState",
        reward: float,
        round_over: bool,
    ) -> None:
        pass
