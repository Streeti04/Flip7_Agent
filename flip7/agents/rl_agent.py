"""Tournament-ready NumPy inference for the trained MaskablePPO actor.

The actor (``Ergebnisse/rl_actor.npz``, v4) expects the 141-dim observation of
:func:`flip7.observation.encode_game_state` for 4 players (incl. the game-end
block). Stale exports with another input size fall back to the heuristic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flip7.action import Action
from flip7.agents.base_agent import BaseAgent
from flip7.agents.heuristic_agent import HeuristicAgent, HeuristicConfig
from flip7.game_state import GameState
from flip7.observation import encode_game_state, legal_action_mask, observation_size

# Bust value for the endgame override (see RLAgent.act). Larger than the
# heuristic's own default: measured best for the RL agent in both evaluation
# fields (VERBESSERUNGSVORSCHLAEGE.md §2.1).
ENDGAME_BUST_VALUE = 25.0


class RLAgent(BaseAgent):
    """Run exported policy weights without a PyTorch/SB3 runtime dependency."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._fallback = HeuristicAgent()
        self._endgame = HeuristicAgent(
            HeuristicConfig(endgame_bust_value=ENDGAME_BUST_VALUE)
        )
        self._weights: list[tuple[np.ndarray, np.ndarray]] = []
        self._n_players = 4
        self._load(model_path)

    @staticmethod
    def _default_model_paths() -> tuple[Path, ...]:
        module_path = Path(__file__).resolve()
        project_root = module_path.parents[2]
        return (
            module_path.with_name("rl_actor.npz"),
            project_root / "Ergebnisse" / "rl_actor.npz",
            project_root / "submissions" / "flip7_rl" / "rl_actor.npz",
        )

    def _load(self, model_path: str | Path | None) -> None:
        candidates = (Path(model_path),) if model_path is not None else self._default_model_paths()
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            return

        try:
            with np.load(path, allow_pickle=False) as data:
                self._n_players = int(data["n_players"].item())
                layer_count = int(data["layer_count"].item())
                self._weights = [
                    (
                        np.asarray(data[f"weight_{index}"], dtype=np.float32),
                        np.asarray(data[f"bias_{index}"], dtype=np.float32),
                    )
                    for index in range(layer_count)
                ]
        except Exception:
            # Corrupt or incompatible weights must never crash a tournament;
            # act() degrades to the heuristic fallback via _compatible_weights.
            self._weights = []

    @property
    def model_loaded(self) -> bool:
        return bool(self._weights)

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if not legal_actions:
            # Never raise: an empty legal list has no meaningful choice.
            return self._fallback.act(gs, legal_actions)
        if (
            Action.STOP in legal_actions
            and Action.DRAW in legal_actions
            and gs.my_total_score + gs.my_round_score >= 200
        ):
            # Endgame override: banking now would end the game, while a bust
            # only costs this round. The policy learned this rare pattern only
            # partially (too cautious with 3-5 cards), so these Stop/Draw
            # decisions go to the endgame-aware heuristic.
            return self._endgame.act(gs, legal_actions)
        weights = self._compatible_weights(gs)
        if weights is None:
            return self._fallback.act(gs, legal_actions)

        x = encode_game_state(gs, self._n_players)
        for index, (weight, bias) in enumerate(weights):
            x = weight @ x + bias
            if index < len(weights) - 1:
                x = np.tanh(x)

        mask = legal_action_mask(gs)
        x = np.where(mask, x, -np.inf)
        selected = Action(int(np.argmax(x)))
        if selected not in legal_actions:  # final defence against corrupt weights
            return self._fallback.act(gs, legal_actions)
        return selected

    def _compatible_weights(
        self, gs: GameState
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Loaded layers, or None if they cannot apply to this observation.

        Guards against stale npz exports whose input dimension no longer
        matches the encoder: the agent must degrade to the heuristic fallback
        instead of crashing mid-tournament.
        """
        if not self._weights or len(gs.opponents) + 1 != self._n_players:
            return None
        expected_input = self._weights[0][0].shape[-1]
        if expected_input != observation_size(self._n_players):
            return None
        return self._weights


# Convenient name for direct tournament registration.
Agent = RLAgent

