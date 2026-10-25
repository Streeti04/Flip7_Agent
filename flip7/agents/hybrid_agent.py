"""Hybrid agent: RL policy for Stop/Draw, heuristic target selection.

MaskablePPO sees far fewer TARGET_* decisions than Stop/Draw decisions during
training, so its target choices stay under-trained while its Stop/Draw policy
is strong. The hybrid delegates the rare target selections to the tuned
expectimax heuristic and leaves everything else to the trained policy.
"""

from __future__ import annotations

from flip7.action import Action
from flip7.agents.rl_agent import RLAgent


class HybridAgent(RLAgent):
    """RL Stop/Draw decisions + heuristic action-card targeting."""

    def act(self, gs, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions and gs.pending_action_card is not None:
            return self._fallback._choose_target(gs, [Action(a) for a in legal_actions])
        return super().act(gs, legal_actions)
