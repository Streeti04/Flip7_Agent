"""Example submission — copy this folder's structure for your own agent.

Rename the parent folder to your name/matriculation number; the folder name
becomes the display name on the tournament leaderboard.
"""

from flip7.agents.base_agent import BaseAgent, Action


class Agent(BaseAgent):
    """Stops as soon as 4 or more number cards are collected."""

    def act(self, gs, legal_actions):
        if Action.STOP not in legal_actions:
            return legal_actions[0]
        if len(gs.my_number_cards) >= 4:
            return Action.STOP
        return Action.DRAW
