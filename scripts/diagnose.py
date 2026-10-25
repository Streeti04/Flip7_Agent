#!/usr/bin/env python3
"""Summarise learned and heuristic decisions by hand size and safety signals."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flip7.action import Action
from flip7.agents.base_agent import BaseAgent
from flip7.agents.heuristic_agent import HeuristicAgent
from flip7.agents.rl_agent import RLAgent
from flip7.game import Flip7Game
from tournament import ConservativeAgent, ThresholdAgent


class Recorder(BaseAgent):
    def __init__(self, name: str, inner: BaseAgent, counts: dict) -> None:
        self.name = name
        self.inner = inner
        self.counts = counts

    def act(self, game_state, legal_actions):
        action = self.inner.act(game_state, legal_actions)
        if Action.STOP in legal_actions:
            keys = (
                ("hand_size", str(len(game_state.my_number_cards))),
                ("second_chance", str(bool(game_state.my_has_second_chance))),
                ("score_band", str(min(60, game_state.my_round_score // 10 * 10))),
            )
            for context, key in keys:
                bucket = self.counts[(self.name, context, key)]
                bucket["decisions"] += 1
                bucket["draws"] += int(action == Action.DRAW)
        else:
            context = f"target_{game_state.pending_action_card}"
            bucket = self.counts[(self.name, context, str(int(action)))]
            bucket["decisions"] += 1
        return action


def main() -> None:
    counts = defaultdict(lambda: {"decisions": 0, "draws": 0})
    for episode in range(1000):
        seats = [
            Recorder("RLAgent", RLAgent(), counts),
            Recorder("HeuristicAgent", HeuristicAgent(), counts),
            ConservativeAgent(),
            ThresholdAgent(),
        ]
        shift = episode % 4
        seats = seats[shift:] + seats[:shift]
        Flip7Game(n_players=4, seed=40_000 + episode).run_game(seats)

    output = Path("Ergebnisse/decision_diagnostics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent", "context", "value", "decisions", "draw_rate"])
        for (agent, context, value), bucket in sorted(counts.items()):
            decisions = bucket["decisions"]
            writer.writerow(
                [agent, context, value, decisions, bucket["draws"] / decisions]
            )

    for agent in ("RLAgent", "HeuristicAgent"):
        print(agent)
        for (name, context, value), bucket in sorted(counts.items()):
            if name == agent and context in {"hand_size", "second_chance"}:
                rate = bucket["draws"] / bucket["decisions"]
                print(f"  {context}={value}: n={bucket['decisions']} draw={rate:.1%}")


if __name__ == "__main__":
    main()

