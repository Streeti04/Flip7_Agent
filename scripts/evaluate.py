#!/usr/bin/env python3
"""Run the required 1000-episode round-robin and persist the leaderboard."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flip7.agents.heuristic_agent import HeuristicAgent
from flip7.agents.rl_agent import RLAgent
from flip7.agents.base_agent import BaseAgent
from flip7.game import Flip7Game
from tournament import (
    ConservativeAgent,
    RandomAgent,
    ThresholdAgent,
    print_leaderboard,
    run_tournament,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=8675309)
    parser.add_argument(
        "--output", type=Path, default=Path("Ergebnisse/evaluation.csv")
    )
    return parser.parse_args()


def evaluate_regime(
    focal_factory: type[BaseAgent],
    opponent_factory: type[BaseAgent],
    *,
    episodes: int,
    seed: int,
) -> tuple[float, float]:
    total_score = 0.0
    wins = 0
    for episode in range(episodes):
        learner_seat = episode % 4
        agents: list[BaseAgent] = []
        for seat in range(4):
            factory = focal_factory if seat == learner_seat else opponent_factory
            if issubclass(factory, RandomAgent):
                agent = factory(
                    rng=np.random.default_rng(seed + 10_000 * episode + seat)  # type: ignore[call-arg]
                )
            else:
                agent = factory()
            agents.append(agent)
        scores = Flip7Game(n_players=4, seed=seed + episode).run_game(agents)
        focal_score = scores[learner_seat]
        total_score += focal_score
        wins += int(focal_score == max(scores.values()))
    return total_score / episodes, wins / episodes


def main() -> None:
    args = parse_args()
    agents = [
        ("RandomAgent", RandomAgent),
        ("ConservativeAgent", ConservativeAgent),
        ("ThresholdAgent", ThresholdAgent),
        ("HeuristicAgent", HeuristicAgent),
        ("RLAgent", RLAgent),
    ]
    leaderboard = run_tournament(
        agents,
        n_episodes=args.episodes,
        n_players=4,
        seed=args.seed,
    )
    print_leaderboard(leaderboard)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "agent", "mean_score", "win_rate", "games"])
        for rank, result in enumerate(leaderboard, start=1):
            writer.writerow(
                [rank, result.name, result.mean_score, result.win_rate, result.games]
            )

    regime_path = args.output.with_name("evaluation_regimes.csv")
    regimes = []
    opponent_set = (
            ("RandomAgent", RandomAgent),
            ("ConservativeAgent", ConservativeAgent),
            ("ThresholdAgent", ThresholdAgent),
            ("HeuristicAgent", HeuristicAgent),
            ("RLAgent", RLAgent),
    )
    for focal_name, focal in (("RLAgent", RLAgent), ("HeuristicAgent", HeuristicAgent)):
        for opponent_index, (opponent_name, opponent) in enumerate(opponent_set):
            if focal is opponent:
                continue
            mean_score, win_rate = evaluate_regime(
                focal,
                opponent,
                episodes=args.episodes,
                seed=args.seed + 100_000 * (opponent_index + 1),
            )
            regimes.append((focal_name, opponent_name, mean_score, win_rate))

    with regime_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_agent", "three_opponents", "mean_score", "win_rate"])
        writer.writerows(regimes)

    print("\nRegime evaluation (one focal agent vs three identical opponents)")
    for focal, opponent, mean_score, win_rate in regimes:
        print(f"{focal:<16} vs {opponent:<18} {mean_score:7.2f} {win_rate:6.1%}")


if __name__ == "__main__":
    main()
