"""Runs the round-robin tournament over all student submissions plus baselines.

Usage:
    python run_submission_tournament.py

Every game played is recorded to games/ as a JSON GameRecord (seed + full
action sequence), so any individual game can later be reconstructed with
flip7.recorder.replay_game().
"""

from __future__ import annotations
import itertools
from pathlib import Path

import numpy as np

from flip7.game import IllegalActionError
from flip7.recorder import record_game
from submission_loader import load_submissions
from tournament import (
    AgentResult,
    RandomAgent,
    ConservativeAgent,
    ThresholdAgent,
    print_leaderboard,
)

N_EPISODES = 1000
N_PLAYERS = 4
SEED = 42
GAMES_DIR = Path(__file__).parent / "games"

BASELINE_AGENTS: list[tuple[str, type]] = [
    ("RandomAgent", RandomAgent),
    ("ConservativeAgent", ConservativeAgent),
    ("ThresholdAgent", ThresholdAgent),
]


def run_and_record_tournament(
    agent_factories: list[tuple[str, type]],
    n_episodes: int,
    n_players: int,
    seed: int,
    games_dir: Path,
) -> list[AgentResult]:
    games_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    results: dict[str, AgentResult] = {
        name: AgentResult(name=name) for name, _ in agent_factories
    }

    combos = list(itertools.combinations(range(len(agent_factories)), n_players))
    game_idx = 0
    for combo in combos:
        for _ in range(n_episodes):
            ep_seed = int(rng.integers(2**31))
            names = [agent_factories[i][0] for i in combo]
            agents = [agent_factories[i][1]() for i in combo]

            try:
                scores, record = record_game(agents, names, ep_seed)
            except IllegalActionError as exc:
                offender = names[exc.player_id]
                print(
                    f"Warning: {offender} raised IllegalActionError ({exc}); "
                    f"episode discarded, {offender} scored 0."
                )
                exc.partial_record.save(games_dir / f"game_{game_idx:06d}_illegal.json")
                game_idx += 1
                results[offender].total_score += 0
                results[offender].games += 1
                continue

            record.save(games_dir / f"game_{game_idx:06d}.json")
            game_idx += 1

            winning_score = max(scores.values())
            for name, pid in zip(names, scores.keys()):
                results[name].total_score += scores[pid]
                results[name].games += 1
                if scores[pid] == winning_score:
                    results[name].wins += 1

    return sorted(results.values(), key=lambda r: r.mean_score, reverse=True)


if __name__ == "__main__":
    submissions = load_submissions()
    if not submissions:
        print("No submissions found in submissions/ — running baselines only.\n")

    agents = BASELINE_AGENTS + submissions
    print(
        f"Running tournament: {len(agents)} agents "
        f"({len(submissions)} submissions + {len(BASELINE_AGENTS)} baselines), "
        f"{N_EPISODES} episodes per pairing...\n"
    )
    leaderboard = run_and_record_tournament(
        agents,
        n_episodes=N_EPISODES,
        n_players=N_PLAYERS,
        seed=SEED,
        games_dir=GAMES_DIR,
    )
    print_leaderboard(leaderboard)
    print(
        f"\nAll games recorded to {GAMES_DIR}/ (replay with flip7.recorder.replay_game)."
    )
