#!/usr/bin/env python3
"""Random-search tuning of HeuristicConfig against a diverse validation field.

Method: sample configurations, evaluate each by playing the tuned agent in a
rotating seat against a fixed opponent field, then re-rank the top candidates
on disjoint holdout seeds with a *different* opponent field to guard against
seed/field overfitting.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flip7.agents.base_agent import BaseAgent
from flip7.agents.heuristic_agent import HeuristicAgent, HeuristicConfig
from flip7.agents.heuristic_v1 import HeuristicAgentV1
from flip7.game import Flip7Game
from tournament import ConservativeAgent, RandomAgent, ThresholdAgent

# Parameter bounds for the random search.
SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "freeze_value": (0.0, 5.0),
    "flip_three_value": (0.0, 8.0),
    "freeze_weight_flip7": (0.0, 14.0),
    "freeze_weight_second_chance": (0.0, 10.0),
    "freeze_weight_hand": (0.0, 3.0),
    "freeze_weight_score": (0.0, 0.08),
    "flip_three_denial_value": (0.0, 20.0),
    "flip_three_gift_penalty": (0.0, 12.0),
    "seat_continuation": (-0.5, 0.8),
}

VALIDATION_FIELD = (RandomAgent, ConservativeAgent, HeuristicAgentV1)
HOLDOUT_FIELD = (RandomAgent, ThresholdAgent, HeuristicAgentV1)

_ZOO_OPPONENTS: tuple[type[BaseAgent], ...] | None = None


def _zoo_pool() -> tuple[type[BaseAgent], ...]:
    """Diverse opponent pool for field-dependent tuning (eval_zoo registry)."""
    global _ZOO_OPPONENTS
    if _ZOO_OPPONENTS is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import eval_zoo  # noqa: local import; argparse only runs in its main()

        pool: list[type[BaseAgent]] = [
            *eval_zoo.ZOO.values(),
            RandomAgent,
            ConservativeAgent,
            ThresholdAgent,
        ]
        _ZOO_OPPONENTS = tuple(pool)
    return _ZOO_OPPONENTS


def _make_opponent(factory: type[BaseAgent], seed: int) -> BaseAgent:
    try:
        return factory(rng=np.random.default_rng(seed))  # type: ignore[call-arg]
    except TypeError:
        return factory()


def evaluate_config(
    config: HeuristicConfig,
    n_games: int,
    seed: int,
    field: tuple[type[BaseAgent], ...] | None,
    *,
    diverse: bool = False,
) -> float:
    """Mean final score of the config agent across rotating seats.

    ``diverse`` samples three opponents per episode from the zoo pool; the
    opponent sequence depends only on (seed, episode) so every config faces an
    identical opponent mix (paired comparison).
    """
    scores: list[float] = []
    zoo = _zoo_pool() if diverse else ()
    for episode in range(n_games):
        seat = episode % 4
        if diverse:
            pick_rng = np.random.default_rng(seed * 31 + episode)
            opponents = [
                zoo[i]
                for i in pick_rng.integers(len(zoo), size=3)
            ]
        else:
            assert field is not None
            opponents = list(field)
        agents: list[BaseAgent] = []
        opponent_index = 0
        for player_id in range(4):
            if player_id == seat:
                agents.append(HeuristicAgent(config))
            else:
                agents.append(
                    _make_opponent(opponents[opponent_index], seed + episode * 7 + opponent_index)
                )
                opponent_index += 1
        game = Flip7Game(n_players=4, seed=seed + episode)
        result = game.run_game(agents)
        scores.append(float(result[seat]))
    return float(np.mean(scores))


def _job(payload: tuple[dict, int, int, tuple | None, bool]) -> float:
    config_dict, n_games, seed, field, diverse = payload
    return evaluate_config(
        HeuristicConfig(**config_dict), n_games, seed, field, diverse=diverse
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--holdout-games", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("Ergebnisse"))
    parser.add_argument(
        "--center-json",
        type=Path,
        default=None,
        help="Optional config JSON; samples jitter around it instead of globally.",
    )
    parser.add_argument("--jitter", type=float, default=0.35,
                        help="Relative jitter around --center-json (0-1).")
    parser.add_argument("--field", choices=("baseline", "zoo"), default="baseline",
                        help="baseline: fixed 3-agent fields; zoo: random diverse "
                             "opponent mixes per episode (tournament-like).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    candidates: list[dict] = []
    center: dict | None = None
    if args.center_json is not None and args.center_json.is_file():
        center = json.loads(args.center_json.read_text())["config"]
    for index in range(args.samples):
        defaults = asdict(HeuristicConfig())
        if center is not None and index % 5 != 4:  # 80% local, 20% global
            config = {
                name: float(
                    np.clip(
                        center.get(name, defaults[name])
                        * rng.uniform(1 - args.jitter, 1 + args.jitter),
                        low,
                        high,
                    )
                )
                for name, (low, high) in SEARCH_SPACE.items()
            }
        else:
            config = {
                name: float(rng.uniform(low, high))
                for name, (low, high) in SEARCH_SPACE.items()
            }
        candidates.append(config)
    # Baseline (current defaults) always participates.
    candidates.append(asdict(HeuristicConfig()))

    diverse = args.field == "zoo"
    jobs = [
        (config, args.games, args.seed + 1_000_000, VALIDATION_FIELD, diverse)
        for config in candidates
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        validation_scores = list(pool.map(_job, jobs, chunksize=1))

    ranked = sorted(
        zip(validation_scores, candidates), key=lambda pair: pair[0], reverse=True
    )
    print("Top-5 Validation:")
    for score, config in ranked[:5]:
        print(f"  {score:8.2f}  {json.dumps({k: round(v, 3) for k, v in config.items()})}")
    baseline_score = validation_scores[-1]
    print(f"Baseline (defaults): {baseline_score:.2f}")

    shortlist = [config for _, config in ranked[:6]]
    holdout_seed = args.seed + 2_000_000
    jobs = [
        (config, args.holdout_games, holdout_seed, HOLDOUT_FIELD, diverse)
        for config in shortlist
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        holdout_scores = list(pool.map(_job, jobs, chunksize=1))

    best_index = int(np.argmax(holdout_scores))
    best_config = shortlist[best_index]
    print(f"\nHoldout (disjoint seeds, field={args.field}):")
    for score, config in zip(holdout_scores, shortlist):
        marker = "  <- best" if config is best_config else ""
        print(f"  {score:8.2f}  {json.dumps({k: round(v, 3) for k, v in config.items()})}{marker}")

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "tuned_heuristic_config.json"
    output_path.write_text(
        json.dumps({"config": best_config, "holdout_score": holdout_scores[best_index]}, indent=2),
        encoding="utf-8",
    )
    print(f"\nBest config saved to {output_path}")


if __name__ == "__main__":
    main()
