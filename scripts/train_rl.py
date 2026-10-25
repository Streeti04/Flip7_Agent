#!/usr/bin/env python3
"""Train MaskablePPO and export a dependency-light NumPy actor."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import fields
from pathlib import Path
import shutil
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from flip7.agents.base_agent import BaseAgent
from flip7.agents.heuristic_agent import HeuristicAgent, HeuristicConfig
from flip7.agents.heuristic_v1 import HeuristicAgentV1
from flip7.agents.rl_agent import RLAgent
from flip7.env import Flip7Env
from flip7.game import Flip7Game
from tournament import ConservativeAgent, RandomAgent, ThresholdAgent

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def RandomizedHeuristic(rng: np.random.Generator | None = None) -> HeuristicAgent:
    """HeuristicAgent with every config parameter jittered ±50% around its default.

    The multiplicative jitter U(0.5, 1.5) preserves each parameter's sign. (An
    earlier version clamped to >= 1e-9, which silently turned the tuned
    negative ``seat_continuation`` into 0 for every pool opponent.)
    """
    rng = rng or np.random.default_rng()
    default = HeuristicConfig()
    jittered = {
        field.name: float(getattr(default, field.name)) * rng.uniform(0.5, 1.5)
        for field in fields(HeuristicConfig)
    }
    return HeuristicAgent(HeuristicConfig(**jittered))


DEFAULT_OPPONENT_POOL = (
    RandomAgent,
    ConservativeAgent,
    ThresholdAgent,
    HeuristicAgent,
    HeuristicAgentV1,
    RandomizedHeuristic,
    RandomizedHeuristic,
    RandomizedHeuristic,
)

EVAL_OPPONENT_FACTORIES = (ThresholdAgent, ConservativeAgent, HeuristicAgentV1)


class EpisodeStatsCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.returns: list[float] = []
        self.lengths: list[int] = []
        self.scores: list[float] = []
        self.wins: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            self.returns.append(float(episode["r"]))
            self.lengths.append(int(episode["l"]))
            final_scores = info.get("final_scores") or {}
            learner_id = int(info.get("learner_id", -1))
            learner_score = float(final_scores.get(learner_id, 0.0))
            self.scores.append(learner_score)
            self.wins.append(
                int(bool(final_scores) and learner_score == max(final_scores.values()))
            )
        return True


def linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return initial_value * progress_remaining

    return schedule


def make_env(seed: int, opponent_pool=DEFAULT_OPPONENT_POOL, reward_mode: str = "score", terminal_bonus: float = 1.0):
    def factory():
        env = Flip7Env(
            n_players=4,
            opponent_pool=opponent_pool,
            reward_mode=reward_mode,
            terminal_bonus=terminal_bonus,
        )
        env.reset(seed=seed)
        return Monitor(env)

    return factory


def export_actor(model: MaskablePPO, path: Path, n_players: int = 4) -> None:
    policy_layers = [
        layer
        for layer in model.policy.mlp_extractor.policy_net
        if isinstance(layer, nn.Linear)
    ]
    layers = [*policy_layers, model.policy.action_net]
    arrays: dict[str, np.ndarray] = {
        "n_players": np.asarray(n_players, dtype=np.int64),
        "layer_count": np.asarray(len(layers), dtype=np.int64),
    }
    for index, layer in enumerate(layers):
        arrays[f"weight_{index}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"bias_{index}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def evaluate_actor(
    model_path: Path,
    n_episodes: int,
    seed: int,
    n_players: int = 4,
    opponent_factories=EVAL_OPPONENT_FACTORIES,
) -> tuple[float, float]:
    """Mini-tournament: RL actor vs. fixed opponents, learner seat rotates.

    Uses its own seeded rng so evaluation is deterministic and independent of
    training randomness.
    """
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    wins = 0
    for episode in range(n_episodes):
        ep_seed = int(rng.integers(2**31))
        agents: list[BaseAgent] = [
            RLAgent(model_path=model_path),
            *(factory() for factory in opponent_factories),
        ]
        seat = episode % n_players
        arrangement = agents[-seat:] + agents[:-seat] if seat else agents
        game = Flip7Game(n_players=n_players, seed=ep_seed)
        result = game.run_game(arrangement)
        my_score = float(result[seat])
        scores.append(my_score)
        if my_score == max(result.values()):
            wins += 1
    return float(np.mean(scores)), wins / n_episodes


class CheckpointEvalCallback(BaseCallback):
    """Export an actor checkpoint periodically, evaluate it, keep the best."""

    def __init__(
        self,
        output_dir: Path,
        checkpoint_freq: int,
        eval_games: int,
        seed: int,
        n_envs: int,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_freq = max(1, checkpoint_freq // n_envs)
        self.eval_games = eval_games
        self.seed = seed
        self.n_envs = n_envs
        self.best_mean_score = float("-inf")
        self.history: list[tuple[int, float, float]] = []

    def _init_callback(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.output_dir / "checkpoint_evals.csv"
        if not self._csv_path.exists():
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self._csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["step", "mean_score", "win_rate"])

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq != 0:
            return True
        step = self.n_calls * self.n_envs
        checkpoint = self.checkpoint_dir / f"rl_actor_{step}.npz"
        export_actor(self.model, checkpoint)
        mean_score, win_rate = evaluate_actor(
            checkpoint, self.eval_games, seed=self.seed + 999
        )
        self.history.append((step, mean_score, win_rate))
        with self._csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([step, f"{mean_score:.4f}", f"{win_rate:.4f}"])
        print(f"[checkpoint] step={step} mean_score={mean_score:.2f} win_rate={win_rate:.1%}")
        if mean_score > self.best_mean_score:
            self.best_mean_score = mean_score
            shutil.copyfile(checkpoint, self.output_dir / "rl_actor_best.npz")
        return True


def save_learning_outputs(callback: EpisodeStatsCallback, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "return", "length", "final_score", "win"])
        for index, row in enumerate(
            zip(
                callback.returns,
                callback.lengths,
                callback.scores,
                callback.wins,
            ),
            start=1,
        ):
            writer.writerow([index, *row])
    plot_learning_curve(
        csv_path,
        output_dir / "learning_curve.png",
        checkpoint_csv=output_dir / "checkpoint_evals.csv",
    )


def plot_learning_curve(
    metrics_csv: Path,
    output_png: Path,
    *,
    checkpoint_csv: Path | None = None,
    mark_step: int | None = None,
    window: int = 1000,
    eval_games: int = 150,
) -> None:
    """Plot the learning curve from ``training_metrics.csv`` (no retraining needed).

    Panel 1: cumulative reward over episodes (required plot). Panel 2: moving
    average of the episode return. Panel 3 (if ``checkpoint_csv`` exists):
    periodic checkpoint evaluations with a 95 % CI. ``mark_step`` marks the
    exported checkpoint (training step) in all panels.
    """
    with metrics_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    returns = np.asarray([float(row["return"]) for row in rows], dtype=np.float64)
    steps = np.cumsum([int(row["length"]) for row in rows])
    episodes = np.arange(1, returns.size + 1)
    window = max(1, min(window, returns.size))
    rolling = np.convolve(returns, np.ones(window) / window, mode="valid")
    evals = []
    if checkpoint_csv is not None and checkpoint_csv.is_file():
        with checkpoint_csv.open(encoding="utf-8", newline="") as handle:
            evals = [(int(r["step"]), float(r["mean_score"])) for r in csv.DictReader(handle)]
    mark_episode = (
        int(np.searchsorted(steps, mark_step)) + 1 if mark_step is not None else None
    )

    blue, gray = "#1f5fa8", "#6b6b6b"
    panels = 3 if evals else 2
    figure, axes = plt.subplots(panels, 1, figsize=(9, 3.0 * panels))
    top, middle = axes[0], axes[1]
    top.plot(episodes, np.cumsum(returns), color=blue, linewidth=2)
    top.set_ylabel("kumulativer Reward")
    top.set_title("Kumulativer Reward über alle Trainingsepisoden", loc="left", fontsize=11)
    middle.plot(np.arange(window, returns.size + 1), rolling, color=blue, linewidth=1.5)
    middle.set_ylabel("Return (≈ Endscore/100)")
    middle.set_title(
        f"Gleitender Mittelwert des Episodenreturns (Fenster {window} Episoden)",
        loc="left",
        fontsize=11,
    )
    for axis in (top, middle):
        axis.set_xlabel("Episode")
        if mark_episode is not None:
            axis.axvline(mark_episode, color=gray, linestyle="--", linewidth=1)
    if mark_episode is not None:
        middle.annotate(
            f"exportierter Checkpoint\n({mark_step:,} Schritte)".replace(",", "."),
            xy=(mark_episode, float(rolling.min())),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color=gray,
        )
    if evals:
        bottom = axes[2]
        eval_steps = np.asarray([step for step, _ in evals]) / 1e6
        eval_means = np.asarray([mean for _, mean in evals])
        # 95 % CI of an eval_games mean with the observed final-score SD (~46 points).
        half_width = 1.96 * 46.0 / np.sqrt(eval_games)
        bottom.errorbar(
            eval_steps, eval_means, yerr=half_width, color=blue, marker="o",
            markersize=6, linewidth=1.5, capsize=3,
        )
        if mark_step is not None:
            bottom.axvline(mark_step / 1e6, color=gray, linestyle="--", linewidth=1)
        bottom.set_xlabel("Trainingsschritte (Mio.)")
        bottom.set_ylabel("Mean Score")
        bottom.set_title(
            f"Checkpoint-Evaluation ({eval_games} Spiele vs. Threshold, Conservative, "
            "Heuristik V1; 95 %-KI)",
            loc="left",
            fontsize=11,
        )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=180)
    plt.close(figure)


def write_training_info(
    args: argparse.Namespace,
    callback: EpisodeStatsCallback,
    learn_seconds: float,
    total_seconds: float,
) -> None:
    """Log wall time and the run setup for this training run."""
    info = {
        "wall_time_total_min": round(total_seconds / 60.0, 2),
        "wall_time_learn_min": round(learn_seconds / 60.0, 2),
        "steps": args.steps,
        "steps_per_second": round(args.steps / max(learn_seconds, 1e-9), 1),
        "episodes": len(callback.returns),
        "args": {key: str(value) for key, value in vars(args).items()},
    }
    (args.output / "training_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print(
        f"wall_time_total={info['wall_time_total_min']:.1f} min "
        f"(learn {info['wall_time_learn_min']:.1f} min, {info['steps_per_second']:.0f} steps/s)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, default=Path("Ergebnisse"))
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--reward-mode", choices=("competitive", "score"), default="score"
    )
    # 0.0 (v4): with gamma=1 the return is exactly final_score/100, i.e. the
    # tournament metric. A ±1 win bonus (v3 default) was measurably worse with
    # the v4 observation (Ergebnisse/RL_V4_ERGEBNISSE.md).
    parser.add_argument("--terminal-bonus", type=float, default=0.0)
    parser.add_argument("--net", type=str, default="128,128")
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--eval-games", type=int, default=150)
    parser.add_argument("--league-dir", type=Path, default=None)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch CPU threads (0 = torch default); useful for parallel seeds",
    )
    return parser.parse_args()


def main() -> None:
    """Train MaskablePPO on Flip 7 and export NumPy actors.

    Metric alignment: with reward_mode="score" the per-round reward is the
    learner's round score / 100. With gamma=1.0 the discounted episode return
    therefore sums exactly to final_total_score / 100 (+ terminal bonus, 0 by
    default), i.e. the optimised objective matches the tournament metric.
    """
    args = parse_args()
    started = time.perf_counter()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    opponent_pool: list = list(DEFAULT_OPPONENT_POOL)
    if args.league_dir is not None:
        opponent_pool += [
            (lambda path=path: RLAgent(model_path=path))
            for path in sorted(args.league_dir.glob("*.npz"))
        ]
    net_arch = [int(size) for size in args.net.split(",")]
    env = DummyVecEnv(
        [
            make_env(
                args.seed + index,
                opponent_pool=opponent_pool,
                reward_mode=args.reward_mode,
                terminal_bonus=args.terminal_bonus,
            )
            for index in range(args.n_envs)
        ]
    )
    callback = EpisodeStatsCallback()
    checkpoint_callback = CheckpointEvalCallback(
        output_dir=args.output,
        checkpoint_freq=args.checkpoint_freq,
        eval_games=args.eval_games,
        seed=args.seed,
        n_envs=args.n_envs,
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(3e-4),
        n_steps=512,
        batch_size=256,
        n_epochs=8,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "activation_fn": nn.Tanh,
            "net_arch": {"pi": net_arch, "vf": net_arch},
        },
        seed=args.seed,
        verbose=1,
        device="cpu",
    )
    learn_started = time.perf_counter()
    model.learn(
        total_timesteps=args.steps,
        callback=[callback, checkpoint_callback],
        progress_bar=False,
    )
    learn_seconds = time.perf_counter() - learn_started

    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "maskable_ppo_flip7")
    export_actor(model, args.output / "rl_actor.npz")
    save_learning_outputs(callback, args.output)
    write_training_info(args, callback, learn_seconds, time.perf_counter() - started)

    recent_scores = deque(callback.scores[-100:], maxlen=100)
    recent_wins = deque(callback.wins[-100:], maxlen=100)
    print(f"episodes={len(callback.returns)}")
    if recent_scores:
        print(f"last_100_mean_score={np.mean(recent_scores):.2f}")
        print(f"last_100_win_rate={np.mean(recent_wins):.1%}")


if __name__ == "__main__":
    main()
