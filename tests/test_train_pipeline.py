"""Fast unit tests for the RL training pipeline (no long training runs)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_rl.py"


def _load_train_rl():
    spec = importlib.util.spec_from_file_location("train_rl", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_rl():
    return _load_train_rl()


def test_randomized_heuristic_same_rng_same_config(train_rl):
    first = train_rl.RandomizedHeuristic(rng=np.random.default_rng(7))
    second = train_rl.RandomizedHeuristic(rng=np.random.default_rng(7))
    assert isinstance(first, train_rl.HeuristicAgent)
    assert first.config == second.config


def test_randomized_heuristic_different_rng_differs(train_rl):
    configs = [
        train_rl.RandomizedHeuristic(rng=np.random.default_rng(seed)).config
        for seed in (1, 2, 3, 4)
    ]
    assert len(set(configs)) > 1
    default = vars(train_rl.HeuristicConfig())
    for config in configs:
        for name, value in vars(config).items():
            # ±50 % jitter around the default, sign preserved (regression: the
            # negative seat_continuation used to be clamped to ~0).
            low, high = sorted((0.5 * default[name], 1.5 * default[name]))
            assert low <= value <= high, name
        assert config.seat_continuation < 0


def test_evaluate_actor_returns_positive_mean(train_rl, tmp_path):
    missing = tmp_path / "does_not_exist.npz"  # RLAgent falls back to HeuristicAgent
    mean_score, win_rate = train_rl.evaluate_actor(missing, n_episodes=5, seed=1234)
    assert isinstance(mean_score, float)
    assert mean_score > 0
    assert 0.0 <= win_rate <= 1.0


def test_parse_args_defaults(train_rl, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_rl.py"])
    args = train_rl.parse_args()
    assert args.gamma == 1.0
    assert args.reward_mode == "score"
    assert args.terminal_bonus == 0.0
    assert args.steps == 1_500_000
    assert args.net == "128,128"
