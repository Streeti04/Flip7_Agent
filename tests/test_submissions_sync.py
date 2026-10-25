"""Behaviour-equivalence tests: generated submissions must mirror the package code.

The submissions are auto-generated copies (scripts/build_submissions.py); these
tests guarantee that the copies act identically to the development modules and
that they import the way the grader's submission_loader imports them (module
NOT registered in sys.modules before exec — this matters on Python 3.14).
"""

from __future__ import annotations

import base64
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

from flip7.action import Action
from flip7.agents.base_agent import BaseAgent
from flip7.agents.heuristic_agent import HeuristicAgent
from flip7.agents.rl_agent import RLAgent
from flip7.game import Flip7Game
from flip7.observation import encode_game_state

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"


def _load(name: str, root: Path = SUBMISSIONS):
    """Import like submission_loader.load_submissions does (no sys.modules)."""
    agent_file = root / name / "agent.py"
    spec = importlib.util.spec_from_file_location(
        f"submissions.{name}.agent", agent_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _embedded_actor(rl_module, tmp_path: Path) -> Path:
    """Write the actor embedded in the RL submission to an npz file."""
    if not rl_module._ACTOR_NPZ_B64:
        pytest.skip("no actor embedded yet")
    path = tmp_path / "embedded_actor.npz"
    path.write_bytes(base64.b64decode(rl_module._ACTOR_NPZ_B64))
    return path


@pytest.fixture(scope="module")
def heuristic_module():
    return _load("flip7_heuristic")


@pytest.fixture(scope="module")
def rl_module():
    return _load("flip7_rl")


def test_submissions_define_agent_baseagent(heuristic_module, rl_module) -> None:
    for module in (heuristic_module, rl_module):
        assert issubclass(module.Agent, BaseAgent)


def _random_decision_states(n_games: int = 8, seed: int = 99):
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n_games):
        game = Flip7Game(n_players=4, seed=int(rng.integers(2**31)))
        result = game.reset()
        while not result.game_over:
            gs = result.game_state
            if gs.legal_actions:
                states.append((gs, list(gs.legal_actions)))
            legal = list(gs.legal_actions) or [Action.STOP]
            action = legal[int(rng.integers(len(legal)))]
            result = game.step(action)
            while result.round_over and not result.game_over:
                result = game.step(Action.STOP)
    return states


def test_heuristic_submission_actions_identical(heuristic_module) -> None:
    package_agent = HeuristicAgent()
    submission_agent = heuristic_module.Agent()
    for gs, legal in _random_decision_states():
        assert submission_agent.act(gs, legal) == package_agent.act(gs, legal)


def _endgame_decision_states(n_games: int = 12, seed: int = 7):
    """Stop/Draw decisions with total + round >= 200 (endgame override path)."""
    rng = np.random.default_rng(seed)
    heuristic = HeuristicAgent()
    states = []
    for _ in range(n_games):
        game = Flip7Game(n_players=4, seed=int(rng.integers(2**31)))
        result = game.reset()
        while not result.game_over:
            gs = result.game_state
            legal = list(gs.legal_actions)
            if (
                Action.STOP in legal
                and Action.DRAW in legal
                and gs.my_total_score + gs.my_round_score >= 200
            ):
                states.append((gs, legal))
            action = heuristic.act(gs, legal) if legal else Action.STOP
            result = game.step(action)
    return states


def test_rl_submission_fallback_matches_package(rl_module) -> None:
    """With missing weights the RL submission must act like the package RLAgent
    in fallback mode (heuristic + endgame override)."""
    agent = rl_module.Agent()
    agent._layers = []  # force the fallback path deterministically
    package_agent = RLAgent(model_path=SUBMISSIONS / "does_not_exist.npz")
    assert not package_agent.model_loaded
    states = _random_decision_states() + _endgame_decision_states()
    for gs, legal in states:
        assert agent.act(gs, legal) == package_agent.act(gs, legal)


def test_rl_submission_actions_identical(rl_module, tmp_path) -> None:
    """With the embedded actor, submission and package RLAgent must agree —
    including the endgame override decisions."""
    agent = rl_module.Agent()
    package_agent = RLAgent(model_path=_embedded_actor(rl_module, tmp_path))
    assert package_agent.model_loaded
    endgame = _endgame_decision_states()
    assert endgame, "no endgame decisions sampled"
    for gs, legal in _random_decision_states() + endgame:
        assert agent.act(gs, legal) == package_agent.act(gs, legal)


def test_rl_submission_embeds_package_default_actor(rl_module, tmp_path) -> None:
    """The embedded actor is the one the package RLAgent loads by default."""
    embedded = RLAgent(model_path=_embedded_actor(rl_module, tmp_path))
    default = RLAgent(model_path=ROOT / "Ergebnisse" / "rl_actor.npz")
    assert len(embedded._weights) == len(default._weights)
    for (w1, b1), (w2, b2) in zip(embedded._weights, default._weights):
        assert np.array_equal(w1, w2) and np.array_equal(b1, b2)


def test_rl_submission_works_as_single_file(tmp_path) -> None:
    """submissions/README.md: one agent.py per folder. Copied alone into an
    empty folder, the agent must still run the network (embedded weights)."""
    folder = tmp_path / "flip7_rl"
    folder.mkdir()
    shutil.copy2(SUBMISSIONS / "flip7_rl" / "agent.py", folder / "agent.py")
    module = _load("flip7_rl", root=tmp_path)
    if not module._ACTOR_NPZ_B64:
        pytest.skip("no actor embedded yet")
    agent = module.Agent()
    assert agent._layers, "embedded weights not loaded (heuristic fallback)"
    gs, legal = _random_decision_states(n_games=1)[0]
    assert agent._compatible(gs)
    x = module.encode_game_state(gs, agent._n_players)
    for index, (weight, bias) in enumerate(agent._layers):
        x = weight @ x + bias
        if index < len(agent._layers) - 1:
            x = np.tanh(x)
    masked = np.full(len(Action), -np.inf, dtype=np.float32)
    for action in legal:
        masked[int(action)] = x[int(action)]
    assert agent.act(gs, legal) == Action(int(np.argmax(masked)))


def test_rl_submission_corrupt_embed_falls_back(tmp_path) -> None:
    """A corrupt embed (and no npz beside it) degrades to the heuristic."""
    folder = tmp_path / "flip7_rl"
    folder.mkdir()
    shutil.copy2(SUBMISSIONS / "flip7_rl" / "agent.py", folder / "agent.py")
    module = _load("flip7_rl", root=tmp_path)
    module._ACTOR_NPZ_B64 = "bm90IGFuIG5weg=="  # "not an npz"
    agent = module.Agent()
    assert not agent._layers
    package_agent = RLAgent(model_path=tmp_path / "missing.npz")
    for gs, legal in _random_decision_states(n_games=2):
        assert agent.act(gs, legal) == package_agent.act(gs, legal)


def test_rl_submission_encoder_identical(rl_module) -> None:
    for gs, _legal in _random_decision_states():
        embedded = rl_module.encode_game_state(gs, 4)
        package = encode_game_state(gs, 4)
        assert embedded.shape == package.shape
        assert np.array_equal(embedded, package)


def test_rl_submission_uses_actor_when_compatible(rl_module) -> None:
    """If an actor is embedded, Agent must actually run it (not silently fall back)."""
    if not rl_module._ACTOR_NPZ_B64:
        pytest.skip("no actor embedded yet")
    agent = rl_module.Agent()
    assert agent._layers, "embedded actor not loaded"
    gs, legal = _random_decision_states(n_games=2)[0]
    assert agent._compatible(gs), "embedded actor incompatible with current encoder"
    # Actor decision and manual actor decision must agree.
    x = rl_module.encode_game_state(gs, agent._n_players)
    for index, (weight, bias) in enumerate(agent._layers):
        x = weight @ x + bias
        if index < len(agent._layers) - 1:
            x = np.tanh(x)
    masked = np.full(len(Action), -np.inf, dtype=np.float32)
    for action in legal:
        masked[int(action)] = x[int(action)]
    assert agent.act(gs, legal) == Action(int(np.argmax(masked)))
