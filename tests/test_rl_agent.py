"""Tests for the RLAgent endgame override (and its inheritance by HybridAgent)."""

from __future__ import annotations

import pytest

from flip7.action import Action
from flip7.agents.heuristic_agent import HeuristicConfig
from flip7.agents.hybrid_agent import HybridAgent
from flip7.agents.rl_agent import ENDGAME_BUST_VALUE, RLAgent
from tests.test_heuristic_v2 import CALM_RIVALS, make_state

STOP_DRAW = [Action.STOP, Action.DRAW]


class _Recorder:
    """Stand-in for the endgame heuristic that records every delegation."""

    def __init__(self, answer: Action = Action.DRAW) -> None:
        self.calls = 0
        self.answer = answer

    def act(self, gs, legal_actions):
        self.calls += 1
        return self.answer


@pytest.fixture(params=[RLAgent, HybridAgent], ids=["rl", "hybrid"])
def agent(request) -> RLAgent:
    agent = request.param()
    assert agent.model_loaded, "Ergebnisse/rl_actor.npz is required for these tests"
    return agent


def test_endgame_heuristic_is_separate_from_fallback(agent: RLAgent) -> None:
    assert agent._endgame is not agent._fallback
    assert agent._endgame.config == HeuristicConfig(endgame_bust_value=ENDGAME_BUST_VALUE)
    assert agent._fallback.config == HeuristicConfig()


def test_override_delegates_endgame_stop_draw(agent: RLAgent) -> None:
    agent._endgame = _Recorder(Action.DRAW)
    gs = make_state(
        hand=frozenset({5}), seen=frozenset({5}), round_score=5, total_score=195,
        opponents=CALM_RIVALS,
    )
    assert agent.act(gs, STOP_DRAW) == Action.DRAW
    assert agent._endgame.calls == 1


def test_override_not_used_before_endgame(agent: RLAgent) -> None:
    agent._endgame = _Recorder()
    gs = make_state(
        hand=frozenset({5}), seen=frozenset({5}), round_score=5, total_score=194,
        opponents=CALM_RIVALS,
    )
    assert agent.act(gs, STOP_DRAW) in STOP_DRAW
    assert agent._endgame.calls == 0


def test_override_not_used_for_target_selection(agent: RLAgent) -> None:
    agent._endgame = _Recorder()
    targets = [Action.TARGET_0, Action.TARGET_1, Action.TARGET_2]
    gs = make_state(
        total_score=210,
        legal=targets,
        pending="freeze",
        targets=(1, 2, 3),
        opponents=CALM_RIVALS,
    )
    assert agent.act(gs, targets) in targets
    assert agent._endgame.calls == 0


def test_override_not_used_when_only_stop_is_legal(agent: RLAgent) -> None:
    agent._endgame = _Recorder()
    gs = make_state(total_score=210, legal=(Action.STOP,), opponents=CALM_RIVALS)
    assert agent.act(gs, [Action.STOP]) == Action.STOP
    assert agent._endgame.calls == 0


def test_endgame_draws_with_one_card(agent: RLAgent) -> None:
    # Real delegate: banking 5 would end the game at 200, a bust only delays.
    gs = make_state(
        hand=frozenset({5}), seen=frozenset({5}), round_score=5, total_score=195,
        opponents=CALM_RIVALS,
    )
    assert agent.act(gs, STOP_DRAW) == Action.DRAW
