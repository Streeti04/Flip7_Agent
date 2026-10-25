from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from flip7.action import Action
from flip7.agents.heuristic_agent import HeuristicAgent
from flip7.agents.rl_agent import RLAgent
from flip7.env import Flip7Env
from flip7.observation import observation_size
from tournament import ConservativeAgent, RandomAgent, ThresholdAgent, run_tournament


def test_flip7_env_passes_gymnasium_checker():
    check_env(Flip7Env(n_players=4), skip_render_check=True)


def test_observation_and_mask_have_fixed_shapes():
    env = Flip7Env(n_players=4, learner_seat=0)
    observation, info = env.reset(seed=123)
    assert observation.shape == (observation_size(4),)
    assert observation.dtype == np.float32
    assert np.all((-1.0 <= observation) & (observation <= 1.0))
    assert info["action_mask"].shape == (len(Action),)
    assert info["action_mask"].dtype == np.bool_


def test_out_of_mask_action_is_remapped_before_engine_step():
    env = Flip7Env(n_players=4, learner_seat=0)
    env.reset(seed=7)
    mask = env.action_masks()
    illegal = int(np.flatnonzero(~mask)[-1])
    _, _, _, _, info = env.step(illegal)
    assert info["invalid_action"] is True
    assert info["executed_action"] in (Action.STOP, Action.DRAW)


def test_round_reward_is_from_learner_perspective():
    env = Flip7Env(n_players=4, learner_seat=2, reward_mode="competitive")
    env.reset(seed=11)
    reward = env._round_reward({0: 10, 1: 20, 2: 50, 3: 30})
    assert reward == (50 - 20) / 100


def test_agents_always_return_legal_actions():
    env = Flip7Env(n_players=4, learner_seat=0)
    env.reset(seed=99)
    for agent in (HeuristicAgent(), RLAgent()):
        state = env._result.game_state
        action = agent.act(state, list(state.legal_actions))
        assert action in state.legal_actions


def test_tournament_seed_also_reproduces_random_agent():
    factories = [
        ("Random", RandomAgent),
        ("Conservative", ConservativeAgent),
        ("Threshold", ThresholdAgent),
        ("Heuristic", HeuristicAgent),
    ]
    first = run_tournament(factories, n_episodes=20, n_players=4, seed=314)
    second = run_tournament(factories, n_episodes=20, n_players=4, seed=314)
    assert [result.__dict__ for result in first] == [result.__dict__ for result in second]
