"""Tests for flip7.recorder — game recording and deterministic replay."""

import pytest
from flip7.recorder import record_game, replay_game, GameRecord
from tournament import RandomAgent, ConservativeAgent, ThresholdAgent
from tests.test_game import AlwaysStopAgent, AlwaysDrawAgent, DrawNTimesAgent


def test_record_game_matches_run_game_scores():
    agents = [
        AlwaysDrawAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
    ]
    names = ["Drawer", "Stop1", "Stop2", "Stop3"]
    scores, record = record_game(agents, names, seed=42)
    assert record.seed == 42
    assert record.n_players == 4
    assert record.agent_names == names
    assert record.final_scores == scores
    assert len(record.actions) > 0


def test_replay_reproduces_identical_final_scores():
    agents = [DrawNTimesAgent(3), AlwaysStopAgent(), AlwaysStopAgent()]
    names = ["Drawer3", "Stop1", "Stop2"]
    scores, record = record_game(agents, names, seed=7)

    last_result = None
    for result, _action in replay_game(record):
        last_result = result
    assert last_result.final_scores == scores


def test_replay_yields_same_action_sequence():
    agents = [AlwaysDrawAgent(), AlwaysStopAgent(), AlwaysStopAgent()]
    names = ["Drawer", "Stop1", "Stop2"]
    _scores, record = record_game(agents, names, seed=3)

    replayed_actions = [
        (result.acting_player_id, int(action))
        for result, action in replay_game(record)
        if action is not None
    ]
    assert replayed_actions == record.actions


def test_replay_game_states_are_legal_for_well_behaved_agents():
    """For agents that respect legal_actions, every replayed action must stay legal.

    Uses the tournament.py baseline agents (which all correctly branch on
    legal_actions, including TARGET_* selection), across several seeds to
    cover Freeze/Second Chance/Flip Three target-selection decisions too.
    """
    agents_factory = [RandomAgent, ConservativeAgent, ThresholdAgent]
    for seed in range(10):
        agents = [f() for f in agents_factory]
        names = [f.__name__ for f in agents_factory]
        _scores, record = record_game(agents, names, seed=seed)

        for result, action in replay_game(record):
            if action is not None:
                assert action in result.game_state.legal_actions


def test_game_record_json_roundtrip(tmp_path):
    agents = [AlwaysStopAgent()] * 3
    names = ["A", "B", "C"]
    _scores, record = record_game(agents, names, seed=1)

    path = tmp_path / "game.json"
    record.save(path)
    loaded = GameRecord.load(path)

    assert loaded.seed == record.seed
    assert loaded.n_players == record.n_players
    assert loaded.agent_names == record.agent_names
    assert loaded.actions == record.actions
    assert loaded.final_scores == record.final_scores


def test_replay_mismatch_raises_on_wrong_acting_player():
    agents = [AlwaysStopAgent()] * 3
    names = ["A", "B", "C"]
    _scores, record = record_game(agents, names, seed=0)

    # Corrupt the recorded acting player id for the first action.
    pid, action_value = record.actions[0]
    record.actions[0] = ((pid + 1) % record.n_players, action_value)

    with pytest.raises(ValueError):
        list(replay_game(record))


def test_record_and_replay_across_multiple_rounds():
    """Intermediate round boundaries must be recorded correctly and replayed
    (as visible, action=None yields) without being counted as real decisions."""
    agents = [RandomAgent(), ConservativeAgent(), ThresholdAgent(), RandomAgent()]
    names = ["R1", "C", "T", "R2"]
    scores, record = record_game(agents, names, seed=1)

    replayed = list(replay_game(record))

    real_action_pairs = [
        (result.acting_player_id, int(action))
        for result, action in replayed
        if action is not None
    ]
    assert real_action_pairs == record.actions

    intermediate_boundaries = [
        result
        for result, action in replayed
        if action is None and result.round_over and not result.game_over
    ]
    assert len(intermediate_boundaries) > 0, (
        "replay must pass through at least one intermediate round boundary"
    )
    for result in intermediate_boundaries:
        assert result.game_state.round_scores is not None

    last_result = replayed[-1][0]
    assert last_result.final_scores == scores
