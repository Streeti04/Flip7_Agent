"""Tests for game.py — base mechanics (no action cards)."""

import pytest
from flip7.agents.base_agent import BaseAgent, Action
from flip7.game import Flip7Game, IllegalActionError
from flip7.game_state import GameState
from flip7.player import PlayerStatus


class AlwaysStopAgent(BaseAgent):
    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions:
            return legal_actions[0]  # pending target selection
        return Action.STOP


class AlwaysDrawAgent(BaseAgent):
    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.DRAW not in legal_actions:
            return legal_actions[0]  # pending target selection
        return Action.DRAW


class DrawNTimesAgent(BaseAgent):
    """Draw exactly n times per round, then stop."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._drawn = 0

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions and Action.DRAW not in legal_actions:
            return legal_actions[0]  # pending target selection
        if self._drawn < self._n:
            self._drawn += 1
            return Action.DRAW
        return Action.STOP

    def observe(self, gs, action, next_gs, reward, round_over):
        if round_over:
            self._drawn = 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def make_game(n_players: int, seed: int = 0) -> Flip7Game:
    return Flip7Game(n_players=n_players, seed=seed)


def run_round(agents: list[BaseAgent], seed: int = 0) -> dict[int, int]:
    """Run exactly one round and return round scores."""
    game = Flip7Game(n_players=len(agents), seed=seed)
    result = game.reset()
    round_scores: dict[int, int] | None = None
    while not result.game_over:
        pid = result.acting_player_id
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        result = game.step(action)
        agents[pid].observe(
            gs, action, result.game_state, result.reward, result.round_over
        )
        if result.round_over:
            round_scores = result.game_state.round_scores
            break
    if round_scores is None and result.game_over:
        round_scores = result.game_state.round_scores
    return round_scores or {}


# ------------------------------------------------------------------
# Basic game mechanics
# ------------------------------------------------------------------


def test_run_round_returns_scores():
    scores = run_round([AlwaysStopAgent(), AlwaysStopAgent(), AlwaysStopAgent()])
    assert set(scores.keys()) == {0, 1, 2}


def test_player_count_validation():
    with pytest.raises(ValueError):
        Flip7Game(n_players=2, seed=0)  # 2 players — below minimum


def test_all_stop_no_bust():
    game = make_game(4, seed=42)
    agents = [AlwaysStopAgent()] * 4
    result = game.reset()
    while not result.game_over and not result.round_over:
        pid = result.acting_player_id
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        result = game.step(action)
    for p in game._players:
        assert p.status != PlayerStatus.BUSTED


def test_draw_until_bust_eventually():
    """A player who always draws must end the round either busted or with Flip 7."""
    agents = [AlwaysDrawAgent(), AlwaysStopAgent(), AlwaysStopAgent()]
    game = Flip7Game(n_players=3, seed=7)
    result = game.reset()
    while not result.game_over and not result.round_over:
        pid = result.acting_player_id
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        result = game.step(action)
    p0 = game._players[0]
    assert p0.status == PlayerStatus.BUSTED or p0.flip7_bonus


def test_scores_accumulate_across_rounds():
    """seed=1 with AlwaysStopAgent()*4 is confirmed to span >=2 rounds."""
    agents = [AlwaysStopAgent()] * 4
    game = Flip7Game(n_players=4, seed=1)

    # Round 1
    result = game.reset()
    while not result.round_over:
        pid = result.acting_player_id
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        result = game.step(action)
    assert result.game_over is False, (
        "seed=1 with AlwaysStopAgent()*4 must not win outright in round 1"
    )
    totals_r1 = [p.total_score for p in game._players]

    # Drain the intermediate round-boundary notification, then play round 2.
    result = game.step(Action.STOP)
    while not result.round_over and not result.game_over:
        pid = result.acting_player_id
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        result = game.step(action)
    totals_r2 = [p.total_score for p in game._players]

    for t1, t2 in zip(totals_r1, totals_r2):
        assert t2 >= t1
    assert any(t2 > t1 for t1, t2 in zip(totals_r1, totals_r2)), (
        "round 2 must actually add points for at least one player"
    )


def test_flip7_ends_round_and_gives_bonus():
    """Force Flip 7 by drawing 7 times with a seeded deck that gives distinct numbers."""
    agents = [
        DrawNTimesAgent(7),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
    ]
    game = Flip7Game(n_players=4, seed=0)
    game.run_game(agents)
    flip7_player = next((p for p in game._players if p.flip7_bonus), None)
    if flip7_player is not None:
        assert game._players[flip7_player.id].total_score >= 15
        for p in game._players:
            if p.id != flip7_player.id:
                assert not p.is_in_round()


def test_run_game_terminates():
    """run_game must terminate — use agents that stop early so rounds are fast."""
    agents = [AlwaysStopAgent()] * 4
    game = Flip7Game(n_players=4, seed=99)
    result = game.run_game(agents)
    assert len(result) == 4
    assert any(score >= 200 for score in result.values())


def test_second_chance_prevents_bust():
    """Player with Second Chance should survive one duplicate draw."""
    from flip7.card import NumberCard

    game = Flip7Game(n_players=3, seed=5)
    game.reset()
    p = game._players[0]
    p.has_second_chance = True
    p.number_cards = [NumberCard(3)]
    game._apply_number_card(p, NumberCard(3))
    assert p.status == PlayerStatus.ACTIVE
    assert not p.has_second_chance
    assert len(p.number_cards) == 1


def test_legal_actions_in_game_state():
    """GameState must carry legal_actions for action masking."""
    game = Flip7Game(n_players=3, seed=0)
    result = game.reset()
    # First decision point must offer STOP and DRAW
    assert set(result.game_state.legal_actions) == {Action.STOP, Action.DRAW}


def test_round_scores_in_final_state():
    """round_scores must be populated in the round-end StepResult."""
    agents = [AlwaysStopAgent()] * 3
    game = Flip7Game(n_players=3, seed=0)
    result = game.reset()
    while not result.round_over and not result.game_over:
        pid = result.acting_player_id
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        result = game.step(action)
    assert result.round_over or result.game_over
    assert result.game_state.round_scores is not None


def test_step_raises_on_illegal_action():
    """An action outside the current legal_actions must raise IllegalActionError."""
    game = Flip7Game(n_players=3, seed=0)
    result = game.reset()
    assert Action.TARGET_0 not in result.game_state.legal_actions
    with pytest.raises(IllegalActionError) as exc_info:
        game.step(Action.TARGET_0)
    assert exc_info.value.player_id == result.acting_player_id
    assert exc_info.value.action == Action.TARGET_0
    assert exc_info.value.legal_actions == result.game_state.legal_actions


def test_step_raises_on_illegal_action_during_target_selection():
    """STOP/DRAW must be rejected while a TARGET_* decision is pending.

    seed=0 with n_players=3 is known to reach a multi-candidate target
    selection (Freeze/Second Chance/Flip Three) within 200 steps, across
    however many round boundaries need to be drained along the way.
    """
    game = Flip7Game(n_players=3, seed=0)
    result = game.reset()
    for _ in range(200):
        if result.game_over:
            break
        gs = result.game_state
        if gs.pending_action_card is not None:
            break
        if result.round_over:
            result = game.step(Action.STOP)  # drain intermediate round boundary
            continue
        action = Action.DRAW if Action.DRAW in gs.legal_actions else gs.legal_actions[0]
        result = game.step(action)

    assert result.game_state.pending_action_card is not None, (
        "test fixture assumption broken: seed=0 no longer reaches a pending "
        "target selection within 200 steps"
    )
    assert Action.STOP not in result.game_state.legal_actions
    with pytest.raises(IllegalActionError):
        game.step(Action.STOP)


# ------------------------------------------------------------------
# Round-boundary signaling (round_over must fire at every round, not just
# the game's final round — see IllegalActionError-adjacent fix history)
# ------------------------------------------------------------------


class CountingAgent(BaseAgent):
    """AlwaysStopAgent variant that counts round_over notifications it observes.

    Only the player whose action ends a round receives that round's
    observe(..., round_over=True) call (run_game() calls observe() on the
    acting player only) — so counts are checked in aggregate across all
    agents, not per agent.
    """

    def __init__(self) -> None:
        self.round_over_count = 0

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions:
            return legal_actions[0]
        return Action.STOP

    def observe(self, gs, action, next_gs, reward, round_over):
        if round_over:
            self.round_over_count += 1


def test_round_over_fires_every_round():
    """round_over must fire once per round boundary, not only at game end."""
    agents = [CountingAgent() for _ in range(4)]
    game = Flip7Game(n_players=4, seed=1)
    game.run_game(agents)

    total_round_over_notifications = sum(a.round_over_count for a in agents)
    assert total_round_over_notifications >= 2, (
        "round_over must fire for intermediate rounds, not only the final one"
    )
    assert total_round_over_notifications == game._current_round


def test_step_returns_distinct_result_at_intermediate_round_boundary():
    """An intermediate round boundary is its own StepResult, separate from
    the next round's first decision."""
    agents = [AlwaysStopAgent()] * 4
    game = Flip7Game(n_players=4, seed=1)
    result = game.reset()
    while not result.round_over:
        pid = result.acting_player_id
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        result = game.step(action)

    assert result.game_over is False
    assert result.game_state.legal_actions == ()
    assert result.game_state.round_scores is not None
    assert set(result.game_state.round_scores.keys()) == {0, 1, 2, 3}

    next_result = game.step(Action.STOP)
    assert next_result.round_over is False
    assert next_result.game_state.legal_actions != ()


def test_illegal_action_error_not_raised_on_round_boundary_drain():
    """The placeholder action used to drain a round boundary is never
    validated against legal_actions — any Action value must be accepted."""
    agents = [AlwaysStopAgent()] * 4
    game = Flip7Game(n_players=4, seed=1)
    result = game.reset()
    while not result.round_over:
        pid = result.acting_player_id
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        result = game.step(action)
    assert result.game_over is False

    # Action.DRAW is not in the (empty) legal_actions of the boundary
    # notification, yet draining with it must not raise IllegalActionError.
    next_result = game.step(Action.DRAW)
    assert next_result.round_over is False
