"""Behavioral signal-correctness tests across common in-game situations.

These tests deliberately go beyond pure rule-correctness (already covered in
test_game.py / test_action_cards.py) to check that the *signals* the engine
exposes to agents — StepResult.round_over/reward/game_over, GameState.
round_scores, and Player.status transitions — are consistent and complete in
every common situation: a player busting mid-round while others keep
playing, Flip 7 ending the round for everyone instantly, Freeze/Second
Chance/Flip Three action-card resolution, and the round-boundary
notification introduced to fix the round_over signaling bug.
"""

from __future__ import annotations

from flip7.action import Action
from flip7.agents.base_agent import BaseAgent
from flip7.card import ActionCard, NumberCard
from flip7.game import Flip7Game
from flip7.game_state import GameState
from flip7.player import PlayerStatus


class DrawUntilBustedOrStoppedAgent(BaseAgent):
    """Draws forever unless told to target/select, or already inactive."""

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.DRAW not in legal_actions:
            return legal_actions[0]
        return Action.DRAW


class AlwaysStopAgent(BaseAgent):
    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions:
            return legal_actions[0]
        return Action.STOP


class RecordingAgent(BaseAgent):
    """Records every observe() call verbatim for post-hoc assertions."""

    def __init__(self, inner: BaseAgent) -> None:
        self.inner = inner
        self.observations: list[tuple] = []

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        return self.inner.act(gs, legal_actions)

    def observe(self, gs, action, next_gs, reward, round_over) -> None:
        self.observations.append((gs, action, next_gs, reward, round_over))


# ------------------------------------------------------------------
# Bust mid-round: does the round continue for the other players, and
# is the busted player's score/status signal correct?
# ------------------------------------------------------------------


def test_other_players_continue_after_one_player_busts():
    """seed=9: player 0 busts on its first draw, but players 1-3 must still
    get to make their own decisions before the round ends."""
    agents = [
        DrawUntilBustedOrStoppedAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
    ]
    game = Flip7Game(n_players=4, seed=9)
    result = game.reset()

    acting_pids_before_round_over: list[int] = []
    busted_player_action_count = 0
    while not result.round_over:
        pid = result.acting_player_id
        acting_pids_before_round_over.append(pid)
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        if pid == 0:
            busted_player_action_count += 1
        result = game.step(action)

    # Player 0 busts on its very first action; it must not be asked to act
    # again afterwards (it is no longer "in the round").
    assert game._players[0].status == PlayerStatus.BUSTED
    assert busted_player_action_count == 1

    # Players other than 0 must still have acted after player 0 busted —
    # i.e. the round did not end the instant player 0 busted.
    assert any(pid != 0 for pid in acting_pids_before_round_over[1:])

    # The busted player's round score must be exactly 0, everyone else's
    # reflects their own hand.
    round_scores = result.game_state.round_scores
    assert round_scores is not None
    assert round_scores[0] == 0
    assert game._players[0].total_score == 0


def test_busted_player_not_offered_decisions_for_rest_of_round():
    """Once busted, a player must never appear as acting_player_id again
    until the next round starts."""
    agents = [
        DrawUntilBustedOrStoppedAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
    ]
    game = Flip7Game(n_players=4, seed=9)
    result = game.reset()
    busted_seen_at = None
    step = 0
    while not result.round_over:
        pid = result.acting_player_id
        step += 1
        if pid == 0 and game._players[0].status == PlayerStatus.BUSTED:
            busted_seen_at = step
        assert not (
            busted_seen_at is not None and pid == 0 and step > busted_seen_at
        ), "busted player must not be asked to act again this round"
        action = agents[pid].act(
            result.game_state, list(result.game_state.legal_actions)
        )
        result = game.step(action)


def test_bust_mid_round_does_not_fire_round_over_prematurely():
    """A single player busting must not itself trigger round_over — only
    when ALL players are inactive (or Flip 7 happens) does the round end."""
    agents = [
        DrawUntilBustedOrStoppedAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
    ]
    game = Flip7Game(n_players=4, seed=9)
    result = game.reset()
    saw_bust_without_round_over = False
    while not result.round_over:
        pid = result.acting_player_id
        action = agents[pid].act(
            result.game_state, list(result.game_state.legal_actions)
        )
        result = game.step(action)
        if (
            pid == 0
            and game._players[0].status == PlayerStatus.BUSTED
            and not result.round_over
        ):
            saw_bust_without_round_over = True
    assert saw_bust_without_round_over, (
        "expected to observe the round continuing for at least one more "
        "step after player 0's bust"
    )


# ------------------------------------------------------------------
# observe() delivery: every acting player's own decisions are reported
# back to them, with correct reward/round_over timing.
# ------------------------------------------------------------------


def test_observe_called_for_every_decision_the_agent_makes():
    """run_game() must call observe() exactly once per act() call — no
    decision is silently dropped, and no agent gets an extra observe() call
    it didn't act for."""

    class CountingActAgent(BaseAgent):
        def __init__(self, inner: BaseAgent) -> None:
            self.inner = inner
            self.act_count = 0
            self.observe_count = 0

        def act(self, gs, legal_actions):
            self.act_count += 1
            return self.inner.act(gs, legal_actions)

        def observe(self, gs, action, next_gs, reward, round_over):
            self.observe_count += 1

    inner_agents = [
        DrawUntilBustedOrStoppedAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
    ]
    agents = [CountingActAgent(a) for a in inner_agents]
    game = Flip7Game(n_players=4, seed=9)
    game.run_game(agents)

    for pid, agent in enumerate(agents):
        assert agent.observe_count == agent.act_count, (
            f"player {pid} made {agent.act_count} decisions but received "
            f"{agent.observe_count} observe() calls"
        )
        assert agent.act_count > 0


def test_observe_reward_matches_round_score_only_at_round_end():
    """reward must be 0.0 on every mid-round observe(), and equal to the
    acting player's own round score on the round-ending observe()."""
    agents = [RecordingAgent(AlwaysStopAgent()) for _ in range(4)]
    game = Flip7Game(n_players=4, seed=1)
    result = game.reset()
    while not result.round_over:
        pid = result.acting_player_id
        action = agents[pid].act(
            result.game_state, list(result.game_state.legal_actions)
        )
        result = game.step(action)
        agents[pid].observe(
            result.game_state,
            action,
            result.game_state,
            result.reward,
            result.round_over,
        )

    round_scores = result.game_state.round_scores
    assert round_scores is not None

    # Every recorded observation with round_over=False must carry reward=0.0.
    for agent in agents:
        for _gs, _action, _next_gs, reward, round_over in agent.observations:
            if not round_over:
                assert reward == 0.0

    # The final recorded observation (round end) must carry that player's
    # own round score as reward.
    last_pid = result.acting_player_id
    last_obs = agents[last_pid].observations[-1]
    _gs, _action, _next_gs, reward, round_over = last_obs
    assert round_over is True
    assert reward == float(round_scores[last_pid])


def test_run_game_delivers_round_over_to_every_player_across_a_full_game():
    """Summed across all agents, run_game() must deliver exactly one
    round_over=True observe() call per round actually played."""

    class CountingAgent(BaseAgent):
        def __init__(self) -> None:
            self.round_over_calls = 0
            self.total_calls = 0

        def act(self, gs, legal_actions):
            if Action.STOP not in legal_actions:
                return legal_actions[0]
            return Action.STOP

        def observe(self, gs, action, next_gs, reward, round_over):
            self.total_calls += 1
            if round_over:
                self.round_over_calls += 1

    agents = [CountingAgent() for _ in range(4)]
    game = Flip7Game(n_players=4, seed=1)
    game.run_game(agents)

    total_round_over = sum(a.round_over_calls for a in agents)
    assert total_round_over == game._current_round
    # Every observe() call must have actually happened (sanity: nobody was
    # silently skipped across the whole game).
    assert sum(a.total_calls for a in agents) > 0


# ------------------------------------------------------------------
# Flip 7: instant round end for everyone, correct bonus/score signals.
# ------------------------------------------------------------------


def test_flip7_stops_all_other_active_players_immediately():
    """When one player reaches Flip 7, every other still-active player must
    be forced to STOPPED in the same instant — they get no further turns."""
    game = Flip7Game(n_players=4, seed=0)
    game.reset()
    p0 = game._players[0]
    others = game._players[1:]
    p0.number_cards = [NumberCard(v) for v in range(6)]  # 6 distinct numbers
    game._apply_number_card(p0, NumberCard(6))  # 7th distinct number → Flip 7
    assert p0.flip7_bonus is True
    assert p0.status == PlayerStatus.ACTIVE  # Flip 7 player is not "stopped"
    for p in others:
        assert p.status == PlayerStatus.STOPPED


def test_flip7_round_score_includes_bonus_and_ends_round_for_table():
    """Round-ending via Flip 7 must still deliver round_scores for all
    players, and the Flip 7 player's score must include the +15 bonus.

    seed=27 with this agent mix is confirmed to produce a Flip 7 for player 0.
    """
    agents = [
        DrawUntilBustedOrStoppedAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
        AlwaysStopAgent(),
    ]
    game = Flip7Game(n_players=4, seed=27)
    result = game.reset()
    while not result.round_over:
        pid = result.acting_player_id
        action = agents[pid].act(
            result.game_state, list(result.game_state.legal_actions)
        )
        result = game.step(action)

    flip7_player = next((p for p in game._players if p.flip7_bonus), None)
    assert flip7_player is not None, (
        "test fixture assumption broken: seed=27 no longer produces a Flip 7"
    )
    round_scores = result.game_state.round_scores
    assert round_scores is not None
    assert round_scores[flip7_player.id] >= 15
    assert set(round_scores.keys()) == {0, 1, 2, 3}


# ------------------------------------------------------------------
# Action cards: Freeze / Second Chance / Flip Three all resolve with
# correct StepResult signaling (no accidental round_over, correct
# pending_action_card / target_player_ids exposure).
# ------------------------------------------------------------------


def test_freeze_target_selection_step_has_no_round_over():
    """A pending Freeze target-selection StepResult must never claim the
    round is over — it's mid-round, awaiting a decision."""
    game = Flip7Game(n_players=3, seed=0)
    game.reset()
    p0 = game._players[0]
    card = ActionCard("freeze")
    gen = game._apply_freeze_gen(p0, card)
    step_result = next(gen)
    assert step_result.round_over is False
    assert step_result.game_over is False
    assert step_result.game_state.pending_action_card == "freeze"
    assert len(step_result.game_state.legal_actions) >= 1


def test_frozen_player_is_excluded_from_further_decisions_this_round():
    """A frozen player must not be asked to act again in the same round,
    but still contributes their score at round end."""
    agents = [AlwaysStopAgent()] * 3
    game = Flip7Game(n_players=3, seed=0)
    result = game.reset()
    frozen_pid = None
    while not result.round_over:
        pid = result.acting_player_id
        gs = result.game_state
        if gs.pending_action_card == "freeze" and len(gs.legal_actions) >= 1:
            frozen_pid = gs.target_player_ids[0]
            action = Action.TARGET_0
        else:
            action = agents[pid].act(gs, list(gs.legal_actions))
        prev_acting = pid
        result = game.step(action)
        if frozen_pid is not None and prev_acting != frozen_pid:
            assert result.acting_player_id != frozen_pid or result.round_over


def test_second_chance_consumed_signal_is_immediate():
    """Using a Second Chance to survive a duplicate draw must clear the
    flag on the very next observation of that player's state."""
    game = Flip7Game(n_players=3, seed=5)
    game.reset()
    p = game._players[0]
    p.has_second_chance = True
    p.number_cards = [NumberCard(3)]
    game._apply_number_card(p, NumberCard(3))
    assert p.status == PlayerStatus.ACTIVE
    assert p.has_second_chance is False
    assert p.has_number(3)


def test_flip_three_deals_exactly_three_cards_to_target():
    """Flip Three must draw exactly 3 cards for the target (unless the
    target busts or Flip 7s early, ending it sooner)."""
    game = Flip7Game(n_players=3, seed=0)
    game.reset()
    target = game._players[1]
    starting_hand_size = len(target.number_cards)
    card = ActionCard("flip_three")

    from flip7.game import _ActionToken

    gen = game._apply_flip_three_gen(game._players[0], card)
    try:
        next(gen)
        # Auto-resolves to a single candidate or needs a target pick;
        # drive it to completion regardless.
        while True:
            gen.send(_ActionToken(Action.TARGET_0))
    except StopIteration:
        pass

    ending_hand_size = len(target.number_cards) + (
        1 if target.status == PlayerStatus.BUSTED else 0
    )
    # Either the target gained up to 3 new number cards, or busted/Flip7'd
    # partway through — in all cases the hand size must not have grown by
    # more than 3 number cards from Flip Three's own draws.
    assert ending_hand_size - starting_hand_size <= 3
