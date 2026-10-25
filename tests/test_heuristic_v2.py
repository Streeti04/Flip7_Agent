"""Tests for the V2 expectimax heuristic agent."""

from __future__ import annotations

import numpy as np
import pytest

from flip7.action import Action
from flip7.agents.base_agent import BaseAgent
from flip7.agents.heuristic_agent import HeuristicAgent, HeuristicConfig
from flip7.card import BonusCard
from flip7.game import Flip7Game
from flip7.game_state import GameState, PlayerView
from flip7.player import PlayerStatus


def make_state(
    *,
    hand=frozenset(),
    seen=frozenset(),
    deck_remaining: int = 80,
    bonus_cards=(),
    has_second_chance: bool = False,
    round_score: int = 0,
    total_score: int = 0,
    legal=(Action.STOP, Action.DRAW),
    pending: str | None = None,
    targets=(),
    opponents=(),
) -> GameState:
    return GameState(
        my_player_id=0,
        my_number_cards=frozenset(hand),
        my_bonus_cards=tuple(bonus_cards),
        my_has_second_chance=has_second_chance,
        my_total_score=total_score,
        my_status=PlayerStatus.ACTIVE,
        my_round_score=round_score,
        deck_remaining=deck_remaining,
        numbers_seen_this_game=frozenset(seen),
        current_round=1,
        legal_actions=tuple(legal),
        pending_action_card=pending,
        target_player_ids=tuple(targets),
        round_scores=None,
        opponents=opponents,
    )


def opponent(player_id: int, cards: int, *, sc: bool = False, total: int = 0) -> PlayerView:
    return PlayerView(
        player_id=player_id,
        num_number_cards=cards,
        has_bonus_card=False,
        has_second_chance=sc,
        status=PlayerStatus.ACTIVE,
        total_score=total,
    )


@pytest.fixture()
def agent() -> HeuristicAgent:
    return HeuristicAgent()


def test_subclasses_base_agent(agent: HeuristicAgent) -> None:
    assert isinstance(agent, BaseAgent)


def test_draws_with_small_safe_hand(agent: HeuristicAgent) -> None:
    gs = make_state(hand=frozenset({0, 1}), seen=frozenset({0, 1}), round_score=1)
    assert agent.act(gs, [Action.STOP, Action.DRAW]) == Action.DRAW


def test_stops_with_big_risky_hand(agent: HeuristicAgent) -> None:
    # Bank 50 with a duplicate-heavy hand: drawing is clearly -EV.
    gs = make_state(
        hand=frozenset({8, 9, 10, 11, 12}),
        seen=frozenset({8, 9, 10, 11, 12}),
        round_score=50,
    )
    assert agent.act(gs, [Action.STOP, Action.DRAW]) == Action.STOP


def opponent_with_status(
    player_id: int, cards: int, status: PlayerStatus, *, total: int = 0
) -> PlayerView:
    return PlayerView(
        player_id=player_id,
        num_number_cards=cards,
        has_bonus_card=False,
        has_second_chance=False,
        status=status,
        total_score=total,
    )


# Rivals far from 200: my bust cannot hand them the game this round.
CALM_RIVALS = tuple(opponent(i, 1, total=100) for i in (1, 2, 3))
# Active rival at 190 with 3 cards: 190 + 8 * (3 + 1) >= 200.
THREATENING_RIVALS = (opponent(1, 3, total=190),) + CALM_RIVALS[1:]


def test_endgame_draws_with_one_card_when_no_rival_threatens(agent: HeuristicAgent) -> None:
    # Banking 5 would end the game at 200; a bust only costs this round.
    gs = make_state(
        hand=frozenset({5}),
        seen=frozenset({5}),
        round_score=5,
        total_score=195,
        opponents=CALM_RIVALS,
    )
    assert agent.act(gs, [Action.STOP, Action.DRAW]) == Action.DRAW


def test_endgame_draws_in_tie_break_round_with_empty_hand(agent: HeuristicAgent) -> None:
    # Tie-break extra round: total already >= 200 before any card is drawn.
    gs = make_state(total_score=205, opponents=CALM_RIVALS)
    assert agent.act(gs, [Action.STOP, Action.DRAW]) == Action.DRAW


def test_endgame_bust_value_tips_marginal_decision(agent: HeuristicAgent) -> None:
    # Same hand, three contexts: mid-game pure EV stops; in the endgame the
    # bust is not worth 0 and tips the decision to Draw — unless a rival could
    # end the game this round, then the pure-EV decision (Stop) applies.
    hand = frozenset({0, 6, 10, 11})
    midgame = make_state(
        hand=hand, seen=hand, round_score=27, total_score=50, opponents=CALM_RIVALS
    )
    endgame = make_state(
        hand=hand, seen=hand, round_score=27, total_score=178, opponents=CALM_RIVALS
    )
    threatened = make_state(
        hand=hand, seen=hand, round_score=27, total_score=178, opponents=THREATENING_RIVALS
    )
    legal = [Action.STOP, Action.DRAW]
    assert agent.act(midgame, legal) == Action.STOP
    assert agent.act(endgame, legal) == Action.DRAW
    assert agent.act(threatened, legal) == Action.STOP


def test_endgame_bust_value_zero_restores_pure_ev() -> None:
    hand = frozenset({0, 6, 10, 11})
    endgame = make_state(
        hand=hand, seen=hand, round_score=27, total_score=178, opponents=CALM_RIVALS
    )
    pure_ev = HeuristicAgent(HeuristicConfig(endgame_bust_value=0.0))
    assert pure_ev.act(endgame, [Action.STOP, Action.DRAW]) == Action.STOP


@pytest.mark.parametrize(
    ("rival", "threat"),
    [
        (opponent_with_status(1, 6, PlayerStatus.BUSTED, total=199), False),
        (opponent_with_status(1, 1, PlayerStatus.STOPPED, total=191), False),
        (opponent_with_status(1, 2, PlayerStatus.STOPPED, total=184), True),
        (opponent_with_status(1, 1, PlayerStatus.FROZEN, total=184), False),
        (opponent_with_status(1, 1, PlayerStatus.ACTIVE, total=183), False),
        (opponent_with_status(1, 1, PlayerStatus.ACTIVE, total=184), True),
        (opponent_with_status(1, 0, PlayerStatus.ACTIVE, total=200), True),
    ],
)
def test_rival_may_end_game_estimate(
    agent: HeuristicAgent, rival: PlayerView, threat: bool
) -> None:
    gs = make_state(total_score=200, opponents=(rival,) + CALM_RIVALS[1:])
    assert agent._rival_may_end_game(gs) is threat


def test_never_raises_and_always_legal_in_fuzzed_games(agent: HeuristicAgent) -> None:
    rng = np.random.default_rng(7)
    for game_index in range(60):
        game = Flip7Game(n_players=4, seed=int(rng.integers(2**31)))
        result = game.reset()
        while not result.game_over:
            gs = result.game_state
            legal = list(gs.legal_actions)
            action = agent.act(gs, legal)
            assert action in legal
            result = game.step(action)
            while result.round_over and not result.game_over:
                result = game.step(Action.STOP)


def test_empty_legal_actions_returns_stop_without_raising(agent: HeuristicAgent) -> None:
    gs = make_state()
    assert agent.act(gs, []) == Action.STOP


def test_freeze_targets_flip7_threat_over_high_scorer(agent: HeuristicAgent) -> None:
    # V1 targeted the biggest score; V2 denies the near-Flip-7 rival.
    gs = make_state(
        seen=frozenset({5}),
        legal=[Action.TARGET_0, Action.TARGET_1],
        pending="freeze",
        targets=[1, 2],
        opponents=(opponent(1, 2, total=150), opponent(2, 6, total=60)),
    )
    assert agent.act(gs, [Action.TARGET_0, Action.TARGET_1]) == Action.TARGET_1


def test_second_chance_goes_to_weakest(agent: HeuristicAgent) -> None:
    gs = make_state(
        seen=frozenset({4}),
        legal=[Action.TARGET_0, Action.TARGET_1],
        pending="second_chance",
        targets=[1, 2],
        opponents=(opponent(1, 6, total=180), opponent(2, 1, total=10)),
    )
    assert agent.act(gs, [Action.TARGET_0, Action.TARGET_1]) == Action.TARGET_1


def test_flip_three_denies_near_flip7_opponent(agent: HeuristicAgent) -> None:
    gs = make_state(
        seen=frozenset({4}),
        legal=[Action.TARGET_0, Action.TARGET_1],
        pending="flip_three",
        targets=[1, 2],
        opponents=(opponent(1, 3, total=40), opponent(2, 6, total=90)),
    )
    assert agent.act(gs, [Action.TARGET_0, Action.TARGET_1]) == Action.TARGET_1


def test_flip_three_self_target_when_own_hand_is_safe(agent: HeuristicAgent) -> None:
    # Own hand is tiny; three controlled cards beat harassing a mid-size rival.
    gs = make_state(
        hand=frozenset({0, 1}),
        seen=frozenset({0, 1}),
        round_score=1,
        legal=[Action.TARGET_0, Action.TARGET_1],
        pending="flip_three",
        targets=[0, 2],
        opponents=(opponent(2, 3, total=30),),
    )
    assert agent.act(gs, [Action.TARGET_0, Action.TARGET_1]) == Action.TARGET_0


def test_second_chance_enables_riskier_draws(agent: HeuristicAgent) -> None:
    # With SC, a duplicate only costs the SC — draw when it would stop without.
    # Consistent late-game state: all values seen, dup mass is significant.
    base = dict(
        hand=frozenset({10, 11, 12}),
        seen=frozenset(range(13)),
        round_score=33,
        deck_remaining=25,
    )
    without_sc = make_state(**base)
    with_sc = make_state(**base, has_second_chance=True)
    assert agent.act(without_sc, [Action.STOP, Action.DRAW]) == Action.STOP
    assert agent.act(with_sc, [Action.STOP, Action.DRAW]) == Action.DRAW


def test_flip7_chance_outweighs_small_bank(agent: HeuristicAgent) -> None:
    # Six cards, the 6 is still unseen: drawing wins +15 and ends the round.
    gs = make_state(
        hand=frozenset({0, 1, 2, 3, 4, 5}),
        seen=frozenset({0, 1, 2, 3, 4, 5}),
        round_score=15,
    )
    assert agent.act(gs, [Action.STOP, Action.DRAW]) == Action.DRAW
