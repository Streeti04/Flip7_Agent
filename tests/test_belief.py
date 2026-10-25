"""Tests for the belief-state model over public information."""

from __future__ import annotations

import numpy as np
import pytest

from flip7.belief import (
    ALL_BONUS_CARDS,
    NUMBER_COUNTS,
    PileBelief,
    TOTAL_CARDS,
    build_belief,
)
from flip7.card import BonusCard
from flip7.game_state import GameState
from flip7.player import PlayerStatus
from flip7.action import Action


def make_state(
    *,
    hand=frozenset(),
    seen=frozenset(),
    deck_remaining: int = 94,
    bonus_cards=(),
    has_second_chance: bool = False,
    round_score: int = 0,
    total_score: int = 0,
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
        legal_actions=(Action.STOP, Action.DRAW),
        pending_action_card=None,
        target_player_ids=(),
        round_scores=None,
        opponents=opponents,
    )


def test_fresh_deck_belief_is_exact() -> None:
    belief = build_belief(make_state())
    assert belief.pile_size == 94
    for value, count in NUMBER_COUNTS.items():
        assert belief.expected_copies[value] == float(count)
    assert sum(belief.bonus_counts.values()) == float(len(ALL_BONUS_CARDS))
    assert sum(belief.action_counts.values()) == 9.0


def test_unseen_values_are_exact_even_when_seen_values_are_blurred() -> None:
    # Values 3 and 5 have been drawn once each; the pile is 80 cards.
    state = make_state(hand=frozenset({3}), seen=frozenset({3, 5}), deck_remaining=80)
    belief = build_belief(state)

    unseen_sum = sum(count for v, count in NUMBER_COUNTS.items() if v not in {3, 5})
    assert unseen_sum <= 80  # consistency of the scenario
    # Never-seen values keep their exact multiplicity regardless of the share.
    for value in range(13):
        if value in {3, 5}:
            continue
        assert belief.expected_copies[value] == float(NUMBER_COUNTS[value])


def test_only_unseen_cards_left_means_zero_bust_risk() -> None:
    unseen_sum = sum(
        count for v, count in NUMBER_COUNTS.items() if v not in {0, 1, 2}
    )
    state = make_state(
        hand=frozenset({1}), seen=frozenset({0, 1, 2}), deck_remaining=unseen_sum
    )
    belief = build_belief(state)
    assert belief.prob_duplicate(frozenset({1})) == 0.0
    assert belief.prob_duplicate(frozenset({0, 1, 2})) == 0.0


def test_probabilities_are_bounded_and_sum_to_at_most_one() -> None:
    state = make_state(
        hand=frozenset({2, 9}),
        seen=frozenset({0, 2, 5, 9, 12}),
        deck_remaining=60,
        bonus_cards=(BonusCard("+6", 6),),
        has_second_chance=True,
    )
    belief = build_belief(state)
    total = belief.total_mass()
    assert 0.0 < total <= 60.0 + 1e-9
    for value in range(13):
        assert 0.0 <= belief.prob_value(value) <= 1.0
    assert 0.0 <= belief.prob_bonus() <= 1.0
    assert 0.0 <= belief.prob_action() <= 1.0
    assert belief.prob_duplicate(frozenset({2, 9})) <= 1.0


def test_unseen_mass_stays_exact_under_rescaling() -> None:
    # Post-recycle-like state: many seen values, small pile. Seen/bonus/action
    # masses are rescaled, but unseen copies must remain exact.
    seen = frozenset(range(13))
    state = make_state(hand=frozenset({10, 11, 12}), seen=seen, deck_remaining=25)
    belief = build_belief(state)
    assert belief.total_mass() <= 25.0 + 1e-6
    # all values seen → every expectation is approximate, no exact assertion here
    state2 = make_state(hand=frozenset({0}), seen=frozenset({1, 2}), deck_remaining=79)
    belief2 = build_belief(state2)
    # unseen_sum = 78 - 3 = 75 <= 79 → exactness preserved
    for value in (0, 3, 4, 5):
        assert belief2.expected_copies[value] == float(NUMBER_COUNTS[value])


def test_withdraw_reduces_mass_monotonically() -> None:
    belief = build_belief(make_state(seen=frozenset({4}), deck_remaining=85))
    before = belief.prob_value(4)
    after = belief.withdraw(4).prob_value(4)
    assert after < before
    assert belief.withdraw(4).pile_size == belief.pile_size - 1


def test_belief_calibration_against_real_games() -> None:
    """Empirical duplicate frequency must track the belief estimate.

    Statistical test: bucket decisions by estimated duplicate probability and
    compare with the actual top-of-pile card (white-box peek). Loose tolerance
    because buckets are small.
    """
    from flip7.game import Flip7Game
    from flip7.agents.base_agent import Action
    from flip7.card import NumberCard

    rng = np.random.default_rng(1234)
    buckets: dict[int, list[int]] = {}

    for game_index in range(250):
        game = Flip7Game(n_players=4, seed=int(rng.integers(2**31)))
        result = game.reset()
        while not result.game_over:
            gs = result.game_state
            legal = list(gs.legal_actions)
            if gs.legal_actions == (Action.STOP, Action.DRAW):
                belief = build_belief(gs)
                estimated = belief.prob_duplicate(gs.my_number_cards)
                # White-box peek at the card the engine will pop next; the pile
                # can legitimately be empty at a decision point (recycle pending).
                if game._deck._draw_pile:
                    actual = game._deck._draw_pile[-1]
                    is_duplicate = (
                        isinstance(actual, NumberCard)
                        and actual.value in gs.my_number_cards
                    )
                    key = min(int(estimated * 5), 4)
                    buckets.setdefault(key, []).append(int(is_duplicate))
                action = (
                    Action.DRAW
                    if estimated > 0.28
                    else Action(rng.choice([Action.STOP.value, Action.DRAW.value]))
                )
            else:
                action = legal[0]
            result = game.step(action)
            while result.round_over and not result.game_over:
                result = game.step(Action.STOP)

    checked = 0
    for key, outcomes in buckets.items():
        if len(outcomes) < 100:
            continue
        empirical = float(np.mean(outcomes))
        estimated = (key + 0.5) / 5.0
        assert abs(empirical - estimated) < 0.15, (
            f"bucket {key}: estimated {estimated:.2f}, empirical {empirical:.2f} "
            f"({len(outcomes)} samples)"
        )
        checked += len(outcomes)
    assert checked >= 200, f"too few calibrated samples: {checked}"
