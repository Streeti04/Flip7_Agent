from flip7.card import (
    NumberCard,
    BonusCard,
    ActionCard,
    DECK_COMPOSITION,
    TOTAL_CARDS,
)


def test_total_card_count():
    assert TOTAL_CARDS == 94


def test_number_card_counts():
    number_total = sum(
        count
        for card, count in DECK_COMPOSITION.items()
        if isinstance(card, NumberCard)
    )
    assert number_total == 79


def test_bonus_card_counts():
    bonus_total = sum(
        count for card, count in DECK_COMPOSITION.items() if isinstance(card, BonusCard)
    )
    assert bonus_total == 6


def test_action_card_counts():
    action_total = sum(
        count
        for card, count in DECK_COMPOSITION.items()
        if isinstance(card, ActionCard)
    )
    assert action_total == 9


def test_number_card_0_appears_once():
    assert DECK_COMPOSITION[NumberCard(0)] == 1


def test_number_card_count_equals_value():
    for v in range(1, 13):
        assert DECK_COMPOSITION[NumberCard(v)] == v


def test_cards_are_hashable():
    # Needed for frozenset usage in GameState
    s = {NumberCard(1), BonusCard("+2", 2), ActionCard("freeze")}
    assert len(s) == 3
