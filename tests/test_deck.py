import numpy as np
import pytest
from flip7.card import NumberCard, TOTAL_CARDS
from flip7.deck import Deck


def make_deck(seed: int = 0) -> Deck:
    return Deck(np.random.default_rng(seed))


def test_initial_size():
    deck = make_deck()
    assert deck.cards_remaining() == TOTAL_CARDS


def test_draw_reduces_count():
    deck = make_deck()
    deck.draw()
    assert deck.cards_remaining() == TOTAL_CARDS - 1


def test_draw_all_cards():
    deck = make_deck()
    drawn = [deck.draw() for _ in range(TOTAL_CARDS)]
    assert len(drawn) == TOTAL_CARDS
    assert deck.cards_remaining() == 0


def test_recycle_after_empty():
    deck = make_deck()
    drawn = [deck.draw() for _ in range(TOTAL_CARDS)]
    deck.discard(drawn)
    # Draw one more — should trigger recycle
    card = deck.draw()
    assert card is not None
    assert deck.cards_remaining() == TOTAL_CARDS - 1


def test_recycle_without_discard_raises():
    deck = make_deck()
    [deck.draw() for _ in range(TOTAL_CARDS)]
    with pytest.raises(RuntimeError):
        deck.draw()


def test_seen_numbers_tracks_number_cards():
    deck = make_deck(42)
    seen: set[int] = set()
    for _ in range(TOTAL_CARDS):
        card = deck.draw()
        if isinstance(card, NumberCard):
            seen.add(card.value)
    assert deck.seen_numbers() == frozenset(seen)


def test_seen_numbers_persists_across_recycle():
    deck = make_deck(1)
    drawn = [deck.draw() for _ in range(TOTAL_CARDS)]
    seen_before = deck.seen_numbers()
    deck.discard(drawn)
    deck.draw()
    # seen_numbers only grows, never shrinks
    assert seen_before <= deck.seen_numbers()


def test_discard_does_not_affect_draw_pile():
    deck = make_deck()
    before = deck.cards_remaining()
    deck.discard([NumberCard(5)])
    assert deck.cards_remaining() == before
