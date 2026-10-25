from __future__ import annotations
import numpy as np
from .card import Card, NumberCard, DECK_COMPOSITION


class Deck:
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._draw_pile: list[Card] = []
        self._discard_pile: list[Card] = []
        self._seen_numbers: set[int] = set()
        self._build_fresh()

    def _build_fresh(self) -> None:
        cards: list[Card] = []
        for card, count in DECK_COMPOSITION.items():
            cards.extend([card] * count)
        self._rng.shuffle(cards)  # type: ignore[arg-type]
        self._draw_pile = cards

    def draw(self) -> Card:
        if not self._draw_pile:
            self._recycle()
        card = self._draw_pile.pop()
        if isinstance(card, NumberCard):
            self._seen_numbers.add(card.value)
        return card

    def discard(self, cards: list[Card]) -> None:
        self._discard_pile.extend(cards)

    def _recycle(self) -> None:
        """Shuffle the discard pile into a new draw pile.

        Cards currently in play (not yet discarded) remain untouched.
        Only call when the draw pile is empty.
        """
        if not self._discard_pile:
            raise RuntimeError("Both draw and discard piles are empty.")
        cards = self._discard_pile[:]
        self._discard_pile.clear()
        self._rng.shuffle(cards)  # type: ignore[arg-type]
        self._draw_pile = cards

    def cards_remaining(self) -> int:
        return len(self._draw_pile)

    def seen_numbers(self) -> frozenset[int]:
        return frozenset(self._seen_numbers)
