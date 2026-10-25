"""Belief state over the hidden draw pile, derived from public information.

The engine exposes no card counts, but two facts enable principled inference:

1. Cards can only enter the discard pile by being drawn, so a number value that
   has never been seen (``numbers_seen_this_game``) has all of its physical
   copies in the current draw pile *exactly* (recycling only shuffles the
   discard back into the pile and never touches unseen copies).
2. Before the first recycle every drawn copy is provably outside the pile
   (hands or discard). After a recycle, discarded copies can return. We model
   the expected in-pile copies of a seen value as ``(count(v) - 1) * share``
   where ``share`` is the pile's share of the "undrawn or recycled" card pool.
   This is exact before the first recycle when no extra duplicates have been
   drawn and degrades gracefully afterwards.

All queries are derived solely from :class:`~flip7.game_state.GameState` — no
private information is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .card import ActionCard, BonusCard, DECK_COMPOSITION, NumberCard
from .game_state import GameState

NUMBER_COUNTS: dict[int, int] = {
    card.value: count
    for card, count in DECK_COMPOSITION.items()
    if isinstance(card, NumberCard)
}
ALL_BONUS_CARDS: tuple[BonusCard, ...] = tuple(
    card for card in DECK_COMPOSITION if isinstance(card, BonusCard)
)
ACTION_CARD_COUNTS: dict[str, int] = {
    card.action_type: count
    for card, count in DECK_COMPOSITION.items()
    if isinstance(card, ActionCard)
}
TOTAL_CARDS = sum(DECK_COMPOSITION.values())


@dataclass(frozen=True)
class PileBelief:
    """Expected composition of the current draw pile."""

    expected_copies: dict[int, float]  # value 0-12 → expected copies in pile
    pile_size: int  # deck_remaining
    bonus_counts: dict[BonusCard, float]  # bonus card → expected copies in pile
    action_counts: dict[str, float]  # action type → expected copies in pile

    def total_mass(self) -> float:
        """Expected number of cards in the pile (all categories combined)."""
        return (
            sum(self.expected_copies.values())
            + sum(self.bonus_counts.values())
            + sum(self.action_counts.values())
        )

    # -- basic probabilities -------------------------------------------------

    def prob_value(self, value: int) -> float:
        """P(next card is number ``value``)."""
        if self.pile_size <= 0:
            return 0.0
        return min(max(self.expected_copies.get(value, 0.0), 0.0) / self.pile_size, 1.0)

    def prob_bonus(self, bonus: BonusCard | None = None) -> float:
        """P(next card is a bonus card, or a specific one)."""
        if self.pile_size <= 0:
            return 0.0
        mass = (
            float(self.bonus_counts.get(bonus, 0.0))
            if bonus is not None
            else sum(self.bonus_counts.values())
        )
        return min(mass / self.pile_size, 1.0)

    def prob_action(self, action_type: str | None = None) -> float:
        """P(next card is an action card, or a specific type)."""
        if self.pile_size <= 0:
            return 0.0
        mass = (
            float(self.action_counts.get(action_type, 0.0))
            if action_type is not None
            else sum(self.action_counts.values())
        )
        return min(mass / self.pile_size, 1.0)

    def prob_duplicate(self, hand_values: frozenset[int]) -> float:
        """P(next number card duplicates a value already in the given hand)."""
        return sum(self.prob_value(v) for v in hand_values)

    def safe_new_values(self, hand_values: frozenset[int]) -> list[int]:
        """Values that would not bust the hand and still have positive mass."""
        return [
            v for v in range(13) if v not in hand_values and self.prob_value(v) > 0.0
        ]

    # -- belief update -------------------------------------------------------

    def withdraw(self, value: int) -> "PileBelief":
        """Belief after one copy of ``value`` is removed from the pile (drawn)."""
        copies = dict(self.expected_copies)
        copies[value] = max(copies.get(value, 0.0) - 1.0, 0.0)
        return replace(self, expected_copies=copies, pile_size=max(self.pile_size - 1, 0))

    def withdraw_bonus(self, bonus: BonusCard) -> "PileBelief":
        counts = dict(self.bonus_counts)
        counts[bonus] = max(counts.get(bonus, 0.0) - 1.0, 0.0)
        return replace(self, bonus_counts=counts, pile_size=max(self.pile_size - 1, 0))

    def withdraw_action(self, action_type: str) -> "PileBelief":
        counts = dict(self.action_counts)
        counts[action_type] = max(counts.get(action_type, 0.0) - 1.0, 0.0)
        return replace(self, action_counts=counts, pile_size=max(self.pile_size - 1, 0))


def build_belief(game_state: GameState) -> PileBelief:
    """Infer the pile belief from a public :class:`GameState` snapshot."""
    seen = game_state.numbers_seen_this_game
    hand = game_state.my_number_cards
    pile = max(int(game_state.deck_remaining), 0)

    unseen_sum = sum(
        count for value, count in NUMBER_COUNTS.items() if value not in seen
    )
    # Pile slots left for seen-value copies, bonus cards, and action cards.
    pool_slots = max(pile - unseen_sum, 0)

    # Bonus/action cards visibly held by any player are not in the pile.
    visible_bonuses = len(game_state.my_bonus_cards) + sum(
        1 for op in game_state.opponents if op.has_bonus_card
    )
    visible_second_chances = int(game_state.my_has_second_chance) + sum(
        1 for op in game_state.opponents if op.has_second_chance
    )
    bonus_pool = max(len(ALL_BONUS_CARDS) - visible_bonuses, 0)
    action_pool = max(
        sum(ACTION_CARD_COUNTS.values()) - visible_second_chances, 0
    )

    # Undrawn copies of seen values plus the bonus/action pool. One copy of each
    # seen value is provably outside the pile (it was drawn at least once).
    pool_mass = sum(
        NUMBER_COUNTS[value] - 1 for value in seen
    ) + bonus_pool + action_pool
    share = pool_slots / pool_mass if pool_mass > 0 else 0.0
    share = min(max(share, 0.0), 1.0)

    expected: dict[int, float] = {}
    for value in range(13):
        if value not in seen:
            expected[value] = float(NUMBER_COUNTS[value])  # exact
        else:
            expected[value] = (NUMBER_COUNTS[value] - 1) * share

    # Which specific bonus types remain is unknown; assume the highest-value
    # ones (deterministic and keeps the EV model from underestimating draws).
    bonus_candidates = sorted(ALL_BONUS_CARDS, key=lambda b: b.value, reverse=True)
    bonus_counts = {bonus: share for bonus in bonus_candidates[:bonus_pool]}
    action_counts = {
        action_type: share * float(count)
        for action_type, count in ACTION_CARD_COUNTS.items()
    }

    belief = PileBelief(
        expected_copies=expected,
        pile_size=pile,
        bonus_counts=bonus_counts,
        action_counts=action_counts,
    )

    # Approximations can push the pool mass (seen values + bonus + action) above
    # the pile slots actually available to it. Rescale only the pool categories:
    # unseen masses stay exact because those cards are provably in the pile.
    raw_pool = sum(bonus_counts.values()) + sum(action_counts.values()) + sum(
        mass for value, mass in expected.items() if value in seen
    )
    available_pool = max(pile - unseen_sum, 0)
    if raw_pool > available_pool > 0 or (available_pool == 0 and raw_pool > 0):
        factor = available_pool / raw_pool if raw_pool > 0 else 0.0
        bonus_counts = {b: m * factor for b, m in bonus_counts.items()}
        action_counts = {t: m * factor for t, m in action_counts.items()}
        expected = {
            value: (mass * factor if value in seen else mass)
            for value, mass in expected.items()
        }
        belief = PileBelief(
            expected_copies=expected,
            pile_size=pile,
            bonus_counts=bonus_counts,
            action_counts=action_counts,
        )
    return belief

