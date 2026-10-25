from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class NumberCard:
    value: int  # 0–12


@dataclass(frozen=True)
class BonusCard:
    bonus_type: str  # "x2" | "+2" | "+4" | "+6" | "+8" | "+10"
    value: int  # numeric value (0 for x2, additive for the rest)


@dataclass(frozen=True)
class ActionCard:
    action_type: str  # "freeze" | "second_chance" | "flip_three"


Card = Union[NumberCard, BonusCard, ActionCard]

# Complete deck composition: card → count
DECK_COMPOSITION: dict[Card, int] = {
    **{NumberCard(v): (1 if v == 0 else v) for v in range(13)},
    BonusCard("x2", 0): 1,
    BonusCard("+2", 2): 1,
    BonusCard("+4", 4): 1,
    BonusCard("+6", 6): 1,
    BonusCard("+8", 8): 1,
    BonusCard("+10", 10): 1,
    ActionCard("freeze"): 3,
    ActionCard("second_chance"): 3,
    ActionCard("flip_three"): 3,
}

TOTAL_CARDS = sum(DECK_COMPOSITION.values())  # 94
