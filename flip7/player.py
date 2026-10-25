from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from .card import NumberCard, BonusCard


class PlayerStatus(Enum):
    ACTIVE = auto()
    STOPPED = auto()
    BUSTED = auto()
    FROZEN = auto()


@dataclass
class Player:
    id: int
    total_score: int = 0

    # Round state — reset each round
    number_cards: list[NumberCard] = field(default_factory=list)
    bonus_cards: list[BonusCard] = field(default_factory=list)
    has_second_chance: bool = False
    status: PlayerStatus = PlayerStatus.ACTIVE
    flip7_bonus: bool = False

    # --- Round helpers ---

    def is_in_round(self) -> bool:
        return self.status == PlayerStatus.ACTIVE

    def has_number(self, value: int) -> bool:
        return any(c.value == value for c in self.number_cards)

    def number_values(self) -> frozenset[int]:
        return frozenset(c.value for c in self.number_cards)

    def round_score(self) -> int:
        """Scoring order per rules: sum → ×2 → +X bonuses → +15 Flip 7."""
        if self.status == PlayerStatus.BUSTED:
            return 0
        total = sum(c.value for c in self.number_cards)
        if any(b.bonus_type == "x2" for b in self.bonus_cards):
            total *= 2
        total += sum(b.value for b in self.bonus_cards if b.bonus_type != "x2")
        if self.flip7_bonus:
            total += 15
        return total

    def reset_round(self) -> None:
        self.number_cards = []
        self.bonus_cards = []
        self.has_second_chance = False
        self.status = PlayerStatus.ACTIVE
        self.flip7_bonus = False
