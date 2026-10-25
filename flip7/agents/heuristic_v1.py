"""Pre-V2 expected-value heuristic (frozen baseline for comparison/zoo)."""

from __future__ import annotations

from dataclasses import dataclass

from flip7.action import Action
from flip7.agents.base_agent import BaseAgent
from flip7.card import DECK_COMPOSITION, BonusCard, NumberCard
from flip7.game_state import GameState, PlayerView


@dataclass(frozen=True)
class HeuristicConfig:
    """Small set of interpretable risk parameters."""

    minimum_bank_score: float = 10.0
    continuation_bonus: float = 1.35
    trailing_aggression: float = 8.0
    second_chance_value: float = 3.0


class HeuristicAgentV1(BaseAgent):
    """Choose Draw by one-step EV and resolve action cards tactically."""

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if not legal_actions:
            raise ValueError("HeuristicAgent was asked to act without legal actions.")
        legal_actions = [Action(action) for action in legal_actions]

        if Action.STOP not in legal_actions:
            return self._choose_target(gs, legal_actions)
        if Action.DRAW not in legal_actions:
            return Action.STOP

        return Action.DRAW if self._should_draw(gs) else Action.STOP

    def _choose_target(self, gs: GameState, legal_actions: list[Action]) -> Action:
        indexed: list[tuple[Action, int]] = []
        for action in legal_actions:
            target_index = int(action) - int(Action.TARGET_0)
            if 0 <= target_index < len(gs.target_player_ids):
                indexed.append((action, gs.target_player_ids[target_index]))
        if not indexed:
            return legal_actions[0]

        opponents = {op.player_id: op for op in gs.opponents}

        def threat(player_id: int) -> float:
            if player_id == gs.my_player_id:
                return (
                    3.5 * len(gs.my_number_cards)
                    + 4.0 * bool(gs.my_bonus_cards)
                    + 5.0 * gs.my_has_second_chance
                    + 0.04 * gs.my_total_score
                )
            op = opponents[player_id]
            return (
                3.5 * op.num_number_cards
                + 4.0 * op.has_bonus_card
                + 5.0 * op.has_second_chance
                + 0.04 * op.total_score
            )

        if gs.pending_action_card == "second_chance":
            # A forced benefit goes to the currently least dangerous candidate.
            return min(indexed, key=lambda pair: threat(pair[1]))[0]

        if gs.pending_action_card == "freeze":
            # Freeze banks a rival's live score, so prioritise visible round strength
            # and a rival close to winning.
            return max(indexed, key=lambda pair: threat(pair[1]))[0]

        if gs.pending_action_card == "flip_three":
            def flip_three_pressure(player_id: int) -> float:
                if player_id == gs.my_player_id:
                    count = len(gs.my_number_cards)
                    second_chance = gs.my_has_second_chance
                    total = gs.my_total_score
                    is_opponent = 0.0
                else:
                    op = opponents[player_id]
                    count = op.num_number_cards
                    second_chance = op.has_second_chance
                    total = op.total_score
                    is_opponent = 4.0
                return (
                    1.8 * count * count
                    - 8.0 * second_chance
                    + 0.025 * total
                    + is_opponent
                )

            return max(indexed, key=lambda pair: flip_three_pressure(pair[1]))[0]

        return legal_actions[0]

    @staticmethod
    def _estimated_card_masses(gs: GameState) -> tuple[dict[int, float], list[BonusCard], float]:
        """Return approximate remaining number, bonus, and action-card masses.

        The public signal records only whether a value has ever appeared, not its
        exact draw count. We therefore remove one known copy for seen values and
        blend this estimate with the undepleted prior when the current pile is small.
        """
        prior = {value: float(max(value, 1)) for value in range(13)}
        depleted: dict[int, float] = {}
        for value, count in prior.items():
            known_copies = 1.0 if value in gs.numbers_seen_this_game else 0.0
            if value in gs.my_number_cards:
                known_copies = max(known_copies, 1.0)
            depleted[value] = max(0.0, count - known_copies)

        confidence = min(1.0, max(0.15, gs.deck_remaining / 94.0))
        masses = {
            value: confidence * depleted[value]
            + (1.0 - confidence) * max(0.0, count - float(value in gs.my_number_cards))
            for value, count in prior.items()
        }

        held_bonus_types = {bonus.bonus_type for bonus in gs.my_bonus_cards}
        bonuses = [
            card
            for card in DECK_COMPOSITION
            if isinstance(card, BonusCard) and card.bonus_type not in held_bonus_types
        ]
        # Action cards are always safe on the immediate number-duplicate event.
        action_mass = 9.0
        return masses, bonuses, action_mass

    @staticmethod
    def _score_with(gs: GameState, *, extra_number: int | None = None, bonus: BonusCard | None = None) -> float:
        number_sum = float(sum(gs.my_number_cards))
        if extra_number is not None:
            number_sum += extra_number

        has_x2 = any(card.bonus_type == "x2" for card in gs.my_bonus_cards)
        additive = float(
            sum(card.value for card in gs.my_bonus_cards if card.bonus_type != "x2")
        )
        if bonus is not None:
            if bonus.bonus_type == "x2":
                has_x2 = True
            else:
                additive += bonus.value

        score = number_sum * (2.0 if has_x2 else 1.0) + additive
        if extra_number is not None and len(gs.my_number_cards) == 6:
            score += 15.0
        return score

    def _should_draw(self, gs: GameState) -> bool:
        current = float(gs.my_round_score)
        hand_size = len(gs.my_number_cards)
        opponent_scores = [op.total_score for op in gs.opponents]
        leading_opponent = max(opponent_scores, default=0)

        if gs.my_total_score + current >= 200:
            return False
        if current < self.config.minimum_bank_score:
            return True

        number_masses, bonuses, action_mass = self._estimated_card_masses(gs)
        total_mass = sum(number_masses.values()) + len(bonuses) + action_mass
        if total_mass <= 0:
            return False

        expected = 0.0
        for value, mass in number_masses.items():
            if value in gs.my_number_cards:
                outcome = current if gs.my_has_second_chance else 0.0
            else:
                outcome = self._score_with(gs, extra_number=value)
            expected += mass * outcome
        for bonus in bonuses:
            expected += self._score_with(gs, bonus=bonus)
        expected += action_mass * current
        expected /= total_mass

        if gs.my_has_second_chance:
            expected -= self.config.second_chance_value

        # Drawing retains the option to stop later; the bonus is largest with a
        # small hand and vanishes near Flip 7 where the immediate EV already
        # contains the +15 outcome.
        option_value = self.config.continuation_bonus * max(0, 5 - hand_size)
        score_deficit = max(0.0, leading_opponent - gs.my_total_score)
        urgency = min(1.0, score_deficit / 100.0)
        if leading_opponent >= 180:
            urgency = max(urgency, 0.75)
        aggression = self.config.trailing_aggression * urgency

        # When comfortably ahead near the finish, prefer lower variance.
        safety_margin = 0.0
        if gs.my_total_score > leading_opponent and gs.my_total_score >= 160:
            safety_margin = 4.0

        return expected + option_value + aggression >= current + safety_margin

