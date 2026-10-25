"""Expectimax heuristic for Flip 7 over a public-information belief state.

V2 rewrite. Key properties:

- **Metric-aligned:** the in-game tournament ranks agents by *mean score*, so
  the agent maximises the expected value of its own banked round score. Win-rate
  terms (trailing aggression, safety margins) are deliberately absent.
- **Endgame-aware:** when banking would end the game (total + round >= 200),
  a bust only delays my finish instead of costing the game, so the bust
  outcome is valued above 0 unless a rival may end the game first.
- **Belief state:** draw probabilities come from :mod:`flip7.belief`, which uses
  the exact guarantee that never-seen values are still in the pile and
  normalises consistently across number/bonus/action outcomes.
- **Depth-2 expectimax:** the value of drawing accounts for the option to stop
  (or keep drawing) after a safe outcome, instead of a hand-tuned bonus.
- **Correct action-card semantics:** Freeze does *not* bank a rival's score — it
  only locks them out of further draws; Flip Three aimed at oneself is a
  controlled 3-card package; a surplus Second Chance goes to the weakest rival.
- Never raises: any internal error falls back to a safe legal action.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from flip7.action import Action
from flip7.agents.base_agent import BaseAgent
from flip7.belief import PileBelief, build_belief
from flip7.card import BonusCard
from flip7.game_state import GameState
from flip7.player import PlayerStatus

FLIP7_SIZE = 7
FLIP7_BONUS = 15.0


@dataclass(frozen=True)
class HeuristicConfig:
    """Interpretable parameters (tuned via scripts/tune_heuristic.py, see
    Ergebnisse/tune_round*; finale Vielfeld-Selektion siehe PLAN_V2)."""

    # Value of drawing Freeze/Flip Three for oneself (round-continuation and
    # controlled-cards gains that are not expressible as own round score).
    freeze_value: float = 3.151375540022036
    flip_three_value: float = 8.0
    # Targeting weights for Freeze: deny rivals who gain most from drawing more.
    freeze_weight_flip7: float = 5.684201532084
    freeze_weight_second_chance: float = 9.173870048151775
    freeze_weight_hand: float = 0.6106965222127275
    freeze_weight_score: float = 0.017461368615843027
    # Flip Three targeting: denial of a rival Flip 7 vs. gifting weak hands.
    flip_three_denial_value: float = 6.075361995013242
    flip_three_gift_penalty: float = 8.818716734823115
    # Seat awareness: scales the continuation value by
    # 1 + seat_continuation * centred turn position (-1 = act first among
    # active players, +1 = act last). The tuned value is negative: whoever
    # acts early gets a larger continuation premium (plays more aggressively),
    # whoever acts last a smaller one.
    seat_continuation: float = -0.3656124323459342
    # Endgame: when banking would end the game (total + round >= 200) and no
    # rival threatens to end it this round, a bust only delays my finish —
    # the game goes on and I keep scoring, so the bust is worth roughly one
    # more round's gain instead of 0 (see VERBESSERUNGSVORSCHLAEGE.md §1.1).
    endgame_bust_value: float = 15.0
    # Rival threat estimate: total_score + this many points per live number
    # card (+1 card if the rival may still draw) reaching 200.
    endgame_threat_per_card: float = 8.0


@dataclass(frozen=True)
class _RoundState:
    """Own live round state as relevant for scoring."""

    hand: frozenset[int]
    has_x2: bool
    additive: float
    has_second_chance: bool

    @classmethod
    def from_game_state(cls, gs: GameState) -> "_RoundState":
        return cls(
            hand=frozenset(gs.my_number_cards),
            has_x2=any(card.bonus_type == "x2" for card in gs.my_bonus_cards),
            additive=float(
                sum(card.value for card in gs.my_bonus_cards if card.bonus_type != "x2")
            ),
            has_second_chance=gs.my_has_second_chance,
        )

    def bank(self) -> float:
        return float(sum(self.hand)) * (2.0 if self.has_x2 else 1.0) + self.additive


class HeuristicAgent(BaseAgent):
    """Choose Draw by depth-2 expectimax; resolve action cards tactically."""

    def __init__(self, config: HeuristicConfig | None = None) -> None:
        self.config = config or HeuristicConfig()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        try:
            return self._act(gs, [Action(a) for a in legal_actions])
        except Exception:
            # The tournament disqualifies the whole run if act() raises; any
            # internal failure must degrade to a safe legal action.
            if not legal_actions:
                return Action.STOP
            return (
                Action.STOP if Action.STOP in legal_actions else Action(legal_actions[0])
            )

    def _act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if not legal_actions:
            return Action.STOP
        if Action.STOP not in legal_actions:
            return self._choose_target(gs, legal_actions)
        if Action.DRAW not in legal_actions:
            return Action.STOP
        return Action.DRAW if self._should_draw(gs) else Action.STOP

    # ------------------------------------------------------------------
    # Stop / draw decision (depth-2 expectimax)
    # ------------------------------------------------------------------

    def _should_draw(self, gs: GameState) -> bool:
        belief = build_belief(gs)
        state = _RoundState.from_game_state(gs)
        value = self._draw_value(belief, state, depth=1, seat=self._seat_factor(gs))
        if (
            gs.my_total_score + gs.my_round_score >= 200
            and not state.has_second_chance
            and not self._rival_may_end_game(gs)
        ):
            # Banking now ends the game and freezes my final score, whereas a
            # bust only costs this round: the game continues and I keep
            # scoring. The search values a bust at 0, so credit the immediate
            # duplicate mass with endgame_bust_value (one-ply correction).
            p_dup = sum(belief.expected_copies[v] for v in state.hand) / max(
                belief.total_mass(), 1e-9
            )
            value += self.config.endgame_bust_value * p_dup
        return value > state.bank() + 1e-9

    def _rival_may_end_game(self, gs: GameState) -> bool:
        """Could a rival reach 200 this round (then my bust is not "free")?

        Busted rivals bank nothing this round. For the others the round gain is
        estimated as ``endgame_threat_per_card`` points per live number card,
        plus one more card if the rival is still active.
        """
        per_card = self.config.endgame_threat_per_card
        for op in gs.opponents:
            if op.status == PlayerStatus.BUSTED:
                continue
            estimate = op.total_score + per_card * (
                op.num_number_cards + (1 if op.status == PlayerStatus.ACTIVE else 0)
            )
            if estimate >= 200:
                return True
        return False

    def _seat_factor(self, gs: GameState) -> float:
        """Continuation-premium factor implied by my turn position.

        The decision sweep always runs in ascending player id, so the rank of
        ``my_player_id`` among active players is fully observable. The factor
        scales the option premium of continuing (1.0 = seat-neutral); with the
        tuned negative ``seat_continuation`` early actors get a factor > 1
        (more aggressive), late actors < 1.
        """
        scale = self.config.seat_continuation
        if scale == 0.0:
            return 1.0
        active_opponents = [
            op for op in gs.opponents if op.status == PlayerStatus.ACTIVE
        ]
        n_active = 1 + len(active_opponents)
        if n_active <= 1:
            return 1.0
        my_rank = 1 + sum(
            1 for op in active_opponents if op.player_id < gs.my_player_id
        )
        centred = (my_rank - 1) / (n_active - 1) * 2.0 - 1.0
        return min(max(1.0 + scale * centred, 0.0), 1.5)

    def _draw_value(
        self, belief: PileBelief, state: _RoundState, depth: int, seat: float = 1.0
    ) -> float:
        """EV of drawing one card, playing on with at most ``depth`` further searches.

        ``seat`` is a continuation-premium factor (see :meth:`_seat_factor`)
        applied to the option value of playing on after a safe outcome;
        1.0 is seat-neutral.
        """
        bank = state.bank()
        total_mass = belief.total_mass()
        if belief.pile_size <= 0 or total_mass <= 1e-9:
            return bank
        if depth <= 0:
            return self._one_ply_value(belief, state, bank, total_mass)

        def cont(next_bank: float, next_ev: float) -> float:
            """max(next_bank, next_ev) with the option premium seat-scaled."""
            premium = next_ev - next_bank
            if premium <= 0.0:
                return next_bank
            return next_bank + seat * premium

        ev = 0.0

        for value in range(13):
            mass = belief.expected_copies[value]
            prob = mass / total_mass
            if prob <= 1e-9:
                continue
            if value in state.hand:
                if state.has_second_chance:
                    # Bust averted: card discarded, Second Chance consumed, and
                    # the option to stop or keep drawing remains.
                    outcome = cont(
                        bank,
                        self._draw_value(
                            belief.withdraw(value),
                            replace(state, has_second_chance=False),
                            depth - 1,
                            seat,
                        ),
                    )
                else:
                    outcome = 0.0  # bust: round score lost
            else:
                hand2 = state.hand | {value}
                if len(hand2) >= FLIP7_SIZE:
                    outcome = float(sum(hand2)) * (2.0 if state.has_x2 else 1.0) + state.additive + FLIP7_BONUS
                else:
                    next_state = replace(state, hand=hand2)
                    outcome = cont(
                        next_state.bank(),
                        self._draw_value(
                            belief.withdraw(value), next_state, depth - 1, seat
                        ),
                    )
            ev += prob * outcome

        for bonus, mass in belief.bonus_counts.items():
            prob = mass / total_mass
            if prob <= 1e-9:
                continue
            next_state = self._with_bonus(state, bonus)
            outcome = cont(
                next_state.bank(),
                self._draw_value(
                    belief.withdraw_bonus(bonus), next_state, depth - 1, seat
                ),
            )
            ev += prob * outcome

        for action_type, mass in belief.action_counts.items():
            prob = mass / total_mass
            if prob <= 1e-9:
                continue
            if action_type == "second_chance":
                next_state = replace(state, has_second_chance=True)
                outcome = cont(
                    bank,
                    self._draw_value(
                        belief.withdraw_action(action_type), next_state, depth - 1, seat
                    ),
                )
            else:
                gain = (
                    self.config.freeze_value
                    if action_type == "freeze"
                    else self.config.flip_three_value
                )
                outcome = bank + gain
            ev += prob * outcome

        return ev

    def _one_ply_value(
        self, belief: PileBelief, state: _RoundState, bank: float, total_mass: float
    ) -> float:
        """Analytic one-ply EV (no recursion, no belief copies) — hot search leaf."""
        hand = state.hand
        mult = 2.0 if state.has_x2 else 1.0
        base = float(sum(hand))
        triggers_flip7 = len(hand) >= FLIP7_SIZE - 1
        copies = belief.expected_copies

        dup_mass = 0.0
        safe_acc = 0.0
        for value in range(13):
            mass = copies[value]
            if mass <= 1e-9:
                continue
            if value in hand:
                dup_mass += mass
            elif triggers_flip7:
                safe_acc += mass * ((base + value) * mult + state.additive + FLIP7_BONUS)
            else:
                safe_acc += mass * ((base + value) * mult + state.additive)

        bonus_acc = 0.0
        for bonus, mass in belief.bonus_counts.items():
            if bonus.bonus_type == "x2":
                bank2 = base * 2.0 + state.additive
            else:
                bank2 = base * mult + state.additive + float(bonus.value)
            bonus_acc += mass * max(bank2, bank)

        action_acc = 0.0
        for action_type, mass in belief.action_counts.items():
            if action_type == "second_chance":
                # Insurance only pays off in later decisions; at leaf level the
                # banked score is unchanged.
                action_acc += mass * bank
            else:
                gain = (
                    self.config.freeze_value
                    if action_type == "freeze"
                    else self.config.flip_three_value
                )
                action_acc += mass * (bank + gain)

        dup_outcome = bank if state.has_second_chance else 0.0
        return (dup_mass * dup_outcome + safe_acc + bonus_acc + action_acc) / total_mass

    @staticmethod
    def _with_bonus(state: _RoundState, bonus: BonusCard) -> _RoundState:
        if bonus.bonus_type == "x2":
            return replace(state, has_x2=True)
        return replace(state, additive=state.additive + float(bonus.value))

    # ------------------------------------------------------------------
    # Target selection for action cards
    # ------------------------------------------------------------------

    def _choose_target(self, gs: GameState, legal_actions: list[Action]) -> Action:
        indexed: list[tuple[Action, int]] = []
        for action in legal_actions:
            target_index = int(action) - int(Action.TARGET_0)
            if 0 <= target_index < len(gs.target_player_ids):
                indexed.append((action, gs.target_player_ids[target_index]))
        if not indexed:
            return legal_actions[0]

        opponents = {op.player_id: op for op in gs.opponents}

        if gs.pending_action_card == "freeze":
            cfg = self.config

            def freeze_value(player_id: int) -> float:
                if player_id == gs.my_player_id:
                    return -1e6  # engine never offers self, guard anyway
                op = opponents[player_id]
                return (
                    cfg.freeze_weight_flip7 * float(op.num_number_cards >= FLIP7_SIZE - 1)
                    + cfg.freeze_weight_second_chance * float(op.has_second_chance)
                    + cfg.freeze_weight_hand * op.num_number_cards
                    + cfg.freeze_weight_score * op.total_score
                )

            return max(indexed, key=lambda pair: freeze_value(pair[1]))[0]

        if gs.pending_action_card == "second_chance":
            # A surplus Second Chance helps its holder; give it to the weakest
            # candidate (smallest live hand, lowest score).
            def weakness(player_id: int) -> float:
                if player_id == gs.my_player_id:
                    return 1e6
                op = opponents[player_id]
                return op.num_number_cards + 0.01 * op.total_score

            return min(indexed, key=lambda pair: weakness(pair[1]))[0]

        if gs.pending_action_card == "flip_three":
            cfg = self.config
            belief = build_belief(gs)
            state = _RoundState.from_game_state(gs)

            def flip_three_value(player_id: int) -> float:
                if player_id == gs.my_player_id:
                    return self._flip_three_self_value(belief, state)
                op = opponents[player_id]
                size = op.num_number_cards
                if size >= FLIP7_SIZE - 1:
                    # Forcing cards on a near-Flip-7 rival most likely busts
                    # them and denies the +15 / early round end.
                    return cfg.flip_three_denial_value
                if size <= 1:
                    # Three free cards help a weak rival improve — avoid.
                    return -cfg.flip_three_gift_penalty
                return 0.0

            return max(indexed, key=lambda pair: flip_three_value(pair[1]))[0]

        return legal_actions[0]

    def _flip_three_self_value(self, belief: PileBelief, state: _RoundState) -> float:
        """EV of aiming Flip Three at oneself: 3 forced cards, no stop between."""
        def forced_value(current_belief: PileBelief, current_state: _RoundState, remaining: int) -> float:
            bank = current_state.bank()
            total_mass = current_belief.total_mass()
            if remaining <= 0 or total_mass <= 1e-9:
                return bank
            if remaining == 1:
                return self._forced_one_ply(current_belief, current_state, bank, total_mass)
            ev = 0.0
            for value in range(13):
                mass = current_belief.expected_copies[value]
                prob = mass / total_mass
                if prob <= 1e-9:
                    continue
                if value in current_state.hand:
                    if current_state.has_second_chance:
                        ev += prob * forced_value(
                            current_belief.withdraw(value),
                            replace(current_state, has_second_chance=False),
                            remaining - 1,
                        )
                    # else: bust → 0
                else:
                    hand2 = current_state.hand | {value}
                    if len(hand2) >= FLIP7_SIZE:
                        ev += prob * (
                            float(sum(hand2)) * (2.0 if current_state.has_x2 else 1.0)
                            + current_state.additive
                            + FLIP7_BONUS
                        )
                    else:
                        ev += prob * forced_value(
                            current_belief.withdraw(value),
                            replace(current_state, hand=hand2),
                            remaining - 1,
                        )
            for bonus, mass in current_belief.bonus_counts.items():
                prob = mass / total_mass
                if prob <= 1e-9:
                    continue
                ev += prob * forced_value(
                    current_belief.withdraw_bonus(bonus),
                    self._with_bonus(current_state, bonus),
                    remaining - 1,
                )
            # Action cards among the three are resolved by another player, not
            # by us — no own value; the card is simply consumed.
            consumed = sum(current_belief.action_counts.values()) / total_mass
            ev += consumed * forced_value(
                _withdraw_any_action(current_belief), current_state, remaining - 1
            )
            return ev

        return forced_value(belief, state, 3)

    def _forced_one_ply(
        self, belief: PileBelief, state: _RoundState, bank: float, total_mass: float
    ) -> float:
        """Analytic EV of one *forced* card (no stop option) — hot search leaf."""
        hand = state.hand
        mult = 2.0 if state.has_x2 else 1.0
        base = float(sum(hand))
        triggers_flip7 = len(hand) >= FLIP7_SIZE - 1
        copies = belief.expected_copies

        acc = 0.0
        for value in range(13):
            mass = copies[value]
            if mass <= 1e-9:
                continue
            if value in hand:
                if state.has_second_chance:
                    acc += mass * bank  # bust averted, forced part over
                # else bust → 0
            elif triggers_flip7:
                acc += mass * ((base + value) * mult + state.additive + FLIP7_BONUS)
            else:
                acc += mass * ((base + value) * mult + state.additive)
        for bonus, mass in belief.bonus_counts.items():
            if bonus.bonus_type == "x2":
                acc += mass * (base * 2.0 + state.additive)
            else:
                acc += mass * (base * mult + state.additive + float(bonus.value))
        # Action cards are resolved by another player: own value unchanged.
        acc += sum(belief.action_counts.values()) * bank
        return acc / total_mass


def _withdraw_any_action(belief: PileBelief) -> PileBelief:
    for action_type, mass in belief.action_counts.items():
        if mass > 0.0:
            return belief.withdraw_action(action_type)
    return belief
