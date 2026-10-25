from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .card import NumberCard, BonusCard, ActionCard
from .deck import Deck
from .player import Player, PlayerStatus
from .game_state import GameState
from .action import Action
from .agents.base_agent import BaseAgent

WIN_SCORE = 200
FLIP7_SIZE = 7
FLIP7_BONUS = 15


class IllegalActionError(Exception):
    """Raised by Flip7Game.step() when an action is not in the current legal_actions.

    Carries player_id so tournament runners can attribute the fault. Callers
    that record games (see recorder.record_game) may attach the partial
    GameRecord as `partial_record` for post-mortem debugging.
    """

    def __init__(self, player_id: int, action: Action, legal_actions: tuple):
        self.player_id = player_id
        self.action = action
        self.legal_actions = legal_actions
        self.partial_record: object | None = None
        super().__init__(
            f"Player {player_id} chose illegal action {action!r}; "
            f"legal_actions were {legal_actions!r}."
        )


@dataclass
class StepResult:
    """Returned by Flip7Game.step() and Flip7Game.reset().

    acting_player_id: which agent must supply the next action.
    game_state:       snapshot from that agent's perspective.
    reward:           round score just awarded (non-zero only at round end).
    round_over:       True when a round just finished.
    game_over:        True when a player reached WIN_SCORE.
    final_scores:     player_id → total score; non-None only when game_over.
    """

    acting_player_id: int
    game_state: GameState
    reward: float
    round_over: bool
    game_over: bool
    final_scores: dict[int, int] | None


# Sentinel passed through the generator to deliver an action
class _ActionToken:
    def __init__(self, action: Action) -> None:
        self.action = action


class Flip7Game:
    """Flip 7 game engine with a step-based API suitable for RL environments.

    Usage (step-based)::

        result = game.reset()            # starts first round
        while not result.game_over:
            action = my_policy(result)
            result = game.step(action)

    Usage (agent-based, non-RL)::

        scores = game.run_game(agents)
    """

    def __init__(self, n_players: int, seed: int | None = None) -> None:
        if not (3 <= n_players <= 18):
            raise ValueError("Flip 7 requires 3–18 players.")
        self._n = n_players
        self._rng = np.random.default_rng(seed)
        self._players: list[Player] = [Player(id=i) for i in range(n_players)]
        self._deck = Deck(self._rng)
        self._current_round = 0
        self._dealer_idx = int(self._rng.integers(n_players))
        self._flip7_player: int | None = None

        # Generator state
        self._gen: object | None = None  # active round generator
        self._pending_pid: int | None = None  # who must act next
        self._pending_legal_actions: tuple = ()  # legal actions for _pending_pid
        self._game_over = False
        self._final_scores: dict[int, int] | None = None
        # True right after an intermediate (non-game-ending) round-over
        # notification, until the next step() call starts the new round.
        self._round_boundary_pending = False

    # ------------------------------------------------------------------
    # Step-based public API
    # ------------------------------------------------------------------

    def reset(self) -> StepResult:
        """Start a new game from scratch and return the first decision point."""
        self._players = [Player(id=i) for i in range(self._n)]
        self._deck = Deck(self._rng)
        self._current_round = 0
        self._dealer_idx = int(self._rng.integers(self._n))
        self._flip7_player = None
        self._game_over = False
        self._final_scores = None
        self._round_boundary_pending = False
        return self._start_round()

    def step(self, action: Action) -> StepResult:
        """Advance the game by one agent decision. Returns the next decision point.

        Raises IllegalActionError if action is not in the legal_actions of the
        StepResult most recently returned to the acting player.

        A round boundary (intermediate or final) is surfaced as its own
        StepResult with round_over=True and legal_actions=() — no decision is
        being requested. The following step() call advances into the next
        round's first real decision; its `action` argument is not validated
        (any value is accepted and ignored) since there was nothing legal to
        choose from on the notification itself.
        """
        if self._game_over:
            raise RuntimeError("Game is over. Call reset() to start a new game.")
        if self._gen is None:
            raise RuntimeError("Call reset() before step().")

        if self._round_boundary_pending:
            self._round_boundary_pending = False
            return self._start_round()

        if action not in self._pending_legal_actions:
            raise IllegalActionError(
                self._pending_pid, action, self._pending_legal_actions
            )

        result = self._send_to_gen(action)
        if result is not None:
            return result

        # Round finished — check for game winner and start next round if needed
        round_scores = self._end_round()
        self._dealer_idx = (self._dealer_idx + 1) % self._n

        winner = self._check_game_winner()
        # Return from the perspective of the last acting player; reward is their score
        pid = self._pending_pid if self._pending_pid is not None else 0
        gs = self._make_game_state(
            self._players[pid],
            legal_actions=(),
            round_scores=round_scores,
        )
        if winner is not None:
            self._game_over = True
            self._final_scores = {p.id: p.total_score for p in self._players}
            return StepResult(
                acting_player_id=pid,
                game_state=gs,
                reward=float(round_scores.get(pid, 0)),
                round_over=True,
                game_over=True,
                final_scores=self._final_scores,
            )

        # Intermediate round boundary: notify the caller now, defer starting
        # the next round until the following step() call.
        self._round_boundary_pending = True
        self._pending_legal_actions = ()
        return StepResult(
            acting_player_id=pid,
            game_state=gs,
            reward=float(round_scores.get(pid, 0)),
            round_over=True,
            game_over=False,
            final_scores=None,
        )

    # ------------------------------------------------------------------
    # Agent-based public API (non-RL convenience wrapper)
    # ------------------------------------------------------------------

    def run_game(self, agents: list[BaseAgent]) -> dict[int, int]:
        """Play a full game with the given agents. Returns player_id → total score."""
        if len(agents) != self._n:
            raise ValueError(f"Expected {self._n} agents, got {len(agents)}.")
        result = self.reset()
        while True:
            pid = result.acting_player_id
            gs = result.game_state
            action = agents[pid].act(gs, list(gs.legal_actions))
            prev_gs = gs
            result = self.step(action)
            agents[pid].observe(
                prev_gs, action, result.game_state, result.reward, result.round_over
            )
            if result.game_over:
                break
            while result.round_over and not result.game_over:
                # Intermediate round boundary: no decision was requested for
                # this StepResult (legal_actions is empty). Advance into the
                # next round; the placeholder action is ignored by step()'s
                # handling of the pending round-boundary flag.
                result = self.step(Action.STOP)
                if result.game_over:
                    break
        return {p.id: p.total_score for p in self._players}

    # ------------------------------------------------------------------
    # Internal — round generator machinery
    # ------------------------------------------------------------------

    def _start_round(self) -> StepResult:
        self._current_round += 1
        self._flip7_player = None
        for p in self._players:
            p.reset_round()
        self._gen = self._round_generator()
        result = next(self._gen)  # type: ignore[arg-type]
        self._pending_pid = result.acting_player_id
        self._pending_legal_actions = result.game_state.legal_actions
        return result

    def _send_to_gen(self, action: Action) -> StepResult | None:
        """Send action into the generator. Returns StepResult or None if round ended."""
        try:
            result = self._gen.send(_ActionToken(action))  # type: ignore[union-attr]
            self._pending_pid = result.acting_player_id
            self._pending_legal_actions = result.game_state.legal_actions
            return result
        except StopIteration:
            return None

    def _round_generator(self):
        """Coroutine that yields a StepResult each time an agent decision is needed.

        Caller sends back an _ActionToken. The generator runs to StopIteration
        when the round is over; _end_round() is then called by step().
        """
        # --- Initial deal ---
        order = self._deal_order()
        for pid in order:
            if self._round_over():
                break
            player = self._players[pid]
            card = self._deck.draw()
            yield from self._apply_card_gen(player, card, during_initial_deal=True)

        if self._round_over():
            return

        # --- Decision phase ---
        still_deciding = [p.id for p in self._players if p.is_in_round()]
        while still_deciding and not self._round_over():
            next_deciding = []
            for pid in still_deciding:
                if self._round_over():
                    break
                player = self._players[pid]
                if not player.is_in_round():
                    continue

                gs = self._make_game_state(
                    player, legal_actions=(Action.STOP, Action.DRAW)
                )
                token: _ActionToken = yield StepResult(
                    acting_player_id=pid,
                    game_state=gs,
                    reward=0.0,
                    round_over=False,
                    game_over=False,
                    final_scores=None,
                )
                action = token.action

                if action == Action.STOP:
                    player.status = PlayerStatus.STOPPED
                else:
                    card = self._deck.draw()
                    yield from self._apply_card_gen(
                        player, card, during_initial_deal=False
                    )
                    if player.is_in_round():
                        next_deciding.append(pid)

            still_deciding = next_deciding

    def _apply_card_gen(
        self, player: Player, card: object, *, during_initial_deal: bool
    ):
        """Generator wrapper around card application; yields when target selection is needed.

        The trailing `return; yield` is an intentional Python idiom to make this
        function a generator in all branches, even those that never yield.
        """
        if isinstance(card, NumberCard):
            self._apply_number_card(player, card)
        elif isinstance(card, BonusCard):
            player.bonus_cards.append(card)
        elif isinstance(card, ActionCard):
            yield from self._apply_action_card_gen(
                player, card, during_initial_deal=during_initial_deal
            )
        return
        yield  # noqa: F821 — unreachable; makes every branch a generator (Python generator idiom)

    def _apply_action_card_gen(
        self, player: Player, card: ActionCard, *, during_initial_deal: bool
    ):
        if card.action_type == "freeze":
            yield from self._apply_freeze_gen(player, card)
        elif card.action_type == "second_chance":
            yield from self._apply_second_chance_gen(player, card)
        elif card.action_type == "flip_three":
            yield from self._apply_flip_three_gen(player, card)
        else:
            self._deck.discard([card])
        return
        yield  # noqa: F821 — unreachable; makes every branch a generator (Python generator idiom)

    # ------------------------------------------------------------------
    # Internal — action card generators
    # ------------------------------------------------------------------

    def _apply_freeze_gen(self, acting_player: Player, card: ActionCard):
        targets = [
            p for p in self._players if p.is_in_round() and p.id != acting_player.id
        ]
        if not targets:
            # Only active player — must freeze self (per rules)
            acting_player.status = PlayerStatus.FROZEN
            self._deck.discard([card])
            return
        target = yield from self._pick_target_gen(
            acting_player, targets, action_card="freeze"
        )
        target.status = PlayerStatus.FROZEN
        self._deck.discard([card])
        return
        yield  # noqa: F821 — unreachable; makes every branch a generator (Python generator idiom)

    def _apply_second_chance_gen(self, receiving_player: Player, card: ActionCard):
        if not receiving_player.has_second_chance:
            receiving_player.has_second_chance = True
            self._deck.discard([card])
        else:
            # Already holds one — pass to another active player without a duplicate SC
            others = [
                p
                for p in self._players
                if p.is_in_round()
                and p.id != receiving_player.id
                and not p.has_second_chance
            ]
            if others:
                target = yield from self._pick_target_gen(
                    receiving_player, others, action_card="second_chance"
                )
                target.has_second_chance = True
            self._deck.discard([card])
        return
        yield  # noqa: F821 — unreachable; makes every branch a generator (Python generator idiom)

    def _apply_flip_three_gen(self, acting_player: Player, card: ActionCard):
        self._deck.discard([card])
        targets = [p for p in self._players if p.is_in_round()]
        if not targets:
            return
        target = yield from self._pick_target_gen(
            acting_player, targets, action_card="flip_three", include_self=True
        )

        deferred_action_cards: list[ActionCard] = []
        for _ in range(3):
            if self._round_over():
                break
            drawn = self._deck.draw()
            if isinstance(drawn, ActionCard):
                deferred_action_cards.append(drawn)
            else:
                yield from self._apply_card_gen(
                    target, drawn, during_initial_deal=False
                )
                if not target.is_in_round():
                    break

        for ac in deferred_action_cards:
            if self._round_over():
                self._deck.discard([ac])
            else:
                resolver = self._find_active_player_other_than(target)
                if resolver is not None:
                    yield from self._apply_action_card_gen(
                        resolver, ac, during_initial_deal=False
                    )
                else:
                    self._deck.discard([ac])
        return
        yield  # noqa: F821 — unreachable; makes every branch a generator (Python generator idiom)

    def _pick_target_gen(
        self,
        acting_player: Player,
        candidates: list[Player],
        *,
        action_card: str,
        include_self: bool = False,
    ):
        if not include_self:
            candidates = [p for p in candidates if p.id != acting_player.id]
        if not candidates:
            return acting_player
        if len(candidates) == 1:
            return candidates[0]

        legal = tuple(Action(Action.TARGET_0 + i) for i in range(len(candidates)))
        candidate_ids = tuple(p.id for p in candidates)
        gs = self._make_game_state(
            acting_player,
            legal_actions=legal,
            pending_action_card=action_card,
            target_player_ids=candidate_ids,
        )
        token: _ActionToken = yield StepResult(
            acting_player_id=acting_player.id,
            game_state=gs,
            reward=0.0,
            round_over=False,
            game_over=False,
            final_scores=None,
        )
        target_idx = int(token.action) - int(Action.TARGET_0)
        if target_idx not in range(len(candidates)):
            target_idx = 0
        return candidates[target_idx]

    # ------------------------------------------------------------------
    # Internal — round helpers
    # ------------------------------------------------------------------

    def _apply_number_card(self, player: Player, card: NumberCard) -> None:
        if player.has_number(card.value):
            if player.has_second_chance:
                player.has_second_chance = False
                self._deck.discard([card])
            else:
                player.status = PlayerStatus.BUSTED
                self._deck.discard(player.number_cards + player.bonus_cards + [card])
                player.number_cards = []
                player.bonus_cards = []
        else:
            player.number_cards.append(card)
            if self._check_flip7(player):
                player.flip7_bonus = True
                self._flip7_player = player.id
                for p in self._players:
                    if p.is_in_round() and p.id != player.id:
                        p.status = PlayerStatus.STOPPED

    def _deal_order(self) -> list[int]:
        n = len(self._players)
        return [(self._dealer_idx + 1 + i) % n for i in range(n)]

    def _check_flip7(self, player: Player) -> bool:
        return len(player.number_cards) == FLIP7_SIZE

    def _round_over(self) -> bool:
        return self._flip7_player is not None or all(
            not p.is_in_round() for p in self._players
        )

    def _find_active_player_other_than(self, player: Player) -> Player | None:
        return next(
            (p for p in self._players if p.is_in_round() and p.id != player.id), None
        )

    def _end_round(self) -> dict[int, int]:
        round_scores: dict[int, int] = {}
        cards_to_discard: list = []
        for player in self._players:
            score = player.round_score()
            round_scores[player.id] = score
            player.total_score += score
            cards_to_discard.extend(player.number_cards)
            cards_to_discard.extend(player.bonus_cards)
            # Second Chance cards are discarded at round end whether used or not
            if player.has_second_chance:
                player.has_second_chance = False
        self._deck.discard(cards_to_discard)
        return round_scores

    def _check_game_winner(self) -> int | None:
        max_score = max(p.total_score for p in self._players)
        if max_score >= WIN_SCORE:
            leaders = [p for p in self._players if p.total_score == max_score]
            if len(leaders) == 1:
                return leaders[0].id
        return None

    def _make_game_state(
        self,
        player: Player,
        legal_actions: tuple = (),
        pending_action_card: str | None = None,
        round_scores: dict[int, int] | None = None,
        target_player_ids: tuple[int, ...] = (),
    ) -> GameState:
        return GameState.from_game(
            perspective=player,
            all_players=self._players,
            deck_remaining=self._deck.cards_remaining(),
            numbers_seen=self._deck.seen_numbers(),
            current_round=self._current_round,
            legal_actions=legal_actions,
            pending_action_card=pending_action_card,
            round_scores=round_scores,
            target_player_ids=target_player_ids,
        )
