from __future__ import annotations
from dataclasses import dataclass
from .card import BonusCard
from .player import Player, PlayerStatus
from .action import Action


@dataclass(frozen=True)
class PlayerView:
    """Partial opponent info visible to all players (mirrors physical game)."""

    player_id: int
    num_number_cards: int
    has_bonus_card: bool
    has_second_chance: bool
    status: PlayerStatus
    total_score: int


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of the game passed to agents after every turn.

    Designed for direct use in RL environments:
    - ``legal_actions`` enables action masking without a separate call.
    - ``pending_action_card`` identifies why target-selection is requested.
    - ``target_player_ids`` maps TARGET_i to the corresponding player id.
    - ``my_round_score`` gives the current round score for intermediate rewards.
    - ``my_player_id`` identifies which player this snapshot belongs to.
    - ``round_scores`` is non-None only at round end (round_over=True in StepResult).
    """

    # Full own state
    my_player_id: int
    my_number_cards: frozenset[int]  # values of number cards in front of me
    my_bonus_cards: tuple[BonusCard, ...]
    my_has_second_chance: bool
    my_total_score: int
    my_status: PlayerStatus
    my_round_score: int  # live round score (0 if busted, else current sum)

    # Global game information
    deck_remaining: int
    numbers_seen_this_game: frozenset[int]  # all numbers drawn — enables card counting
    current_round: int

    # Legal actions for the agent who is about to act (enables action masking).
    # Empty tuple when this state is delivered as a post-action observation.
    legal_actions: tuple[Action, ...]

    # Set to "freeze" / "second_chance" / "flip_three" when the agent is asked to
    # choose a target for an action card; None during normal Stop/Draw decisions.
    pending_action_card: str | None

    # Maps TARGET_i → player_id for the current target-selection request.
    # target_player_ids[i] is the player that Action.TARGET_0 + i refers to.
    # Empty tuple when pending_action_card is None.
    target_player_ids: tuple[int, ...]

    # Non-None only in the final StepResult of a round (round_over=True).
    round_scores: dict[int, int] | None

    # Opponents: exact cards unknown (only count visible)
    opponents: tuple[PlayerView, ...]

    @staticmethod
    def from_game(
        perspective: Player,
        all_players: list[Player],
        deck_remaining: int,
        numbers_seen: frozenset[int],
        current_round: int,
        legal_actions: tuple[Action, ...] = (),
        pending_action_card: str | None = None,
        target_player_ids: tuple[int, ...] = (),
        round_scores: dict[int, int] | None = None,
    ) -> "GameState":
        opponents = tuple(
            PlayerView(
                player_id=p.id,
                num_number_cards=len(p.number_cards),
                has_bonus_card=len(p.bonus_cards) > 0,
                has_second_chance=p.has_second_chance,
                status=p.status,
                total_score=p.total_score,
            )
            for p in all_players
            if p.id != perspective.id
        )
        return GameState(
            my_player_id=perspective.id,
            my_number_cards=perspective.number_values(),
            my_bonus_cards=tuple(perspective.bonus_cards),
            my_has_second_chance=perspective.has_second_chance,
            my_total_score=perspective.total_score,
            my_status=perspective.status,
            my_round_score=perspective.round_score(),
            deck_remaining=deck_remaining,
            numbers_seen_this_game=numbers_seen,
            current_round=current_round,
            legal_actions=tuple(Action(a) for a in legal_actions),
            pending_action_card=pending_action_card,
            target_player_ids=target_player_ids,
            round_scores=round_scores,
            opponents=opponents,
        )
