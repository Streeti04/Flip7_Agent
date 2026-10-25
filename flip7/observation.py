"""Shared, fixed-length observation encoding for training and tournament play."""

from __future__ import annotations

import numpy as np

from .action import Action
from .agents.heuristic_agent import HeuristicAgent, _RoundState
from .belief import NUMBER_COUNTS, build_belief
from .game_state import GameState, PlayerView
from .player import PlayerStatus

ACTION_COUNT = len(Action)
MAX_ROUND_SCORE = 171.0
SCORE_SCALE = 400.0
BELIEF_FEATURE_COUNT = 13
ENDGAME_FEATURE_COUNT = 9
WINNING_SCORE = 200
# Rival threat estimate (mirrors HeuristicConfig.endgame_threat_per_card):
# points per live number card, +1 card if the rival may still draw.
THREAT_PER_CARD = 8.0
# Normalisation of the heuristic draw advantage EV(draw) - bank (round points).
DRAW_ADVANTAGE_SCALE = 30.0

_STATUSES = (
    PlayerStatus.ACTIVE,
    PlayerStatus.STOPPED,
    PlayerStatus.BUSTED,
    PlayerStatus.FROZEN,
)
_PENDING_TYPES = (None, "freeze", "second_chance", "flip_three")


# Fixed default-config heuristic whose expectimax value feeds ``draw_advantage``.
_EV_HEURISTIC = HeuristicAgent()


def observation_size(n_players: int = 4) -> int:
    """Return the vector length for a fixed number of players."""
    if not 3 <= n_players <= 18:
        raise ValueError("Flip 7 observations require 3-18 players.")
    # own cards + seen values + own features + global + pending-card one-hot
    # + opponent slots + target-candidate slots + belief block + game-end block
    return (
        13 + 13 + 11 + 4 + 4 + 10 * (n_players - 1) + 11 * n_players
        + BELIEF_FEATURE_COUNT + ENDGAME_FEATURE_COUNT
    )


def legal_action_mask(game_state: GameState) -> np.ndarray:
    """Encode ``legal_actions`` as the boolean mask expected by MaskablePPO."""
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    for action in game_state.legal_actions:
        mask[int(action)] = True
    return mask


def _status_one_hot(status: PlayerStatus) -> list[float]:
    return [float(status is candidate) for candidate in _STATUSES]


def _clip(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _opponent_features(
    opponent: PlayerView,
    *,
    my_id: int,
    my_score: int,
    n_players: int,
) -> list[float]:
    relative_seat = (opponent.player_id - my_id) % n_players
    return [
        relative_seat / max(1, n_players - 1),
        min(opponent.num_number_cards, 7) / 7.0,
        float(opponent.has_bonus_card),
        float(opponent.has_second_chance),
        *_status_one_hot(opponent.status),
        _clip(opponent.total_score / SCORE_SCALE),
        _clip((opponent.total_score - my_score) / SCORE_SCALE),
    ]


def _target_features(
    player_id: int | None,
    *,
    game_state: GameState,
    players: dict[int, PlayerView],
    n_players: int,
) -> list[float]:
    if player_id is None:
        return [0.0] * 11

    if player_id == game_state.my_player_id:
        return [
            1.0,
            1.0,
            0.0,
            min(len(game_state.my_number_cards), 7) / 7.0,
            float(bool(game_state.my_bonus_cards)),
            float(game_state.my_has_second_chance),
            *_status_one_hot(game_state.my_status),
            _clip(game_state.my_total_score / SCORE_SCALE),
        ]

    opponent = players[player_id]
    relative_seat = (player_id - game_state.my_player_id) % n_players
    return [
        1.0,
        0.0,
        relative_seat / max(1, n_players - 1),
        min(opponent.num_number_cards, 7) / 7.0,
        float(opponent.has_bonus_card),
        float(opponent.has_second_chance),
        *_status_one_hot(opponent.status),
        _clip(opponent.total_score / SCORE_SCALE),
    ]


def _belief_features(game_state: GameState) -> list[float]:
    """Engineered belief features derived solely from public information.

    Block layout (order matters, documented for report part B):

    0. ``p_bust`` — P(next number card duplicates a hand value)
       = belief.prob_duplicate(my_number_cards).
    1. ``expected_duplicate_mass`` — expected in-pile copies of hand values
       = Σ_v∈hand belief.expected_copies[v] / 40.
    2. ``unseen_count`` — number values never drawn this game / 13.
    3. ``unseen_mass_share`` — expected copies of unseen values
       / max(deck_remaining, 1).
    4. ``hand_sum`` — sum of own number card values / 78.
    5. ``active_opponents`` — opponents with status ACTIVE / 3.
    6. ``opponents_total_cards`` — Σ opponent num_number_cards / 21.
    7. ``pile_pressure`` — pile slots left for seen-value copies
       = clip((deck_remaining - unseen_mass) / 40).
    8. ``p_bonus`` — P(next card is a bonus card) = belief.prob_bonus().
    9. ``p_action`` — P(next card is an action card) = belief.prob_action().
    10. ``flip7_proximity`` — 1.0 iff own hand holds exactly 6 number values.
    11. ``my_turn_index`` — decision-sweep position among active players,
        normalised to [0, 1] (0 = act first, 1 = act last). The sweep always
        runs in ascending player id, so this captures the positional advantage
        of acting late.
    12. ``active_players`` — number of active players (incl. myself) / 4.
    """
    belief = build_belief(game_state)
    hand = game_state.my_number_cards
    unseen = [v for v in range(13) if v not in game_state.numbers_seen_this_game]
    unseen_mass = sum(belief.expected_copies[v] for v in unseen)
    pile = max(int(game_state.deck_remaining), 1)
    return [
        belief.prob_duplicate(hand),
        sum(belief.expected_copies[v] for v in hand) / 40.0,
        len(unseen) / 13.0,
        unseen_mass / pile,
        sum(hand) / 78.0,
        sum(
            1 for op in game_state.opponents if op.status == PlayerStatus.ACTIVE
        ) / 3.0,
        sum(op.num_number_cards for op in game_state.opponents) / 21.0,
        _clip(
            (game_state.deck_remaining
             - sum(NUMBER_COUNTS[v] for v in unseen)) / 40.0
        ),
        belief.prob_bonus(),
        belief.prob_action(),
        float(len(hand) == 6),
        *_turn_position_features(game_state),
    ]


def _turn_position_features(game_state: GameState) -> list[float]:
    """Own position in the ascending-id decision sweep among active players."""
    active_opponents = [
        op.player_id
        for op in game_state.opponents
        if op.status == PlayerStatus.ACTIVE
    ]
    n_active = 1 + len(active_opponents)
    if n_active <= 1:
        return [0.0, n_active / 4.0]
    my_rank = 1 + sum(1 for pid in active_opponents if pid < game_state.my_player_id)
    return [(my_rank - 1) / (n_active - 1), n_active / 4.0]


def _rival_threat_margin(game_state: GameState) -> float:
    """Largest estimated end-of-round total of a non-busted rival minus 200.

    A rival banks ``THREAT_PER_CARD`` points per live number card (+1 card if
    still active); busted rivals bank nothing this round. ``>= 0`` means the
    rival may end the game this round (same rule as
    ``HeuristicAgent._rival_may_end_game``). ``-200`` if no rival can bank.
    """
    margins = [
        op.total_score
        + THREAT_PER_CARD
        * (op.num_number_cards + (1 if op.status == PlayerStatus.ACTIVE else 0))
        - WINNING_SCORE
        for op in game_state.opponents
        if op.status != PlayerStatus.BUSTED
    ]
    return float(max(margins, default=-WINNING_SCORE))


def _draw_advantage(game_state: GameState) -> float:
    """Heuristic EV(draw) - bank in round points (0 when not active).

    Uses the default ``HeuristicAgent``'s depth-2 expectimax over the public
    belief state (incl. its seat factor), *without* the endgame bust
    correction: the game-end block exposes that context separately.
    """
    if game_state.my_status != PlayerStatus.ACTIVE:
        return 0.0
    state = _RoundState.from_game_state(game_state)
    value = _EV_HEURISTIC._draw_value(
        build_belief(game_state),
        state,
        depth=1,
        seat=_EV_HEURISTIC._seat_factor(game_state),
    )
    return float(value - state.bank())


def _endgame_features(game_state: GameState) -> list[float]:
    """Game-end control block (v4), public information only.

    Block layout (order matters, documented for report part B); ``banked`` =
    ``my_total_score + my_round_score`` (my total if I stop now), ``best_opp``
    = highest opponent total score:

    0. ``bank_margin`` — clip((banked - 200) / 50).
    1. ``bank_ends_game`` — 1.0 iff banked >= 200 (stopping ends the game).
    2. ``bank_near_miss`` — 1.0 iff 190 <= banked < 200 (stopping lands just
       below the finish line).
    3. ``best_opp_distance`` — clip((200 - best_opp) / 100).
    4. ``rival_may_end_game`` — 1.0 iff a non-busted rival may reach 200 this
       round (see :func:`_rival_threat_margin`).
    5. ``rival_threat_margin`` — clip(threat margin / 50).
    6. ``lead`` — clip((my_total_score - best_opp) / 50).
    7. ``round_score`` — clip(my_round_score / 50).
    8. ``draw_advantage`` — clip((heuristic EV(draw) - bank) / 30), see
       :func:`_draw_advantage`.
    """
    banked = game_state.my_total_score + game_state.my_round_score
    best_opp = max(
        (op.total_score for op in game_state.opponents),
        default=game_state.my_total_score,
    )
    threat = _rival_threat_margin(game_state)
    return [
        _clip((banked - WINNING_SCORE) / 50.0),
        float(banked >= WINNING_SCORE),
        float(WINNING_SCORE - 10 <= banked < WINNING_SCORE),
        _clip((WINNING_SCORE - best_opp) / 100.0),
        float(threat >= 0.0),
        _clip(threat / 50.0),
        _clip((game_state.my_total_score - best_opp) / 50.0),
        _clip(game_state.my_round_score / 50.0),
        _clip(_draw_advantage(game_state) / DRAW_ADVANTAGE_SCALE),
    ]


def encode_game_state(game_state: GameState, n_players: int = 4) -> np.ndarray:
    """Encode a player-perspective snapshot as a bounded float32 vector.

    Opponent slots use relative seat order, so rotating all absolute player IDs does
    not create a new strategic state. Target slots follow ``target_player_ids``
    exactly: slot zero describes the player selected by ``TARGET_0``.
    """
    if len(game_state.opponents) != n_players - 1:
        raise ValueError(
            f"Expected {n_players - 1} opponents, got {len(game_state.opponents)}."
        )

    values: list[float] = []
    values.extend(float(v in game_state.my_number_cards) for v in range(13))
    values.extend(float(v in game_state.numbers_seen_this_game) for v in range(13))

    additive_bonus = sum(
        card.value for card in game_state.my_bonus_cards if card.bonus_type != "x2"
    )
    values.extend(
        [
            float(any(card.bonus_type == "x2" for card in game_state.my_bonus_cards)),
            additive_bonus / 30.0,
            min(len(game_state.my_bonus_cards), 6) / 6.0,
            float(game_state.my_has_second_chance),
            *_status_one_hot(game_state.my_status),
            min(len(game_state.my_number_cards), 7) / 7.0,
            _clip(game_state.my_round_score / MAX_ROUND_SCORE),
            _clip(game_state.my_total_score / SCORE_SCALE),
        ]
    )

    opponent_scores = [op.total_score for op in game_state.opponents]
    max_score = max(opponent_scores, default=game_state.my_total_score)
    mean_score = float(np.mean(opponent_scores)) if opponent_scores else 0.0
    values.extend(
        [
            _clip(game_state.deck_remaining / 94.0),
            _clip(game_state.current_round / 50.0),
            _clip((game_state.my_total_score - max_score) / SCORE_SCALE),
            _clip((game_state.my_total_score - mean_score) / SCORE_SCALE),
        ]
    )
    values.extend(
        float(game_state.pending_action_card == pending)
        for pending in _PENDING_TYPES
    )

    opponents = sorted(
        game_state.opponents,
        key=lambda op: (op.player_id - game_state.my_player_id) % n_players,
    )
    for opponent in opponents:
        values.extend(
            _opponent_features(
                opponent,
                my_id=game_state.my_player_id,
                my_score=game_state.my_total_score,
                n_players=n_players,
            )
        )

    players = {op.player_id: op for op in game_state.opponents}
    for index in range(n_players):
        player_id = (
            game_state.target_player_ids[index]
            if index < len(game_state.target_player_ids)
            else None
        )
        values.extend(
            _target_features(
                player_id,
                game_state=game_state,
                players=players,
                n_players=n_players,
            )
        )

    values.extend(_belief_features(game_state))
    values.extend(_endgame_features(game_state))

    observation = np.asarray(values, dtype=np.float32)
    expected = observation_size(n_players)
    if observation.shape != (expected,):
        raise RuntimeError(
            f"Observation encoder produced {observation.shape}, expected {(expected,)}."
        )
    return np.clip(observation, -1.0, 1.0).astype(np.float32, copy=False)

