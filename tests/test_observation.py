"""Tests for the observation encoder, incl. the appended belief and game-end blocks.

The first 119 entries are the legacy base encoding and must stay bit-identical;
the belief block (13) and the v4 game-end block (9) are strictly appended at
the end (see equivalence tests below).
"""

from __future__ import annotations

import numpy as np

from flip7.agents.heuristic_agent import HeuristicAgent, _RoundState
from flip7.belief import NUMBER_COUNTS, build_belief
from flip7.game import Flip7Game
from flip7.game_state import GameState, PlayerView
from flip7.observation import (
    ACTION_COUNT,
    BELIEF_FEATURE_COUNT,
    ENDGAME_FEATURE_COUNT,
    encode_game_state,
    legal_action_mask,
    observation_size,
)
from flip7.player import PlayerStatus
from tournament import RandomAgent

LEGACY_SIZE = 119

# Feature indices inside the belief block.
P_BUST = 0
EXPECTED_DUP_MASS = 1
UNSEEN_COUNT = 2
UNSEEN_MASS_SHARE = 3
HAND_SUM = 4
ACTIVE_OPPONENTS = 5
OPPONENTS_TOTAL_CARDS = 6
PILE_PRESSURE = 7
P_BONUS = 8
P_ACTION = 9
FLIP7_PROXIMITY = 10


def make_state(
    *,
    hand=frozenset(),
    seen=frozenset(),
    deck_remaining: int = 94,
    n_players: int = 4,
    opponent_cards=(1, 1, 1),
    opponent_statuses=None,
) -> GameState:
    statuses = opponent_statuses or (PlayerStatus.ACTIVE,) * len(opponent_cards)
    opponents = tuple(
        PlayerView(
            player_id=i + 1,
            num_number_cards=cards,
            has_bonus_card=False,
            has_second_chance=False,
            status=status,
            total_score=0,
        )
        for i, (cards, status) in enumerate(zip(opponent_cards, statuses))
    )
    assert len(opponents) == n_players - 1
    return GameState(
        my_player_id=0,
        my_number_cards=frozenset(hand),
        my_bonus_cards=(),
        my_has_second_chance=False,
        my_total_score=0,
        my_status=PlayerStatus.ACTIVE,
        my_round_score=sum(hand),
        deck_remaining=deck_remaining,
        numbers_seen_this_game=frozenset(seen),
        current_round=1,
        legal_actions=(),
        pending_action_card=None,
        target_player_ids=(),
        round_scores=None,
        opponents=opponents,
    )


# ----------------------------------------------------------------------
# Frozen copy of the pre-belief 119-dim encoder, as the equivalence
# reference for the legacy base blocks.
# ----------------------------------------------------------------------


def _status_one_hot(status):
    statuses = (
        PlayerStatus.ACTIVE,
        PlayerStatus.STOPPED,
        PlayerStatus.BUSTED,
        PlayerStatus.FROZEN,
    )
    return [float(status is candidate) for candidate in statuses]


def _clip(value):
    return float(np.clip(value, -1.0, 1.0))


def _legacy_encode_119(game_state, n_players=4):
    pending_types = (None, "freeze", "second_chance", "flip_three")

    def opponent_features(opponent, *, my_id, my_score, n_players):
        relative_seat = (opponent.player_id - my_id) % n_players
        return [
            relative_seat / max(1, n_players - 1),
            min(opponent.num_number_cards, 7) / 7.0,
            float(opponent.has_bonus_card),
            float(opponent.has_second_chance),
            *_status_one_hot(opponent.status),
            _clip(opponent.total_score / 400.0),
            _clip((opponent.total_score - my_score) / 400.0),
        ]

    def target_features(player_id, *, game_state, players, n_players):
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
                _clip(game_state.my_total_score / 400.0),
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
            _clip(opponent.total_score / 400.0),
        ]

    values = []
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
            _clip(game_state.my_round_score / 171.0),
            _clip(game_state.my_total_score / 400.0),
        ]
    )

    opponent_scores = [op.total_score for op in game_state.opponents]
    max_score = max(opponent_scores, default=game_state.my_total_score)
    mean_score = float(np.mean(opponent_scores)) if opponent_scores else 0.0
    values.extend(
        [
            _clip(game_state.deck_remaining / 94.0),
            _clip(game_state.current_round / 50.0),
            _clip((game_state.my_total_score - max_score) / 400.0),
            _clip((game_state.my_total_score - mean_score) / 400.0),
        ]
    )
    values.extend(
        float(game_state.pending_action_card == pending) for pending in pending_types
    )

    opponents = sorted(
        game_state.opponents,
        key=lambda op: (op.player_id - game_state.my_player_id) % n_players,
    )
    for opponent in opponents:
        values.extend(
            opponent_features(
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
            target_features(
                player_id,
                game_state=game_state,
                players=players,
                n_players=n_players,
            )
        )

    observation = np.asarray(values, dtype=np.float32)
    return np.clip(observation, -1.0, 1.0).astype(np.float32, copy=False)


def collect_decision_snapshots(n_snapshots: int = 20, seed: int = 0):
    """Random real games; one snapshot per decision point (round_over=False)."""
    rng = np.random.default_rng(seed)
    snapshots = []
    while len(snapshots) < n_snapshots:
        game = Flip7Game(n_players=4, seed=int(rng.integers(2**31)))
        result = game.reset()
        guard = 0
        while not result.game_over and guard < 10_000:
            guard += 1
            if not result.round_over:
                snapshots.append(result.game_state)
            legal = list(result.game_state.legal_actions)
            action = legal[int(rng.integers(len(legal)))] if legal else None
            if action is None:
                break
            result = game.step(action)
    return snapshots[:n_snapshots]


# ----------------------------------------------------------------------
# Shape / range / determinism
# ----------------------------------------------------------------------


def test_observation_size_includes_belief_block():
    base = 13 + 13 + 11 + 4 + 4 + 10 * 3 + 11 * 4
    assert base == LEGACY_SIZE
    assert observation_size(4) == LEGACY_SIZE + BELIEF_FEATURE_COUNT + ENDGAME_FEATURE_COUNT
    assert BELIEF_FEATURE_COUNT == 13  # 11 belief + 2 turn-position features
    assert ENDGAME_FEATURE_COUNT == 9
    assert observation_size(4) == 141


def test_encoded_length_matches_observation_size_for_all_player_counts():
    for n_players in range(3, 19):
        state = make_state(
            n_players=n_players,
            opponent_cards=tuple(1 for _ in range(n_players - 1)),
        )
        encoded = encode_game_state(state, n_players)
        assert encoded.shape == (observation_size(n_players),)
        assert encoded.dtype == np.float32


def test_encoded_values_are_bounded_and_deterministic():
    snapshots = collect_decision_snapshots(n_snapshots=20)
    for state in snapshots:
        first = encode_game_state(state, 4)
        second = encode_game_state(state, 4)
        assert np.array_equal(first, second)
        assert np.all((-1.0 <= first) & (first <= 1.0))


# ----------------------------------------------------------------------
# Belief block correctness on handcrafted states (values recomputed
# from first principles, not via build_belief).
# ----------------------------------------------------------------------


def _belief_block(state):
    encoded = encode_game_state(state, 4)
    assert encoded.shape == (observation_size(4),)
    return encoded[LEGACY_SIZE:LEGACY_SIZE + BELIEF_FEATURE_COUNT]


def test_belief_block_mid_round_values():
    # seen={3,5,7}, hand={3,5}, deck=80: unseen mass 64, share = 16/27.
    state = make_state(
        hand=frozenset({3, 5}),
        seen=frozenset({3, 5, 7}),
        deck_remaining=80,
        opponent_cards=(2, 3, 4),
    )
    block = _belief_block(state)
    share = 16 / 27
    pile = 80.0
    unseen_mass = sum(NUMBER_COUNTS[v] for v in range(13) if v not in {3, 5, 7})
    assert unseen_mass == 64
    assert np.isclose(block[P_BUST], (2 * share + 4 * share) / pile)
    assert np.isclose(block[EXPECTED_DUP_MASS], (2 * share + 4 * share) / 40.0)
    assert np.isclose(block[UNSEEN_COUNT], 10 / 13)
    assert np.isclose(block[UNSEEN_MASS_SHARE], 64 / pile)
    assert np.isclose(block[HAND_SUM], 8 / 78)
    assert np.isclose(block[ACTIVE_OPPONENTS], 1.0)
    assert np.isclose(block[OPPONENTS_TOTAL_CARDS], 9 / 21)
    assert np.isclose(block[PILE_PRESSURE], (80 - 64) / 40.0)
    assert np.isclose(block[P_BONUS], 6 * share / pile)
    assert np.isclose(block[P_ACTION], 9 * share / pile)
    assert block[FLIP7_PROXIMITY] == 0.0


def test_belief_block_full_pool_share_and_mixed_statuses():
    # seen={12}, hand={12}, deck=93: pool slots 26 == pool mass 26 → share 1.
    state = make_state(
        hand=frozenset({12}),
        seen=frozenset({12}),
        deck_remaining=93,
        opponent_cards=(1, 1, 1),
        opponent_statuses=(
            PlayerStatus.ACTIVE,
            PlayerStatus.STOPPED,
            PlayerStatus.BUSTED,
        ),
    )
    block = _belief_block(state)
    pile = 93.0
    assert np.isclose(block[P_BUST], 11 / pile)
    assert np.isclose(block[EXPECTED_DUP_MASS], 11 / 40.0)
    assert np.isclose(block[UNSEEN_COUNT], 12 / 13)
    assert np.isclose(block[UNSEEN_MASS_SHARE], 67 / pile)
    assert np.isclose(block[HAND_SUM], 12 / 78)
    assert np.isclose(block[ACTIVE_OPPONENTS], 1 / 3)
    assert np.isclose(block[OPPONENTS_TOTAL_CARDS], 3 / 21)
    assert np.isclose(block[PILE_PRESSURE], (93 - 67) / 40.0)
    assert np.isclose(block[P_BONUS], 6 / pile)
    assert np.isclose(block[P_ACTION], 9 / pile)
    assert block[FLIP7_PROXIMITY] == 0.0


def test_belief_block_flip7_proximity_and_low_hand_values():
    # hand={0..5} (6 values → Flip 7 next), seen adds 6, deck=72 → share 0.5.
    hand = frozenset({0, 1, 2, 3, 4, 5})
    state = make_state(hand=hand, seen=hand | {6}, deck_remaining=72)
    block = _belief_block(state)
    pile = 72.0
    hand_mass = (0 + 0 + 1 + 2 + 3 + 4) * 0.5  # (count-1) * share per hand value
    assert np.isclose(block[P_BUST], hand_mass / pile)
    assert np.isclose(block[EXPECTED_DUP_MASS], hand_mass / 40.0)
    assert np.isclose(block[UNSEEN_COUNT], 6 / 13)
    assert np.isclose(block[UNSEEN_MASS_SHARE], 57 / pile)
    assert np.isclose(block[HAND_SUM], 15 / 78)
    assert block[FLIP7_PROXIMITY] == 1.0
    assert np.isclose(block[PILE_PRESSURE], 15 / 40.0)
    assert np.isclose(block[P_BONUS], 3 / pile)
    assert np.isclose(block[P_ACTION], 4.5 / pile)


def test_belief_block_on_real_game_snapshots_stays_bounded():
    for state in collect_decision_snapshots(n_snapshots=5):
        block = _belief_block(state)
        assert np.all((-1.0 <= block) & (block <= 1.0))


# ----------------------------------------------------------------------
# Equivalence: appending the belief block must not change the legacy
# base 119 entries on real decision-point snapshots.
# ----------------------------------------------------------------------


def test_base_119_entries_unchanged_on_real_snapshots():
    for state in collect_decision_snapshots(n_snapshots=20):
        encoded = encode_game_state(state, 4)
        legacy = _legacy_encode_119(state, 4)
        assert np.array_equal(encoded[:LEGACY_SIZE], legacy), (
            "legacy base encoding changed"
        )


def test_legal_action_mask_and_action_count_untouched():
    state = make_state()
    mask = legal_action_mask(state)
    assert mask.shape == (ACTION_COUNT,)
    assert mask.dtype == np.bool_
    assert not mask.any()


# ----------------------------------------------------------------------
# v4 game-end block (appended after the belief block)
# ----------------------------------------------------------------------

ENDGAME_OFFSET = LEGACY_SIZE + BELIEF_FEATURE_COUNT
BANK_MARGIN, BANK_ENDS_GAME, BANK_NEAR_MISS, BEST_OPP_DISTANCE = 0, 1, 2, 3
RIVAL_MAY_END, RIVAL_THREAT_MARGIN, LEAD, ROUND_SCORE, DRAW_ADVANTAGE = 4, 5, 6, 7, 8


def make_scored_state(
    *,
    hand,
    my_total: int,
    opponent_totals=(0, 0, 0),
    opponent_cards=(1, 1, 1),
    opponent_statuses=None,
    my_status=PlayerStatus.ACTIVE,
) -> GameState:
    base = make_state(
        hand=hand,
        seen=frozenset(hand),
        deck_remaining=80,
        opponent_cards=opponent_cards,
        opponent_statuses=opponent_statuses,
    )
    opponents = tuple(
        PlayerView(
            player_id=op.player_id,
            num_number_cards=op.num_number_cards,
            has_bonus_card=False,
            has_second_chance=False,
            status=op.status,
            total_score=total,
        )
        for op, total in zip(base.opponents, opponent_totals)
    )
    from dataclasses import replace

    return replace(base, my_total_score=my_total, opponents=opponents, my_status=my_status)


def _endgame_block(state):
    encoded = encode_game_state(state, 4)
    return encoded[ENDGAME_OFFSET:]


def _expected_advantage(state):
    heuristic = HeuristicAgent()
    round_state = _RoundState.from_game_state(state)
    value = heuristic._draw_value(
        build_belief(state), round_state, depth=1, seat=heuristic._seat_factor(state)
    )
    return float(np.clip((value - round_state.bank()) / 30.0, -1.0, 1.0))


def test_endgame_block_near_miss_without_threat():
    # banked = 185 + 12 = 197; rivals 150 (2 cards, active), 120 (3, stopped),
    # 199 (busted -> ignored for the threat).
    state = make_scored_state(
        hand=frozenset({3, 4, 5}),
        my_total=185,
        opponent_totals=(150, 120, 199),
        opponent_cards=(2, 3, 1),
        opponent_statuses=(PlayerStatus.ACTIVE, PlayerStatus.STOPPED, PlayerStatus.BUSTED),
    )
    block = _endgame_block(state)
    assert block.shape == (ENDGAME_FEATURE_COUNT,)
    assert np.isclose(block[BANK_MARGIN], -3 / 50)
    assert block[BANK_ENDS_GAME] == 0.0
    assert block[BANK_NEAR_MISS] == 1.0
    assert np.isclose(block[BEST_OPP_DISTANCE], (200 - 199) / 100)
    # threat margins: 150 + 8*3 - 200 = -26, 120 + 8*3 - 200 = -56
    assert block[RIVAL_MAY_END] == 0.0
    assert np.isclose(block[RIVAL_THREAT_MARGIN], -26 / 50)
    assert np.isclose(block[LEAD], (185 - 199) / 50)
    assert np.isclose(block[ROUND_SCORE], 12 / 50)
    assert np.isclose(block[DRAW_ADVANTAGE], _expected_advantage(state), atol=1e-6)


def test_endgame_block_bank_ends_game_with_rival_threat():
    # banked = 195 + 12 = 207; rival at 190 with 1 card still active:
    # 190 + 8*2 - 200 = +6 -> may end the game.
    state = make_scored_state(
        hand=frozenset({3, 4, 5}),
        my_total=195,
        opponent_totals=(190, 100, 50),
        opponent_cards=(1, 0, 0),
    )
    block = _endgame_block(state)
    assert np.isclose(block[BANK_MARGIN], 7 / 50)
    assert block[BANK_ENDS_GAME] == 1.0
    assert block[BANK_NEAR_MISS] == 0.0
    assert block[RIVAL_MAY_END] == 1.0
    assert np.isclose(block[RIVAL_THREAT_MARGIN], 6 / 50)
    assert np.isclose(block[LEAD], 5 / 50)


def test_endgame_block_draw_advantage_zero_when_not_active():
    state = make_scored_state(
        hand=frozenset({3, 4}), my_total=50, my_status=PlayerStatus.STOPPED
    )
    assert _endgame_block(state)[DRAW_ADVANTAGE] == 0.0


def test_endgame_block_bounded_on_real_snapshots():
    for state in collect_decision_snapshots(n_snapshots=30, seed=3):
        block = _endgame_block(state)
        assert np.all((-1.0 <= block) & (block <= 1.0))


def test_endgame_block_is_strictly_appended():
    """The game-end block only extends the vector; the base entries are unchanged."""
    for state in collect_decision_snapshots(n_snapshots=20):
        encoded = encode_game_state(state, 4)
        assert encoded.shape == (ENDGAME_OFFSET + ENDGAME_FEATURE_COUNT,)
        legacy = _legacy_encode_119(state, 4)
        assert np.array_equal(encoded[:LEGACY_SIZE], legacy)
