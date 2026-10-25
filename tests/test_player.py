from flip7.card import NumberCard, BonusCard
from flip7.player import Player, PlayerStatus


def make_player(id: int = 0) -> Player:
    return Player(id=id)


# --- Scoring ---


def test_score_number_cards_only():
    p = make_player()
    p.number_cards = [NumberCard(3), NumberCard(5), NumberCard(7)]
    assert p.round_score() == 15


def test_score_with_x2_applied_before_bonuses():
    p = make_player()
    p.number_cards = [NumberCard(4)]
    p.bonus_cards = [BonusCard("x2", 0), BonusCard("+6", 6)]
    # (4 × 2) + 6 = 14
    assert p.round_score() == 14


def test_score_bonus_only_no_x2():
    p = make_player()
    p.number_cards = [NumberCard(2)]
    p.bonus_cards = [BonusCard("+4", 4), BonusCard("+8", 8)]
    assert p.round_score() == 14


def test_score_zero_card():
    p = make_player()
    p.number_cards = [NumberCard(0)]
    assert p.round_score() == 0


def test_score_busted_is_zero():
    p = make_player()
    p.number_cards = [NumberCard(10)]
    p.status = PlayerStatus.BUSTED
    assert p.round_score() == 0


def test_score_flip7_adds_15():
    p = make_player()
    p.number_cards = [NumberCard(v) for v in range(7)]
    p.flip7_bonus = True
    base = sum(range(7))
    assert p.round_score() == base + 15


def test_score_x2_then_bonus_then_flip7():
    p = make_player()
    p.number_cards = [NumberCard(v) for v in range(7)]  # 0+1+…+6 = 21
    p.bonus_cards = [BonusCard("x2", 0), BonusCard("+10", 10)]
    p.flip7_bonus = True
    # (21 × 2) + 10 + 15 = 67
    assert p.round_score() == 67


# --- State helpers ---


def test_has_number():
    p = make_player()
    p.number_cards = [NumberCard(3), NumberCard(7)]
    assert p.has_number(3)
    assert not p.has_number(5)


def test_number_values():
    p = make_player()
    p.number_cards = [NumberCard(1), NumberCard(5)]
    assert p.number_values() == frozenset({1, 5})


def test_is_in_round_only_when_active():
    p = make_player()
    assert p.is_in_round()
    p.status = PlayerStatus.STOPPED
    assert not p.is_in_round()


def test_reset_round():
    p = make_player()
    p.total_score = 50
    p.number_cards = [NumberCard(9)]
    p.bonus_cards = [BonusCard("+2", 2)]
    p.has_second_chance = True
    p.status = PlayerStatus.BUSTED
    p.flip7_bonus = True
    p.reset_round()
    assert p.number_cards == []
    assert p.bonus_cards == []
    assert not p.has_second_chance
    assert p.status == PlayerStatus.ACTIVE
    assert not p.flip7_bonus
    assert p.total_score == 50  # preserved across rounds
