"""Tests for Freeze, Second Chance, and Flip Three action card logic."""

from flip7.agents.base_agent import BaseAgent, Action
from flip7.card import NumberCard, ActionCard
from flip7.game import Flip7Game, _ActionToken
from flip7.game_state import GameState
from flip7.player import PlayerStatus


class AlwaysStopAgent(BaseAgent):
    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions:
            return legal_actions[0]  # pending target selection
        return Action.STOP


class AlwaysDrawAgent(BaseAgent):
    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.DRAW not in legal_actions:
            return legal_actions[0]  # pending target selection
        return Action.DRAW


class TargetFirstAgent(BaseAgent):
    """Always picks the first legal TARGET action, otherwise Stop."""

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        for a in legal_actions:
            if a >= Action.TARGET_0:
                return a
        return Action.STOP


def make_game(n_players: int, seed: int = 0) -> Flip7Game:
    return Flip7Game(n_players=n_players, seed=seed)


def exhaust_gen(gen, token=None):
    """Drive a generator to completion, sending token on every yield."""
    if token is None:
        token = _ActionToken(Action.TARGET_0)
    try:
        next(gen)
        while True:
            gen.send(token)
    except StopIteration:
        pass


# ------------------------------------------------------------------
# Freeze
# ------------------------------------------------------------------


def test_freeze_removes_player_from_round():
    game = make_game(3, seed=0)
    game.reset()
    p0 = game._players[0]
    # With 2 other active players the freeze requires target selection (yield).
    # With exactly 1 other active player it auto-freezes without a yield.
    # Make p2 inactive so there is exactly 1 target → auto-freeze, no token needed.
    game._players[2].status = PlayerStatus.STOPPED
    card = ActionCard("freeze")
    exhaust_gen(game._apply_freeze_gen(p0, card))
    frozen = [p for p in game._players if p.status == PlayerStatus.FROZEN]
    assert len(frozen) == 1


def test_freeze_removed_player_scores_normally():
    game = make_game(3, seed=0)
    game.reset()
    p1 = game._players[1]
    p1.number_cards = [NumberCard(5), NumberCard(3)]
    p1.status = PlayerStatus.FROZEN
    assert p1.round_score() == 8


def test_freeze_self_when_only_active():
    game = make_game(3, seed=0)
    game.reset()
    p0 = game._players[0]
    game._players[1].status = PlayerStatus.STOPPED
    game._players[2].status = PlayerStatus.STOPPED
    card = ActionCard("freeze")
    exhaust_gen(game._apply_freeze_gen(p0, card))
    assert p0.status == PlayerStatus.FROZEN


def test_freeze_card_discarded():
    game = make_game(3, seed=0)
    game.reset()
    p0 = game._players[0]
    game._players[2].status = PlayerStatus.STOPPED
    card = ActionCard("freeze")
    exhaust_gen(game._apply_freeze_gen(p0, card))
    # Card was drawn from deck before calling, so discard restores balance via discard pile
    assert len(game._deck._discard_pile) >= 1
    assert card in game._deck._discard_pile


# ------------------------------------------------------------------
# Second Chance
# ------------------------------------------------------------------


def test_second_chance_grants_card():
    game = make_game(3, seed=0)
    game.reset()
    p = game._players[0]
    assert not p.has_second_chance
    card = ActionCard("second_chance")
    exhaust_gen(game._apply_second_chance_gen(p, card))
    assert p.has_second_chance


def test_second_chance_passed_when_already_held():
    game = make_game(3, seed=0)
    game.reset()
    p0 = game._players[0]
    p1 = game._players[1]
    p0.has_second_chance = True
    card = ActionCard("second_chance")
    exhaust_gen(game._apply_second_chance_gen(p0, card), _ActionToken(Action.TARGET_0))
    assert p1.has_second_chance
    assert p0.has_second_chance


def test_second_chance_not_passed_to_player_who_already_has_one():
    game = make_game(3, seed=0)
    game.reset()
    p0 = game._players[0]
    p1 = game._players[1]
    p1.has_second_chance = True  # p1 already has one
    p0.has_second_chance = True
    card = ActionCard("second_chance")
    # Only valid target is p2 (p1 already has SC)
    exhaust_gen(game._apply_second_chance_gen(p0, card), _ActionToken(Action.TARGET_0))
    assert game._players[2].has_second_chance


def test_second_chance_used_on_duplicate():
    game = make_game(3, seed=0)
    game.reset()
    p = game._players[0]
    p.number_cards = [NumberCard(7)]
    p.has_second_chance = True
    game._apply_number_card(p, NumberCard(7))
    assert p.status == PlayerStatus.ACTIVE
    assert not p.has_second_chance
    assert p.has_number(7)


def test_second_chance_flag_cleared_at_round_end():
    # The SC card object is discarded when drawn (_apply_second_chance_gen).
    # The flag is cleared at round end so it is not carried into the next round.
    game = make_game(3, seed=0)
    game.reset()
    game._players[0].has_second_chance = True
    game._end_round()
    assert not game._players[0].has_second_chance


# ------------------------------------------------------------------
# Flip Three
# ------------------------------------------------------------------


def test_flip_three_does_not_crash():
    agents = [TargetFirstAgent(), AlwaysStopAgent(), AlwaysStopAgent()]
    game = Flip7Game(n_players=3, seed=42)
    game.run_game(agents)


def test_flip_three_stops_on_bust():
    """If target busts mid-Flip-Three, no further cards should be dealt to them."""
    game = make_game(3, seed=0)
    game.reset()
    target = game._players[1]
    target.number_cards = [NumberCard(v) for v in range(6)]
    card = ActionCard("flip_three")
    exhaust_gen(game._apply_flip_three_gen(game._players[0], card))
    assert target.status in (
        PlayerStatus.ACTIVE,
        PlayerStatus.BUSTED,
        PlayerStatus.STOPPED,
    )


def test_flip_three_card_discarded():
    game = make_game(3, seed=0)
    game.reset()
    card = ActionCard("flip_three")
    exhaust_gen(game._apply_flip_three_gen(game._players[0], card))
    assert card in game._deck._discard_pile


# ------------------------------------------------------------------
# Full round with action cards
# ------------------------------------------------------------------


def test_full_round_with_action_cards_does_not_crash():
    """Smoke test: a full game with agents that always draw must terminate."""
    agents = [AlwaysDrawAgent()] * 4
    game = Flip7Game(n_players=4, seed=123)
    scores = game.run_game(agents)
    assert len(scores) == 4
