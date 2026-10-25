"""Prints a Flip 7 game move by move — hands, draws, busts, and round scores.

Run this before writing your own Gym wrapper or agent to see what a game
actually looks like: what each player holds, when a bust happens, how the
round score is computed, and how a game spans multiple rounds until someone
reaches 200 points.

Usage:
    python demo.py [--seed SEED] [--players N] [--rounds N]
"""

from __future__ import annotations
import argparse

from flip7.action import Action
from flip7.game import Flip7Game
from flip7.player import PlayerStatus
from tournament import RandomAgent, ConservativeAgent, ThresholdAgent


def describe_hand(game: Flip7Game, player_id: int) -> str:
    player = game._players[player_id]
    numbers = sorted(c.value for c in player.number_cards)
    bonuses = [f"{b.bonus_type}" for b in player.bonus_cards]
    parts = [f"numbers={numbers}"]
    if bonuses:
        parts.append(f"bonuses={bonuses}")
    if player.has_second_chance:
        parts.append("second_chance")
    return ", ".join(parts)


def play_one_round(game: Flip7Game, agents: list, names: list[str], result):
    """Play decisions until the current round ends. Returns the round-boundary
    StepResult (round_over=True) — game_over may or may not be True."""
    prev_status = {i: PlayerStatus.ACTIVE for i in range(len(agents))}

    while not result.round_over:
        pid = result.acting_player_id
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        print(f"{names[pid]}: hand before action -> {describe_hand(game, pid)}")
        print(f"  -> chooses {action.name}")

        result = game.step(action)

        player = game._players[pid]
        if (
            player.status == PlayerStatus.BUSTED
            and prev_status[pid] == PlayerStatus.ACTIVE
        ):
            print(f"  BUST! {names[pid]} scores 0 for this round.")
        prev_status[pid] = player.status

    return result


def run_demo(n_players: int, seed: int, max_rounds: int) -> None:
    agent_factories = [RandomAgent, ConservativeAgent, ThresholdAgent, RandomAgent]
    agents = [agent_factories[i % len(agent_factories)]() for i in range(n_players)]
    names = [f"P{i} ({type(a).__name__})" for i, a in enumerate(agents)]

    game = Flip7Game(n_players=n_players, seed=seed)
    result = game.reset()
    print(f"=== Game start, seed={seed}, players={names} ===")

    round_num = 0
    while not result.game_over and round_num < max_rounds:
        round_num += 1
        print(f"\n--- Round {round_num} ---")
        result = play_one_round(game, agents, names, result)

        round_scores = result.game_state.round_scores or {}
        print(f"--- Round {round_num} over ---")
        for pid, score in round_scores.items():
            print(
                f"{names[pid]}: round_score={score}, "
                f"total_score={game._players[pid].total_score}"
            )

        if result.game_over:
            break
        if round_num < max_rounds:
            # Drain the intermediate round-boundary notification before the
            # next round's first decision can be requested.
            result = game.step(Action.STOP)

    if result.game_over:
        print(f"\n=== Game over after {round_num} round(s)! ===")
        print(f"Final scores: {result.final_scores}")
    else:
        print(f"\n=== Stopped after {round_num} round(s) (--rounds limit reached) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="max number of rounds to show (the game may end sooner)",
    )
    args = parser.parse_args()
    run_demo(n_players=args.players, seed=args.seed, max_rounds=args.rounds)
