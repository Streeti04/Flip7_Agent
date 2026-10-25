"""Round-robin tournament for Flip 7 agents.

Usage:
    python tournament.py

Add agents to the AGENTS list at the bottom of this file.
"""

from __future__ import annotations
import itertools
from dataclasses import dataclass
import numpy as np

from flip7.agents.base_agent import BaseAgent, Action
from flip7.agents.heuristic_agent import HeuristicAgent
from flip7.agents.rl_agent import RLAgent
from flip7.game import Flip7Game, IllegalActionError
from flip7.game_state import GameState


# ---------------------------------------------------------------------------
# Built-in baseline agents (students implement their own in separate files)
# ---------------------------------------------------------------------------


class RandomAgent(BaseAgent):
    """Stop or Draw with equal probability."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng or np.random.default_rng()

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        sd = [a for a in legal_actions if a in (Action.STOP, Action.DRAW)]
        if sd:
            return Action(self._rng.choice([a.value for a in sd]))
        return Action(self._rng.choice([a.value for a in legal_actions]))


class ConservativeAgent(BaseAgent):
    """Stop as soon as 3 or more number cards are in hand."""

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions:
            return legal_actions[0]
        if len(gs.my_number_cards) >= 3:
            return Action.STOP
        return Action.DRAW


class ThresholdAgent(BaseAgent):
    """Stop when estimated bust probability exceeds 50%."""

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions:
            return legal_actions[0]
        remaining = gs.deck_remaining
        if remaining == 0:
            return Action.STOP
        bust_cards = sum(max(v, 1) - 1 for v in gs.my_number_cards)
        bust_prob = bust_cards / remaining
        if bust_prob > 0.5:
            return Action.STOP
        return Action.DRAW


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    name: str
    total_score: float = 0.0
    wins: int = 0
    games: int = 0

    @property
    def mean_score(self) -> float:
        return self.total_score / self.games if self.games else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


def run_tournament(
    agent_factories: list[tuple[str, type[BaseAgent]]],
    n_episodes: int = 1000,
    n_players: int = 4,
    seed: int = 42,
) -> list[AgentResult]:
    """Round-robin: every combination of n_players agents plays n_episodes games.

    agent_factories: list of (name, AgentClass) pairs.
    Returns a leaderboard sorted by mean score descending.
    """
    rng = np.random.default_rng(seed)
    results: dict[str, AgentResult] = {
        name: AgentResult(name=name) for name, _ in agent_factories
    }

    combos = list(itertools.combinations(range(len(agent_factories)), n_players))
    if not combos:
        base = list(range(len(agent_factories)))
        combo: tuple[int, ...] = tuple(
            (base * (n_players // len(base) + 1))[:n_players]
        )
        combos = [combo]

    for combo in combos:
        for _ in range(n_episodes):
            ep_seed = int(rng.integers(2**31))
            agents: list[BaseAgent] = []
            for seat, agent_index in enumerate(combo):
                factory = agent_factories[agent_index][1]
                # Seed the stochastic baseline without changing the constructor
                # contract expected from arbitrary student agents.
                if issubclass(factory, RandomAgent):
                    agent = factory(
                        rng=np.random.default_rng(ep_seed + seat + 1)  # type: ignore[call-arg]
                    )
                else:
                    agent = factory()
                agents.append(agent)
            names = [agent_factories[i][0] for i in combo]
            game = Flip7Game(n_players=n_players, seed=ep_seed)
            try:
                scores = game.run_game(agents)
            except IllegalActionError as exc:
                offender = names[exc.player_id]
                print(
                    f"Warning: {offender} raised IllegalActionError ({exc}); "
                    f"episode discarded, {offender} scored 0."
                )
                results[offender].total_score += 0
                results[offender].games += 1
                continue
            winning_score = max(scores.values())
            for name, pid in zip(names, scores.keys()):
                results[name].total_score += scores[pid]
                results[name].games += 1
                if scores[pid] == winning_score:
                    results[name].wins += 1

    return sorted(results.values(), key=lambda r: r.mean_score, reverse=True)


def print_leaderboard(leaderboard: list[AgentResult]) -> None:
    header = (
        f"{'Rank':<5} {'Agent':<25} {'Mean Score':>12} {'Win Rate':>10} {'Games':>8}"
    )
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(leaderboard, start=1):
        print(
            f"{rank:<5} {r.name:<25} {r.mean_score:>12.2f} {r.win_rate:>9.1%} {r.games:>8}"
        )


if __name__ == "__main__":
    AGENTS: list[tuple[str, type[BaseAgent]]] = [
        ("RandomAgent", RandomAgent),
        ("ConservativeAgent", ConservativeAgent),
        ("ThresholdAgent", ThresholdAgent),
        ("HeuristicAgent", HeuristicAgent),
        ("RLAgent", RLAgent),
    ]

    print(f"Running tournament: {len(AGENTS)} agents, 1000 episodes per pairing...\n")
    leaderboard = run_tournament(AGENTS, n_episodes=1000, n_players=min(4, len(AGENTS)))
    print_leaderboard(leaderboard)
