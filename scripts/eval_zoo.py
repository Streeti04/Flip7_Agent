#!/usr/bin/env python3
"""Evaluate focal agents against a zoo of plausible foreign agents.

Default mode: round-robin over every 4-player combination drawn from
[focal agents + zoo + 3 baselines], run once per seed and aggregated
(report per seed + mean). --head-to-head A,B,... plays only the named
agents, padded with baselines to 4 players.

--robust N runs a robustness check per focal agent: N games against random
zoo mixes (seed 5150), counting IllegalActionError (attributed via
exc.player_id) and any other exception, and measuring act() latency of the
focal agent through a TimingAgent wrapper.

Output: CSV (default Ergebnisse/zoo_evaluation.csv) with columns
agent, seed, games, mean_score, win_rate, rank — one row per seed plus a
"mean" row per agent — and a markdown table on stdout sorted by mean score.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flip7.agents.base_agent import Action, BaseAgent
from flip7.agents.heuristic_agent import HeuristicAgent, HeuristicConfig
from flip7.agents.heuristic_v1 import HeuristicAgentV1
from flip7.agents.hybrid_agent import HybridAgent
from flip7.agents.rl_agent import RLAgent
from flip7.game import Flip7Game, IllegalActionError
from flip7.game_state import GameState
from tournament import (
    AgentResult,
    ConservativeAgent,
    RandomAgent,
    ThresholdAgent,
    run_tournament,
)

ROBUST_SEED = 5150


# ---------------------------------------------------------------------------
# Zoo: plausible foreign agents (stand-ins for unknown student submissions)
# ---------------------------------------------------------------------------


class ThresholdT(BaseAgent):
    """Stop once the estimated bust probability exceeds ``threshold``."""

    threshold: float = 0.5

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions:
            return legal_actions[0]
        remaining = gs.deck_remaining
        if remaining == 0:
            return Action.STOP
        bust_cards = sum(max(v, 1) - 1 for v in gs.my_number_cards)
        return Action.STOP if bust_cards / remaining > self.threshold else Action.DRAW


class ThresholdT02(ThresholdT):
    threshold = 0.2


class ThresholdT03(ThresholdT):
    threshold = 0.3


class ThresholdT04(ThresholdT):
    threshold = 0.4


class ThresholdT06(ThresholdT):
    threshold = 0.6


class UltraConservative(BaseAgent):
    """Stop as soon as two number cards are in hand."""

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions:
            return legal_actions[0]
        return Action.STOP if len(gs.my_number_cards) >= 2 else Action.DRAW


class Flip7Hunter(BaseAgent):
    """Draw until five number cards, then stop (chases the Flip 7 bonus)."""

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions:
            return legal_actions[0]
        return Action.STOP if len(gs.my_number_cards) >= 5 else Action.DRAW


class DrawBias(RandomAgent):
    """Draw with fixed probability ``draw_probability`` (rng-seeded like RandomAgent)."""

    draw_probability: float = 0.5

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        stop_or_draw = [a for a in legal_actions if a in (Action.STOP, Action.DRAW)]
        if stop_or_draw:
            return Action.DRAW if self._rng.random() < self.draw_probability else Action.STOP
        return Action(self._rng.choice([a.value for a in legal_actions]))


class DrawBias03(DrawBias):
    draw_probability = 0.3


class DrawBias07(DrawBias):
    draw_probability = 0.7


class HeuristicVarA(HeuristicAgent):
    """Jitter: high action-card values, aggressive targeting."""

    def __init__(self) -> None:
        super().__init__(HeuristicConfig(
            freeze_value=1.2,
            flip_three_value=8.0,
            freeze_weight_flip7=6.0,
            freeze_weight_second_chance=0.6,
            freeze_weight_hand=2.0,
            freeze_weight_score=0.02,
            flip_three_denial_value=8.0,
            flip_three_gift_penalty=6.0,
        ))


class HeuristicVarB(HeuristicAgent):
    """Jitter: low action-card values, near-Flip-7-focused and gift-averse."""

    def __init__(self) -> None:
        super().__init__(HeuristicConfig(
            freeze_value=0.15,
            flip_three_value=2.0,
            freeze_weight_flip7=12.0,
            freeze_weight_second_chance=0.1,
            freeze_weight_hand=0.5,
            freeze_weight_score=0.08,
            flip_three_denial_value=3.0,
            flip_three_gift_penalty=16.0,
        ))


class HeuristicVarC(HeuristicAgent):
    """Jitter: score-driven freeze weights, moderate Flip Three handling."""

    def __init__(self) -> None:
        super().__init__(HeuristicConfig(
            freeze_value=0.6,
            flip_three_value=5.0,
            freeze_weight_flip7=9.7,
            freeze_weight_second_chance=0.353,
            freeze_weight_hand=1.12,
            freeze_weight_score=0.15,
            flip_three_denial_value=5.473,
            flip_three_gift_penalty=11.397,
        ))


class TargeterDefault(ThresholdAgent):
    """Threshold stop/draw, but always picks the first offered target."""

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        if Action.STOP not in legal_actions:
            if Action.TARGET_0 in legal_actions:
                return Action.TARGET_0
            return legal_actions[0]
        return super().act(gs, legal_actions)


ZOO: dict[str, type[BaseAgent]] = {
    "threshold_02": ThresholdT02,
    "threshold_03": ThresholdT03,
    "threshold_04": ThresholdT04,
    "threshold_06": ThresholdT06,
    "ultra_conservative": UltraConservative,
    "flip7_hunter": Flip7Hunter,
    "draw_bias_03": DrawBias03,
    "draw_bias_07": DrawBias07,
    "heuristic_var_a": HeuristicVarA,
    "heuristic_var_b": HeuristicVarB,
    "heuristic_var_c": HeuristicVarC,
    "heuristic_v1": HeuristicAgentV1,
    "targeter_default": TargeterDefault,
}

BASELINES: dict[str, type[BaseAgent]] = {
    "random": RandomAgent,
    "conservative": ConservativeAgent,
    "threshold": ThresholdAgent,
}

FOCUS_REGISTRY: dict[str, type[BaseAgent]] = {
    "heuristic": HeuristicAgent,
    "heuristic_v1": HeuristicAgentV1,
    "rl": RLAgent,
}


def _bound_factory(builder: Callable[[], BaseAgent]) -> type[BaseAgent]:
    """Wrap a zero-arg builder (e.g. RLAgent(model_path=...)) as an agent class."""

    class BoundAgent(BaseAgent):
        def __init__(self) -> None:
            self._agent = builder()

        def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
            return self._agent.act(gs, legal_actions)

        def observe(self, gs, action, next_gs, reward, round_over) -> None:
            self._agent.observe(gs, action, next_gs, reward, round_over)

        @property
        def model_loaded(self) -> bool:
            # Surface the wrapped agent's model status for diagnostics.
            return bool(getattr(self._agent, "model_loaded", True))

    return BoundAgent


def resolve_agent(token: str) -> type[BaseAgent]:
    if token.startswith("rl:"):
        path = token[3:]
        return _bound_factory(lambda: RLAgent(model_path=path))
    if token.startswith("hybrid:"):
        path = token[7:]
        return _bound_factory(lambda: HybridAgent(model_path=path))
    if token == "hybrid":
        return HybridAgent
    for registry in (FOCUS_REGISTRY, ZOO, BASELINES):
        if token in registry:
            return registry[token]
    known = sorted(set(FOCUS_REGISTRY) | set(ZOO) | set(BASELINES))
    raise SystemExit(
        f"Unknown agent '{token}'. Available: {', '.join(known)}, rl:<path>, "
        "hybrid:<path>, all"
    )


# ---------------------------------------------------------------------------
# Multi-seed tournament (reuses tournament.run_tournament per seed)
# ---------------------------------------------------------------------------


def aggregate(per_seed: dict[int, list[AgentResult]]) -> list[AgentResult]:
    totals: dict[str, AgentResult] = {}
    for leaderboard in per_seed.values():
        for result in leaderboard:
            agg = totals.setdefault(result.name, AgentResult(name=result.name))
            agg.total_score += result.total_score
            agg.wins += result.wins
            agg.games += result.games
    return sorted(totals.values(), key=lambda r: r.mean_score, reverse=True)


def print_markdown(
    totals: list[AgentResult],
    per_seed: dict[int, list[AgentResult]],
    seeds: list[int],
) -> None:
    by_seed = {seed: {r.name: r for r in lb} for seed, lb in per_seed.items()}
    header = (
        f"| Rank | Agent | Mean | "
        + " | ".join(f"Seed {s}" for s in seeds)
        + " | WinRate | Games |"
    )
    print(header)
    print("|" + "---|" * (4 + len(seeds) + 2))
    for rank, agg in enumerate(totals, start=1):
        seed_cells = " | ".join(f"{by_seed[s][agg.name].mean_score:.2f}" for s in seeds)
        print(
            f"| {rank} | {agg.name} | {agg.mean_score:.2f} | {seed_cells} "
            f"| {agg.win_rate:.1%} | {agg.games} |"
        )


def write_csv(
    path: Path,
    totals: list[AgentResult],
    per_seed: dict[int, list[AgentResult]],
) -> None:
    overall_rank = {agg.name: rank for rank, agg in enumerate(totals, start=1)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent", "seed", "games", "mean_score", "win_rate", "rank"])
        for agg in totals:
            writer.writerow(
                [
                    agg.name,
                    "mean",
                    agg.games,
                    f"{agg.mean_score:.4f}",
                    f"{agg.win_rate:.4f}",
                    overall_rank[agg.name],
                ]
            )
            for seed, leaderboard in per_seed.items():
                result = next(r for r in leaderboard if r.name == agg.name)
                seed_rank = leaderboard.index(result) + 1
                writer.writerow(
                    [
                        agg.name,
                        seed,
                        result.games,
                        f"{result.mean_score:.4f}",
                        f"{result.win_rate:.4f}",
                        seed_rank,
                    ]
                )


# ---------------------------------------------------------------------------
# Robustness check: random zoo mixtures + exception counting + act() timing
# ---------------------------------------------------------------------------


class TimingAgent(BaseAgent):
    """Delegate to an inner agent and record every act() duration."""

    def __init__(self, inner: BaseAgent) -> None:
        self._inner = inner
        self.durations: list[float] = []

    def act(self, gs: GameState, legal_actions: list[Action]) -> Action:
        start = time.perf_counter()
        try:
            return self._inner.act(gs, legal_actions)
        finally:
            self.durations.append(time.perf_counter() - start)

    def observe(self, gs, action, next_gs, reward, round_over) -> None:
        self._inner.observe(gs, action, next_gs, reward, round_over)


def _spawn(cls: type[BaseAgent], seat: int, ep_seed: int) -> BaseAgent:
    if issubclass(cls, RandomAgent):
        return cls(rng=np.random.default_rng(ep_seed + seat + 1))
    return cls()


def robustness_check(
    focus: list[tuple[str, type[BaseAgent]]],
    foreign: list[tuple[str, type[BaseAgent]]],
    n_games: int,
) -> None:
    rng = np.random.default_rng(ROBUST_SEED)
    for name, factory in focus:
        illegal_focus = illegal_other = other_errors = 0
        offenders: Counter[str] = Counter()
        durations: list[float] = []
        for _ in range(n_games):
            focus_seat = int(rng.integers(4))
            picks = rng.choice(len(foreign), size=3, replace=False)
            opponents = [foreign[i] for i in picks]
            ep_seed = int(rng.integers(2**31))
            timer = TimingAgent(factory())
            agents: list[BaseAgent] = []
            seat_names: list[str] = []
            k = 0
            for seat in range(4):
                if seat == focus_seat:
                    agents.append(timer)
                    seat_names.append(name)
                else:
                    opponent_name, opponent_cls = opponents[k]
                    k += 1
                    agents.append(_spawn(opponent_cls, seat, ep_seed))
                    seat_names.append(opponent_name)
            try:
                Flip7Game(n_players=4, seed=ep_seed).run_game(agents)
            except IllegalActionError as exc:
                if exc.player_id == focus_seat:
                    illegal_focus += 1
                else:
                    illegal_other += 1
                    offenders[seat_names[exc.player_id]] += 1
            except Exception:
                other_errors += 1
            durations.extend(timer.durations)

        stats = np.asarray(durations)
        p50, p99, mx = (
            (np.percentile(stats, 50), np.percentile(stats, 99), stats.max())
            if stats.size
            else (0.0, 0.0, 0.0)
        )
        total_errors = illegal_focus + illegal_other + other_errors
        verdict = "OK" if total_errors == 0 else "FAIL"
        detail = f"focus={illegal_focus}, others={illegal_other}"
        if offenders:
            detail += " [" + ", ".join(
                f"{agent}×{count}" for agent, count in offenders.most_common()
            ) + "]"
        print(
            f"{name:<18} {verdict}  games={n_games} errors={total_errors} "
            f"(illegal: {detail}; other_exceptions={other_errors}) "
            f"act_calls={stats.size} "
            f"p50={p50:.3e}s p99={p99:.3e}s max={mx:.3e}s"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def warn_on_missing_model(focus: list[tuple[str, type[BaseAgent]]]) -> None:
    for name, factory in focus:
        if (
        (name == "rl" or name.startswith("rl:") or name == "hybrid"
         or name.startswith("hybrid:"))
        and not factory().model_loaded
    ):
            print(f"WARNING: {name}: no compatible rl_actor.npz found — "
                  f"RLAgent silently falls back to HeuristicAgent.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--focus", default="heuristic,rl",
                        help="comma-separated focal agents: registry names, zoo "
                             "names, baselines, rl:<path>, or 'all'")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", default="42,1337,2026")
    parser.add_argument("--output", type=Path, default=Path("Ergebnisse/zoo_evaluation.csv"))
    parser.add_argument("--robust", type=int, default=200,
                        help="robustness games per focal agent (0 disables)")
    parser.add_argument("--head-to-head", dest="head_to_head", default=None,
                        help="comma-separated agents; padded with baselines to 4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.head_to_head:
        tokens = list(dict.fromkeys(t.strip() for t in args.head_to_head.split(",") if t.strip()))
        pool = {token: resolve_agent(token) for token in tokens}
        focus = list(pool.items())
        for baseline_name, cls in BASELINES.items():
            if len(pool) >= 4:
                break
            pool.setdefault(baseline_name, cls)
    else:
        tokens = [t.strip() for t in args.focus.split(",") if t.strip()]
        if tokens == ["all"]:
            tokens = list(dict.fromkeys([*FOCUS_REGISTRY, *ZOO, *BASELINES]))
        focus = [(token, resolve_agent(token)) for token in dict.fromkeys(tokens)]
        pool = dict(focus)
        pool.update(ZOO)
        pool.update(BASELINES)
    pool_list = list(pool.items())
    warn_on_missing_model(focus)

    pairings = math.comb(len(pool_list), 4)
    print(f"Pool ({len(pool_list)} agents, {pairings} pairings per seed): "
          f"{', '.join(name for name, _ in pool_list)}")

    per_seed: dict[int, list[AgentResult]] = {}
    for seed in seeds:
        print(f"\n[seed {seed}] {args.episodes} episodes × {pairings} pairings ...",
              flush=True)
        per_seed[seed] = run_tournament(
            pool_list, n_episodes=args.episodes, n_players=4, seed=seed
        )

    totals = aggregate(per_seed)
    print(f"\n### Zoo evaluation ({args.episodes} episodes/seed, seeds {seeds})\n")
    print_markdown(totals, per_seed, seeds)
    write_csv(args.output, totals, per_seed)
    print(f"\nCSV written to {args.output}")

    if args.robust > 0:
        foreign = [*ZOO.items(), *BASELINES.items()]
        print(f"\n### Robustness ({args.robust} games vs random zoo mixes, "
              f"rng seed {ROBUST_SEED}; errors=0 erwartet)")
        robustness_check(focus, foreign, args.robust)

    print(f"\nTotal runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
