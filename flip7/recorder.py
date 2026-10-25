"""Game recording and replay for Flip 7.

A ``GameRecord`` stores everything needed to deterministically reconstruct a
game: the seed, player count, agent names, and the full sequence of
(player_id, action) decisions. Since ``Flip7Game`` only draws randomness from
its seeded ``np.random.Generator``, replaying the same actions against a game
constructed with the same seed reproduces the exact same sequence of
``GameState`` snapshots.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .action import Action
from .agents.base_agent import BaseAgent
from .game import Flip7Game, IllegalActionError


@dataclass
class GameRecord:
    seed: int
    n_players: int
    agent_names: list[str]
    actions: list[tuple[int, int]] = field(
        default_factory=list
    )  # (player_id, action_value)
    final_scores: dict[int, int] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(data: str) -> "GameRecord":
        obj = json.loads(data)
        obj["actions"] = [tuple(a) for a in obj["actions"]]
        if obj["final_scores"] is not None:
            obj["final_scores"] = {int(k): v for k, v in obj["final_scores"].items()}
        return GameRecord(**obj)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @staticmethod
    def load(path: str | Path) -> "GameRecord":
        return GameRecord.from_json(Path(path).read_text())


def record_game(
    agents: list[BaseAgent], agent_names: list[str], seed: int
) -> tuple[dict[int, int], GameRecord]:
    """Play one game while recording every (player_id, action) decision.

    Returns (final_scores, GameRecord). The record can later be replayed with
    ``replay_game()`` to reconstruct the exact sequence of GameStates.

    Intermediate round boundaries (StepResult.round_over=True,
    game_over=False) are drained internally with a placeholder action and are
    NOT appended to ``record.actions`` — only real agent decisions are
    recorded, since those are all ``replay_game()`` needs to reconstruct the
    game deterministically.

    Raises IllegalActionError if an agent chooses an action outside
    game_state.legal_actions. The partial record up to (and including) the
    offending action is available on the exception as ``partial_record``, so
    callers can still save it for debugging.
    """
    game = Flip7Game(n_players=len(agents), seed=seed)
    record = GameRecord(seed=seed, n_players=len(agents), agent_names=list(agent_names))

    result = game.reset()
    while True:
        pid = result.acting_player_id
        gs = result.game_state
        action = agents[pid].act(gs, list(gs.legal_actions))
        record.actions.append((pid, int(action)))
        prev_gs = gs
        try:
            result = game.step(action)
        except IllegalActionError as exc:
            exc.partial_record = record
            raise
        agents[pid].observe(
            prev_gs, action, result.game_state, result.reward, result.round_over
        )
        if result.game_over:
            break
        while result.round_over and not result.game_over:
            try:
                result = game.step(Action.STOP)
            except IllegalActionError as exc:
                exc.partial_record = record
                raise
            if result.game_over:
                break

    record.final_scores = result.final_scores
    return result.final_scores, record


def replay_game(record: GameRecord):
    """Replay a recorded game step by step, yielding (StepResult, action_taken).

    Reconstructs the same Flip7Game (same seed, same n_players) and feeds back
    the exact recorded actions instead of consulting agents. Useful to inspect
    or visualize how a past tournament game unfolded.

    ``record.actions`` only contains real agent decisions (see
    ``record_game()``) — intermediate round-boundary notifications
    (StepResult.round_over=True, game_over=False) are not recorded as
    actions, but they ARE yielded here (with ``action=None``, like the final
    StepResult) so a consumer can still see every round's ``round_scores``
    before the notification is drained with a placeholder step and replay
    continues with the next recorded action.
    """
    game = Flip7Game(n_players=record.n_players, seed=record.seed)
    result = game.reset()
    for pid, action_value in record.actions:
        while result.round_over and not result.game_over:
            yield result, None
            result = game.step(Action.STOP)  # drain the pending boundary
        if result.acting_player_id != pid:
            raise ValueError(
                f"Replay mismatch: expected player {result.acting_player_id}, "
                f"recorded action was for player {pid}."
            )
        action = Action(action_value)
        yield result, action
        result = game.step(action)
    yield result, None
