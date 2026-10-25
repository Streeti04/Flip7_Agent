"""Discovers and loads student agent submissions from submissions/<name>/agent.py.

Each submission folder must contain an agent.py defining exactly one class
named `Agent` that subclasses BaseAgent. Folders starting with "_" (e.g. the
bundled _example) are skipped.
"""

from __future__ import annotations
import importlib.util
from pathlib import Path

from flip7.agents.base_agent import BaseAgent

SUBMISSIONS_DIR = Path(__file__).parent / "submissions"


def load_submissions(
    submissions_dir: Path | str = SUBMISSIONS_DIR,
) -> list[tuple[str, type[BaseAgent]]]:
    """Import every submissions/<name>/agent.py and return [(name, AgentClass), ...].

    Raises RuntimeError (with the offending folder name) if a submission's
    agent.py is missing, fails to import, or does not define a class `Agent`
    subclassing BaseAgent.
    """
    submissions_dir = Path(submissions_dir)
    agents: list[tuple[str, type[BaseAgent]]] = []

    for folder in sorted(submissions_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        agent_file = folder / "agent.py"
        if not agent_file.exists():
            raise RuntimeError(f"{folder.name}: no agent.py found.")

        module_name = f"submissions.{folder.name}.agent"
        spec = importlib.util.spec_from_file_location(module_name, agent_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"{folder.name}: could not load agent.py.")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise RuntimeError(
                f"{folder.name}: agent.py raised on import: {exc}"
            ) from exc

        agent_cls = getattr(module, "Agent", None)
        if agent_cls is None or not (
            isinstance(agent_cls, type) and issubclass(agent_cls, BaseAgent)
        ):
            raise RuntimeError(
                f"{folder.name}: agent.py must define a class `Agent(BaseAgent)`."
            )
        agents.append((folder.name, agent_cls))

    return agents
