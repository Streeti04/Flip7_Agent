"""Tests for submission_loader — discovering and importing student agent.py files."""

import pytest
from flip7.agents.base_agent import BaseAgent
from submission_loader import load_submissions

VALID_AGENT = """
from flip7.agents.base_agent import BaseAgent, Action

class Agent(BaseAgent):
    def act(self, gs, legal_actions):
        return legal_actions[0]
"""

MISSING_CLASS_AGENT = """
from flip7.agents.base_agent import BaseAgent

class NotAgent(BaseAgent):
    def act(self, gs, legal_actions):
        return legal_actions[0]
"""

NOT_A_BASEAGENT = """
class Agent:
    def act(self, gs, legal_actions):
        return legal_actions[0]
"""

BROKEN_IMPORT = """
raise RuntimeError("boom")
"""


def _write_submission(tmp_path, folder_name, source):
    d = tmp_path / folder_name
    d.mkdir()
    (d / "agent.py").write_text(source)
    return d


def test_loads_valid_submission(tmp_path):
    _write_submission(tmp_path, "student_a", VALID_AGENT)
    agents = load_submissions(tmp_path)
    assert len(agents) == 1
    name, cls = agents[0]
    assert name == "student_a"
    assert issubclass(cls, BaseAgent)


def test_skips_underscore_prefixed_folders(tmp_path):
    _write_submission(tmp_path, "_example", VALID_AGENT)
    agents = load_submissions(tmp_path)
    assert agents == []


def test_multiple_submissions_sorted_by_name(tmp_path):
    _write_submission(tmp_path, "zeta", VALID_AGENT)
    _write_submission(tmp_path, "alpha", VALID_AGENT)
    agents = load_submissions(tmp_path)
    assert [name for name, _ in agents] == ["alpha", "zeta"]


def test_missing_agent_file_raises(tmp_path):
    d = tmp_path / "student_b"
    d.mkdir()
    with pytest.raises(RuntimeError, match="no agent.py"):
        load_submissions(tmp_path)


def test_missing_agent_class_raises(tmp_path):
    _write_submission(tmp_path, "student_c", MISSING_CLASS_AGENT)
    with pytest.raises(RuntimeError, match="Agent"):
        load_submissions(tmp_path)


def test_agent_not_subclassing_base_agent_raises(tmp_path):
    _write_submission(tmp_path, "student_d", NOT_A_BASEAGENT)
    with pytest.raises(RuntimeError, match="Agent"):
        load_submissions(tmp_path)


def test_broken_import_raises(tmp_path):
    _write_submission(tmp_path, "student_e", BROKEN_IMPORT)
    with pytest.raises(RuntimeError, match="raised on import"):
        load_submissions(tmp_path)
