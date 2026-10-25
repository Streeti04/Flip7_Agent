# Submissions

Each student submits one folder here containing a single `agent.py` file:

```
submissions/
├── jdoe/
│   └── agent.py
├── asmith/
│   └── agent.py
└── ...
```

`agent.py` must define exactly one class named `Agent` that subclasses
`flip7.agents.base_agent.BaseAgent`:

```python
from flip7.agents.base_agent import BaseAgent, Action

class Agent(BaseAgent):
    def act(self, gs, legal_actions):
        return Action.DRAW
```

The folder name becomes the agent's display name on the leaderboard.

`act()` must return a value from `legal_actions` (also available as `gs.legal_actions`).
Returning anything else — e.g. `STOP`/`DRAW` while a `TARGET_*` selection is pending — raises
`IllegalActionError` and scores that single episode as 0 for your agent; it does not stop the
tournament or affect other episodes.

Run the full tournament over all submissions with:

```bash
python run_submission_tournament.py
```
