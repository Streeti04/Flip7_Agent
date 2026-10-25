# Flip 7 — Game Engine and Reinforcement Learning Agent

A deterministic Python implementation of Flip 7 with pluggable agents, a Gymnasium
training environment, tournament tools, and a trained MaskablePPO policy.

## Features

- Seeded game engine for 3–18 players with a step-based API and replayable games.
- Partial observability: agents see public state and opponents' card counts, not their hands.
- Baseline and depth-2 expectimax agents, plus a trained RL agent.
- Fixed-length observation encoding with legal-action masking.
- Standalone tournament submissions and a NumPy inference path that does not need PyTorch.

## Install

Python 3.9 or newer is required.

```bash
git clone https://github.com/Streeti04/Flip7_Agent.git
cd Flip7_Agent
pip install -e '.[dev]'
```

To train the policy, install the optional RL dependencies:

```bash
pip install -e '.[rl]'
python scripts/train_rl.py
```

The checked-in actor weights are in `Ergebnisse/rl_actor.npz`. Game play and tournament
inference use NumPy; the training dependencies are only needed to train a new policy.

## Run

```bash
python demo.py
python tournament.py
python run_submission_tournament.py
pytest
```

The test suite covers the rules engine, observations, environment, agents, and generated
standalone submissions.

## Project layout

```text
flip7/            game engine, observations, environment, and agents
scripts/          training, tuning, evaluation, and submission generation
submissions/      standalone tournament agents
Ergebnisse/       trained actor weights used by the RL agent
tests/            automated tests
```
