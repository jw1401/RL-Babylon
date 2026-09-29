# Soft Actor-Critic (SAC)

The current SAC implementation trains an agent for the continuous-control Cube-Ball demo. It uses a Gaussian actor, twin Q critics, a replay buffer, target critic updates, and automatic entropy tuning.

## Run

Start the static server described in the [Getting Started guide](../../.docs/GETTING_STARTED.md), then run from the `rl-trainers` directory:

```bash
python sac/main.py
```

The entry point currently uses a hardcoded environment URL, one environment, and training settings defined in `trainer.py`. The environment must provide numeric state observations and accept two continuous action values (`x` and `z`).

## Current limitations

- The trainer has no configuration file or command-line options yet.
- Model checkpoint saving and loading are not wired into the training loop.
- SAC is not covered by an automated test suite in this repository.
