# Babylon.js environments

This directory contains the browser environments and shared Babylon.js support files used by RL-Babylon. Serve this directory over HTTP so browser pages can load their scripts and assets.

From the repository root, a simple local server can be started with:

```bash
python -m http.server 5500 --directory .environments
```

Then open an environment page, for example:

```text
http://localhost:5500/babylon-environments/Cube-Ball/Vector-Obs/Environment.html
```

## Included environments

- `Cube-Ball/Vector-Obs` — vector observations
- `Cube-Ball/Visual-Obs` — image observations
- `Cube-Ball/Continous-SAC-Demo` — continuous-action demo used by the current SAC entry point
- `Cube-Ball/Blender-Demo`
- `Cart-Pole`
- `Balancing-Ball`
- `Lunar-Lander`

To train, start the appropriate Python trainer from `rl-trainers` after starting the static server. The browser environment communicates with the trainer using the project's WebSocket/BSON protocol. See the [Getting Started guide](../.docs/GETTING_STARTED.md) for trainer commands and compatibility details.

When creating an environment, implement the scene setup and the reset/step behavior, and return data in the protocol format expected by the Python client. Existing pages and scripts in `babylon-environments/` and `lib/` are the practical references.
