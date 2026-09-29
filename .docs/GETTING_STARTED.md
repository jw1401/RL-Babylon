# Getting started

This guide uses the files currently present in this repository. It walks through launching a Babylon.js environment and connecting a PPO trainer. The SAC trainer is described separately below.

## Requirements

- Python 3 with a PyTorch-compatible version
- Google Chrome (the trainer attempts to open Chrome automatically)
- A local static web server

From the repository root, install the Python packages:

```bash
python -m pip install -r requirements.txt
```

## Start an environment server

The HTML environments need to be served over HTTP. From the repository root, run:

```bash
python -m http.server 5500 --directory .environments
```

This serves the environment files under `http://localhost:5500/babylon-environments/...`. Keep this terminal open. You can open an environment manually to check it, for example:

```text
http://localhost:5500/babylon-environments/Cube-Ball/Vector-Obs/Environment.html
```

## Train with PPO

In a second terminal, change to `rl-trainers`:

```bash
cd rl-trainers
```

Run PPO with numeric observations:

```bash
python ppo/main.py --trainer vector --config_path ./ppo/configs/ppo_config_vec_single_env.json
```

Run PPO with image observations:

```bash
python ppo/main.py --trainer visual --config_path ./ppo/configs/ppo_config_vis_single_env.json
```

The included numeric configuration points to Lunar Lander. The visual configuration points to Cube-Ball Visual-Obs. The trainer opens the configured number of Chrome windows, starts a WebSocket server at `ws://localhost:8765`, and waits for the browser environments to connect. Keep the web server terminal running as well.

To train with multiple browser environments, use the `mult_env_vector` or `mult_env_visual` trainer and the corresponding `*_multi_env.json` configuration. The default configurations use three environments. Each browser tab connects to the same WebSocket server.

### Configuration and saving

The PPO configuration files are in `rl-trainers/ppo/configs/`. They include the environment `URL`, `NUM_ENVS`, `EPISODES`, observation stacking, learning parameters, device, and model paths. Change `URL` to an environment that provides the observation type and action space expected by the selected trainer. Visual training defaults to CUDA; set `DEVICE` to `cpu` when CUDA is unavailable. The current device check uses PyTorch CUDA support; Apple MPS is reported but is not selected automatically.

During training, press `s` in the trainer terminal to save the model to `SAVE_PATH`. Existing paths are relative to the `rl-trainers` directory, so the default model files are under `rl-trainers/.models/`.

## Environments in this checkout

The HTML entry points currently present are:

- `Cube-Ball/Vector-Obs/Environment.html` — numeric observations and discrete actions
- `Cube-Ball/Visual-Obs/Environment.html` — image observations and discrete actions
- `Cube-Ball/Continous-SAC-Demo/Environment.html` — continuous-control demo for SAC
- `Cube-Ball/Blender-Demo/Environment.html`
- `Cart-Pole/Environment.html`
- `Balancing-Ball/Environment.html`
- `Lunar-Lander/Environment.html`

Not every environment has a matching trainer configuration. Check the environment's observation and action spaces before changing a training URL.

## SAC

The SAC entry point is `rl-trainers/sac/main.py`. It currently hardcodes the `Cube-Ball/Continous-SAC-Demo` URL and uses one environment. Start the web server above, then run from `rl-trainers`:

```bash
python sac/main.py
```

SAC expects continuous two-dimensional actions (`x` and `z`) and numeric state observations. Its hyperparameters are constants in `sac/trainer.py`; unlike PPO, it does not currently have a JSON configuration or a documented model save/load flow.

## Troubleshooting

- **Connection refused:** confirm the static server is still running on port 5500 and the URL opens in a browser.
- **The environment does not connect:** check the browser developer console and confirm the page implements the RL-Babylon WebSocket/BSON protocol. The Python server listens on port 8765.
- **Chrome does not open:** open the configured URL manually in Chrome. The automatic launcher uses the default Google Chrome installation path.
- **CUDA errors:** use `"DEVICE": "cpu"` in the selected PPO config unless PyTorch and CUDA are installed and available.
- **Windows keyboard support:** `msvcrt` is part of Python on Windows and is not installed from `requirements.txt`; macOS uses the separate `keyboard_mac` implementation.
- **Import error for BSON:** reinstall the repository requirements; BSON support is supplied by the `pymongo` package.

## Known documentation gaps

The old communicator-test instructions referred to `.communicator-test/`, which is not present in this checkout. There is no standalone communicator test documented here. The PPO multi-environment trainers are present, but the documentation does not claim that they have been validated for every environment.
