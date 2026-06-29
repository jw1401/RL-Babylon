# Getting Started with RL-Babylon

This guide walks you through running the communicator tests and the PPO RL trainer.

## Prerequisites

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jw1401/RL-Babylon.git
   ```

2. **Install Python dependencies:**
   ```bash
   cd RL-Babylon
   pip install -r requirements.txt
   ```

## Step 1: Start the Environment Server

The Babylon.js environments must be served via a local web server.

1. Open `RL-Babylon-Environments` in VS Code
2. Install the **Live Server** extension (if not already installed)
3. Right-click on any `.html` file in the environment folder and select **"Open with Live Server"**
   - The server will start on `http://localhost:5500`

## Step 2: Run the Communicator Test

The communicator test verifies basic WebSocket communication between Python and the Babylon environment. For more details check here [Communicator Test](Communicator.md)

```bash
cd RL-Babylon/rl-trainers
python .communicator-test/main.py
```

**What happens:**
- A Chrome window opens automatically with the Simple Navigation Vector environment
- The trainer establishes a WebSocket connection
- Random actions are sent to the environment for 10,000 episodes
- Check the terminal for episode scores

## Step 3: Run the PPO Trainer

Train an agent using Proximal Policy Optimization (PPO).

### Vector Observations (State-based)

```bash
cd RL-Babylon/rl-trainers
python ppo/main.py --trainer vector --config_path ./ppo/configs/ppo_config_vec_single_env.json
```

### Visual Observations (Pixel-based)

```bash
python ppo/main.py --trainer visual --config_path ./ppo/configs/ppo_config_vis_single_env.json
```

### Multi-Environment Training

For faster training across multiple parallel environments:

```bash
# Vector observations
python ppo/main.py --trainer mult_env_vector --config_path ./ppo/configs/ppo_config_vec_multi_env.json

# Visual observations
python ppo/main.py --trainer mult_env_visual --config_path ./ppo/configs/ppo_config_vis_multi_env.json
```

**What happens:**
- Browser windows open automatically for each environment
- The trainer starts a WebSocket server on `ws://localhost:8765`
- Environments connect and training begins
- Press **`s`** in the terminal to save the trained model
- Check the terminal for training stats (episode, score, loss, etc.)

## Configuration

Config files are located in `rl-trainers/ppo/configs/`:
- `ppo_config_vec_single_env.json` - Vector observation configuration
- `ppo_config_vis_single_env.json` - Visual observation configuration

Adjust hyperparameters like:
- `LR` - Learning rate
- `EPISODES` - Total training episodes
- `NUM_ENVS` - Number of parallel environments
- `URL` - Environment URL to connect to
- `DEVICE` - `"cpu"` or `"cuda"` (if GPU available)

## Available Environments

Located in `RL-Babylon-Environments/babylon-environments/`:
- **/Cube Ball/Vector Obs** - Agent navigates to a target (vector observations)
- **/Cube Ball/Visual Obs** - Agent navigates to a target (visual/pixel observations)
- **Cart Pole** - Balancing a pole on a cart
- **Balancing Ball** - Balancing a ball on a platform
- **Lunar Lander** - Balancing a ball on a platform
- **and more**

Update the `URL` in the config file to switch environments.

## Troubleshooting

**Connection refused error?**
- Ensure the Live Server is running and accessible at `http://localhost:5500`

**Trainer starts but no environment opens?**
- Check that Chrome is installed at the default location
- The trainer expects `http://localhost:5500/babylon-environments/{environment-path}/Environment.html`

**WebSocket connection timeout?**
- Verify the environment's `Environment.html` properly implements the RL-Babylon protocol
- Check browser console for errors (F12 in the opened environment window)

**GPU not being used?**
- Set `"DEVICE": "cuda"` in the config file (requires NVIDIA GPU + CUDA)
- Check with `python -c "import torch; print(torch.cuda.is_available())"`
