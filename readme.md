# RL-Babylon - Reinforcement Learning with Babylon.js

![rl-babylon](./.docs/rl-babylon.jpg)

**RL-Babylon** is a Python framework that enables Reinforcement Learning (RL) agents to interact with **Babylon.js environments** via WebSockets. It provides a flexible, asynchronous interface for training RL algorithms in visually rich 3D environments rendered directly in the browser. With RL-Babylon, you can easily integrate **state-based** or **pixel-based observations** into your RL workflow.

## Features

- **WebSocket-based Environment** – Real-time communication with Babylon.js scenes  
- **Visual & Numeric Observations** – Supports raw pixel frames or structured state vectors  
- **Gym-like Interface** – Easy `init()`, `reset()`, and `step()` API  

## Getting started

- [Getting Started](.docs/GETTING_STARTED.md)

## Trainers

- [PPO](./rl-trainers/ppo/ppo.md)
- [SAC](./rl-trainers/sac/sac.md)

## RL-Babylon-Environments

RL-Babylon-Environments can be found in the [.environments](./.environments/README.md) folder. Simply start a local server (for example using Live Server in VS Code), open your browser (recommended: Chrome), and navigate to the Environment.html file inside the desired environment folder.

Environments:

 - Simple Navigation Vector (vector obs)
 - Simple Navigation Visual (visual obs)
 - Cart Pole (vector obs)
 - Balancing Ball (vector obs)
