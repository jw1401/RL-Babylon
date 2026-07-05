// Config file für babylon.js training environment

export default {

    name:               "Simple Visual (84x84) Navigation Environment",
    version:            "1.0",
    author:             "jw",
    goal:               "Move the cube agent toward the target sphere.",
    objects:            "A controllable cube agent and a target sphere which is the goal position.",
    reward:             "Negative distance to target, +1 when reached, -5 when out of bounds.",
    notes:              "Discrete environment used for PPO training in BabylonJS via WebSocket interface.",
  
    VIS_OBS:            true,
    state_space:        [84, 84, 4],  // Width x Height x RGBA
    action_space:       4,
    MAX_STEPS:          50

  };