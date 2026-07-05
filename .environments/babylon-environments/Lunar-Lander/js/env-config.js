// Config file für babylon.js training environment

export default {

    name:               "Lunar Lander",
    version:            "1.0",
    author:             "jw",
    goal:               "Move the lander to landing platform",
    objects:            "A controllable lander agent and a target palttform which is the goal position.",
    reward:             "time penalty, landing reward for soft landing, out of bound penalty and crhas penalty",
    notes:              "Discrete environment used for PPO training in BabylonJS via WebSocket interface.",
  
    VIS_OBS:            false,
    state_space:        5,  // Width x Height x RGBA
    action_space:       4,
    MAX_STEPS:          500

  };