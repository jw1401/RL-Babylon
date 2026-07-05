// Config file für babylon.js training environment

export default {

    name:               "Continous SAC Demo",
    version:            "1.0",
    author:             "jw",
    goal:               "Move the cube agent toward the target sphere.",
    objects:            "A controllable cube agent and a target sphere which is the goal position.",
    reward:             "Negative distance to target, +1 when reached, -5 when out of bounds.",
    notes:              "Continous environment used for SAC training in BabylonJS via WebSocket interface.",
  
    VIS_OBS:            false,
    state_space:        6,  
    action_space:       2,
    MAX_STEPS:          300

  };