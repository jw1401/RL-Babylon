// Config file für babylon.js training environment

export default {

    name:               "Simple Vector Navigation Environment - Blender",
    version:            "1.0",
    author:             "jw1401",
    goal:               "Move the cube agent toward the target sphere.",
    objects:            "A controllable cube agent and a target sphere which is the goal position.",
    reward:             "Negative distance to target, +5 when reached, -5 when out of bounds.",
    notes:              "Discrete environment",
  
    VIS_OBS:            false,
    state_space:        4,  
    action_space:       4,
    MAX_STEPS:          50

  };