// Config file für babylon.js training environment

export default {

    name:               "Cart-Pole",
    version:            "1.0",
    author:             "jw",
    goal:               "Balance a pole on a cart",
    objects:            "Cart and pole on ground with limits", 
    reward:             "Until not terminated reward=+1; if terminated reward=-1",
    notes:              "Discrete Environment used for PPO training",
  
    VIS_OBS:            false,
    state_space:        4,
    action_space:       2,
    MAX_STEPS:          400

  };