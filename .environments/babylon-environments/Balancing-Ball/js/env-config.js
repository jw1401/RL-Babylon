// Config file für babylon.js training environment

export default {

    name:               "Balancing-Ball",
    version:            "1.0",
    author:             "jw",
    goal:               "Balance a ball on a plattform",
    objects:            "Ball and plattform", 
    reward:             "distance reward, -1 for fall off plattform",
    notes:              "Discrete Environment used for PPO training",
  
    VIS_OBS:            false,
    state_space:        8,
    action_space:       4,
    MAX_STEPS:          2000

  };