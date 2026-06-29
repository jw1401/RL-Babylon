// main.js --> Main entry point for RL Scene 

import { BabylonEnv } from "./babylon-env.js";

const env = new BabylonEnv("ws://localhost:8765", "renderCanvas");

// Optional: manual reset on key press
window.addEventListener("keydown", (e) => {

  if (e.key === "r") {
    env.realtime = !env.realtime;
  }

});

// Adjust when window is resized
window.addEventListener("resize", () => {

    env.engine.resize();
    env.scene.render();

});