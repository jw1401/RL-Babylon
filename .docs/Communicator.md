## Communicator Test

Before starting training, make sure the Babylon environments are being served by a web server (for example VS Code Live Server).
The trainer automatically launches the required browser environments when it starts.

### Start the Communicator-Test 

Start the main.py file in the folder `/rl-trainers/.communicator-test`. Change the env_case variable for switching between single_env, mult_env and continous_env tests.

```python

import sys, pathlib, asyncio

script_dir = pathlib.Path(__file__).parent      # Folder of this file
root = script_dir.parent                        # Root of Project (one level up)
sys.path.append(str(root / "src"))              # add src to sys-paths 
sys.path.append(str(script_dir))                # add script-folder to sys-paths 

env_case = 1                                    # 1 --> single, 2 --> mult, 3 --> continous

from utils import run_envs
 
if __name__ == "__main__":

    if env_case == 1:
        from single_env_test import RLTrainerServer     
        trainer_server = RLTrainerServer()
        run_envs(num_envs = 1, url = "http://localhost:5500/babylon-environments/Cube-Ball/Vector-Obs/Environment.html")

    elif env_case ==2:
        from mult_env_test import RLTrainerServer
        trainer_server = RLTrainerServer(num_envs = 3)
        run_envs(num_envs = 3, url = "http://localhost:5500/babylon-environments/Cube-Ball/Vector-Obs/Environment.html")

    elif env_case ==3:
        from continous_env_test import RLTrainerServer
        trainer_server = RLTrainerServer()
        run_envs(num_envs = 1, url = "http://localhost:5500/babylon-environments/Cube-Ball/Continous-SAC-Demo/Environment.html")

    try:
        asyncio.run(trainer_server.start_server())
        
    except Exception as e:
      print(f"Error: {e}")

```

### Environment Requirements

Each Babylon environment

* Connects to the trainer via WebSocket.
* Implements the RL-Babylon communication protocol.
* Provides observations, rewards, and termination signals.

Once connected, the environment can be used through the Gym-like API:

```python

import websockets, random
from environment import BabylonEnv, BaseRLTrainerServer

class RLTrainerServer(BaseRLTrainerServer):

    async def trainer(self, websocket: websockets):

        try:
            env = BabylonEnv(websocket)
            await env.init()

            # Enter main training loop
            for episode in range (10000):
                    
                terminated = False
                truncated = False

                await env.reset()

                while not terminated or not truncated:
                        
                    # Get action - gives int numbers 0,1,2,3
                    action  = random.randrange(env.action_dim)   
                    
                    # Step Environment
                    next_state, reward, terminated, truncated = await env.step(action)
                    
                    # if terminated or truncated break the episode
                    if terminated or truncated:
                        break
                    
        except Exception as e:      
           print(f"Unexpected error: {e}")


```

You can study the `/rl-trainers/.communicator-test` examples and see how the training loop works
