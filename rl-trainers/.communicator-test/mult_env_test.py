import websockets, random, asyncio
from environment import BabylonEnv, BaseRLTrainerServer

class RLTrainerServer(BaseRLTrainerServer):

    def __init__(self, num_envs = 1):

        super().__init__()

        self.active_envs = []
        self.num_envs= num_envs

    async def trainer(self, websocket: websockets):

        try:
            env = BabylonEnv(websocket)
            await env.init()

            self.active_envs.append(env)    
                
            if len(self.active_envs) == self.num_envs: 
                asyncio.create_task(self.global_training_loop()) 
            
            await asyncio.Event().wait()

        except Exception as e:      
            print(f"Unexpected error in trainer: {e}")


    async def global_training_loop(self):

        try:
            first_env = self.active_envs[0]
            env_scores = {env: 0.0 for env in self.active_envs}

            for episode in range(10000):

                # reset all envs
                raw_frames = await asyncio.gather(*(env.reset() for env in self.active_envs))
                
                while True:
                    
                    # send and receive step for all envs
                    step_tasks = [env.step(random.randrange(first_env.action_dim)) for i, env in enumerate(self.active_envs)]
                    results = await asyncio.gather(*step_tasks)
                    
                    # iterate through all envs and get env results
                    for i, env in enumerate(self.active_envs):

                        next_raw_frame, reward, terminated, truncated = results[i]
                        env_scores[env] += reward

                        if terminated or truncated:

                            print(f"Env {i} in Episode {episode} beendet mit Score: {env_scores[env]}")
                            env_scores[env] = 0.0

                            # async reset
                            fresh_raw_frame = await env.reset()

        except Exception as e:
            print(f"Error in global-training-loop: {e}")

        