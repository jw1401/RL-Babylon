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
                        
                    # Get action - gives random numbers from -1.0 to 1.0
                    action  = {"x": random.uniform(-1.0, 1.0), "z": random.uniform(-1.0, 1.0)} 
                    
                    # Step Environment
                    next_state, reward, terminated, truncated = await env.step(action)
                    
                    # if terminated or truncated break the episode
                    if terminated or truncated:
                        break
                    
        except Exception as e:      
           print(f"Unexpected error: {e}")

   