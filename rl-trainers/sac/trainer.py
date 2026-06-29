import websockets, traceback
from environment import BabylonEnv, BaseRLTrainerServer
from utils import Logger as Log
from sac import SACAgent

EPISODES = 100000
UPDATE_CYCLE = 4
GRADIENT_STEPS = 2

class RLTrainerServer(BaseRLTrainerServer):

    async def trainer(self, websocket: websockets):

        try:
            env = BabylonEnv(websocket)
            await env.init()

            agent = SACAgent(state_dim = env.state_dim, action_dim = env.action_dim)
            batch_size = 256

            # Enter main trainings loop
            for episode in range (EPISODES):

                state = await env.reset()

                accumulated_steps = 0
                episode_reward = 0
                done = False

                while not done:

                    # Select action
                    action = agent.select_action(state)

                    # Format the payload
                    action_payload = {
                        "x": float(action[0]), 
                        "z": float(action[1])
                    }
                    
                    # Step the world
                    next_state, reward, terminated, truncated = await env.step(action_payload)
                    done = terminated or truncated
                    
                    # Save transaction
                    agent.replay_buffer.push(state, action, reward, next_state, float(terminated))
                    
                    state = next_state
                    episode_reward += reward

                    accumulated_steps += 1

                    if accumulated_steps % UPDATE_CYCLE == 0:
                        for _ in range(GRADIENT_STEPS): agent.update_parameters(batch_size)

                Log.logInfo(f"Episode: {episode + 1} | Total Reward: {episode_reward:.2f} | Alpha: {agent.alpha:.4f}")


        except Exception as e:        
            Log.logError(f"Unexpected error: {traceback.format_exc()}")


