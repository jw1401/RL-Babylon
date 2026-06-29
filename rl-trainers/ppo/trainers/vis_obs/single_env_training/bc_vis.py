import websockets, traceback
from utils import Keyboard, Logger as Log
from environment import BabylonEnv, BaseRLTrainerServer
from core import RolloutBuffer, StateStacker, BehavioralCloning, make_policy
from trainers import register_trainer

# ========================================= #
# RL Training Loop with WebSocketEnv        #
# ========================================= #

@register_trainer("bc_vis")
class RLTrainerServer(BaseRLTrainerServer):
    
    async def trainer(self, websocket: websockets):

        try:

            env = BabylonEnv(websocket)
            await env.init()

            obs_shape = tuple(env.state_dim[:2])                         # --> (84,84)
            state_dim = (self.config["NUM_STACK"], *obs_shape)           # --> (4,84,84) # --> (84,84) --> (4, *(84,84)) --> (4,84,84) unpacks the tuple                      

            policy = make_policy("cnn_shared_seperate_heads", input_dim = state_dim, action_dim = env.action_dim, in_channels = 4, out_dim = 128)
            trainer = BehavioralCloning(model = policy, config = self.config)
            
            stacker = StateStacker(state_dim, num_stack = self.config["NUM_STACK"], device=self.config["DEVICE"])
            buffer = RolloutBuffer(buffer_size = self.config["BUFFER_SIZE"], buffer_shape = state_dim, device=self.config["DEVICE"])

            Logger = Log(print_interval=50)
            Keys = Keyboard(mode="train")

            score = 0.0

            data = buffer.loadBuffer(path=self.config["BUFFER_LOAD_PATH"])

            if data is not None:
                trainer.train_policy_bc(data, epochs = 5000)
                trainer.save(path=self.config["SAVE_PATH"])
                await self.eval(trainer,env,stacker)

            for episode in range (self.config["EPISODES"]):

                first_state = await env.reset()
                state = stacker.reset(first_state)
                state = state.reshape(1, 4, 84, 84)         # (batch = 1, 16, 84, 84) or (batch = 1, 4, 84, 84)
                
                terminated = False
                truncated = False
                reward = 0.0

                if buffer.is_full(): 
                    Logger.logInfo("Buffer reset!")
                    buffer.reset()

                score = 0.0

                while not terminated or not truncated or not buffer.is_full():
                    
                    action = Keys.getActionFromInstructor()
                    if type(action) is not int:
                        break

                    # Step Environment
                    next_state, reward, terminated, truncated = await env.step(action)

                    # Fill the buffer
                    buffer.add(state, action, 0, reward, terminated, 0) 
                        
                    # add the next state to stacker for next cycle, if only one stack then state = next_state
                    state = stacker.push(next_state)
                    state = state.reshape(1, 4, 84, 84)         


                    # add reward to score
                    score += reward

                    # if terminated or truncated break the episode
                    if terminated or truncated:
                        break

                Logger.logInfo(f"Score: {score:.2f}")

                if Keys.mode == "save":
                    buffer.saveBuffer(path = self.config["BUFFER_SAVE_PATH"])
                    Keys.mode = "train"

        except Exception as e:
            Log.logError(f"Unexpected error: {traceback.format_exc()}")


    async def eval(self, trainer: BehavioralCloning, env: BabylonEnv, stacker: StateStacker):

        for episode in range (self.config["EPISODES"]):

            first_state = await env.reset()
            state = stacker.reset(first_state)
            state = state.reshape(1, 4, 84, 84)         

            terminated = False
            truncated = False
            reward = 0.0
            score = 0.0

            while not terminated or not truncated:
                                        
                # Step env
                action, _, _ = trainer.select_action(state)
                next_state, reward, terminated, truncated = await env.step(action.item())
                
                # add the next state to stacker for next cycle, if only one stack then state = next_state
                state = stacker.push(next_state)
                state = state.reshape(1, 4, 84, 84)  

                # Add reward to score
                score += reward

                # if terminated or truncated break the episode
                if terminated or truncated:
                    break

            Log.logInfo(f"Score: {score:.2f}")

       




