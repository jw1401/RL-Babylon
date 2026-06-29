import websockets, traceback
from utils import Keyboard, Logger as Log
from environment import BabylonEnv, BaseRLTrainerServer
from core import RolloutBuffer, StateStacker, BehavioralCloning, make_policy
from trainers import register_trainer

# ========================================= #
# Behavioral Cloning with WebSocketEnv      #
# ========================================= #

@register_trainer("bc_vec")
class RLTrainerServer(BaseRLTrainerServer):

    async def trainer(self, websocket: websockets):

        try:

            env = BabylonEnv(websocket)
            await env.init()   

            state_dim = env.state_dim * self.config["NUM_STACK"]             # 16
            stacker_shape = (env.state_dim, self.config["NUM_STACK"])        # --> (4,4)    

            policy = make_policy("mlp_shared", input_dim = state_dim, action_dim = env.action_dim, hidden_sizes = self.config["HIDDEN_SIZES"])
            trainer = BehavioralCloning(model = policy, config = self.config)
            
            stacker = StateStacker(stacker_shape, num_stack = self.config["NUM_STACK"], device=self.config["DEVICE"])
            buffer = RolloutBuffer(buffer_size = self.config["BUFFER_SIZE"], buffer_shape = (state_dim,), device=self.config["DEVICE"]) # --> (16,)

            Logger = Log(print_interval=50)
            Keys = Keyboard()

            data = buffer.loadBuffer(path = self.config["BUFFER_LOAD_PATH"])

            if data is not None:
                trainer.train_policy_bc(data, epochs = 1028)
                trainer.save(path=self.config["SAVE_PATH"])
                await self.eval(trainer,env,stacker)
            
            for episode in range (self.config["EPISODES"]):

                first_state = await env.reset()
                state = stacker.reset(first_state)
                state = state.flatten()

                terminated = False
                truncated = False
                reward = 0.0

                if buffer.is_full(): 
                    Logger.logInfo("Buffer reset")
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
                    state = state.flatten()

                    # Add reward to score
                    score += reward

                    # if terminated or truncated break the episode
                    if terminated or truncated:
                        break

                Log.logInfo(f"Score: {score:.2f}")

                if Keys.mode == "save":
                    buffer.saveBuffer(path= self.config["BUFFER_SAVE_PATH"])
                    Keys.mode = "train"
            
        except Exception as e:        
            Log.logError(f"Unexpected error: {traceback.format_exc()}")


    async def eval(self, trainer: BehavioralCloning, env: BabylonEnv, stacker: StateStacker):

        for episode in range (self.config["EPISODES"]):

            first_state = await env.reset()
            state = stacker.reset(first_state)
            state = state.flatten()

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
                state = state.flatten()

                # Add reward to score
                score += reward

                # if terminated or truncated break the episode
                if terminated or truncated:
                    break

            Log.logInfo(f"Score: {score:.2f}")

    

