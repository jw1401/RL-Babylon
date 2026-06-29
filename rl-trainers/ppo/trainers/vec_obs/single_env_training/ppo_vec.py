import websockets, traceback
from environment import BabylonEnv, BaseRLTrainerServer
from core import RolloutBuffer, StateStacker, PPO, make_policy
from utils import Keyboard, Logger as Log
from trainers import register_trainer

# ========================================= #
# RL Training Loop with WebSocketEnv        #
# ========================================= #

@register_trainer("vector")
class RLTrainerServer(BaseRLTrainerServer):

    async def trainer(self, websocket: websockets):

        try:
            env = BabylonEnv(websocket)
            await env.init()

            state_dim = env.state_dim * self.config["NUM_STACK"]             # 16
            stacker_shape = (self.config["NUM_STACK"], env.state_dim)        # --> (4,4)                          

            policy = make_policy("mlp_shared", input_dim = state_dim, action_dim = env.action_dim, hidden_sizes = self.config["HIDDEN_SIZES"])
            trainer = PPO(model = policy, config = self.config)
            
            stacker = StateStacker(stacker_shape, num_stack = self.config["NUM_STACK"],device=self.config["DEVICE"])
            buffer = RolloutBuffer(buffer_size = self.config["BUFFER_SIZE"], buffer_shape = (state_dim,),device=self.config["DEVICE"]) # --> (16,)
            Logger = Log(print_interval = 50)
            kb = Keyboard()

            # loads only if path set
            trainer.load(path = self.config["LOAD_PATH"])

            score = 0.0

            level = await env.curriculum_level(0)
            Log.logInfo(f"Curriculum Level: {level}")
            
            # Enter main trainings loop
            for episode in range (self.config["EPISODES"]):

                first_state = await env.reset()
                state = stacker.reset(first_state)
                state = state.flatten()
                    
                terminated = False
                truncated = False

                buffer.reset()

                while not terminated or not truncated or not buffer.is_full():
                        
                    # Get action
                    action, log_prob, value = trainer.select_action(state)
                        
                    # Step Environment
                    next_state, reward, terminated, truncated = await env.step(action.item())
                    
                    # Fill the buffer
                    buffer.add(state, action, log_prob, reward, terminated, value) 
                        
                    # add the next state to stacker for next cycle, if only one stack then state = next_state
                    state = stacker.push(next_state)
                    state = state.flatten()
                        
                    # add reward to score
                    score += reward

                    # if terminated or truncated break the episode
                    if terminated or truncated:
                        break

                # add value of next state for bootstraping
                _, _, next_value = trainer.select_action(state)
                buffer.bootstrap(next_value)

                # train the network
                loss, policy_loss, value_loss, entropy = trainer.train_net(buffer)

                # Output score every print interval and resets score
                if (Logger.logStats(episode, score, loss.item(), policy_loss.item(), value_loss.item(), entropy.item())): score = 0.0


                if score/Logger.print_interval > 4 and level == 0:
                    level = await env.curriculum_level(2)
                    Log.logInfo(f"Curriculum Level: {level}")

                # Save model weights
                if (kb.checkMode() == "save"):
                    trainer.save(path = self.config["SAVE_PATH"])
                    kb.mode = "train"
                    
        except Exception as e:        
            Log.logError(f"Unexpected error: {traceback.format_exc()}")
