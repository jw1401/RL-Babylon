import websockets, traceback
from environment import BabylonEnv, BaseRLTrainerServer
from core import RolloutBuffer, StateStacker, PPO, make_policy
from utils import Keyboard, Logger as Log
from trainers import register_trainer

# ========================================= #
# RL Training Loop with WebSocketEnv        #
# ========================================= #

@register_trainer("visual")
class RLTrainerServer(BaseRLTrainerServer):

    async def trainer(self, websocket: websockets):
        
        try:

            env = BabylonEnv(websocket)
            await env.init()

            obs_shape = tuple(env.state_dim[:2])                    # --> (84,84)
            state_dim = (self.config["NUM_STACK"], *obs_shape)           # --> (4,84,84) # --> (84,84) --> (4, *(84,84)) --> (4,84,84) unpacks the tuple

            policy = make_policy("cnn_shared_seperate_heads", input_dim = state_dim, action_dim = env.action_dim, in_channels = 4, out_dim = 128)
            trainer = PPO(model = policy, config = self.config)

            stacker = StateStacker(state_dim, num_stack = self.config["NUM_STACK"], device=self.config["DEVICE"])
            buffer = RolloutBuffer(buffer_size = self.config["BUFFER_SIZE"], buffer_shape = state_dim, device=self.config["DEVICE"])
            Logger = Log (print_interval = 50)
            Keys = Keyboard(mode="train")

            trainer.load(path = self.config["LOAD_PATH"])

            score = 0.0
            
            for episode in range (self.config["EPISODES"]):

                first_state = await env.reset()
                state = stacker.reset(first_state)
                state = state.reshape(1, 4, 84, 84)         # (batch = 1, 16, 84, 84) or (batch = 1, 4, 84, 84)
                
                terminated = False
                truncated = False

                buffer.reset()

                while not terminated or not truncated or not buffer.is_full:
                    
                    # Get action
                    action, log_prob, value = trainer.select_action(state)

                    # Step Environment
                    next_state, reward, terminated, truncated = await env.step(action.item())

                    # Fill the buffer
                    buffer.add(state, action, log_prob, reward, terminated, value) 
                    
                    # add the next state to stacker for next cycle, if only one stack then state = next_state
                    state = stacker.push(next_state)
                    state = state.reshape(1, 4, 84, 84)         # (batch = 1, 16, 84, 84) or (batch = 1, 4, 84, 84)
                    
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

                # Save model weights
                if (Keys.checkMode() == "save"):
                    trainer.save(path=self.config["SAVE_PATH"])
                    Keys.mode = "train"

        except Exception as e:        
            Log.logError(f"Unexpected error: {traceback.format_exc()}")

