import websockets, traceback, asyncio, torch
from environment import BabylonEnv, BaseRLTrainerServer
from core import RolloutBuffer, StateStacker, PPO, make_policy
from utils import Logger as Log
from trainers import register_trainer

# ========================================= #
# RL Training Loop with WebSocketEnv        #
# ========================================= #

@register_trainer("mult_env_visual")
class RLTrainerServer(BaseRLTrainerServer):

    def __init__(self, config_path = ""):

        super().__init__(config_path = config_path)
        self.active_envs = []


    async def trainer(self, websocket: websockets):

        try:
            env = BabylonEnv(websocket)
            await env.init()

            self.active_envs.append(env)    
                
            if len(self.active_envs) == self.config["NUM_ENVS"]:
                asyncio.create_task(self.global_training_loop()) 

            await asyncio.Event().wait()

        except Exception as e:      
            Log.logError(f"Unexpected error in trainer: {traceback.format_exc()}")


    async def global_training_loop(self):
        
        Log.logInfo("Starting gobal-training-loop...")

        try:

            logger = Log(print_interval = 20)
            global_running_score = 0.0
            total_episodes = 0

            first_env = self.active_envs[0]

            obs_shape = tuple(first_env.state_dim[:2])  
            state_dim = (self.config["NUM_STACK"], *obs_shape) 

            rollout_buffer = RolloutBuffer(buffer_size = self.config["BUFFER_SIZE"], buffer_shape = state_dim, device=self.config["DEVICE"])       

            policy = make_policy("cnn_shared_seperate_heads", input_dim = state_dim, action_dim = first_env.action_dim, in_channels = 4, out_dim = 128)
            ppo_trainer = PPO(model = policy, config = self.config)

            env_stackers = {env: StateStacker(state_dim, num_stack = self.config["NUM_STACK"], device=self.config["DEVICE"]) for env in self.active_envs}
            local_memories = {env: [] for env in self.active_envs}
            env_scores = {env: 0.0 for env in self.active_envs}

            for episode in range(self.config["EPISODES"]):
                
                # Alle Envs parallel zurücksetzen asyncio.gather führt env.reset() für alle Envs gleichzeitig aus
                raw_frames = await asyncio.gather(*(env.reset() for env in self.active_envs))
                current_stacked_states = [env_stackers[env].reset(raw_frames[i]).reshape(4,84,84) for i, env in enumerate(self.active_envs)]

                rollout_buffer.reset()

                counter_env_endings = 0

                while not rollout_buffer.is_full() and counter_env_endings < self.config["NUM_ENVS"]:

                    # Convert the list of states into a single Batch-Tensor
                    state_tensor = torch.stack(current_stacked_states).to(self.config["DEVICE"]).float()

                    # get actions here
                    actions, log_probs, values = ppo_trainer.select_action(state_tensor)
                        
                    # Konvertieren für die Netzwerk-Tasks
                    actions_list = actions.cpu().tolist()
                    
                    # Parallel senden & empfangen via WebSockets
                    step_tasks = [env.step(actions_list[i]) for i, env in enumerate(self.active_envs)]
                    results = await asyncio.gather(*step_tasks)

                    next_stacked_states = []
                    
                    for i, env in enumerate(self.active_envs):

                        next_raw_frame, reward, terminated, truncated = results[i]

                        stacker = env_stackers[env]
                        next_stacked_state = stacker.push(next_raw_frame).reshape(4,84,84)

                        env_scores[env] += reward

                        local_memories[env].append((
                            current_stacked_states[i],
                            actions[i],
                            log_probs[i],
                            reward,
                            terminated,
                            values[i]
                        ))

                        if terminated or truncated:

                            remaining = rollout_buffer.buffer_size - rollout_buffer.ptr

                            if len(local_memories[env]) <= remaining:
                                s, a, lp, r, d, v = zip(*local_memories[env])
                                rollout_buffer.add_batch(states=torch.stack(s), actions=a,log_probs=lp,rewards=r, dones=d,values=v)
                        
                            # Lokalen Speicher für das nächste Mal leeren
                            local_memories[env].clear()

                            # Sofortiger asynchroner Reset für diese eine Umgebung via WebSocket
                            fresh_raw_frame = await env.reset()
                            fresh_stacked_state = stacker.reset(fresh_raw_frame).reshape(4,84,84)
                            next_stacked_states.append(fresh_stacked_state)

                            global_running_score += env_scores[env]
                            counter_env_endings +=1
                            total_episodes +=1
                            env_scores[env] = 0.0
                            
                        else:
                            next_stacked_states.append(next_stacked_state)

                    # Update variables for the next iteration step
                    current_stacked_states = next_stacked_states


                # *******************************************************************   
                #               Buffer full - start training                        *
                # *******************************************************************

                last_state_tensor = current_stacked_states[0]
                _, _, next_value = ppo_trainer.select_action(last_state_tensor.reshape(1,4,84,84))
                last_state_tensor.squeeze(0)
                rollout_buffer.bootstrap(next_value)

                loss, policy_loss, value_loss, entropy = ppo_trainer.train_net(rollout_buffer)

                if (logger.logStats(episode, global_running_score/counter_env_endings, loss.item(), policy_loss.item(), value_loss.item(), entropy.item())): 
                    global_running_score = 0.0
                    Log.logInfo(f"Total Episodes {total_episodes}")


        except Exception as e:
            Log.logError(f"Error in global-training-loop: {traceback.format_exc()}")
