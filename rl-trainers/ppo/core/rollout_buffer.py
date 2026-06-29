import torch, numpy as np
from utils import Logger       
from .model_io import load_buffer, save_buffer

# ============================================== #
# RolloutBuffer Class                            #
# Saves Data of environment Rollouts and         #
# converts it to tensors                         #
# ============================================== #

class RolloutBuffer:

    def __init__(self, buffer_size, buffer_shape, device = "cpu"):
        
        self.buffer_size = buffer_size
        self.device = device

        # Preallocate tensors
        self.states     = torch.zeros((buffer_size, *buffer_shape), dtype=torch.float32, device=self.device) # (2048, 4, 84, 84)
        self.actions    = torch.zeros(buffer_size, dtype=torch.long, device=self.device)
        self.log_probs  = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.rewards    = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.dones      = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.values     = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)

        self.ptr = 0
        self.next_value = 0.0  # bootstrap value


    def _to_tensor(self, x, dtype = torch.float32):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype = dtype, device=self.device)
        elif not torch.is_tensor(x):
            x = torch.tensor(x, dtype = dtype, device=self.device)

        return x.detach() 


    def add(self, state, action, log_prob, reward, done, value):

        if self.ptr >= self.buffer_size:
            raise IndexError(f"Buffer full! ptr={self.ptr}, buffer_size={self.buffer_size}")

        self.states[self.ptr]    = self._to_tensor(state, torch.float32)
        self.actions[self.ptr]   = self._to_tensor(action, torch.long)
        self.log_probs[self.ptr] = self._to_tensor(log_prob, torch.float32)
        self.rewards[self.ptr]   = self._to_tensor(reward, torch.float32)
        self.dones[self.ptr]     = self._to_tensor(done, torch.float32)
        self.values[self.ptr]    = self._to_tensor(value, torch.float32)

        self.ptr += 1

    def add_batch(self, states, actions, log_probs, rewards, dones, values):

        states = self._to_tensor(states, torch.float32) # states = self._to_tensor(np.array(states), torch.float32)
        actions = self._to_tensor(actions, torch.long)
        log_probs = self._to_tensor(log_probs, torch.float32)
        rewards = self._to_tensor(rewards, torch.float32)
        dones = self._to_tensor(dones, torch.float32)
        values = self._to_tensor(values, torch.float32)
        # values = values.squeeze(-1)

        batch_size = len(states)

        end = self.ptr + batch_size

        self.states[self.ptr:end] = states
        self.actions[self.ptr:end] = actions
        self.log_probs[self.ptr:end] = log_probs
        self.rewards[self.ptr:end] = rewards
        self.dones[self.ptr:end] = dones
        self.values[self.ptr:end] = values

        self.ptr += batch_size


    def bootstrap(self, value):

        # Store the last value for GAE computation
        self.next_value = self._to_tensor(value, torch.float32,).squeeze()  # scalar


    def compute_gae(self, gamma = 0.99, lam = 0.95):
        
        # Compute advantages and returns

        T = self.ptr
        rewards = self.rewards[:T]
        dones   = self.dones[:T]
        values  = self.values[:T]

        # Append bootstrap value virtually (no need to modify preallocated buffer)
        extended_values = torch.zeros(T + 1, dtype=torch.float32, device=self.device)
        extended_values[:T] = values
        extended_values[T] = self.next_value

        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = 0.0

        for t in reversed(range(T)):

            delta = rewards[t] + gamma * extended_values[t+1] * (1 - dones[t]) - extended_values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + extended_values[:T]

        return advantages.detach(), returns.detach()
    

    def is_full(self):

        return self.ptr >= self.buffer_size


    def get(self, device = None):

        T = self.ptr

        data = {
            "states": self.states[:T].detach(), 
            "actions": self.actions[:T].detach(),
            "log_probs": self.log_probs[:T].detach(), 
            "rewards": self.rewards[:T].detach(), 
            "dones": self.dones[:T].detach(), 
            "values": self.values[:T].detach(), 
        }

        return data


    def reset(self):

        self.ptr = 0
        self.next_value = 0.0


    def saveBuffer(self, path = ""):
        save_buffer(data = self.get(), path = path)


    def loadBuffer(self, path = ""):
        data = load_buffer(path = path)
        return data