import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from .rollout_buffer import RolloutBuffer
from .model_io import load_parameters, save
from utils import Logger

# ======================================================== #
# Class PPO:                                               #
# Implements the Proximal Policy Optimization algorithm    #
# ======================================================== #

class PPO(nn.Module):

    """ 
    ## Proximal Policy Optimization (PPO)
    Implements the Proximal Policy Optimization algorithm.

    PPO (Proximal Policy Optimization) is a reinforcement learning algorithm that
    balances policy improvement with stability by constraining updates to stay
    close to the old policy.

    See documentation: [./Docs/PPO.md](./Docs/PPO.md) 
    """

    def __init__(self, model = None, config = None): 
        super().__init__()

        Logger.logInfo("Proximal-Policy-Optimization (PPO)")

        self.model = model
        self.config = config
        self.model.to(self.config["DEVICE"])
        self.optimizer  = optim.Adam(self.model.parameters() , lr = self.config["LR"])

    # --------------------------------------------- #
    # Select action to step environment             #
    # --------------------------------------------- #

    def select_action(self, state):

        policy_logits, value = self.model(state)

        distribution = Categorical(logits = policy_logits)
        action = distribution.sample() # sample action from categorical distribution
        log_prob = distribution.log_prob(action) 

        return action.detach(), log_prob.detach(), value.detach() # return detached tensors to avoid backprob

    # --------------------------------------------------------------- #
    # Make a rondom shuffled mini batch if needed for better training #
    # --------------------------------------------------------------- #

    def make_batch(self, buffer: RolloutBuffer):

        # Util now only hands through buffer.get()
        return buffer.get()

    # ------------------------------------------------------- #
    # Method train_net() --> See reference: ./Docs/PPO.md     #
    # ------------------------------------------------------- #

    def train_net(self, buffer:RolloutBuffer):
       
        data = self.make_batch(buffer)  

        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        
        # Bootstrap value in RolloutBuffer and compute advantage and returns
        advantage_tensor, returns_tensor = buffer.compute_gae(gamma = self.config["GAMMA"], lam = self.config["LAMBDA"]) 
        
        # Normalize advantage --> not working nan Tensor is exploding
        # adv = (adv - adv.mean()) / (adv.std() + 1e-7)  

        for i in range(self.config["UPDATE_EPOCHS"]):

            policy_logits, values = self.model(states)         
            distribution = Categorical(logits = policy_logits)
            new_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean() # Entropy for exploration
            
            # Ratio
            ratio = torch.exp( new_log_probs - old_log_probs )     

            # Unclipped objective term --> same as vanilla policy gradient 
            surrogate_1 = ratio * advantage_tensor

            # Clipped Objective
            surrogate_2 = torch.clamp(ratio, 1 - self.config["CLIP_EPS"], 1 + self.config["CLIP_EPS"]) * advantage_tensor     
            
            # Policy loss
            policy_loss = - torch.min(surrogate_1, surrogate_2).mean()

            # Critic or Value loss  --> Mean Squared error
            value_loss = F.mse_loss(values.squeeze(-1), returns_tensor) 

            # Combined loss
            loss = policy_loss + self.config["VALUE_COEFF"] * value_loss - self.config["ENTROPY_COEFF"] * entropy
            
            # Backward pass through optimizer for gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, policy_loss, value_loss, entropy


    def load(self, path = ""):

        params = load_parameters(path = path)
        if params is not None:
            self.model.load_state_dict(params)
            self.model.to(self.config["DEVICE"])


    def save(self, path = ""):
        save(model = self.model, path = path)
                