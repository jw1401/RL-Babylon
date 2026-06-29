import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from .rollout_buffer import RolloutBuffer
from .model_io import load_parameters, save
from utils import Logger
 
# ======================================================== #
# Class BehavioralCloning:                                 #
# Implements the ...                                       #
# ======================================================== #

class BehavioralCloning(nn.Module):

    def __init__(self, model = None, config = None): 
        super().__init__()

        Logger.logInfo("Behavioral-Cloning (BC)")

        self.model = model
        self.config = config
        self.model.to(self.config["DEVICE"])
        self.device = torch.device(self.config["DEVICE"])
        self.optimizer  = optim.Adam(self.model.parameters() , lr = self.config["LR"])

    # --------------------------------------------- #
    # Select action to step environment             #
    # --------------------------------------------- #

    def select_action(self, state):

        policy_logits, value = self.model(state)

        distribution = Categorical(logits = policy_logits)
        # sample action from categorical distribution
        action = distribution.sample() 
        log_prob = distribution.log_prob(action) 

        # return detached tensors to avoid backprob
        return action.detach(), log_prob.detach(), value.detach() 

    # --------------------------------------------------------------- #
    # Make a rondom shuffled mini batch if needed for better training #
    # --------------------------------------------------------------- #

    # def make_batch(self, buffer: RolloutBuffer):
    #     # Util now only hands through buffer.get()
    #     return buffer.get()
       

    def train_policy_bc(self, data, epochs = 2048, breakpoint = 0.3):

        states = data["states"]
        actions = data["actions"]

        for i in range(epochs):    # 128 für VEC 2048 für VIS für PPO und BC Warmstart

            policy_logits, _ = self.model(states)            
            
            # Policy loss
            loss = F.cross_entropy(policy_logits, actions)

            # Backward pass through optimizer for gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Loss: {loss.item():.3f}", end="\r")

            if loss.item() < breakpoint: 
                print(f"Loss: {loss.item():.3f}")
                break

        return loss


    def load(self, path = ""):

        params = load_parameters(path = path)
        if params is not None:
            self.model.load_state_dict(params)
            self.model.to(self.config["DEVICE"])

    def save(self, path = ""):
        save(model = self.model, path = path)

