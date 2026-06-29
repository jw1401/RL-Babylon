import torch
import torch.nn as nn
import torch.nn.functional as F
from core import register_policy

torch.manual_seed(42)

# ============================================== #
# PPOActorCritic Class                           #
# Defines the models with networks for policy    #
# and value with shared layers                   #
# ============================================== #

@register_policy("mlp_shared")
class ActorCritic_VEC(nn.Module):

    def __init__(self, input_dim, action_dim, hidden_sizes = [64, 64], activation = nn.ReLU):

        super().__init__()

        # ------------------------------------------------------------------------------------- #
        # Shared Feature-Extractor for Value and Policy                                         #
        #                                                                                       #
        # Fully Connecetd Shared Network for Policy and Value                                   # 
        # --> Faster laerning less parameters                                                   #
        #                                                                                       #
        # Not suitable for advanced scenarios --> use seperate networks                         #
        # ------------------------------------------------------------------------------------- #
        
        shared_layers = []
        last_dim = input_dim

        for h in hidden_sizes:

            shared_layers.append(nn.Linear(last_dim, h))
            shared_layers.append(activation())
            last_dim = h

        self.shared = nn.Sequential(*shared_layers)

        # ----------------------------------------------------------------- #
        # Separate heads for Policy and Value                               #
        # ----------------------------------------------------------------- #

        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Linear(last_dim, 1)

    # ----------------------------------------------------------------- #
    # Define Policy and Value forwad pass --> return logits and value   #
    # ----------------------------------------------------------------- #

    def forward(self, x):

        features = self.shared(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
