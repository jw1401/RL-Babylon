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

@register_policy("cnn_fc_shared_no_pooling")
class ActorCritic_VIS_V1(nn.Module):

    def __init__(self, input_dim, action_dim, in_channels = 4, out_dim = 512):

        super().__init__()

        # Convolutional network
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),    # (16,84,84) → (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),             # → (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),             # → (64,7,7)
            nn.ReLU(),
            nn.Flatten()
            
        )

        # Berechne flatten Größe dynamisch
        with torch.no_grad():
            dummy = torch.zeros(1, *input_dim)
            n_flatten = self.conv(dummy).view(1, -1).size(1)
            # print(n_flatten)

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, out_dim),
            nn.ReLU()
        ) 

        # ----------------------------------------------------------------- #
        # Separate heads for Policy and Value                               #
        # ----------------------------------------------------------------- #

        self.policy_head = nn.Linear(out_dim, action_dim)
        self.value_head = nn.Linear(out_dim, 1)

    # ----------------------------------------------------------------- #
    # Define Policy and Value forwad pass --> return logits and value   #
    # ----------------------------------------------------------------- #

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value
    
    
