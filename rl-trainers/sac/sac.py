import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# --- REPLAY BUFFER ---
class ReplayBuffer:

    def __init__(self, capacity, state_dim, action_dim):

        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):

        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[idxs]),
            torch.FloatTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_states[idxs]),
            torch.FloatTensor(self.dones[idxs])
        )


# --- NETWORKS ---
class CriticNetwork(nn.Module):

    """Dual Q-Network implementation for Twin-Delayed / SAC style value estimation."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):

        sa = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class ActorNetwork(nn.Module):

    """Gaussian Policy Network mapping states to action distributions bounded by tanh."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(ActorNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):

        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):

        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Enforcing Action Bound via Reparameterization Trick
        # rsample handles backprop gradients through the distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()                                      
        action = torch.tanh(x_t)
        
        # Compute Log Probability adjusted for tanh squishing
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean)
        
        return action, log_prob, mean


# --- SAC AGENT ---
class SACAgent:

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=1000000):

        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Critic initialization
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Actor initialization
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Automatic Entropy Tuning (Target Entropy = -|A|)
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    def select_action(self, state, evaluate=False):

        """Returns a plain NumPy array bounded between [-1.0, 1.0] ready for transmission."""

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size):

        if self.replay_buffer.size < batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)
        
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # 1. Update Critic Network
        with torch.no_grad():
            next_state_action, next_state_log_prob, _ = self.actor.sample(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_state_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_state_log_prob
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, next_q_value) + F.mse_loss(current_q2, next_q_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Update Actor Network
        pi, log_prob, _ = self.actor.sample(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = ((self.alpha * log_prob) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Tune Temperature Parameter (Alpha)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()

        # 4. Soft Update Target Critic Network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


