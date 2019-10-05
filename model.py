import torch
import torch.nn as nn
from torch.distributions import Normal

n_board = 2 * 28 * 28 * 7 
n_meta = 14
num_outputs = 714 # todo envs.action_space.shape[0]
hidden_size = 100

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(n_board + n_meta, 12000),
            nn.ReLU(),
            nn.Linear(12000, 18000),
            nn.ReLU(),
            nn.Linear(18000, 12000),
            nn.ReLU(),
            nn.Linear(12000, (28 + 210) * 3),
            nn.Sigmoid()
        )
    def forward(self, state):
        return self.seq(state)


class ActorCritic(nn.Module):
    def __init__(self, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(n_meta + n_board, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = Actor()
        self.log_std = nn.Parameter(torch.ones(num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, state):
        value = self.critic(state)
        mu    = self.actor(state)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
