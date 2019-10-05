import torch
import torch.nn as nn
from torch.distributions import Normal

n_board = 28 * 28 * 7 
n_meta = 4
num_outputs = 714 # todo envs.action_space.shape[0]
hidden_size = 100

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.boardBlock = nn.Sequential(
            nn.Linear(n_board, 12000),
            nn.ReLU(),
            nn.Linear(12000, 18000),
            nn.ReLU()
        )
        self.secondBlock = nn.Sequential(
            nn.Linear(18000 + n_meta, 12000),
            nn.ReLU(),
            nn.Linear(12000, (28 + 210) * 3),
            nn.Sigmoid()
        )
    def forward(self, Qboard, Qmeta):
        boardRep = self.boardBlock(Qboard)
        return self.secondBlock(torch.cat(boardRep, Qmeta))


class ActorCritic(nn.Module):
    def __init__(self, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(n_meta + n_board, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = Actor()
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, Qboard, Qmeta):
        value = self.critic(torch.cat(Qboard, Qmeta))
        mu    = self.actor(Qboard, Qmeta)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
