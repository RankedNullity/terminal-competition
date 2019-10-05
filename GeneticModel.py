import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

class TerminalAI(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(4, 32, (8, 8), stride=4),
                                      nn.LeakyReLU(),
                                      # 13 x 6 x 32
                                      nn.Conv2d(32, 64, (4, 4), stride=1),
                                      nn.LeakyReLU(),
                                      # 10 x 3 x 64
                                      nn.Conv2d(64, 128, (2, 2), stride=1),
                                      nn.LeakyReLU()
                                      # 9 x 2 x 128
                                      )
            self.fc = nn.Sequential(nn.Linear(9 * 2 * 128 + 14, 2700),
                                    nn.LeakyReLU(),
                                    nn.Linear(2700, 3300),
                                    nn.LeakyReLU(),
                                    nn.Linear(3300, 3300),
                                    nn.LeakyReLU(),
                                    nn.Linear(3300, 3780)
                                    )
                                    
            self.add_tensors = {}
            for name, tensor in self.named_parameters():
                if tensor.size() not in self.add_tensors:
                    self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
                if 'weight' in name:
                    nn.init.kaiming_normal(tensor)
                else:
                    tensor.data.zero_()

        def forward(self, conv_input, linear_input):
                '''Input size 56 x 28 x 2. Output size (14*15*18) x 1'''
                output = self.conv(conv_input)
                output = output.view(-1, 1)
                output = torch.cat((output.squeeze(), linear_input), 0)
                return self.fc(output)
                

def init_weights(m):
    
        # nn.Conv2d weights are of shape [16, 1, 3, 3] i.e. # number of filters, 1, stride, stride
        # nn.Conv2d bias is of shape [16] i.e. # number of filters
        
        # nn.Linear weights are of shape [32, 24336] i.e. # number of input features, number of output features
        # nn.Linear bias is of shape [32] i.e. # number of output features
        
        if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.00)

def mutate(agent, mutation_power=0.02):
    child_agent = copy.deepcopy(agent)
    # mutation_power = 0.002 hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf        
    for name, tensor in child_agent.named_parameters():
        to_add = child_agent.add_tensors[tensor.size()]
        to_add.normal_(0.0, mutation_power)
        tensor.data.add_(to_add)

    return child_agent
