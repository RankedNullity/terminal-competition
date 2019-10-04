import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from GeneticModel import *d

def return_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        agent = TerminalAI()
        for param in agent.parameters():
            param.requires_grad = False
        init_weights(agent)
        agents.append(agent)
    return array(agents)

#disable gradients as we will not use them
torch.set_grad_enabled(False)

# initialize N number of agents
num_agents = 1000
agents = return_random_agents(num_agents)

# How many top agents to consider as parents
top_limit = 50

# run evolution until X generations
generations = 100

elite_index = None

for generation in range(generations):

    # return rewards of agents
    rewards = run_agents_n_times(agents, 3) #return average of 3 runs

    # sort by rewards
    sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] #reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
    print("")
    print("")
    
    top_rewards = []
    for best_parent in sorted_parent_indexes:
        top_rewards.append(rewards[best_parent])
    
    print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
    #print(rewards)
    print("Top ",top_limit," scores", sorted_parent_indexes)
    print("Rewards for top: ",top_rewards)
    
    # setup an empty list for containing children agents
    children_agents, elite_index = return_children(agents, sorted_parent_indexes, elite_index)

    # kill all agents, and replace them with their children
    agents = children_agents
