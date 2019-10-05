import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import glob
import os
import subprocess
import sys
import json

from GeneticModel import *

def return_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        agent = TerminalAI()
        for param in agent.parameters():
            param.requires_grad = False
        init_weights(agent)
        agents.append(agent)
    return agents

def run_single_game():
    process_command = "java -jar engine.jar work .\\run_gene1.ps1 .\\run_gene2.ps1"
    print("Start running a match")
    p = subprocess.Popen(
        process_command,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr
        )
    # daemon necessary so game shuts down if this script is shut down by user
    p.daemon = 1
    p.wait()
    print("Finished running match")

#disable gradients as we will not use them
torch.set_grad_enabled(False)
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# initialize N number of agents
num_agents = 250
agents = return_random_agents(num_agents)

# How many top agents to consider as parents
top_limit = num_agents / 50

# run evolution until X generations
generations = 100


min_agent_games = 1
total_games = num_agents * min_agent_games + 100


replayDir = os.getcwd() + '\\replays'
eliteDir = os.getcwd() + '\\models\\elites'

def update_stats(i, j):
    games_played[i] += 1
    games_played[j] += 1
    replays = os.listdir(replayDir)
    current_replay_file = replays[0]
    with open(replayDir + '\\' + current_replay_file, 'r') as file:
        data = file.read()
        if (data.find('"winner":1') != -1):
            rewards[i] += 1
            rewards[j] -= 1
        elif (data.find('"winner":2') != -1):
            rewards[i] -= 1
            rewards[j] += 1
    os.remove(replayDir + '\\' + current_replay_file)
    
def choose_n_gen_elites(n):
    elites = os.listdir(eliteDir)
    ans = []
    indices = np.random.choice(len(elites), n)
    for index in indices:
        agent = TerminalAI()
        agent.load_state_dict(torch.load('models\\elites\\' + elites[index]))
        ans.append(agent)
    return ans
    

for generation in range(generations):
    rewards = np.zeros(num_agents)
    games_played = np.zeros(num_agents)
    print(" Generation {} start".format(generation))
    for i in range(len(agents)):
    
        matchups = np.random.choice(len(agents), min_agent_games, replace=False)
        agent1 = agents[i]
        torch.save(agent1.state_dict(), 'models\\temp_model_1')
        for j in matchups:
            print("playing mandatory game: vs agent{} for agent {}".format(j, i))
            agent2 = agents[j]
            torch.save(agent2.state_dict(), 'models\\temp_model_2')
            run_single_game()
            update_stats(i, j)
                    
    for game in range(total_games - min_agent_games * num_agents):
        match_pairing = np.random.choice(len(agents), 2, replace=False) 
         print("playing random game: agent {} vs agent {}".format(match_pairing[0], match_pairing[1]))
        # Running a single game
        agent1 = agents[match_pairing[0]]
        agent2 = agents[match_pairing[1]]
        torch.save(agent1.state_dict(), 'models\\temp_model_1')
        torch.save(agent2.state_dict(), 'models\\temp_model_2')
        run_single_game()
        update_stats(i, j)
            
    # sort by rewards
    reward_ratio = rewards / games_played
    sorted_parent_indexes = np.argsort(reward_ratio)[::-1][:top_limit] #reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
    
    top_rewards = []
    new_pop = []
    
    for best_parent in sorted_parent_indexes:
        top_rewards.append(reward_ratio[best_parent])
        new_pop.append(agents[best_parent])
        torch.save(agents[best_parent], 'models\\elites\\generation_' + generation + '_' + top_rewards[0])
    new_pop_array = new_pop.copy()
    
    print("Generation ", generation, " | Mean rewards: ", np.mean(reward_ratio), " | Mean of top 5: ",np.mean(top_rewards[:5]))
    print("Top ",top_limit," scorers", sorted_parent_indexes)
    print("Rewards for top: ", top_rewards)
    
    # Save the top performing agent
    torch.save(agents[sorted_parent_indexes[0]], 'models\\elites\\generation_' + generation + '_' + top_rewards[0])

    # fill the new generation with children
    children = []
    for i in range(num_agents * 8 / 10):
        children.append(mutate(new_pop_array[i % top_limit]))
    new_pop = new_pop + children
    new_pop = new_pop + return_random_agents(num_agents / 10)

    elites = os.listdir(eliteDir)
    num_old_elites = num_agents * 2 / 25
    if len(elites) < num_old_elites:
        for i in range(num_old_elites):
            new_pop.append(mutate(new_pop_array[i % top_limit], 0.05))
    else:
        new_pop = new_pop + choose_n_gen_elites(num_old_elites)

    # kill all agents, and replace them with their children
    agents = new_pop
