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
import copy

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

def run_single_game(flip=False):
    if flip:
        agent2 = ".\\run_gene1.ps1"
        agent1 = ".\\run_gene2.ps1"

    else:
        agent1 = ".\\run_gene1.ps1"
        agent2 = ".\\run_gene2.ps1"
    process_command = "java -jar engine.jar work {} {}".format(agent1, agent2)
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
num_agents = 50
agents = return_random_agents(num_agents)

# How many top agents to consider as parents
top_limit = 3

# run evolution until X generations
generations = 10


min_agent_games = 5
additional_games = 0
total_games = num_agents * min_agent_games + additional_games


replayDir = os.getcwd() + '\\replays'
eliteDir = os.getcwd() + '\\models\\elites'
f = open(os.getcwd() + "\\models\\training_log.txt", 'w+')
f.write("Starting Genetic Evolution. Population size: {} Generations: {}\n".format(num_agents, generations))
f.close()

def update_stats(i, j):
    games_played[i] += 1
    games_played[j] += 1
    replays = os.listdir(replayDir)
    current_replay_file = replays[0]
    with open(replayDir + '\\' + current_replay_file, 'r') as file:
        data = file.read()
        turns_indicator = '"turns":'
        start_index = data.find(turns_indicator) + len(turns_indicator)
        rest_string = data[start_index:]
        end_index = rest_string.find('}')
        turns_end = int(data[start_index:start_index + end_index])
        if (data.find('"winner":1') != -1):
            rewards[i] += 1 / turns_end
            rewards[j] -= 1 / turns_end
        elif (data.find('"winner":2') != -1):
            rewards[i] -= 1 / turns_end
            rewards[j] += 1 / turns_end
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
            print("playing mandatory game: agent {} vs agent {}".format(i, j))
            agent2 = agents[j]
            torch.save(agent2.state_dict(), 'models\\temp_model_2')
            run_single_game(False)
            update_stats(i, j)
                    
    for game in range(total_games - min_agent_games * num_agents):
        match_pairing = np.random.choice(len(agents), 2, replace=False) 
        print("playing random game: agent {} vs agent {}".format(match_pairing[0], match_pairing[1]))
        # Running a single game
        agent1 = agents[match_pairing[0]]
        agent2 = agents[match_pairing[1]]
        torch.save(agent1.state_dict(), 'models\\temp_model_1')
        torch.save(agent2.state_dict(), 'models\\temp_model_2')
        run_single_game(False)
        update_stats(i, j)
            
    # sort by rewards
    reward_ratio = rewards
    sorted_parent_indexes = np.argsort(reward_ratio)[::-1][:top_limit] #reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
    
    top_rewards = []
    new_pop = []
    
    for best_parent in sorted_parent_indexes:
        top_rewards.append(reward_ratio[best_parent])
        new_pop.append(agents[best_parent])
        torch.save(agents[best_parent], 'models\\elites\\generation_' + str(generation) + '_' + str(top_rewards[0]))
    new_pop_array = new_pop.copy()
    
    file = open(os.getcwd() + "\\models\\training_log.txt", 'a+')
    file.write("Generation {} | Mean rewards: {} | Mean of top 5: {}\n".format(generation, np.mean(reward_ratio), np.mean(top_rewards[:5])))
    file.write("Top {} scorers: {} \n".format(top_limit, sorted_parent_indexes))
    file.write("Rewards for top: {} \n".format(top_rewards))
    file.close()
    
    # Save the top performing agent
    #torch.save(agents[sorted_parent_indexes[0]], 'models\\elites\\generation_' + generation + '_' + top_rewards[0])

    print("Creating new population")
    # fill the new generation with children
    children = []
    print("Generating descendants")
    for i in range(int(round(num_agents * 8 / 10))):
        children.append(mutate(new_pop_array[i % top_limit]))
    new_pop = new_pop + children
    print("Adding Random Agents")
    new_pop = new_pop + return_random_agents(int(round(num_agents / 10)))
    
    elites = os.listdir(eliteDir)
    
    num_old_elites = num_agents - len(new_pop)
    print("Adding Generational Elites")
    if len(elites) < num_old_elites:
        for i in range(num_old_elites):
            new_pop.append(mutate(new_pop_array[i % top_limit], 0.05))
    else:
        new_pop = new_pop + choose_n_gen_elites(num_old_elites)
    # kill all agents, and replace them with their children
    agents = new_pop
