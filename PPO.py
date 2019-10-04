import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import glob
import os
import subprocess
import sys
import json

from model import ActorCritic


def run_single_game(process_command):
    print("Start run a match")
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

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
        return total_reward
                                                                                            
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns
                                            
def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)
            
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            
            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


#Hyper params:
hidden_size      = 256
lr               = 3e-4
num_steps        = 20
mini_batch_size  = 5
ppo_epochs       = 4
threshold_reward = -200

model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

frame_idx  = 0
test_rewards = []


N_GAMES = 100

for game_idx in range(N_GAMES):
    run_single_game("cd {} && java -jar engine.jar work {} {}".format(".", "algo_strategy_ppo.py", "algo_strategy_ppo.py"))

    list_of_action_replays = glob.glob("action_replay/*.pickle")
    latest_action = max(list_of_action_replays, key=os.path.getctime)
    with open(latest_action):
        actions = pickle.load(latest_action)
    list_of_files = glob.glob("replays/*.replay") # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    assert latest_file.endswith(".replay")
    with open(latest_file) as f:
        for line in f:
            line = line.replace("\n", "")
            line = line.replace("\t", "")
            if line != "":
                states.push(json.loads(line))

    assert len(states) == len(actions), "Found {} states in replay file, but {} actions in action replay file."

    log_probs = []
    values    = []
    states    = []
    rewards   = []
    masks     = []
    entropy = 0

    for i, (state, action) in enumerate(zip(states, actions)):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)
        
        reward = 0 # compute reward here, using the next state if needed
        done = 1 if i == len(states) - 1 else 0
        # next_state, reward, done, _ = envs.step(action.cpu().numpy())
        
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        # states.append(state)
        # actions.append(action)
        
        # state = next_state
        # frame_idx += 1
                    

    # next_state = torch.FloatTensor(next_state).to(device)
    # _, next_value = model(next_state)
    # use last value
    returns = compute_gae(value, rewards, masks, values)
    
    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values
    
    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

    torch.save(model.state_dict(), "run/weights")

    if game_idx % 1 == 0:
        print("Completed game {}/{}, total reward = {}".format(game_idx + 1, N_GAMES, total_reward))
        # test_reward = np.mean([test_env() for _ in range(10)])
        # test_rewards.append(test_reward)
        # plot(frame_idx, test_rewards)
        # if test_reward > threshold_reward: early_stop = True
            
