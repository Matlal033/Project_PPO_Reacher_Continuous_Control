import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

from ActorCritic import ActorCritic
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from tqdm import tqdm

class collect_trajectories:
    def __init__(self, batch_size, num_agents):
        self.l_states = []
        self.l_actions = []
        self.l_log_probs_old = []
        self.l_rewards = []
        self.l_values = []
        self.l_dones = []
        self.batch_size = batch_size
        self.num_agents = num_agents

    def clear_trajectories(self):
        self.l_states = []
        self.l_actions = []
        self.l_log_probs_old = []
        self.l_rewards = []
        self.l_values = []
        self.l_dones = []

    def add_to_trajectory(self, states, actions, log_probs_old, rewards, values, dones):
        self.l_states.append(states)
        self.l_actions.append(actions)
        self.l_log_probs_old.append(log_probs_old)
        self.l_rewards.append(rewards)
        self.l_values.append(values)
        self.l_dones.append(dones)

    def return_trajectories(self):
        return (self.l_states), \
            (self.l_actions), \
            (self.l_log_probs_old), \
            (self.l_rewards), \
            (self.l_values), \
            (self.l_dones)

    def return_batches_idx(self):
        qty_states = (len(self.l_states)-1) * self.num_agents
        all_idx = np.arange(0, qty_states)
        np.random.shuffle(all_idx)
        batches_first_idx = np.arange(0, qty_states, self.batch_size)

        batches_idx = []
        for i in batches_first_idx:
            batches_idx.append(all_idx[i:i+self.batch_size])

        return batches_idx

class Agent:
    def __init__(self, action_size, state_size, device, episode=1000, discount_rate=.99,
        tau=0.95, surrogate_clip=0.2, beta=0.1, tmax=1000, SGD_epoch=1,
        LR=3e-4, adam_epsilon=1e-5, batch_size=2000, hidden_size=512,num_agents=20, gradient_clip=5):

        self.discount_rate = discount_rate
        self.surrogate_clip = surrogate_clip
        self.qty_epoch = SGD_epoch
        self.tau = tau
        self.trajectories = collect_trajectories(batch_size, num_agents)

        self.ActorCritic = ActorCritic(state_size, action_size, hidden_size, LR, adam_epsilon).to(device)
        self.device = device
        self.num_agents = num_agents
        self.gradient_clip = gradient_clip

    def add_to_trajectory(self, states, actions, log_probs_old, rewards, values, dones):
        self.trajectories.add_to_trajectory(states, actions, log_probs_old, rewards, values, dones)

    def select_action(self, states):
        states = torch.tensor(states,dtype=torch.float,device=self.device)

        a_dist, values = self.ActorCritic(states)
        actions = a_dist.sample().detach()
        #actions = actions.clamp(-1,1)

        values = values.detach()

        log_probs_old = a_dist.log_prob(actions)
        log_probs_old = torch.sum(log_probs_old, dim=1, keepdim=True)

        return actions, log_probs_old, values

    def learn(self, surrogate_clip, beta):

        states,\
        actions,\
        log_probs_old,\
        rewards,\
        values,\
        dones = \
            self.trajectories.return_trajectories()

        processed = [None] * (len(states)-1)
        r_advantages = torch.tensor(np.zeros((self.num_agents,1))).to(self.device)

        last_states = torch.tensor(states[len(states)-1], dtype = torch.float).to(self.device)
        _, r_returns = self.ActorCritic(last_states) #return from last state is value at last step
        r_returns = r_returns.unsqueeze(1).detach()

        for r in reversed(range(len(states)-1)):
            r_dones = torch.tensor(dones[r]).unsqueeze(1).to(self.device)
            r_rewards = torch.tensor(rewards[r]).unsqueeze(1).to(self.device)
            r_actions = torch.tensor(actions[r]).to(self.device)
            r_states = torch.tensor(states[r], dtype = torch.float).to(self.device)
            r_value = torch.tensor(values[r]).unsqueeze(1).to(self.device)
            r_next_value = torch.tensor(values[r+1]).unsqueeze(1).to(self.device)
            r_log_probs_old = torch.tensor(log_probs_old[r]).to(self.device)

            r_returns = r_rewards + self.discount_rate * r_returns  * r_dones
            td_error = r_rewards + self.discount_rate * r_dones * r_next_value - r_value
            r_advantages = r_advantages * self.tau * self.discount_rate * r_dones + td_error

            processed[r] = [\
                r_states, \
                r_actions, \
                r_log_probs_old, \
                r_advantages, \
                r_returns]

        #Normalizing advantages
        states, actions, log_probs_old, advantages, returns = map(lambda x: torch.cat(x, dim=0), zip(*processed))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for e in range(self.qty_epoch):
            batches_idx = self.trajectories.return_batches_idx()
            n_batches = 0
            batch_avg_critic_loss = 0
            batch_avg_actor_loss = 0
            for batch_idx in batches_idx:
                batch_idx = torch.tensor(batch_idx).long().to(self.device)
                sampled_states = states[batch_idx]
                sampled_old_log_probs = log_probs_old[batch_idx]
                sampled_actions = actions[batch_idx]
                sampled_advantages = advantages[batch_idx]
                sampled_returns = returns[batch_idx]

                dist, critic_value = self.ActorCritic(sampled_states)

                sampled_new_log_probs = dist.log_prob(sampled_actions)
                sampled_new_log_probs= torch.sum(sampled_new_log_probs, dim=1, keepdim=True)

                ratio = (sampled_new_log_probs-sampled_old_log_probs).exp()

                obj = ratio*sampled_advantages
                obj_clip = ratio.clamp(1.0-surrogate_clip, 1.0+surrogate_clip) * sampled_advantages

                clipped_sur_neg = -torch.min(obj,obj_clip).mean(0)

                entropy = dist.entropy().mean()
                actor_loss = clipped_sur_neg - (beta*entropy)
                critic_loss = 0.5*(sampled_returns - critic_value).pow(2).mean()

                self.ActorCritic.optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                nn.utils.clip_grad_norm_(self.ActorCritic.parameters(), self.gradient_clip)
                self.ActorCritic.optimizer.step()

        #clear memory
        self.trajectories.clear_trajectories()
