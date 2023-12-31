import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rollout_buffer import RolloutBuffer
from utils.utils import log


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, std_init):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softplus(),
        )
        self.std = std_init

    def forward(self, s):
        mu = self.net(s)
        dist = torch.distributions.Normal(mu, self.std)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        return a, log_prob

    def act_without_exploration(self, s):
        return self.net(s)

    def evaluate(self, s, a):
        mu = self.net(s)
        dist = torch.distributions.Normal(mu, self.std)
        log_prob = dist.log_prob(a)
        entropy = dist.entropy()
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, s):
        return self.net(s)
    

class PPO(object):
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, c1, c2, 
                 K_epochs, gamma, eps_clip, std_init, std_decay, std_min, device=torch.device('cpu')):
        self.actor = Actor(state_dim, action_dim, std_init).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ])
        self.c1 = c1
        self.c2 = c2
        self.K_epochs = K_epochs
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.device = device
        self.std_decay = std_decay
        self.std_min = std_min
        self.buffer = RolloutBuffer()


    def decay_action_std(self):
        self.actor.std = max(self.actor.std * self.std_decay, self.std_min)


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        a, log_prob = self.actor(state)
        sv = self.critic(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(a)
        self.buffer.logprobs.append(log_prob)
        self.buffer.state_values.append(sv)
        return a.cpu().data.numpy().flatten()

    def select_action_without_exploration(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            a = self.actor.act_without_exploration(state)
        return a.cpu().data.numpy().flatten()

    def update(self):
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) # ? normalize rewards or not
        rewards = rewards.reshape(-1, 1)
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(self.device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(self.device)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.K_epochs):
            logprobs, dist_entropy = self.actor.evaluate(old_states, old_actions)
            state_values = self.critic(old_states)

            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            mse_loss = nn.MSELoss()(state_values, rewards)
            loss = -torch.min(surr1, surr2) + self.c1 * mse_loss - self.c2 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.buffer.clear()
