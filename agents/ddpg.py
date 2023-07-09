import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            # nn.Tanh()
        )
        self.initialize_weights()

    def forward(self, state):
        # print(self.net(state), state)
        # return self.net(state) * state
        return self.net(state)
    
    # def forward_without_max_action(self, state):
    #     return self.net(state)

    def act(self, state):
        with torch.no_grad():
            action = self.forward(state).cpu().data.numpy().flatten()
        return action
  
    def act_with_noise(self, state, std):
        with torch.no_grad():
            action = self.forward(state).cpu().data.numpy().flatten()
        action_with_exp = np.random.normal(action, np.abs(std * action))
        # action_with_exp = action_with_exp * state.data.numpy()

        return action_with_exp.clip(0, state)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.initialize_weights()

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
  
class DDPG(object):
  def __init__(self, state_dim, action_dim, 
               lr, gamma, tau, batch_size, device=torch.device('cpu')):
    self.actor = Actor(state_dim, action_dim).to(device)
    self.actor_target = Actor(state_dim, action_dim).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)

    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

    self.device = device
    self.gamma = gamma
    self.tau = tau
    self.batch_size = batch_size
    self.epsilon = 1

  def select_action(self, state):
    state = torch.FloatTensor(state).reshape(1,-1).to(self.device)
    return self.actor.act(state)
  
  def select_action_with_noise(self, state, std, epsilon_min, decay):
    state = torch.FloatTensor(state).reshape(1,-1).to(self.device)
    self.epsilon = max(self.epsilon * decay, epsilon_min)
    eps = np.random.random()
    if eps < self.epsilon:
      return self.actor.act_with_noise(state, std)
    else:
      return self.actor.act(state)
  
  def update(self, replay_buffer, iterations):
    for _ in range(iterations):
      # Sample replay buffer
      s, a, r = replay_buffer.sample(self.batch_size)
      state = torch.FloatTensor(s).to(self.device)
      action = torch.FloatTensor(a).to(self.device)
      reward = torch.FloatTensor(r).to(self.device)

      # Compute the target Q value
      state = state.unsqueeze(1)
      action = action.unsqueeze(1)
      target_Q = reward

      # Get current Q estimate
      current_Q = self.critic(state, action)

      # Compute critic loss
      critic_loss = F.mse_loss(current_Q, target_Q)

      # Optimize the critic
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      # Compute actor loss
      actor_loss = -self.critic(state, self.actor(state)).mean()

      # Optimize the actor
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      # Update the frozen target models
      for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

      for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

  def save(self, filename):
    torch.save(self.actor.state_dict(), filename + "_actor.pt")
    torch.save(self.critic.state_dict(), filename + "_critic.pt")

  def load(self, filename):
    self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
    self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.critic_target.load_state_dict(self.critic.state_dict())

  def set_eval(self):
    self.actor.eval()
    self.actor_target.eval()
    self.critic.eval()
    self.critic_target.eval()