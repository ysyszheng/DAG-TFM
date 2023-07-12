import numpy as np
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
            nn.Softplus()
        )
        self.initialize_weights()

    def forward(self, state):
        return self.net(state)

    def act(self, state):
        with torch.no_grad():
            action = self.forward(state).cpu().data.numpy().flatten().clip(0, state)
        return action

    def act_with_exp(self, state, std):
        with torch.no_grad():
            action = self.forward(state).cpu().data.numpy().flatten()

        action_with_exp = np.random.normal(action, np.abs(std * state))
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
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr,
                 gamma,
                 tau,
                 batch_size,
                 epsilon_min,
                 epsilon_decay,
                 device=torch.device('cpu')):
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
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        return self.actor.act(state)

    def select_action_with_exp(self, state, std):
        state = torch.FloatTensor(state).reshape(1, -1).to(self.device)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return self.actor.act_with_exp(state, std) \
          if np.random.rand() > self.epsilon else self.actor.act(state)

    def update(self, replay_buffer, iterations):
        for _ in range(iterations):
            s, a, r = replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(s).to(self.device)
            action = torch.FloatTensor(a).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            state = state.unsqueeze(1)
            action = action.unsqueeze(1)
            target_Q = reward
            current_Q = self.critic(state, action)

            critic_loss = F.mse_loss(current_Q, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

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
