import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, num_agents, num_actions, lr=1e-3):
        super(Net, self).__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.fc1 = nn.Linear(num_agents, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = None
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def learn(self, actions, optimal_actions):
        loss = torch.max(torch.abs(actions - optimal_actions)/actions, dim=1).values
        loss = torch.mean(loss)
        
        self.loss = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
