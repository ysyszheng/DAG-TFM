import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DoubleNN(nn.Module):
  def __init__(self, num_agents, num_actions, lr_1, lr_2):
    super(DoubleNN, self).__init__()
    self.prob_net = nn.Sequential(
      nn.Linear(num_agents, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, num_actions),
      nn.Tanh(),
    )
    self.fee_net = nn.Sequential(
      nn.Linear(num_agents, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, num_actions),
      nn.Softplus(),
    )
    self.optimizer1 = torch.optim.Adam(self.prob_net.parameters(), lr=lr_1)
    self.optimizer2 = torch.optim.Adam(self.fee_net.parameters(), lr=lr_2)
    self.prob_loss = None
    self.rev_loss = None

  def forward(self, x, m):
    fee = self.fee_net(x)
    prob = self.prob_net(fee.detach())
    with torch.no_grad():
      prob_ = self.prob_net(fee)
    rev = (1 - (1 - prob_) ** m) * (x - fee)
    return prob, fee, rev

  def calculate_prob(self, x):
    return self.prob_net(x)

  def calculate_fee(self, x):
    return self.fee_net(x)

  def learn(self, prob, true_prob, rev, opt_rev):
    self.prob_loss = torch.mean(torch.max(torch.abs(prob - true_prob), dim=1).values)
    self.rev_loss = torch.mean(torch.max(torch.div((rev - opt_rev), rev + 1e-8), dim=1).values)
    
    self.optimizer1.zero_grad()
    self.prob_loss.backward()
    self.optimizer1.step()

    self.optimizer2.zero_grad()
    self.rev_loss.backward()
    self.optimizer2.step()
