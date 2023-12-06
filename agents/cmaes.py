import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal as N


class Net(nn.Module):
    def __init__(self, num_agents, num_actions, hidden_layer_size=(16,16)):
        super(Net, self).__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions

        self.fc1 = nn.Linear(num_agents, hidden_layer_size[0])
        self.fc2 = nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        self.fc3 = nn.Linear(hidden_layer_size[1], num_actions)
        # self.fc_mean = nn.Linear(hidden_layer_size[1], num_actions)
        # self.fc_std = nn.Linear(hidden_layer_size[1], num_actions)

        torch.set_grad_enabled(False)

    def forward(self, s):
        assert torch.all(s > 0)

        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        a = x * s
        # mean = self.fc_mean(x)
        # raw_std = self.fc_std(x)
        # std = F.softplus(raw_std) + 1e-8

        # dist = N(mean, std)
        # a = dist.sample()
        return a


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.utils import fix_seed

    net = Net(num_agents=1, num_actions=1)
    net.load_state_dict(torch.load(r'./results/models/cmaes.pth'))
    print(sum(p.numel() for p in net.parameters()))
    for p in net.named_parameters():
        print(p)
