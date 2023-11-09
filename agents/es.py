import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_agents, num_actions):
        super(Net, self).__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.fc1 = nn.Linear(num_agents, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_actions)

        self.xavier_init()
        for param in self.parameters():
            param.data = param.data.to(torch.float64)


    def forward(self, s):
        assert torch.all(s > 0)
        with torch.no_grad():
            a = F.relu(self.fc1(s))
            a = F.relu(self.fc2(a))
            a = self.fc3(a)
            a = torch.min(torch.max(a, torch.zeros_like(a)), s)
        return a


    def xavier_init(self):
        for param in self.parameters():
            if len(param.data.shape) == 2:
                nn.init.xavier_normal_(param.data)

    
    def kaiming_init(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            if hasattr(layer, "weight"):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.utils import fix_seed

    fix_seed(3407)
    net = Net(num_agents=1, num_actions=1)

    # for param in net.parameters():
    #     param.data += 0.06707526

    print(net(torch.FloatTensor([655.0]).to(torch.float64)))
