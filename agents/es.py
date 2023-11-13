import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import warpper


class Net(nn.Module):
    def __init__(self, num_agents, num_actions):
        super(Net, self).__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.fc1 = nn.Linear(num_agents, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_actions)
        self.adam_param = {
            "t": 0,
            "m": torch.tensor(0),
            "v": torch.tensor(0),
            "m_hat": torch.tensor(0),
            "v_hat": torch.tensor(0),
        }

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

    
    def update(self, grad, alpha, beta, eps=1e-8):
        beta1, beta2 = beta
        grad = warpper(grad)

        self.adam_param['t'] += 1
        self.adam_param['m'] = beta1 * self.adam_param['m'] + (1 - beta1) * grad
        self.adam_param['v'] = beta2 * self.adam_param['v'] + (1 - beta2) * grad ** 2
        self.adam_param['m_hat'] = self.adam_param['m'] / (1 - beta1 ** (self.adam_param['t']))
        self.adam_param['v_hat'] = self.adam_param['v'] / (1 - beta2 ** (self.adam_param['t']))

        delta_param = alpha * self.adam_param['m_hat'] / (torch.sqrt(self.adam_param['v_hat']) + eps)

        with torch.no_grad():
            for param, delta in zip(self.parameters(), delta_param):
                param.data.add_(delta)


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(Net, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict.update({prefix + 'adam_param.' + key: value for key, value in self.adam_param.items()})
        return state_dict


    def load_state_dict(self, state_dict, strict=True):
        adam_param_state_dict = {key.replace('adam_param.', ''): value for key, value in state_dict.items() if 'adam_param.' in key}
        self.adam_param.update(adam_param_state_dict)
        state_dict = {key: value for key, value in state_dict.items() if 'adam_param.' not in key}
        super(Net, self).load_state_dict(state_dict, strict)


    def xavier_init(self):
        for param in self.parameters():
            if len(param.data.shape) == 2:
                nn.init.xavier_normal_(param.data)

    
    def kaiming_init(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            if hasattr(layer, "weight"):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')


if __name__ == '__main__':
    from utils.utils import fix_seed

    fix_seed(3407)
    net = Net(num_agents=1, num_actions=1)

    for param in net.named_parameters():
        print(param[0], param[1])

    print("*********")
    print(net.state_dict())
